import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
from datetime import datetime

from structure_model.model_factory import create_model

logger = logging.getLogger(__name__)

class ModelComparator:
    """
    Clase para comparar múltiples modelos PINN en términos de rendimiento y métricas.
    Permite listar modelos disponibles, evaluarlos y generar informes comparativos.
    """
    def __init__(self, output_dir, evaluator):
        """
        Args:
            output_dir (str): Directorio donde guardar los resultados de la comparación
            evaluator: Instancia de ModelEvaluator para evaluar modelos individuales
        """
        self.output_dir = output_dir
        self.evaluator = evaluator
        self.model_catalog = {}
        self.comparison_results = {}
        os.makedirs(output_dir, exist_ok=True)
        
    def scan_for_models(self, models_root_dir):
        """
        Escanea recursivamente un directorio para encontrar modelos entrenados,
        organizados por problema físico.
        
        Args:
            models_root_dir (str): Directorio raíz donde buscar modelos
            
        Returns:
            dict: Catálogo de modelos por problema físico
        """
        self.model_catalog = {}
        
        # Buscar directorios de modelos (contienen model/weights_model.pth)
        model_dirs = []
        for root, dirs, files in os.walk(models_root_dir):
            if "model" in dirs and os.path.isfile(os.path.join(root, "model", "weights_model.pth")):
                model_dirs.append(root)
                
        # Clasificar modelos por problema físico
        for model_dir in model_dirs:
            # Intentar leer los hyperparams de entrenamiento para identificar el problema físico
            hyperparams_path = os.path.join(model_dir, "train_results", "hyperparams_train.json")
            if not os.path.exists(hyperparams_path):
                continue
                
            try:
                with open(hyperparams_path, 'r') as f:
                    hyperparams = json.load(f)
                
                # Identificar problema físico (si existe en los hyperparams)
                physics_type = hyperparams.get("physics", {}).get("physics_type", "unknown")
                
                # Leer estadísticas de entrenamiento
                stats_path = os.path.join(model_dir, "train_results", "training_stats.json")
                stats = {}
                if os.path.exists(stats_path):
                    with open(stats_path, 'r') as f:
                        stats = json.load(f)
                
                # Obtener nombre del modelo de la estructura del directorio
                model_name = os.path.basename(model_dir)
                
                # Guardar información del modelo
                model_info = {
                    "path": os.path.join(model_dir, "model", "weights_model.pth"),
                    "hyperparams": hyperparams,
                    "stats": stats,
                    "name": model_name
                }
                
                # Añadir al catálogo por problema físico
                if physics_type not in self.model_catalog:
                    self.model_catalog[physics_type] = []
                self.model_catalog[physics_type].append(model_info)
            
            except Exception as e:
                logger.warning(f"Error al analizar modelo en {model_dir}: {str(e)}")
        
        return self.model_catalog
    
    def list_available_models(self):
        """
        Muestra los modelos disponibles organizados por problema físico.
        
        Returns:
            dict: Catálogo de modelos
        """
        if not self.model_catalog:
            logger.warning("No hay modelos catalogados. Ejecute primero scan_for_models().")
            return {}
        
        # Imprimir resumen de los modelos disponibles
        total_models = 0
        print("\n=== MODELOS DISPONIBLES POR PROBLEMA FÍSICO ===")
        
        for physics_type, models in self.model_catalog.items():
            print(f"\n-- {physics_type.upper()} ({len(models)} modelos) --")
            
            for i, model in enumerate(models, 1):
                name = model["name"]
                accuracy = model["stats"].get("final_accuracy", "N/A")
                train_time = model["stats"].get("total_training_time_sec", "N/A")
                
                print(f"{i}. {name}")
                print(f"   Precisión: {accuracy}")
                print(f"   Tiempo de entrenamiento: {train_time}s")
            
            total_models += len(models)
        
        print(f"\nTotal: {total_models} modelos encontrados en {len(self.model_catalog)} problemas físicos.")
        return self.model_catalog
    
    def evaluate_all_models(self, data_path, nu):
        """
        Evalúa todos los modelos disponibles con un conjunto de datos común.
        
        Args:
            data_path (str): Ruta al archivo de datos para evaluación
            nu (float): Coeficiente de viscosidad para la evaluación
            
        Returns:
            pd.DataFrame: Tabla comparativa con resultados
        """
        if not self.model_catalog:
            logger.warning("No hay modelos catalogados. Ejecute primero scan_for_models().")
            return None
        
        # Preparar estructura para resultados
        results = []
        
        # Evaluar cada modelo
        for physics_type, models in self.model_catalog.items():
            for model_info in models:
                try:
                    model_path = model_info["path"]
                    model_name = model_info["name"]
                    
                    logger.info(f"Evaluando modelo: {model_name} ({physics_type})")
                    
                    # Evaluar usando el evaluator proporcionado
                    metrics = self.evaluator.evaluate_model(
                        model_path=model_path,
                        data_path=data_path,
                        nu=nu,
                        model_name=model_name
                    )
                    
                    # Añadir información del modelo y tipo físico
                    metrics["physics_type"] = physics_type
                    metrics["model_name"] = model_name
                    
                    # Extraer información de hiperparámetros relevantes
                    hyperparams = model_info["hyperparams"]
                    if "models" in hyperparams and "selected_model" in hyperparams:
                        selected_model = hyperparams["selected_model"]
                        model_config = hyperparams["models"].get(selected_model, {})
                        metrics["layers"] = str(model_config.get("layers", ""))
                        metrics["activation"] = model_config.get("activation_function", "")
                    
                    # Añadir a resultados
                    results.append(metrics)
                    
                except Exception as e:
                    logger.error(f"Error al evaluar {model_name}: {str(e)}")
        
        # Convertir a DataFrame
        if results:
            df_results = pd.DataFrame(results)
            
            # Guardar como CSV
            csv_path = os.path.join(self.output_dir, f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df_results.to_csv(csv_path, index=False)
            logger.info(f"Comparación guardada en {csv_path}")
            
            # Mostrar tabla comparativa
            self._generate_comparison_report(df_results)
            
            return df_results
        else:
            logger.warning("No se pudo evaluar ningún modelo.")
            return None
    
    def _generate_comparison_report(self, df_results):
        """
        Genera reporte visual comparativo entre modelos evaluados.
        
        Args:
            df_results (pd.DataFrame): Resultados de la evaluación
        """
        if df_results is None or len(df_results) == 0:
            logger.warning("No hay resultados para generar informe comparativo.")
            return
            
        # Crear gráficos comparativos
        plt.figure(figsize=(12, 10))
        
        # 1. Comparar MSE por modelo
        plt.subplot(2, 2, 1)
        ax = df_results.sort_values("MSE")["MSE"].plot(kind="bar", color="skyblue")
        plt.xlabel("Modelo")
        plt.ylabel("MSE")
        plt.title("Error Cuadrático Medio por Modelo")
        ax.set_xticklabels(df_results.sort_values("MSE")["model_name"], rotation=90)
        
        # 2. Comparar tiempo de inferencia
        plt.subplot(2, 2, 2)
        ax = df_results.sort_values("inference_time")["inference_time"].plot(kind="bar", color="lightgreen")
        plt.xlabel("Modelo")
        plt.ylabel("Tiempo (s)")
        plt.title("Tiempo de Inferencia por Modelo")
        ax.set_xticklabels(df_results.sort_values("inference_time")["model_name"], rotation=90)
        
        # 3. Gráfico de dispersión MSE vs Tiempo
        plt.subplot(2, 2, 3)
        plt.scatter(df_results["MSE"], df_results["inference_time"], alpha=0.7)
        
        for i, model in enumerate(df_results["model_name"]):
            plt.annotate(model, 
                         (df_results["MSE"].iloc[i], df_results["inference_time"].iloc[i]),
                         fontsize=8)
        plt.xlabel("MSE")
        plt.ylabel("Tiempo de Inferencia (s)")
        plt.title("MSE vs Tiempo de Inferencia")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 4. Comparativa de métricas por física
        plt.subplot(2, 2, 4)
        physics_types = df_results["physics_type"].unique()
        if len(physics_types) > 1:
            pivot = df_results.pivot_table(values="MSE", index="physics_type", aggfunc="mean")
            pivot.plot(kind="bar", color="coral")
            plt.xlabel("Tipo de Física")
            plt.ylabel("MSE Promedio")
            plt.title("MSE Promedio por Tipo de Física")
        else:
            # Si solo hay un tipo de física, mostrar un diagrama de caja con las métricas clave
            boxplot_data = df_results[["MSE", "MAE", "R2"]]
            boxplot_data.plot(kind="box", title="Distribución de Métricas")
        
        plt.tight_layout()
        viz_path = os.path.join(self.output_dir, "model_comparison_metrics.png")
        plt.savefig(viz_path, dpi=300)
        plt.close()
        
        # Generar también una tabla resumida
        summary = df_results.pivot_table(
            values=["MSE", "MAE", "inference_time", "memory_usage_mb"],
            index=["physics_type"],
            aggfunc={"MSE": ["mean", "min", "max"],
                     "MAE": ["mean", "min", "max"],
                     "inference_time": ["mean", "min"],
                     "memory_usage_mb": ["mean"]}
        )
        
        # Guardar resumen como CSV
        summary_csv = os.path.join(self.output_dir, "model_comparison_summary.csv")
        summary.to_csv(summary_csv)
        
        # Guardar DataFrame completo como CSV para referencia
        full_results_csv = os.path.join(self.output_dir, "model_comparison_full.csv")
        df_results.to_csv(full_results_csv, index=False)
        
        # Generar un reporte HTML más completo
        try:
            html_report = os.path.join(self.output_dir, "model_comparison_report.html")
            
            # Crear un informe HTML básico
            with open(html_report, 'w') as f:
                f.write("<html><head>")
                f.write("<title>Informe de Comparación de Modelos</title>")
                f.write("<style>")
                f.write("body { font-family: Arial, sans-serif; margin: 20px; }")
                f.write("table { border-collapse: collapse; width: 100%; }")
                f.write("th, td { text-align: left; padding: 8px; border: 1px solid #ddd; }")
                f.write("th { background-color: #f2f2f2; }")
                f.write("tr:nth-child(even) { background-color: #f9f9f9; }")
                f.write("img { max-width: 100%; height: auto; }")
                f.write("</style>")
                f.write("</head><body>")
                f.write("<h1>Informe de Comparación de Modelos</h1>")
                
                # Top 3 modelos
                f.write("<h2>Top 3 Mejores Modelos</h2>")
                f.write(df_results.sort_values("MSE").head(3).to_html(index=False))
                
                # Gráfico comparativo
                f.write("<h2>Visualizaciones Comparativas</h2>")
                f.write(f'<img src="{os.path.basename(viz_path)}" alt="Gráfico Comparativo">')
                
                # Tabla completa
                f.write("<h2>Todos los Modelos Evaluados</h2>")
                f.write(df_results.to_html(index=False))
                
                f.write("</body></html>")
                
            logger.info(f"Informe HTML generado en: {html_report}")
        except Exception as e:
            logger.warning(f"No se pudo generar el informe HTML: {str(e)}")
        
        logger.info("Informe comparativo generado exitosamente.")
