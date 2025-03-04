import os
import json
import yaml
import shutil
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class OutputManager:
    """
    Administrador centralizado para organizar los resultados del proyecto.
    Crea y gestiona una estructura de directorios consistente para todos los
    resultados de entrenamiento, evaluación y datos.
    """
    
    def __init__(self, project_root=None):
        """
        Inicializa el gestor de salidas.
        
        Args:
            project_root (str, optional): Directorio raíz del proyecto.
                                       Si es None, se detecta automáticamente.
        """
        if project_root is None:
            # Detectar la raíz del proyecto
            project_root = os.path.abspath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                "..", ".."
            ))
        self.project_root = project_root
        
        # Definir el directorio de salida principal
        self.output_root = os.path.join(project_root, "output")
        os.makedirs(self.output_root, exist_ok=True)
        
        # Directorios para cada tipo de problema físico
        self.physics_types = ["burgers", "kovasznay", "taylor_green", "cavity_flow", "other"]
        self.physics_dirs = {}
        
        # Crear directorios para cada tipo de física
        for physics_type in self.physics_types:
            physics_dir = os.path.join(self.output_root, physics_type)
            os.makedirs(physics_dir, exist_ok=True)
            self.physics_dirs[physics_type] = physics_dir
        
        logger.info(f"OutputManager inicializado. Directorio de salida principal: {self.output_root}")

    def create_model_dir(self, model_name, physics_type, epochs, lr, seed):
        """
        Crea un directorio para los resultados de un modelo específico.
        
        Args:
            model_name (str): Nombre del modelo (ej: "pinn_v1")
            physics_type (str): Tipo de problema físico (ej: "burgers")
            epochs (int): Número de épocas
            lr (float): Tasa de aprendizaje
            seed (int): Semilla aleatoria
            
        Returns:
            tuple: (output_dir, model_dir, train_dir) con las rutas completas
        """
        # Verificar si el tipo de física es válido, si no, usar "other"
        if physics_type not in self.physics_types:
            physics_type = "other"
            logger.warning(f"Tipo de física '{physics_type}' no reconocido. Usando 'other'.")
        
        # Crear nombre de directorio del modelo
        model_dirname = f"{physics_type}_{model_name}_{epochs}_{lr}_{seed}"
        output_dir = os.path.join(self.physics_dirs[physics_type], model_dirname)
        
        # Crear estructura de directorios
        os.makedirs(output_dir, exist_ok=True)
        model_dir = os.path.join(output_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        train_dir = os.path.join(output_dir, "train_results")
        os.makedirs(train_dir, exist_ok=True)
        
        # Crear archivo de entorno
        self._create_environment_file(output_dir)
        
        logger.info(f"Directorio de resultados creado: {output_dir}")
        return output_dir, model_dir, train_dir
    
    def create_evaluation_dir(self, model_name, physics_type, target_model_dir=None):
        """
        Crea un directorio para los resultados de evaluación.
        
        Args:
            model_name (str): Nombre del modelo evaluado
            physics_type (str): Tipo de problema físico
            target_model_dir (str, optional): Si se proporciona, los resultados se guardan
                                           dentro del directorio del modelo. Si no, se crea
                                           un directorio específico para la evaluación.
            
        Returns:
            str: Ruta al directorio de evaluación
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if target_model_dir and os.path.exists(target_model_dir):
            # Crear directorio de evaluación dentro del directorio del modelo
            eval_dir = os.path.join(target_model_dir, "evaluations", f"eval_{timestamp}")
        else:
            # Verificar tipo de física
            if physics_type not in self.physics_types:
                physics_type = "other"
            
            # Crear directorio independiente para la evaluación
            eval_dirname = f"eval_{model_name}_{timestamp}"
            eval_dir = os.path.join(self.physics_dirs[physics_type], "evaluations", eval_dirname)
        
        os.makedirs(eval_dir, exist_ok=True)
        
        # Crear archivo de entorno
        self._create_environment_file(eval_dir)
        
        logger.info(f"Directorio de evaluación creado: {eval_dir}")
        return eval_dir
    
    def _create_environment_file(self, directory):
        """
        Crea un archivo con información del entorno.
        
        Args:
            directory (str): Directorio donde crear el archivo
        """
        env_file = os.path.join(directory, "environment.txt")
        
        with open(env_file, 'w') as f:
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Directorio: {directory}\n")
            f.write(f"Proyecto: {self.project_root}\n\n")
            
            # Intentar obtener información del sistema
            try:
                import sys
                import platform
                import torch
                
                f.write(f"Python: {sys.version}\n")
                f.write(f"Sistema: {platform.system()} {platform.release()}\n")
                f.write(f"PyTorch: {torch.__version__}\n")
                f.write(f"CUDA disponible: {torch.cuda.is_available()}\n")
                if torch.cuda.is_available():
                    f.write(f"Dispositivo CUDA: {torch.cuda.get_device_name(0)}\n")
            except ImportError:
                f.write("No se pudo obtener información completa del sistema.\n")
    
    def generate_physics_summary(self, physics_type):
        """
        Genera un resumen de todos los modelos y evaluaciones para un tipo de física específico.
        
        Args:
            physics_type (str): Tipo de problema físico
            
        Returns:
            str: Ruta al archivo de resumen
        """
        if physics_type not in self.physics_types:
            logger.error(f"Tipo de física '{physics_type}' no reconocido.")
            return None
        
        physics_dir = self.physics_dirs[physics_type]
        
        # Recopilar información de modelos
        models_info = []
        model_dirs = [d for d in os.listdir(physics_dir) 
                     if os.path.isdir(os.path.join(physics_dir, d)) and d.startswith(physics_type)]
        
        for model_dir_name in model_dirs:
            model_dir = os.path.join(physics_dir, model_dir_name)
            
            # Buscar estadísticas de entrenamiento
            train_stats_file = os.path.join(model_dir, "train_results", "training_stats.json")
            train_stats = {}
            if os.path.exists(train_stats_file):
                try:
                    with open(train_stats_file, 'r') as f:
                        train_stats = json.load(f)
                except Exception as e:
                    logger.warning(f"Error al leer estadísticas de {train_stats_file}: {str(e)}")
            
            # Buscar evaluaciones
            evaluations = []
            evals_dir = os.path.join(model_dir, "evaluations")
            if os.path.exists(evals_dir):
                eval_dirs = [d for d in os.listdir(evals_dir) if os.path.isdir(os.path.join(evals_dir, d))]
                for eval_dir_name in eval_dirs:
                    eval_dir = os.path.join(evals_dir, eval_dir_name)
                    eval_metrics_files = [f for f in os.listdir(eval_dir) if f.endswith("_metrics.json")]
                    
                    for metrics_file in eval_metrics_files:
                        try:
                            with open(os.path.join(eval_dir, metrics_file), 'r') as f:
                                metrics = json.load(f)
                                evaluations.append(metrics)
                        except Exception as e:
                            logger.warning(f"Error al leer métricas de {metrics_file}: {str(e)}")
            
            # Extraer componentes del nombre del directorio
            parts = model_dir_name.split('_')
            model_info = {
                "model_name": parts[1] if len(parts) > 1 else "unknown",
                "epochs": parts[2] if len(parts) > 2 else "unknown",
                "learning_rate": parts[3] if len(parts) > 3 else "unknown",
                "seed": parts[4] if len(parts) > 4 else "unknown",
                "directory": model_dir,
                "training_stats": train_stats,
                "evaluations": evaluations
            }
            
            models_info.append(model_info)
        
        # Crear DataFrame con la información recopilada
        if models_info:
            df_models = pd.DataFrame(models_info)
            
            # Crear dataframes para métricas de entrenamiento y evaluación
            train_metrics = []
            for _, row in df_models.iterrows():
                stats = row["training_stats"]
                if stats:
                    train_metrics.append({
                        "model_name": row["model_name"],
                        "epochs": row["epochs"],
                        "learning_rate": row["learning_rate"],
                        "final_loss": stats.get("final_loss", "N/A"),
                        "final_accuracy": stats.get("final_accuracy", "N/A"),
                        "training_time": stats.get("total_training_time_sec", "N/A"),
                        "MSE": stats.get("MSE", "N/A"),
                        "R2": stats.get("R2", "N/A")
                    })
            
            eval_metrics = []
            for _, row in df_models.iterrows():
                evaluations = row["evaluations"]
                for eval_data in evaluations:
                    eval_metrics.append({
                        "model_name": row["model_name"],
                        "MSE": eval_data.get("MSE", "N/A"),
                        "MAE": eval_data.get("MAE", "N/A"),
                        "R2": eval_data.get("R2", "N/A"),
                        "inference_time": eval_data.get("inference_time", "N/A")
                    })
            
            # Convertir a DataFrames
            df_train = pd.DataFrame(train_metrics) if train_metrics else None
            df_eval = pd.DataFrame(eval_metrics) if eval_metrics else None
            
            # Crear reporte
            summary_dir = os.path.join(physics_dir, "summary")
            os.makedirs(summary_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = os.path.join(summary_dir, f"{physics_type}_summary_{timestamp}.html")
            
            with open(summary_file, 'w') as f:
                f.write(f"<html><head><title>Resumen de {physics_type}</title>")
                f.write("<style>body{font-family:Arial,sans-serif;margin:20px;}")
                f.write("table{border-collapse:collapse;width:100%;}")
                f.write("th,td{padding:8px;text-align:left;border:1px solid #ddd;}")
                f.write("th{background-color:#f2f2f2;}")
                f.write("h1,h2{color:#333;}</style></head><body>")
                
                f.write(f"<h1>Resumen para Problema Físico: {physics_type}</h1>")
                f.write(f"<p>Generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
                
                # Modelos disponibles
                f.write(f"<h2>Modelos Disponibles ({len(models_info)})</h2>")
                for model in models_info:
                    f.write(f"<p><strong>{model['model_name']}</strong> - ")
                    f.write(f"Épocas: {model['epochs']}, ")
                    f.write(f"LR: {model['learning_rate']}, ")
                    f.write(f"Seed: {model['seed']}</p>")
                
                # Métricas de entrenamiento
                if df_train is not None and not df_train.empty:
                    f.write("<h2>Métricas de Entrenamiento</h2>")
                    f.write(df_train.to_html(index=False))
                    
                    # Guardar también como CSV
                    df_train.to_csv(os.path.join(summary_dir, f"{physics_type}_training_metrics_{timestamp}.csv"), index=False)
                    
                    # Crear gráfico de barras para MSE de entrenamiento
                    if "MSE" in df_train.columns and df_train["MSE"].dtype != object:
                        plt.figure(figsize=(10, 6))
                        df_train.sort_values("MSE")["MSE"].plot(kind="bar")
                        plt.title(f"MSE de Entrenamiento para {physics_type}")
                        plt.ylabel("MSE")
                        plt.tight_layout()
                        chart_file = os.path.join(summary_dir, f"{physics_type}_mse_chart_{timestamp}.png")
                        plt.savefig(chart_file)
                        plt.close()
                        
                        f.write(f'<h3>Gráfico de MSE</h3><img src="{os.path.basename(chart_file)}" alt="MSE Chart">')
                
                # Métricas de evaluación
                if df_eval is not None and not df_eval.empty:
                    f.write("<h2>Métricas de Evaluación</h2>")
                    f.write(df_eval.to_html(index=False))
                    
                    # Guardar también como CSV
                    df_eval.to_csv(os.path.join(summary_dir, f"{physics_type}_evaluation_metrics_{timestamp}.csv"), index=False)
                
                f.write("</body></html>")
            
            logger.info(f"Resumen para {physics_type} generado en: {summary_file}")
            return summary_file
        else:
            logger.warning(f"No se encontraron modelos para el problema físico: {physics_type}")
            return None

    def generate_all_summaries(self):
        """
        Genera resúmenes para todos los tipos de problemas físicos.
        
        Returns:
            list: Lista de rutas a los archivos de resumen generados
        """
        summaries = []
        for physics_type in self.physics_types:
            summary_file = self.generate_physics_summary(physics_type)
            if summary_file:
                summaries.append(summary_file)
        
        return summaries
