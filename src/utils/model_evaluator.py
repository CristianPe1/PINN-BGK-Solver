import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from tabulate import tabulate
import logging

from structure_model.pinn_structure_v1 import PINN_V1
from data_handlers.data_manager import DataManager

class ModelEvaluator:
    """
    Clase para evaluar modelos PINN, calcular métricas de rendimiento,
    visualizar resultados y generar reportes comparativos.
    """
    def __init__(self, output_dir=None, logger=None):
        """
        Inicializa el evaluador.
        
        Args:
            output_dir (str): Directorio para guardar las evaluaciones
            logger: Logger para registrar información (opcional)
        """
        if output_dir is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.output_dir = os.path.join(root_dir, "results", "evaluations", 
                                          f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configurar logger
        self.logger = logger or logging.getLogger(__name__)
        if not self.logger.handlers:
            log_file = os.path.join(self.output_dir, "evaluation.log")
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.metrics_history = []  # Lista para almacenar resultados de evaluaciones
        self.data_manager = DataManager(256, 100)  # Instancia para cargar datos
    
    def load_model(self, model_path, architecture=None):
        """
        Carga un modelo entrenado desde un archivo.
        
        Args:
            model_path (str): Ruta al checkpoint del modelo
            architecture (dict, optional): Diccionario con parámetros de arquitectura
            
        Returns:
            model: El modelo cargado
        """
        self.logger.info(f"Cargando modelo desde: {model_path}")
        
        # Intentar cargar los metadatos para obtener la arquitectura
        if architecture is None:
            try:
                meta_dir = os.path.dirname(model_path)
                meta_file = os.path.join(meta_dir, "hyperparams_train.json")
                if os.path.exists(meta_file):
                    with open(meta_file, 'r') as f:
                        meta_data = json.load(f)
                        architecture = {
                            "layers": meta_data.get("model", {}).get("layers", [2, 50, 50, 50, 50, 1]),
                            "activation": meta_data.get("model", {}).get("activation_function", "Tanh")
                        }
                        self.logger.info(f"Arquitectura cargada de metadatos: {architecture}")
            except Exception as e:
                self.logger.warning(f"Error al cargar metadatos: {str(e)}")
        
        # Usar arquitectura predeterminada si no se encontró
        if architecture is None:
            architecture = {
                "layers": [2, 50, 50, 50, 50, 1],
                "activation": "Tanh"
            }
            self.logger.warning(f"Usando arquitectura predeterminada: {architecture}")
        
        # Crear modelo con la arquitectura
        model = PINN_V1(
            architecture["layers"], 
            architecture["activation"]
        )
        
        # Cargar estado del modelo
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Manejar caso donde el modelo fue entrenado con DataParallel
        if list(checkpoint['model_state_dict'].keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                name = k[7:]  # Eliminar 'module.' prefix
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()  # Cambiar a modo evaluación
        self.logger.info(f"Modelo cargado exitosamente")
        
        return model
    
    def calculate_metrics(self, y_true, y_pred, prefix=""):
        """
        Calcula varias métricas de rendimiento para las predicciones.
        
        Args:
            y_true: Valores reales
            y_pred: Valores predichos
            prefix: Prefijo para las métricas (ej: 'train_', 'val_')
            
        Returns:
            dict: Diccionario con las métricas calculadas
        """
        metrics = {}
        
        # Preparar datos
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.squeeze().detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.squeeze().detach().cpu().numpy()
            
        # Calcular métricas básicas
        metrics[f"{prefix}mae"] = mean_absolute_error(y_true, y_pred)
        metrics[f"{prefix}mse"] = mean_squared_error(y_true, y_pred)
        metrics[f"{prefix}rmse"] = np.sqrt(metrics[f"{prefix}mse"])
        metrics[f"{prefix}r2"] = r2_score(y_true, y_pred)
        
        # Métricas adicionales
        metrics[f"{prefix}max_error"] = np.max(np.abs(y_true - y_pred))
        metrics[f"{prefix}mean_rel_error"] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10)))
        
        # L2 relativo
        metrics[f"{prefix}rel_l2_error"] = np.sqrt(np.sum((y_true - y_pred)**2) / np.sum(y_true**2))
        
        return metrics
    
    def evaluate_model(self, model, data_path, nu=0.01, model_name="model", save_results=True):
        """
        Evalúa un modelo con un conjunto de datos.
        
        Args:
            model: Modelo a evaluar
            data_path (str): Ruta al archivo de datos
            nu (float): Valor de viscosidad
            model_name (str): Nombre para identificar el modelo
            save_results (bool): Si debe guardar los resultados
            
        Returns:
            dict: Métricas del modelo
        """
        self.logger.info(f"Evaluando modelo '{model_name}' con datos: {data_path}")
        
        # Cargar datos de prueba
        try:
            x, t, u_true = self.data_manager.load_data_from_file(data_path)
        except Exception as e:
            self.logger.error(f"Error al cargar datos: {str(e)}")
            raise
            
        # Preparar tensores
        x = DataManager.prepare_tensor(x)
        t = DataManager.prepare_tensor(t)
        u_true = DataManager.prepare_tensor(u_true)
        
        # Realizar predicciones
        with torch.no_grad():
            u_pred = model(x, t)
            
        # Calcular métricas
        metrics = self.calculate_metrics(u_true, u_pred)
        metrics.update({
            'model_name': model_name,
            'data_path': data_path,
            'nu': nu,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Agregar a historial
        self.metrics_history.append(metrics)
        self.logger.info(f"Métricas para {model_name}: MAE={metrics['mae']:.6f}, MSE={metrics['mse']:.6f}, R2={metrics['r2']:.6f}")
        
        # Generar visualizaciones
        if save_results:
            # Guardar métricas
            metrics_file = os.path.join(self.output_dir, f"metrics_{model_name.replace(' ', '_')}.json")
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            self.logger.info(f"Métricas guardadas en: {metrics_file}")
            
            # Generar visualizaciones
            self._plot_prediction_vs_truth(x, t, u_true, u_pred, model_name)
            self._plot_error_heatmap(x, t, u_true, u_pred, model_name)
            
        return metrics
        
    def _plot_prediction_vs_truth(self, x, t, u_true, u_pred, model_name):
        """
        Visualiza la predicción vs valores reales.
        
        Args:
            x, t: Coordenadas
            u_true: Valores reales
            u_pred: Valores predichos 
            model_name: Nombre del modelo
        """
        # Convertir a numpy
        x_np = x.squeeze().detach().cpu().numpy()  
        t_np = t.squeeze().detach().cpu().numpy()
        u_true_np = u_true.squeeze().detach().cpu().numpy()
        u_pred_np = u_pred.squeeze().detach().cpu().numpy()
        
        # Crear visualización
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Solución real
        im1 = axes[0].pcolormesh(t_np, x_np, u_true_np, shading='auto', cmap='viridis')
        axes[0].set_xlabel('t')
        axes[0].set_ylabel('x')
        axes[0].set_title('Solución Real')
        fig.colorbar(im1, ax=axes[0])
        
        # Solución predicha
        im2 = axes[1].pcolormesh(t_np, x_np, u_pred_np, shading='auto', cmap='viridis')
        axes[1].set_xlabel('t')
        axes[1].set_ylabel('x')
        axes[1].set_title('Solución Predicha')
        fig.colorbar(im2, ax=axes[1])
        
        # Error absoluto
        error = np.abs(u_true_np - u_pred_np)
        im3 = axes[2].pcolormesh(t_np, x_np, error, shading='auto', cmap='hot')
        axes[2].set_xlabel('t')
        axes[2].set_ylabel('x')
        axes[2].set_title('Error Absoluto')
        fig.colorbar(im3, ax=axes[2])
        
        plt.suptitle(f"Evaluación del Modelo: {model_name}")
        plt.tight_layout()
        
        # Guardar figura
        fig_path = os.path.join(self.output_dir, f"prediction_vs_truth_{model_name.replace(' ', '_')}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"Visualización guardada en: {fig_path}")
        
    def _plot_error_heatmap(self, x, t, u_true, u_pred, model_name):
        """
        Genera un mapa de calor detallado del error.
        
        Args:
            x, t: Coordenadas
            u_true: Valores reales
            u_pred: Valores predichos
            model_name: Nombre del modelo
        """
        # Convertir a numpy
        x_np = x.squeeze().detach().cpu().numpy()
        t_np = t.squeeze().detach().cpu().numpy()
        u_true_np = u_true.squeeze().detach().cpu().numpy()
        u_pred_np = u_pred.squeeze().detach().cpu().numpy()
        
        # Calcular error
        error = np.abs(u_true_np - u_pred_np)
        
        # Crear visualización
        plt.figure(figsize=(10, 8))
        
        # Usar seaborn para un mapa de calor más detallado
        ax = sns.heatmap(error, cmap='hot', annot=False)
        
        # Añadir líneas en los ejes
        plt.xticks(np.linspace(0, error.shape[1], 5), 
                  np.linspace(min(t_np), max(t_np), 5).round(2))
        plt.yticks(np.linspace(0, error.shape[0], 5), 
                  np.linspace(min(x_np), max(x_np), 5).round(2))
        
        plt.title(f'Mapa de Error Detallado - {model_name}')
        plt.xlabel('t')
        plt.ylabel('x')
        
        # Guardar figura
        fig_path = os.path.join(self.output_dir, f"error_heatmap_{model_name.replace(' ', '_')}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Mapa de error guardado en: {fig_path}")
    
    def compare_models(self, models_results=None, metrics=None, sort_by='rmse'):
        """
        Compara el rendimiento de múltiples modelos.
        
        Args:
            models_results (list): Lista de resultados de modelos (opcional)
            metrics (list): Lista de métricas a comparar
            sort_by (str): Métrica para ordenar la comparación
            
        Returns:
            pd.DataFrame: DataFrame con la comparación
        """
        if models_results is None:
            models_results = self.metrics_history
            
        if not models_results:
            self.logger.warning("No hay modelos para comparar.")
            return None
            
        if metrics is None:
            metrics = ['mae', 'mse', 'rmse', 'r2', 'max_error', 'rel_l2_error']
            
        # Crear DataFrame
        comparison_data = []
        for result in models_results:
            row = {'Modelo': result.get('model_name', 'Desconocido')}
            for metric in metrics:
                if metric in result:
                    row[metric.upper()] = result[metric]
            comparison_data.append(row)
            
        df = pd.DataFrame(comparison_data)
        
        # Ordenar por la métrica especificada
        if sort_by.lower() in [col.lower() for col in df.columns]:
            # Determinar si la métrica es "mejor" cuando es mayor (como R2) o menor (como MSE)
            ascending = True
            if sort_by.lower() == 'r2':
                ascending = False
            df = df.sort_values(by=sort_by.upper(), ascending=ascending)
            
        # Guardar comparación en CSV
        csv_path = os.path.join(self.output_dir, "models_comparison.csv")
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Comparación de modelos guardada en: {csv_path}")
        
        # Crear visualización
        self._plot_model_comparison(df, metrics)
        
        # Crear tabla de texto para el log
        table_str = tabulate(df, headers='keys', tablefmt='grid')
        self.logger.info(f"Comparación de modelos:\n{table_str}")
        
        return df
    
    def _plot_model_comparison(self, df, metrics):
        """
        Crea gráficos de barras comparando las métricas de los modelos.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            metrics (list): Lista de métricas a visualizar
        """
        # Para cada métrica, crear un gráfico de barras
        for metric in metrics:
            metric = metric.upper()
            if metric in df.columns:
                plt.figure(figsize=(12, 6))
                
                # Crear gráfico de barras con seaborn
                ax = sns.barplot(x='Modelo', y=metric, data=df)
                
                # Rotar etiquetas del eje x si hay muchos modelos
                if len(df) > 3:
                    plt.xticks(rotation=45, ha='right')
                
                # Añadir etiquetas con valores
                for i, v in enumerate(df[metric]):
                    ax.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=9)
                
                plt.title(f"Comparación de Modelos - {metric}")
                plt.tight_layout()
                
                # Guardar figura
                fig_path = os.path.join(self.output_dir, f"comparison_{metric}.png")
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Comparación de {metric} guardada en: {fig_path}")
        
        # Gráfico de radar para comparación multimétrica
        self._plot_radar_comparison(df, metrics)
    
    def _plot_radar_comparison(self, df, metrics):
        """
        Crea un gráfico de radar para comparar modelos en múltiples métricas.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            metrics (list): Lista de métricas a visualizar
        """
        # Filtrar métricas disponibles
        available_metrics = [m.upper() for m in metrics if m.upper() in df.columns]
        
        if len(available_metrics) < 3:
            self.logger.warning("Se necesitan al menos 3 métricas para un gráfico de radar.")
            return
            
        # Normalizar datos para el gráfico de radar
        # Para R2, mayor es mejor, para las demás, menor es mejor
        radar_df = df.copy()
        for metric in available_metrics:
            if metric.upper() == 'R2':
                # Para R2, normalizar como (valor - min) / (max - min)
                radar_df[metric] = (radar_df[metric] - radar_df[metric].min()) / (radar_df[metric].max() - radar_df[metric].min() + 1e-10)
            else:
                # Para otras métricas, normalizar como 1 - (valor - min) / (max - min)
                radar_df[metric] = 1 - (radar_df[metric] - radar_df[metric].min()) / (radar_df[metric].max() - radar_df[metric].min() + 1e-10)
        
        # Configurar gráfico
        fig = plt.figure(figsize=(10, 8))
        
        # Número de variables
        N = len(available_metrics)
        
        # Calcular ángulos
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Cerrar el polígono
        
        # Inicializar gráfico
        ax = fig.add_subplot(111, polar=True)
        
        # Para cada modelo
        for i, row in radar_df.iterrows():
            model_name = row['Modelo']
            values = [row[m] for m in available_metrics]
            values += values[:1]  # Cerrar el polígono
            
            # Dibujar polígono
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
            ax.fill(angles, values, alpha=0.1)
        
        # Configurar etiquetas
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_metrics)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])
        ax.set_title("Comparación de Modelos - Gráfico de Radar")
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Guardar figura
        fig_path = os.path.join(self.output_dir, f"radar_comparison.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Gráfico de radar guardado en: {fig_path}")
    
    def generate_full_report(self):
        """
        Genera un informe completo con todas las evaluaciones realizadas.
        
        Returns:
            str: Ruta al archivo de informe
        """
        if not self.metrics_history:
            self.logger.warning("No hay datos de evaluación para generar un informe.")
            return None
            
        # Crear informe
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_models": len(self.metrics_history),
            "evaluations": self.metrics_history,
            "summary": {}
        }
        
        # Resumen por modelo
        models = {}
        for m in self.metrics_history:
            model_name = m.get('model_name', 'Desconocido')
            if model_name not in models:
                models[model_name] = []
            models[model_name].append(m)
        
        # Calcular estadísticas por modelo
        for model_name, results in models.items():
            if len(results) > 0:
                report["summary"][model_name] = {
                    "evaluations_count": len(results),
                    "average_metrics": {}
                }
                
                # Métricas promedio
                for metric in ['mae', 'mse', 'rmse', 'r2']:
                    values = [r.get(metric, 0) for r in results]
                    if values:
                        report["summary"][model_name]["average_metrics"][metric] = sum(values) / len(values)
        
        # Guardar informe
        report_file = os.path.join(self.output_dir, "full_evaluation_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        self.logger.info(f"Informe completo guardado en: {report_file}")
        
        return report_file
    
    def batch_evaluate_models(self, models_config, common_data_path=None):
        """
        Evalúa varios modelos en lote.
        
        Args:
            models_config (list): Lista de diccionarios con configuración de modelos
            common_data_path (str): Ruta común para datos (opcional)
            
        Returns:
            pd.DataFrame: Resultados de la evaluación
        """
        results = []
        
        for cfg in models_config:
            model_path = cfg.get('model_path')
            model_name = cfg.get('name') or f"Modelo_{os.path.basename(model_path)}"
            data_path = cfg.get('data_path') or common_data_path
            nu = cfg.get('nu', 0.01)
            architecture = cfg.get('architecture')
            
            if not model_path or not data_path:
                self.logger.warning(f"Configuración incompleta para {model_name}, saltando evaluación.")
                continue
                
            try:
                self.logger.info(f"Evaluando {model_name} con datos: {data_path}")
                model = self.load_model(model_path, architecture)
                metrics = self.evaluate_model(model, data_path, nu, model_name)
                results.append(metrics)
                self.logger.info(f"Evaluación exitosa para {model_name}")
            except Exception as e:
                self.logger.error(f"Error al evaluar {model_name}: {str(e)}")
        
        # Generar comparación
        if results:
            comparison_df = self.compare_models(results)
            report_file = self.generate_full_report()
            self.logger.info(f"Evaluación por lotes completada. Informe en: {report_file}")
            return comparison_df
        
        return None

# Ejemplo de uso independiente
if __name__ == "__main__":
    # Configurar logger para pruebas
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Crear evaluador
    evaluator = ModelEvaluator()
    
    # Ejemplo de configuración de modelos a evaluar
    models_to_evaluate = [
        {
            "model_path": "/path/to/model1.pth",
            "name": "Modelo Base",
            "data_path": "/path/to/data.mat",
            "nu": 0.01,
        },
        {
            "model_path": "/path/to/model2.pth",
            "name": "Modelo Optimizado",
            "data_path": "/path/to/data.mat",
            "nu": 0.01,
        }
    ]
    
    # Evaluación por lotes
    # results = evaluator.batch_evaluate_models(models_to_evaluate)
    
    print("Para usar el evaluador, configura las rutas correctas a los modelos y datos.")
