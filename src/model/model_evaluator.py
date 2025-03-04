import os
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import psutil
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from datetime import datetime

from data_handlers.data_manager import DataManager
from model_factory import create_model

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Clase para evaluar modelos PINN entrenados.
    """
    def __init__(self, output_dir, logger=None):
        """
        Args:
            output_dir (str): Directorio donde guardar resultados de evaluación
            logger: Logger para mensajes (opcional)
        """
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar resultados de evaluación
        self.evaluation_results = []
    
    def load_model(self, model_path):
        """
        Carga un modelo desde un archivo de pesos.
        
        Args:
            model_path (str): Ruta al archivo de pesos del modelo
            
        Returns:
            nn.Module: Modelo cargado
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
        
        # Cargar checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Intentar identificar la arquitectura del modelo
        model_dir = os.path.dirname(os.path.dirname(model_path))
        arch_path = os.path.join(model_dir, "train_results", "hyperparams_train.json")
        
        model = None
        
        if os.path.exists(arch_path):
            try:
                with open(arch_path, 'r') as f:
                    config = json.load(f)
                    
                # Extraer configuración del modelo
                selected_model = config.get("selected_model", "pinn_v1")
                model_cfg = config.get("models", {}).get(selected_model, {})
                
                # Crear modelo según configuración
                model = create_model(model_cfg)
                self.logger.info(f"Modelo {selected_model} recreado según configuración")
                
            except Exception as e:
                self.logger.warning(f"No se pudo recrear la arquitectura desde el archivo: {str(e)}")
                # Continuar y usar un modelo predeterminado
        
        # Si no se pudo crear el modelo desde la configuración, usar uno predeterminado
        if model is None:
            from structure_model.pinn_structure_v1 import PINN_V1
            model = PINN_V1([2, 50, 50, 50, 50, 1], "Tanh")
            self.logger.warning("Usando modelo predeterminado PINN_V1")
        
        # Verificar si hay incompatibilidad en los nombres de las capas (linears vs linear_layers)
        state_dict = checkpoint['model_state_dict']
        
        # Comprobar si necesitamos adaptación de nombres
        needs_adaptation = any('linears.' in key for key in state_dict.keys()) and not any('linear_layers.' in key for key in state_dict.keys())
        
        if needs_adaptation:
            self.logger.info("Detectada incompatibilidad de nombres de capas. Adaptando estado del modelo...")
            new_state_dict = {}
            
            # Mapear nombres antiguos a nuevos
            for key, value in state_dict.items():
                if 'linears.' in key:
                    new_key = key.replace('linears.', 'linear_layers.')
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
                    
            state_dict = new_state_dict
            self.logger.info("Adaptación de nombres completada.")
        
        # Cargar pesos con manejo de errores flexible
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            self.logger.warning(f"Error al cargar estado con 'strict=True': {str(e)}")
            self.logger.info("Intentando cargar con 'strict=False'...")
            
            # Intentar carga sin estricta
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                self.logger.warning(f"Claves faltantes: {missing_keys}")
            if unexpected_keys:
                self.logger.warning(f"Claves inesperadas: {unexpected_keys}")
        
        model.eval()  # Modo evaluación
        
        self.logger.info(f"Modelo cargado exitosamente desde {model_path}")
        return model
    
    def evaluate_model(self, model_path, data_path, nu, model_name="Modelo"):
        """
        Evalúa un modelo con datos específicos.
        
        Args:
            model_path (str): Ruta al modelo o instancia del modelo
            data_path (str): Ruta a los datos de evaluación
            nu (float): Coeficiente de viscosidad
            model_name (str): Nombre identificativo del modelo
            
        Returns:
            dict: Métricas de evaluación
        """
        # Cargar el modelo si se proporciona una ruta
        if isinstance(model_path, str):
            model = self.load_model(model_path)
        else:
            model = model_path  # Asumir que ya es un modelo
            
        # Cargar datos de evaluación
        data_manager = DataManager(256, 100)
        input_tensor, output_tensor = data_manager.load_data_from_file(data_path)
        
        # Dispositivo: usar GPU si está disponible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        input_tensor = input_tensor.to(device)
        
        # Medir uso de memoria y tiempo de inferencia
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        start_time = time.time()
        
        # Evaluar el modelo
        with torch.no_grad():
            y_pred = model(input_tensor)
            
        # Medir tiempo y memoria final
        inference_time = time.time() - start_time
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_usage = end_memory - start_memory
        
        # Pasar a CPU y numpy para cálculo de métricas
        y_true = output_tensor.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        
        # Calcular métricas
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Crear directorio específico para este modelo
        model_eval_dir = os.path.join(self.output_dir, model_name.replace(" ", "_"))
        os.makedirs(model_eval_dir, exist_ok=True)
        
        # Guardar métricas
        metrics = {
            "model_name": model_name,
            "MAE": float(mae),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "R2": float(r2),
            "inference_time": float(inference_time),
            "memory_usage_mb": float(memory_usage),
            "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_file": data_path,
            "nu": nu
        }
        
        # Guardar métricas como JSON
        metrics_file = os.path.join(model_eval_dir, "evaluation_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Crear visualizaciones
        self._plot_comparison(input_tensor, output_tensor, y_pred, model_eval_dir)
        
        # Añadir a los resultados de evaluación
        self.evaluation_results.append({
            "model_name": model_name,
            "metrics": metrics,
            "output_dir": model_eval_dir
        })
        
        self.logger.info(f"Evaluación completada para {model_name}:")
        self.logger.info(f"  MAE: {mae:.6f}")
        self.logger.info(f"  MSE: {mse:.6f}")
        self.logger.info(f"  RMSE: {rmse:.6f}")
        self.logger.info(f"  R²: {r2:.6f}")
        self.logger.info(f"  Tiempo inferencia: {inference_time:.4f}s")
        self.logger.info(f"  Uso de memoria: {memory_usage:.2f}MB")
        
        return metrics
    
    def _plot_comparison(self, input_tensor, output_tensor, y_pred, output_dir):
        """
        Genera visualizaciones comparativas entre la solución real y la predicha.
        
        Args:
            input_tensor (torch.Tensor): Tensor de entrada (coordenadas)
            output_tensor (torch.Tensor): Tensor de salida real
            y_pred (np.ndarray): Predicciones del modelo
            output_dir (str): Directorio donde guardar las visualizaciones
        """
        # Extraer datos
        x_np = input_tensor[:, 0].detach().cpu().numpy()
        t_np = input_tensor[:, 1].detach().cpu().numpy()
        y_true = output_tensor.detach().cpu().numpy()
        
        try:
            # Reshape para visualización
            nx = int(np.sqrt(len(x_np)))  # Aproximación para visualización
            nt = len(x_np) // nx
            
            X = x_np.reshape(nx, nt)
            T = t_np.reshape(nx, nt)
            U_true = y_true.reshape(nx, nt)
            U_pred = y_pred.reshape(nx, nt)
            
            # Calcular error absoluto
            error = np.abs(U_true - U_pred)
            
            # Gráfico 1: Comparación solución real vs predicción
            plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            plt.pcolormesh(T, X, U_true, shading='auto', cmap='viridis')
            plt.colorbar(label="u(x,t)")
            plt.xlabel("t")
            plt.ylabel("x")
            plt.title("Solución Real")
            
            plt.subplot(132)
            plt.pcolormesh(T, X, U_pred, shading='auto', cmap='viridis')
            plt.colorbar(label="u(x,t)")
            plt.xlabel("t")
            plt.ylabel("x")
            plt.title("Predicción del Modelo")
            
            plt.subplot(133)
            plt.pcolormesh(T, X, error, shading='auto', cmap='hot')
            plt.colorbar(label="Error Absoluto")
            plt.xlabel("t")
            plt.ylabel("x")
            plt.title("Error Absoluto")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "solution_comparison.png"), dpi=300)
            plt.close()
            
            # Gráfico 2: Scatter plot de predicción vs real
            plt.figure(figsize=(8, 6))
            plt.scatter(y_true, y_pred, alpha=0.5)
            
            # Línea diagonal de referencia y=x
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.xlabel("Valores Reales")
            plt.ylabel("Predicciones")
            plt.title("Predicción vs. Valores Reales")
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "prediction_vs_true.png"), dpi=300)
            plt.close()
            
            # NUEVA VISUALIZACIÓN: Perfiles en tiempos específicos
            plt.figure(figsize=(12, 8))
            t_indices = [0, nt//4, nt//2, 3*nt//4, -1]  # Tiempos para perfiles
            t_labels = ["t=0", "t=0.25", "t=0.5", "t=0.75", "t=1.0"]
            
            for i, tidx in enumerate(t_indices):
                plt.subplot(2, 3, i+1)
                plt.plot(X[:, 0], U_true[:, tidx], 'b-', label='Real')
                plt.plot(X[:, 0], U_pred[:, tidx], 'r--', label='Predicción')
                plt.xlabel("x")
                plt.ylabel("u(x,t)")
                plt.title(f"Perfil en {t_labels[i]}")
                plt.grid(True)
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "time_profiles.png"), dpi=300)
            plt.close()
            
            # NUEVA VISUALIZACIÓN: Evolución temporal en posiciones espaciales específicas
            plt.figure(figsize=(12, 8))
            x_indices = [0, nx//4, nx//2, 3*nx//4, -1]  # Posiciones espaciales
            x_labels = ["x=-1.0", "x=-0.5", "x=0.0", "x=0.5", "x=1.0"]
            
            for i, xidx in enumerate(x_indices):
                plt.subplot(2, 3, i+1)
                plt.plot(T[0, :], U_true[xidx, :], 'b-', label='Real')
                plt.plot(T[0, :], U_pred[xidx, :], 'r--', label='Predicción')
                plt.xlabel("t")
                plt.ylabel("u(x,t)")
                plt.title(f"Evolución en {x_labels[i]}")
                plt.grid(True)
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "spatial_evolution.png"), dpi=300)
            plt.close()
            
            # NUEVA VISUALIZACIÓN: Mapa de calor de error relativo (%)
            plt.figure(figsize=(10, 6))
            # Evitar división por cero
            relative_error = 100 * np.abs(U_true - U_pred) / (np.abs(U_true) + 1e-10)
            
            plt.pcolormesh(T, X, relative_error, shading='auto', cmap='hot')
            cbar = plt.colorbar()
            cbar.set_label("Error Relativo (%)")
            plt.xlabel("t")
            plt.ylabel("x")
            plt.title("Error Relativo")
            plt.savefig(os.path.join(output_dir, "relative_error.png"), dpi=300)
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Error al generar visualizaciones: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def generate_full_report(self):
        """
        Genera un informe completo de todas las evaluaciones realizadas.
        
        Returns:
            str: Ruta al informe generado
        """
        if not self.evaluation_results:
            self.logger.warning("No hay resultados para generar informe.")
            return None
        
        # Crear informe JSON completo
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_models_evaluated": len(self.evaluation_results),
            "evaluations": self.evaluation_results
        }
        
        # Guardar informe
        report_file = os.path.join(self.output_dir, f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Informe de evaluación guardado en: {report_file}")
        
        return report_file

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
