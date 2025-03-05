import os
import sys
import json
import yaml
import torch
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Importaciones locales
from model.train import Trainer
from model.model_finder import ModelFinder
from utils.visualization import Visualizer

from model.model_factory import create_model
from model.model_selector import ModelSelector
from utils.output_manager import OutputManager
from model.model_evaluator import ModelEvaluator
from model.loss_functions import get_loss_function
from model.model_comparator import ModelComparator
from data_handlers.data_manager import DataManager



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Establecer PROJECT_ROOT
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configuración de logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levellevel)s - %(message)s')

### ---------------------- Funciones auxiliares ---------------------- ###

def load_config():
    config_path = os.path.join(PROJECT_ROOT, "..", "config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_output_dir(model_name, physics_type, epochs, lr, seed, output_folder):
    """
    Crea directorios organizados para los resultados del modelo.
    
    Args:
        model_name (str): Nombre del modelo
        physics_type (str): Tipo de problema físico
        epochs (int): Número de épocas
        lr (float): Tasa de aprendizaje
        seed (int): Semilla aleatoria
        output_folder (str): Nombre de la carpeta de salida
        
    Returns:
        tuple: (output_dir, model_dir, train_dir) con las rutas completas
    """
    # Usar OutputManager para organizar resultados
    output_manager = OutputManager(project_root=os.path.dirname(PROJECT_ROOT))
    return output_manager.create_model_dir(model_name, physics_type, epochs, lr, seed)


### --------------------- Funciones principales --------------------- ###

def train_model(config):
    """Función principal para entrenar el modelo."""
    # Permitir selección interactiva de modelos
    
    use_interactive = input("¿Desea seleccionar el modelo interactivamente? (s/n): ").lower() == 's'
    if use_interactive:
        
        selector = ModelSelector()  # Usar config por defecto
        model_type, model_id, model_cfg = selector.select_model_interactive()
        
        # Actualizar la configuración con la selección del usuario
        config["selected_model"] = model_id
        
        # Si es un modelo de fluidos, ajustar la configuración
        if model_type == "fluid":
            config["selected_model"] = "fluid_model"  # Marcador genérico
            config["models"]["fluid_model"] = model_cfg
            selected_model = "fluid_model"
        else:
            selected_model = model_id
    else:
        # Usar la selección de la configuración YAML
        selected_model = config.get("selected_model")
        # Mostrar el modelo seleccionado
        print("Modelo seleccionado:", selected_model)
    
    # Seleccionar la función de pérdida según config
    selected_loss = config.get("selected_loss") ## Mirar como hacer para que en cada modelo 
    # lleve su propia función de pérdida
    
    # Verificar que el modelo y la función de pérdida seleccionados existan
    if selected_model not in config.get("models", {}):
        logger.error(f"Modelo seleccionado '{selected_model}' no encontrado en la configuración.")
        return
        
    if selected_loss not in config.get("loss_functions", {}):
        logger.error(f"Función de pérdida seleccionada '{selected_loss}' no encontrada en la configuración.")
        return
    
    # Obtener configuración específica
    model_cfg = config["models"][selected_model]
    loss_cfg = config["loss_functions"][selected_loss]
    train_cfg = config["training"]
    physics_cfg = config["physics"]
    log_cfg = config["logging"]
    data_cfg = config.get("data_generation", {})  # Obtener configuración de datos

    # Extraer parámetros de entrenamiento
    num_epochs = int(train_cfg["epochs"])
    lr = float(model_cfg.get("learning_rate", 0.001))
    seed = int(train_cfg.get("seed", 42))
    batch_size = int(train_cfg.get("batch_size", 32))
    physics_type = physics_cfg.get("physics_type", "burgers")  # Obtener tipo de física
    early_stop_patience = int(train_cfg.get("epochs_no_improve", 20))
    min_loss_improvement = float(train_cfg.get("min_loss_improvement", 1e-5))
    enabled = train_cfg.get("early_stopping", True)

    # Crear directorios de salida
    model_name = selected_model
    OUTPUT_DIR, MODEL_DIR, OUTPUT_TRAIN_DIR = create_output_dir(
        model_name, physics_type, num_epochs, lr, seed, 
        log_cfg.get("output_folder", "output")
    )

    # Cargar datos desde configuración
    eval_cfg = config.get("evaluation", {})
    
    # Obtener ruta del directorio de datos desde config o usar valor por defecto
    data_dir = eval_cfg.get("data_dir", os.path.join(PROJECT_ROOT, "..", "data", "training"))
    
    # Obtener ruta completa del archivo de datos desde config o usar valor por defecto
    data_file_path = eval_cfg.get("data_path", "")
    if not data_file_path or not os.path.exists(data_file_path):
        # Si no se especificó en config o no existe, usar valor por defecto
        default_file = "burgers_shock_mu_01_pi.mat"
        data_file = os.path.join(data_dir, default_file)
        logger.info(f"Usando archivo de datos por defecto: {data_file}")
    else:
        data_file = data_file_path
        logger.info(f"Usando archivo de datos especificado en config: {data_file}")
    
    # Obtener parámetros para el DataManager
    spatial_points = data_cfg.get("spatial_points", 256)
    time_points = data_cfg.get("time_points", 100)
    
    data_manager = DataManager(spatial_points, time_points)
    logger.info(f"DataManager inicializado con {spatial_points} puntos espaciales y {time_points} puntos temporales")
    
    # Cargar tensores de entrada y salida con verificación de errores
    try:
        # Cargar los datos y verificar que son correctos
        input_tensor, output_tensor = data_manager.load_data_from_file(data_file)
        
        
        # Verificación adicional para asegurar que los tensores tienen dimensiones compatibles
        if input_tensor.shape[0] != output_tensor.shape[0]:
            logger.warning(f"¡Advertencia! Incompatibilidad en las dimensiones: input:{input_tensor.shape}, output:{output_tensor.shape}")
            # Ajustar las dimensiones para que coincidan
            min_samples = min(input_tensor.shape[0], output_tensor.shape[0])
            input_tensor = input_tensor[:min_samples]
            output_tensor = output_tensor[:min_samples]
        
        logger.info(f"Datos cargados con éxito: entrada={input_tensor.shape}, salida={output_tensor.shape}")
        
        # Para depuración, mostrar algunos valores
        logger.debug(f"Primeros valores de entrada: {input_tensor[:5]}")
        logger.debug(f"Primeros valores de salida: {output_tensor[:5]}")
        
    except Exception as e:
        logger.error(f"Error al cargar datos: {str(e)}")
        # Si no se pueden cargar los datos del archivo, generar datos sintéticos en su lugar
        logger.info("Intentando generar datos sintéticos como alternativa...")
        nu = float(physics_cfg.get("nu", 0.01))
        input_tensor, output_tensor = data_manager.create_synthetic_data(nu)
    
    # Crear modelo según configuración
    try:
        model = create_model(model_cfg)
        logger.info(f"Modelo {selected_model} creado con éxito")
    except Exception as e:
        logger.error(f"Error al crear modelo: {str(e)}")
        return
    
    # Configurar dispositivo (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        logger.info(f"Usando {torch.cuda.device_count()} GPUs con DataParallel.")
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    input_tensor = input_tensor.to(device)
    output_tensor = output_tensor.to(device)
    logger.info(f"Usando dispositivo: {device}")

    logger.info(f"Arquitectura del modelo:\n{model}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Obtener función de pérdida
    try:
        loss_fn = get_loss_function(loss_cfg)
        logger.info(f"Función de pérdida {selected_loss} cargada")
    except Exception as e:
        logger.error(f"Error al cargar función de pérdida: {str(e)}")
        return
    
    # Configurar parámetros adicionales para la pérdida
    loss_kwargs = loss_cfg.copy()
    loss_kwargs["model"] = model  # Necesario para pérdidas physics-informed
    loss_kwargs["nu"] = float(physics_cfg.get("nu", 0.01))
    
    # Inicializar trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer, 
        epochs=num_epochs,
        batch_size=batch_size,
        memory_limit_gb=float(train_cfg.get("memory_limit_gb", 4)),
        scheduler=None,
        early_stop_params={"enabled": enabled, "patience": early_stop_patience, "min_improvement": min_loss_improvement},       
        loss_fn=loss_fn,
        loss_kwargs=loss_kwargs,
        verbose=True
    )
    logger.info("Iniciando entrenamiento...")
    losses, accuracies, epoch_times, total_training_time, learning_rates = trainer.train(
        input_tensor=input_tensor, 
        output_tensor=output_tensor,
        **loss_kwargs
    )
    #Hacern falta los argumentos de la ecuacion difernecial a escoger
    # Visualizar métricas
    visualizer = Visualizer(OUTPUT_TRAIN_DIR, logger)
    visualizer.plot_metrics(losses, accuracies, epoch_times)

    # Evaluación final y métricas
    with torch.no_grad():
        # Aquí pasamos directamente el tensor de entrada al modelo
        y_pred = model(input_tensor)
    
    y_true = output_tensor.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    # Calcular métricas de regresión
    PRECISION = 4
    training_stats = {
        "final_loss": round(float(losses[-1].item()) if hasattr(losses[-1], "item") else float(losses[-1]), PRECISION),
        "final_accuracy": round(float(accuracies[-1].item()) if hasattr(accuracies[-1], "item") else float(accuracies[-1]), PRECISION),
        "total_training_time_sec": round(float(total_training_time), PRECISION),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), PRECISION),
        "MSE": round(float(mean_squared_error(y_true, y_pred)), PRECISION),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), PRECISION),
        "R2": round(float(r2_score(y_true, y_pred)), PRECISION),
        "losses": [round(float(l.item()) if hasattr(l, "item") else float(l), PRECISION) for l in losses],
        "accuracies": [round(float(a.item()) if hasattr(a, "item") else float(a), PRECISION) for a in accuracies],
        "epoch_times": [round(float(t.item()) if hasattr(t, "item") else float(t), PRECISION) for t in epoch_times]
    }

    stats_path = os.path.join(OUTPUT_TRAIN_DIR, "training_stats.json")
    with open(stats_path, "w") as f:
        json.dump(training_stats, f, indent=4)
    logger.info(f"Training statistics saved to: {stats_path}")

    # Graficar solución - Modificamos esta parte para extraer x, t de input_tensor
    try:
        # Extraer x, t de input_tensor
        x_np = input_tensor[:, 0].detach().cpu().numpy()
        t_np = input_tensor[:, 1].detach().cpu().numpy()
       
        # Obtener dimensiones originales
        nx = 256    
        nt = 100
        
        # Remodelar para la visualización
        x_grid = x_np.reshape(nx, nt)
        t_grid = t_np.reshape(nx, nt)
        u_pred_grid = y_pred.reshape(nx, nt)
        y_true_grid = y_true.reshape(nx, nt)
       
        # Graficar
        visualizer.plot_solution(
            t_grid,
            x_grid,
            u_pred_grid,
            y_true_grid,
            filename="solution_comparison.png"
        )
    except Exception as e:
        logger.error(f"Error al graficar la solución: {str(e)}")
        
        # Alternativa: graficar como scatter plot
        # visualizer
        visualizer.plot_prediction_vs_true(y_pred, y_true, 
                                           filename="prediction_vs_true.png")

    # Guardar modelo
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses.tolist(),
        'accuracies': accuracies.tolist(),
        'epoch_times': epoch_times.tolist()
    }, os.path.join(MODEL_DIR, "weights_model.pth"))
    logger.info(f"Modelo guardado en: {MODEL_DIR}")

    hyperparams_train_path = os.path.join(OUTPUT_TRAIN_DIR, "hyperparams_train.json")
    with open(hyperparams_train_path, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Hiperparámetros de entrenamiento guardados en: {hyperparams_train_path}")

    arch_path = os.path.join(MODEL_DIR, "model_architecture.txt")
    with open(arch_path, "w") as f:
        f.write(model.__repr__())
    logger.info(f"Arquitectura del modelo guardada en: {arch_path}")

    logger.info("Programa finalizado exitosamente.")

    # Al final del entrenamiento, generar informe resumido para este tipo de física
    output_manager = OutputManager(project_root=os.path.dirname(PROJECT_ROOT))
    summary_file = output_manager.generate_physics_summary(physics_type)
    if summary_file:
        logger.info(f"Resumen de modelos para {physics_type} generado en: {summary_file}")



def generate_data(config):
    """
    Genera datos sintéticos basado en la configuración especificada.
    
    Args:
        config (dict): Configuración cargada desde config.yaml
    """
    # Extraer parámetros relevantes del diccionario de configuración
    data_cfg = config.get("data_generation", {})
    physics_cfg = config.get("physics", {})
    
    # Crear directorio para datos generados
    output_dir = os.path.join(PROJECT_ROOT, "..", "data", "synthetic", 
                             f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurar parámetros
    spatial_points = data_cfg.get("spatial_points", 256)
    time_points = data_cfg.get("time_points", 100)
    
    
    # Inicializar generador de datos - Cambiar FluidDataGenerator por DataManager
    logger.info(f"Inicializando generador de datos con {spatial_points} puntos espaciales y {time_points} puntos temporales.")
    generator = DataManager(spatial_points=spatial_points, time_points=time_points, output_dir=output_dir)
    
    # Determinar tipo de datos a generar
    data_type = data_cfg.get("type", "burgers")
    
    if data_type == "burgers":
        # Generar solución de Burgers 1D
        nu = physics_cfg.get("nu", 0.01)
        logger.info(f"Generando solución de Burgers 1D con nu={nu}...")
        x, t, usol = generator.burgers_synthetic_solution(nu=nu, save=True)
        logger.info(f"Solución de Burgers generada: {usol.shape}")
        
    elif data_type == "kovasznay":
        # Parámetros adicionales para Kovasznay
        Re = data_cfg.get("Re", 40)
        N = data_cfg.get("resolution", 32)
        logger.info(f"Generando flujo de Kovasznay con Re={Re}, N={N}...")
        try:
            X, Y, UV, P = generator.generate_kovasznay_flow(Re=Re, N=N, save=True)
            logger.info(f"Solución de Kovasznay generada: {UV.shape}")
        except Exception as e:
            logger.error(f"Error al generar flujo de Kovasznay: {str(e)}")
            
    elif data_type == "taylor_green":
        # Parámetros para Taylor-Green
        nu = data_cfg.get("nu", 0.01)
        U0 = data_cfg.get("U0", 1.0)
        Nx = data_cfg.get("Nx", 32)
        Ny = data_cfg.get("Ny", 32)
        T = data_cfg.get("T", 2.0)
        num_steps = data_cfg.get("num_steps", 50)
        
        logger.info(f"Generando vórtice de Taylor-Green...")
        try:
            X, Y, UV, P = generator.generate_taylor_green_vortex(nu=nu, U0=U0, Nx=Nx, Ny=Ny, 
                                                               T=T, num_steps=num_steps, save=True)
            logger.info(f"Solución de Taylor-Green generada: {UV.shape}")
        except Exception as e:
            logger.error(f"Error al generar vórtice de Taylor-Green: {str(e)}")
            
    elif data_type == "lid_driven_cavity":
        # Parámetros para cavidad con tapa móvil
        nu = data_cfg.get("nu", 0.01)
        U0 = data_cfg.get("U0", 1.0)
        N = data_cfg.get("resolution", 32)
        T = data_cfg.get("T", 2.0)
        num_steps = data_cfg.get("num_steps", 50)
        
        logger.info(f"Generando flujo en cavidad con tapa móvil...")
        try:
            X, Y, UV, P = generator.generate_lid_driven_cavity(nu=nu, U0=U0, N=N, 
                                                            T=T, num_steps=num_steps, save=True)
            logger.info(f"Solución de cavidad generada: {UV.shape}")
        except Exception as e:
            logger.error(f"Error al generar flujo de cavidad: {str(e)}")
            
    else:
        logger.error(f"Tipo de datos '{data_type}' no reconocido.")
        return
    
    # Guardar informe de generación
    report_file = generator.save_generation_report()
    logger.info(f"Informe de generación guardado en: {report_file}")
    logger.info(f"Todos los datos generados se encuentran en: {output_dir}")

# Agregar importación


def evaluate_model(config):
    """
    Evalúa un modelo entrenado según la configuración proporcionada.
    Permite seleccionar entre evaluación de un modelo específico
    o comparación de todos los modelos disponibles.
    
    Args:
        config (dict): Configuración cargada desde config.yaml
    """
    # Extraer parámetros relevantes
    eval_cfg = config.get("evaluation", {})
    physics_cfg = config.get("physics", {})
    physics_type = physics_cfg.get("physics_type", "burgers")  # Tipo de física
    
    # Configurar modelo a evaluar - Ahora busca modelos si no se encuentra el especificado
    model_path = eval_cfg.get("model_path", "")
    
    # Verificar si existe el modelo especificado
    if not os.path.exists(model_path):
        logger.warning(f"No se encontró el modelo en la ruta especificada: {model_path}")
        logger.info("Buscando modelos disponibles en el sistema...")
        
        # Solicitar selección interactiva
        model_path = ModelFinder.select_model_interactive()
        
        if not model_path:
            logger.error("No se seleccionó ningún modelo para evaluar.")
            return
    
    # Parámetros para la evaluación
    data_path = eval_cfg.get("data_path", "")
    output_folder = eval_cfg.get("output_folder", "evaluation_results")
    nu = float(physics_cfg.get("nu", 0.01))
    
    # Buscar archivo de datos si no existe
    import glob
    if not os.path.exists(data_path):
        logger.warning(f"No se encontró el archivo de datos en: {data_path}")
        
        # Intentar buscar en ubicaciones típicas
        data_dir = os.path.join(PROJECT_ROOT, "..", "data", "training")
        data_files = glob.glob(os.path.join(data_dir, "*.mat"))
        
        if data_files:
            print("\n=== ARCHIVOS DE DATOS DISPONIBLES ===")
            for i, file_path in enumerate(data_files, 1):
                print(f"{i}. {os.path.basename(file_path)}")
            
            try:
                choice = int(input("\nSeleccione un archivo de datos (número): ")) - 1
                if 0 <= choice < len(data_files):
                    data_path = data_files[choice]
                    logger.info(f"Usando archivo de datos: {data_path}")
                else:
                    logger.error("Selección no válida.")
                    return
            except ValueError:
                logger.error("Selección no válida.")
                return
        else:
            logger.error("No se encontraron archivos de datos disponibles.")
            return
    
    # Configuración de salida usando OutputManager
    output_manager = OutputManager(project_root=os.path.dirname(PROJECT_ROOT))
    
    # Obtener información del modelo para ver si podemos determinar su tipo de física
    model_info = ModelFinder.get_model_info(model_path)
    model_name = model_info.get("name", os.path.basename(os.path.dirname(model_path)))
    
    # Intentar extraer el tipo de física del modelo si está disponible
    model_physics_type = physics_type
    if model_info.get("hyperparams"):
        try:
            model_physics_type = model_info["hyperparams"].get("physics", {}).get("physics_type", physics_type)
        except:
            pass
    
    # Determinar si el directorio de resultados debe estar en el directorio del modelo
    # o crear uno nuevo específico para esta evaluación
    model_dir = os.path.dirname(os.path.dirname(model_path))
    output_dir = output_manager.create_evaluation_dir(model_name, model_physics_type, model_dir)
    
    # Inicializar evaluador
    evaluator = ModelEvaluator(output_dir=output_dir, logger=logger)
    
    # Extraer nombre del modelo
    model_info = ModelFinder.get_model_info(model_path)
    model_name = model_info.get("name", os.path.basename(os.path.dirname(model_path)))
    
    # Evaluar el modelo
    try:
        logger.info(f"Evaluando modelo: {model_name}")
        metrics = evaluator.evaluate_model(model_path, data_path, nu, model_name)
        logger.info(f"Evaluación completada para {model_name}")
        
        # Generar informe
        report_path = evaluator.generate_full_report()
        logger.info(f"Informe guardado en: {report_path}")
        
        # Mostrar resultados detallados en pantalla
        print("\n" + "="*60)
        print(f"    RESULTADOS DE EVALUACIÓN: {model_name.upper()}")
        print("="*60)
        
        # Información general del modelo
        print("\n[INFORMACIÓN DEL MODELO]")
        print(f"• Nombre: {model_name}")
        print(f"• Ruta: {model_path}")
        
        # Detalles del modelo si están disponibles
        if model_info.get("hyperparams"):
            try:
                physics_type = model_info["hyperparams"].get("physics", {}).get("physics_type", "desconocido")
                selected_model = model_info["hyperparams"].get("selected_model", "desconocido")
                layers = model_info["hyperparams"]["models"][selected_model].get("layers", [])
                activation = model_info["hyperparams"]["models"][selected_model].get("activation_function", "desconocido")
                
                print(f"• Física: {physics_type}")
                print(f"• Tipo: {selected_model}")
                print(f"• Arquitectura: {layers}")
                print(f"• Activación: {activation}")
                
                # Mostrar parámetros de entrenamiento
                if "training" in model_info["hyperparams"]:
                    train_cfg = model_info["hyperparams"]["training"]
                    print(f"• Épocas: {train_cfg.get('epochs', 'N/A')}")
                    print(f"• Batch size: {train_cfg.get('batch_size', 'N/A')}")
            except Exception as e:
                print(f"• No se pudieron extraer todos los detalles: {str(e)}")
        
        # Métricas de rendimiento
        print("\n[MÉTRICAS DE RENDIMIENTO]")
        print(f"• MSE: {metrics.get('MSE', 'N/A'):.6f}")
        print(f"• MAE: {metrics.get('MAE', 'N/A'):.6f}")
        print(f"• RMSE: {metrics.get('RMSE', 'N/A'):.6f}")
        print(f"• R²: {metrics.get('R2', 'N/A'):.6f}")
        
        # Uso de recursos
        print("\n[USO DE RECURSOS]")
        print(f"• Tiempo de inferencia: {metrics.get('inference_time', 'N/A'):.4f} segundos")
        print(f"• Uso de memoria: {metrics.get('memory_usage_mb', 'N/A'):.2f} MB")
        
        # Ubicación de resultados
        print("\n[RESULTADOS GUARDADOS]")
        print(f"• Carpeta de resultados: {output_dir}")
        print(f"• Reporte completo: {report_path}")
        
        # Mostrar las visualizaciones generadas
        print("\n[VISUALIZACIONES GENERADAS]")
        viz_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(output_dir) 
                    for f in filenames if f.endswith('.png')]
        
        for i, viz_file in enumerate(viz_files, 1):
            print(f"• Visualización {i}: {os.path.basename(viz_file)}")
        
        # Si hay visualizaciones, preguntar si se desean mostrar
        import matplotlib.pyplot as plt
        if viz_files and input("\n¿Mostrar visualizaciones ahora? (s/n): ").lower() == 's':
            for viz_file in viz_files:
                if os.path.exists(viz_file):
                    # Usar la biblioteca adecuada para mostrar imágenes según el sistema
                    try:
                        img = plt.imread(viz_file)
                        plt.figure(figsize=(12, 8))
                        plt.imshow(img)
                        plt.axis('off')
                        plt.title(os.path.basename(viz_file))
                        plt.show()
                    except Exception as e:
                        print(f"No se pudo mostrar {os.path.basename(viz_file)}: {str(e)}")
        
        # Preguntar si se desea comparar con otros modelos
        compare = input("\n¿Desea comparar con otros modelos? (s/n): ").lower() == 's'
        if compare:
            # Inicializar comparador
            comparator = ModelComparator(output_dir=output_dir, evaluator=evaluator)
            
            # Directorio donde buscar modelos (output por defecto)
            models_root_dir = os.path.join(PROJECT_ROOT, "output")
            
            # Escanear modelos disponibles
            model_catalog = comparator.scan_for_models(models_root_dir)
            
            if model_catalog:
                # Evaluar todos los modelos
                comparison_df = comparator.evaluate_all_models(data_path, nu)
                
                if comparison_df is not None:
                    logger.info(f"Evaluación comparativa completada. Se analizaron {len(comparison_df)} modelos.")
                    
                    # Ordenar y seleccionar las columnas relevantes
                    top_models = comparison_df.sort_values("MSE")[["model_name", "MSE", "MAE", "R2", "inference_time"]]
                    
                    # Convertir a representación de texto
                    pd.set_option('display.max_columns', None)
                    pd.set_option('display.width', 200)
                    comparison_table = top_models.to_string()
                    
                    # Guardar en un archivo de texto
                    comparison_file = os.path.join(output_dir, "model_comparison_table.txt")
                    with open(comparison_file, 'w') as f:
                        f.write("COMPARACIÓN DE MODELOS\n")
                        f.write("=====================\n\n")
                        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Archivo de datos: {os.path.basename(data_path)}\n")
                        f.write(f"Total de modelos evaluados: {len(comparison_df)}\n\n")
                        f.write(comparison_table)
                        f.write("\n\nNota: Los modelos están ordenados de mejor a peor según MSE.")
                    
                    # Mostrar en el log
                    logger.info("\n[TABLA COMPARATIVA DE MODELOS]")
                    for line in comparison_table.split('\n'):
                        logger.info(line)
                    logger.info(f"Tabla comparativa guardada en: {comparison_file}")
                    
                    # Notificar al usuario
                    print(f"\nTabla comparativa guardada en: {comparison_file}")
                    
                    # Preguntar si se desea ver gráficos comparativos
                    if input("\n¿Mostrar gráficos comparativos? (s/n): ").lower() == 's':
                        comparison_viz = os.path.join(output_dir, "model_comparison_metrics.png")
                        if os.path.exists(comparison_viz):
                            img = plt.imread(comparison_viz)
                            plt.figure(figsize=(14, 10))
                            plt.imshow(img)
                            plt.axis('off')
                            plt.title("Comparación de Modelos")
                            plt.show()
                            
                            # Agregar información al archivo
                            with open(comparison_file, 'a') as f:
                                f.write(f"\n\nGráficos comparativos disponibles en: {os.path.basename(comparison_viz)}")
            else:
                logger.warning("No se encontraron modelos adicionales para comparar.")
        
    except Exception as e:
        logger.error(f"Error al evaluar el modelo: {str(e)}")
        import traceback
        traceback.print_exc()
        
    # Al final de la evaluación, generar informe resumido para este tipo de física
    summary_file = output_manager.generate_physics_summary(model_physics_type)
    if summary_file:
        logger.info(f"Resumen actualizado de modelos para {model_physics_type} generado en: {summary_file}")
    
    return output_dir

def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='PINN-BGK: Red Neuronal con Física Incorporada')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'generate', 'evaluate'],
                       help='Modo de ejecución: entrenar, generar datos o evaluar modelo (default: train)')
    parser.add_argument('--config', type=str, default=None,
                       help='Ruta a un archivo de configuración YAML personalizado (opcional)')
    args = parser.parse_args()

    # Cargar configuración
    if args.config and os.path.exists(args.config):  # Corregido: '&&' a 'and'
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuración cargada desde archivo personalizado: {args.config}")
    else:
        config = load_config()
        logger.info("Configuración cargada desde archivo por defecto.")
    
    # Ejecutar el modo seleccionado
    if args.mode == 'generate':
        logger.info("Modo de generación de datos activado.")
        generate_data(config)
    elif args.mode == 'evaluate':
        logger.info("Modo de evaluación de modelo activado.")
        evaluate_model(config)
    else:
        logger.info("Modo de entrenamiento activado.")
        train_model(config)
    
    logger.info(f"Programa finalizado exitosamente en modo: {args.mode}.")

if __name__ == '__main__':
    # Configurar logger
    if logger.hasHandlers():
        logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    main()
