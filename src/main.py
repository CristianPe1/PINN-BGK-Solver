import os
import sys
import json
import yaml
import torch
import logging
import argparse
import numpy as np
from datetime import datetime

# Importaciones locales
from model.train import Trainer
from model.loss_functions import get_loss_function
from utils.visualization import Visualizer
from utils.model_evaluator import ModelEvaluator
from data_handlers.data_manager import DataManager
from structure_model.model_factory import create_model

# Establecer PROJECT_ROOT
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configuración de logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levellevel)s - %(message)s')

def load_config():
    config_path = os.path.join(PROJECT_ROOT, "..", "config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def create_output_dir(model_name, epochs, lr, seed, output_folder):
    output_dir = os.path.join(PROJECT_ROOT, output_folder, f"Model_{model_name}_{epochs}_{lr}_{seed}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

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

def train_model(config):
    """Función principal para entrenar el modelo."""
    # Seleccionar modelo y función de pérdida según config
    selected_model = config.get("selected_model", "pinn_v1")
    selected_loss = config.get("selected_loss", "mse")
    
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

    # Extraer parámetros de entrenamiento
    num_epochs = int(train_cfg["epochs"])
    lr = float(model_cfg.get("learning_rate", 0.001))
    seed = int(train_cfg.get("seed", 42))
    batch_size = int(train_cfg.get("batch_size", 32))
    early_stop_patience = int(train_cfg.get("epochs_no_improve", 20))
    min_loss_improvement = float(train_cfg.get("min_loss_improvement", 1e-5))
    enabled = train_cfg.get("early_stopping", True)

    # Crear directorios de salida
    model_name = selected_model
    OUTPUT_DIR = create_output_dir(model_name, num_epochs, lr, seed, log_cfg.get("output_folder", "output"))
    OUTPUT_TRAIN_DIR = os.path.join(OUTPUT_DIR, "train_results")
    os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
    MODEL_DIR = os.path.join(OUTPUT_DIR, "model")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Cargar datos
    DATA_DIR = os.path.join(PROJECT_ROOT, "..", "data", "training")
    data_file = os.path.join(DATA_DIR, "burgers_shock_mu_01_pi.mat")
    data_manager = DataManager(256, 100)
    
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

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
        
        print(x_np.shape)
        print(t_np.shape)            
        
        print(x_np.shape)
        print(t_np.shape)
        
        # Obtener dimensiones originales
        nx = 256    
        nt = 100
        
        # Remodelar para la visualización
        x_grid = x_np.reshape(nx, nt)
        t_grid = t_np.reshape(nx, nt)
        u_pred_grid = y_pred.reshape(nx, nt)
        y_true_grid = y_true.reshape(nx, nt)
        
        # print("x_grid")
        # print(x_grid.shape)
        # print("t_grid")
        # print(t_grid.shape)
        # print("u_pred_grid")
        # print(u_pred_grid.shape)
        # print("y_true_grid")
        # print(y_true_grid.shape)
        
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

def evaluate_model(config):
    """
    Evalúa un modelo entrenado según la configuración proporcionada.
    
    Args:
        config (dict): Configuración cargada desde config.yaml
    """
    # Extraer parámetros relevantes
    eval_cfg = config.get("evaluation", {})
    physics_cfg = config.get("physics", {})
    
    # Ruta al modelo a evaluar
    model_path = eval_cfg.get("model_path", "")
    if not os.path.exists(model_path):
        logger.error(f"No se encontró el modelo en: {model_path}")
        return
    
    # Configuración de salida
    output_dir = os.path.join(PROJECT_ROOT, eval_cfg.get("output_folder", "evaluation_results"), 
                            f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluar con el evaluador de modelos
    evaluator = ModelEvaluator(output_dir=output_dir, logger=logger)
    
    # Evaluar modelo individual
    if eval_cfg.get("evaluate_single", True):
        data_path = eval_cfg.get("data_path", "")
        model_name = eval_cfg.get("model_name", "Modelo Principal")
        nu = float(physics_cfg.get("nu", 0.01))
        
        if not os.path.exists(data_path):
            logger.error(f"No se encontró el archivo de datos en: {data_path}")
            return
        
        data_manager = DataManager(256, 100)
        try:
            # Cargar datos como tensores de entrada y salida
            input_tensor, output_tensor = data_manager.load_data_from_file(data_path)
            logger.info(f"Datos de evaluación cargados: entrada={input_tensor.shape}, salida={output_tensor.shape}")
        except Exception as e:
            logger.error(f"Error al cargar datos de evaluación: {str(e)}")
            return
        
        # Cargar y evaluar modelo
        model = evaluator.load_model(model_path)
        metrics = evaluator.evaluate_model(model, data_path, nu, model_name)
        logger.info(f"Evaluación completada para {model_name}")
    
    # Evaluar múltiples modelos si se ha configurado
    if eval_cfg.get("compare_models", False):
        models_config = eval_cfg.get("models_to_compare", [])
        if models_config:
            common_data_path = eval_cfg.get("data_path", "")
            comparison_df = evaluator.batch_evaluate_models(models_config, common_data_path)
            logger.info("Comparación de modelos completada")
    
    # Generar informe completo
    report_file = evaluator.generate_full_report()
    logger.info(f"Informe de evaluación guardado en: {report_file}")
    
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
    if args.config and os.path.exists(args.config):
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
