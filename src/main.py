import os
import sys
import json
import yaml
import torch
import logging
import argparse
import numpy as np
import torch.optim as optim
from datetime import datetime

from model.train import Trainer
from utils.data_loader import DataLoader
from utils.visualization import Visualizer
from structure_model.pinn_structure_v1 import PINN_V1
from data_handlers.fluid_data_generator import FluidDataGenerator

# Establecer PROJECT_ROOT
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configuración de logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    config_path = os.path.join(PROJECT_ROOT, "..", "config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def create_output_dir(model_type, epochs, lr, seed, output_folder):
    output_dir = os.path.join(PROJECT_ROOT, output_folder, f"Model_{model_type}_{epochs}_{lr}_{seed}")
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
    nu = physics_cfg.get("nu", 0.01)
    
    # Inicializar generador de datos
    logger.info(f"Inicializando generador de datos con {spatial_points} puntos espaciales y {time_points} puntos temporales.")
    generator = FluidDataGenerator(spatial_points=spatial_points, time_points=time_points, output_dir=output_dir)
    
    # Determinar tipo de datos a generar
    data_type = data_cfg.get("type", "burgers")
    
    if data_type == "burgers":
        # Generar solución de Burgers 1D
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
    model_cfg    = config["model"]
    train_cfg    = config["training"]
    physics_cfg  = config["physics"]
    log_cfg      = config["logging"]

    num_epochs   = int(train_cfg["epochs"])
    lr           = float(model_cfg["learning_rate"])
    seed         = int(train_cfg["seed"])
    batch_size   = int(train_cfg["batch_size"])
    early_stop_patience = int(train_cfg["epochs_no_improve"])
    min_loss_improvement  = float(train_cfg["min_loss_improvement"])
    layers       = model_cfg["layers"]
    model_type   = model_cfg["type"]

    # Crear directorios de salida basado en la configuración YAML
    OUTPUT_DIR = create_output_dir(model_type, num_epochs, lr, seed, log_cfg["output_folder"])
    OUTPUT_TRAIN_DIR = os.path.join(OUTPUT_DIR, "train_results")
    os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
    MODEL_DIR = os.path.join(OUTPUT_DIR, "model")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Cargar datos
    DATA_DIR = os.path.join(PROJECT_ROOT, "..", "data", "training")
    data_file = os.path.join(DATA_DIR, "burgers_shock_mu_01_pi.mat")
    data_loader = DataLoader(256, 100)
    x_train, t_train, u_train = data_loader.load_data_from_file(data_file)

    # Ajustar dimensiones
    x_train = DataLoader.prepare_tensor(x_train)
    t_train = DataLoader.prepare_tensor(t_train)
    u_train = DataLoader.prepare_tensor(u_train)

    # Inicializar modelo
    logger.info("Inicializando modelo PINN_V1.")
    model = PINN_V1(layers, model_cfg.get("activation_function", "Tanh"))
    if torch.cuda.device_count() > 1:
        logger.info(f"Usando {torch.cuda.device_count()} GPUs con DataParallel.")
        model = torch.nn.DataParallel(model).cuda()
    elif torch.cuda.device_count() == 1:
        logger.info("Usando GPU.")
        model = model.cuda()
    else:
        logger.info("Usando CPU.")

    logger.info(f"Arquitectura del modelo:\n{model}")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Entrenar
    trainer = Trainer(model, optimizer, num_epochs, batch_size, memory_limit_gb=float(train_cfg.get("memory_limit_gb", 4)),
                      early_stop_patience=early_stop_patience,
                      min_loss_improvement=min_loss_improvement, verbose=True)
    logger.info("Iniciando entrenamiento...")
    losses, accuracies, epoch_times, total_training_time = trainer.train(x_train, t_train, u_train, nu=float(physics_cfg["nu"]))

    # Visualizar métricas
    visualizer = Visualizer(OUTPUT_TRAIN_DIR, logger)
    visualizer.plot_metrics(losses, accuracies, epoch_times)

    # Evaluación final y métricas
    with torch.no_grad():
        u_pred = model(x_train, t_train).detach()
    u_train_clean = u_train.squeeze().detach().cpu().numpy()
    u_pred_clean  = u_pred.squeeze().detach().cpu().numpy()

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    PRECISION = 4
    training_stats = {
        "final_loss": round(float(losses[-1].item()) if hasattr(losses[-1], "item") else float(losses[-1]), PRECISION),
        "final_accuracy": round(float(accuracies[-1].item()) if hasattr(accuracies[-1], "item") else float(accuracies[-1]), PRECISION),
        "total_training_time_sec": round(float(total_training_time), PRECISION),
        "MAE": round(float(mean_absolute_error(u_train_clean, u_pred_clean)), PRECISION),
        "MSE": round(float(mean_squared_error(u_train_clean, u_pred_clean)), PRECISION),
        "RMSE": round(float(np.sqrt(mean_squared_error(u_train_clean, u_pred_clean))), PRECISION),
        "R2": round(float(r2_score(u_train_clean, u_pred_clean)), PRECISION),
        "losses": [round(float(l.item()) if hasattr(l, "item") else float(l), PRECISION) for l in losses],
        "accuracies": [round(float(a.item()) if hasattr(a, "item") else float(a), PRECISION) for a in accuracies],
        "epoch_times": [round(float(t.item()) if hasattr(t, "item") else float(t), PRECISION) for t in epoch_times]
    }

    stats_path = os.path.join(OUTPUT_TRAIN_DIR, "training_stats.json")
    with open(stats_path, "w") as f:
        json.dump(training_stats, f, indent=4)
    logger.info(f"Training statistics saved to: {stats_path}")

    # Graficar solución
    visualizer.plot_solution(t_train.squeeze().detach().cpu().numpy(),
                             x_train.squeeze().detach().cpu().numpy(),
                             u_pred.squeeze().detach().cpu().numpy(),
                             u_train.squeeze().detach().cpu().numpy(),
                             filename="solution_comparison.png")

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
    
    # Cargar modelo
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Intentar extraer la arquitectura desde los metadatos o usar la configuración actual
    try:
        model_metadata_path = os.path.join(os.path.dirname(model_path), "hyperparams_train.json")
        if os.path.exists(model_metadata_path):
            with open(model_metadata_path, 'r') as f:
                model_metadata = json.load(f)
            layers = model_metadata.get("model", {}).get("layers", [2, 50, 50, 50, 50, 1])
            activation = model_metadata.get("model", {}).get("activation_function", "Tanh")
            logger.info(f"Arquitectura de modelo cargada desde metadatos: {layers}")
        else:
            layers = config["model"]["layers"]
            activation = config["model"].get("activation_function", "Tanh")
            logger.info(f"Usando arquitectura de modelo de config.yaml: {layers}")
    except Exception as e:
        logger.warning(f"Error al cargar metadatos del modelo: {str(e)}. Usando configuración actual.")
        layers = config["model"]["layers"]
        activation = config["model"].get("activation_function", "Tanh")
    
    # Inicializar modelo con la arquitectura correcta
    model = PINN_V1(layers, activation)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Modo evaluación
    logger.info(f"Modelo cargado desde: {model_path}")
    
    # Cargar datos de evaluación
    eval_data_path = eval_cfg.get("data_path", "")
    if not os.path.exists(eval_data_path):
        logger.error(f"No se encontró el archivo de datos en: {eval_data_path}")
        return
    
    # Cargar datos
    data_loader = DataLoader(256, 100)
    x_eval, t_eval, u_eval = data_loader.load_data_from_file(eval_data_path)
    
    # Preparar tensores
    x_eval = DataLoader.prepare_tensor(x_eval)
    t_eval = DataLoader.prepare_tensor(t_eval)
    u_eval = DataLoader.prepare_tensor(u_eval)
    
    logger.info(f"Datos de evaluación cargados, tamaño: {u_eval.shape}")
    
    # Evaluar modelo
    with torch.no_grad():
        u_pred = model(x_eval, t_eval)
    
    # Calcular métricas
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    u_eval_np = u_eval.squeeze().detach().cpu().numpy()
    u_pred_np = u_pred.squeeze().detach().cpu().numpy()
    
    mae = mean_absolute_error(u_eval_np, u_pred_np)
    mse = mean_squared_error(u_eval_np, u_pred_np)
    rmse = np.sqrt(mse)
    r2 = r2_score(u_eval_np, u_pred_np)
    
    # Guardar resultados
    eval_results = {
        "model_path": model_path,
        "data_path": eval_data_path,
        "metrics": {
            "MAE": float(mae),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "R2": float(r2)
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "physics_params": {
            "nu": float(physics_cfg.get("nu", 0.01))
        }
    }
    
    # Guardar resultados como JSON
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=4)
    logger.info(f"Resultados de evaluación guardados en: {results_path}")
    
    # Visualizar resultados
    visualizer = Visualizer(output_dir, logger)
    visualizer.plot_solution(
        t_eval.squeeze().detach().cpu().numpy(),
        x_eval.squeeze().detach().cpu().numpy(),
        u_pred.squeeze().detach().cpu().numpy(),
        u_eval.squeeze().detach().cpu().numpy(),
        filename="evaluation_comparison.png"
    )
    
    # Calcular error punto a punto
    error = np.abs(u_pred_np - u_eval_np)
    
    # Visualizar error
    visualizer.plot_error_heatmap(
        t_eval.squeeze().detach().cpu().numpy(),
        x_eval.squeeze().detach().cpu().numpy(),
        error,
        filename="error_heatmap.png"
    )
    
    # Imprimir resumen de métricas
    logger.info("\n" + "="*50)
    logger.info("RESULTADOS DE EVALUACIÓN:")
    logger.info(f"MAE:  {mae:.6f}")
    logger.info(f"MSE:  {mse:.6f}")
    logger.info(f"RMSE: {rmse:.6f}")
    logger.info(f"R2:   {r2:.6f}")
    logger.info("="*50)
    
    return eval_results

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
