import os
import sys
import json
import yaml  # Asegúrate de tener PyYAML instalado: pip install pyyaml
import torch
import logging
import numpy as np
import torch.optim as optim

from model.train import Trainer  # Entrenador que usa early stopping integrado.
from utils.data_loader import DataLoader
from utils.visualization import Visualizer
from structure_model.pinn_structure_v1 import PINN_V1

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

def main():
    # Cargar configuración del YAML
    config = load_config()
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

if __name__ == '__main__':
    if logger.hasHandlers():
        logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    main()
