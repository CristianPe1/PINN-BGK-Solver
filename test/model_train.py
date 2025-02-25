import os
import sys
import json  # Nuevo para leer y guardar JSON
import time
import torch
import logging
import platform
import subprocess
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pinn_structure_v1 import PINN_V1
from pinn_structure_v2 import PINN_V2
from diferential_equation_loss import burgers_pde_loss
from device_utils import get_dynamic_batch_size, monitor_device_usage
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Agregar la ruta del directorio raíz del proyecto para forzar la importación del pinn_structure central
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Definir rutas globales
TOTAL_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TOTAL_PATH, "data/training")
os.makedirs(DATA_DIR, exist_ok=True)

# Configurar logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
if logger.hasHandlers():
    logger.handlers.clear()
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

class EarlyStopping:
    def __init__(self, monitor="loss", epochs_no_improve=10, min_loss_improvement=0, verbose=False):
        self.monitor = monitor
        self.epochs_no_improve = epochs_no_improve
        self.min_loss_improvement = min_loss_improvement
        self.verbose = verbose
        
        if monitor in ["loss", "mse", "pde_residual"]:
            self.best_value = float("inf")
            self.improvement = lambda current, best: current < best - self.min_loss_improvement
        else:
            self.best_value = -float("inf")
            self.improvement = lambda current, best: current > best + self.min_loss_improvement
            
        self.counter = 0
        self.early_stop = False

    def __call__(self, current):
        if self.improvement(current, self.best_value):
            self.best_value = current
            self.counter = 0
            if self.verbose:
                logger.info(f"Mejora detectada: nuevo valor {self.best_value:.4f}. Reiniciando contador.")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"Sin mejora significativa: contador {self.counter}/{self.epochs_no_improve}.")
            if self.counter >= self.epochs_no_improve:
                self.early_stop = True
                if self.verbose:
                    logger.info("Se activa Early Stopping.")

# Se añade parámetro model_type para diferenciar carpeta de salida
def create_output_dir(model_type, num_epochs, lr, seed):
    """Crea un directorio para guardar los resultados según el tipo de modelo."""
    output_dir = os.path.join(TOTAL_PATH, "output", f"burgers_{model_type}_{num_epochs}_{lr}_{seed}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

class DataLoader:
    """Genera datos sintéticos para la ecuación de Burgers"""
    def __init__(self, spatial_points, time_points, nu):
        self.spatial_points = spatial_points
        self.time_points = time_points
        self.nu = nu

    def exact_solution(self, x, t):
        return -torch.tanh(x / (2 * self.nu * t + 1e-8))

    def generate_data(self):
        x = np.linspace(-1, 1, self.spatial_points)
        t = np.linspace(0, 1, self.time_points)
        X, T = torch.meshgrid(torch.tensor(x), torch.tensor(t))
        u0 = self.exact_solution(X, T)
        return X, T, u0

    def load_data(self, file_name):
        data = loadmat(file_name)
        x_train = torch.tensor(data['x'].flatten(), dtype=torch.float32)
        t_train = torch.tensor(data['t'].flatten(), dtype=torch.float32)
        u0_train = torch.tensor(data['usol'], dtype=torch.float32)
        X, T = torch.meshgrid(x_train, t_train)
        return X, T, u0_train



def calculate_accuracy(model, features, targets) -> float:
    """
    Calcula precisión como 1 - error relativo normalizado.
    Maneja tensores 3D correctamente.
    """
    model.eval()
    with torch.no_grad():
        if isinstance(features, (tuple, list)):
            predictions = model(*features)
        else:
            predictions = model(features)
        
        # Asegurar que las formas coincidan
        if predictions.shape != targets.shape:
            targets = targets.view_as(predictions)
        
        # Calcular error relativo y evitar divisiones extremas
        denominator = torch.clamp(torch.abs(targets), min=1e-8)
        error = torch.abs(predictions - targets) / denominator
        
        # Limitar el error a un rango razonable
        error = torch.clamp(error, min=0.0, max=1.0)
        
        # Calcular precisión
        accuracy = 1 - torch.mean(error)
        # accuracy = torch.clamp(accuracy, min=0.0, max=1.0)
    
    return max(0.0, accuracy.item())  # Evitar precisiones negativas


def train_model(model, optimizer, num_epochs, early_stopping, x_train, t_train, u_train, 
                nu, batch_size=256, memory_limit_gb=2.0):
    """Entrena el modelo usando la ecuación de Burgers."""
    start_time = time.time()
    loss_history = np.zeros(num_epochs)
    accuracy_history = np.zeros(num_epochs)
    epoch_time_history = np.zeros(num_epochs)
    
    # Monitorear uso de dispositivo y ajustar batch size
    logger.info(f"Usando batch size: {batch_size}")
    device_load = monitor_device_usage()
    batch_size = get_dynamic_batch_size(device_load, batch_size, memory_limit_gb)
    
    
    with tqdm(total=num_epochs, desc="Entrenando la PINN", dynamic_ncols=True) as pbar:
        for epoch in range(num_epochs):
            epoch_start = time.time()
            # optimizer.zero_grad()
            
            # Cálculo de pérdidas
            # loss_data = torch.mean((model(x_train, t_train) - u_train) ** 2)
            # loss_pde = burgers_pde_loss(model, x_train, t_train, nu)
            # loss = loss_data + loss_pde
            
            # loss.backward()
            
            # Reorganizamos los datos para mini-batches: aplanamos las matrices para tener muestras individuales
            inputs = torch.cat((x_train.reshape(-1, 1), t_train.reshape(-1, 1)), dim=1)
            targets = u_train.reshape(-1, 1)
            permutation = torch.randperm(inputs.size(0))
            
            batch_loss = 0.0
            num_batches = 0
            for i in range(0, inputs.size(0), batch_size):
                indices = permutation[i:i+batch_size]
                batch_inputs = inputs[indices]
                batch_x = batch_inputs[:, 0].unsqueeze(1).clone().detach().requires_grad_(True)
                batch_t = batch_inputs[:, 1].unsqueeze(1).clone().detach().requires_grad_(True)
                batch_targets = targets[indices]
                
                optimizer.zero_grad()
                pred = model(batch_x, batch_t)
                loss_data = torch.mean((pred - batch_targets) ** 2)
                loss_pde = burgers_pde_loss(model, batch_x, batch_t, nu)
                loss = loss_data + loss_pde
                loss.backward()
                optimizer.step()
                
                batch_loss += loss.item()
                num_batches += 1
                
            # Promediar la pérdida de todos los mini-batches
            loss = torch.tensor(batch_loss / num_batches)
            
            # Calcular y registrar métricas
            accuracy = calculate_accuracy(model, (x_train, t_train), u_train)
            
            accuracy_history[epoch] = accuracy
            loss_history[epoch] = loss.item()
            epoch_time_history[epoch] = time.time() - epoch_start
            
            # Actualizar barra de progreso
            epoch_time = time.time() - epoch_start
            estimated_time = (epoch_time * (num_epochs - epoch)) / 60
            pbar.set_postfix({
                "Loss": f"{loss.item():.6f}",
                "Acc": f"{accuracy:.6f}",
                "ETC": f"{estimated_time:.2f}m"
            })
            pbar.update(1)
            

            # Early stopping
            early_stopping(loss.item())
            if early_stopping.early_stop:
                logger.info(f"Early stopping activado en la época {epoch}")
                break
    
    total_time = time.time() - start_time
    logger.info(f"Entrenamiento finalizado en {total_time/60:.2f} minutos")
    
    # Encontrar el último elemento de loss_history diferente de cero
    final_loss = np.trim_zeros(loss_history, 'b')[-1] if np.any(loss_history != 0) else 0.0
    final_accuracy = np.trim_zeros(accuracy_history, 'b')[-1] if np.any(accuracy_history != 0) else 0.0
    
    logger.info(f"Pérdida final: {final_loss:.4f}")
    
    logger.info(f"Precisión final: {final_accuracy:.4f}")
    
    return loss_history, accuracy_history, epoch_time_history, total_time

# Función para elegir entre las versiones de PINN
def get_pinn_model(model_type, layers, activation_function="Tanh"):
    """
    Escoge entre dos versiones de PINN:
      - 'v1': retorna PINN_V1
      - 'v2': retorna PINN_V2
    """
    if model_type == "v1":
        logger.info("Usando PINN_V1.")
        return PINN_V1(layers, activation_function)
    elif model_type == "v2":
        logger.info("Usando PINN_V1.")
        return PINN_V2(layers, activation_function)
    else:
        raise ValueError("model_type debe ser 'v1' o 'v2'")

def plot_metrics(losses, accuracies, epoch_times):
    
    # Crear figura y ejes organizados en 2 filas y 3 columnas
    fig, axs = plt.subplots(2, 3, figsize=(15, 7))
    
    # Primera fila de gráficas
    axs[0, 0].plot(losses, label="Pérdida")
    axs[0, 0].set_xlabel("Época")
    axs[0, 0].set_ylabel("Pérdida")
    axs[0, 0].set_title("Pérdida vs Época")
    axs[0, 0].grid(True)

    axs[0, 1].plot(accuracies, label="Precisión")
    axs[0, 1].set_xlabel("Época")
    axs[0, 1].set_ylabel("Precisión (%)")
    axs[0, 1].set_title("Precisión vs Época")
    axs[0, 1].grid(True)

    axs[0, 2].plot(epoch_times, label="Tiempo")
    axs[0, 2].set_xlabel("Época")
    axs[0, 2].set_ylabel("Tiempo (s)")
    axs[0, 2].set_title("Tiempo vs Época")
    axs[0, 2].grid(True)
    
    acumulated_time = np.cumsum(epoch_times)

    # Segunda fila de gráficas
    axs[1, 0].plot(acumulated_time, losses)
    axs[1, 0].set_xlabel("Tiempo (s)")
    axs[1, 0].set_ylabel("Pérdida")
    axs[1, 0].set_title("Pérdida vs Tiempo")
    axs[1, 0].grid(True)

    axs[1, 1].plot(acumulated_time, accuracies)
    axs[1, 1].set_xlabel("Tiempo (s)")
    axs[1, 1].set_ylabel("Precisión ")
    axs[1, 1].set_title("Precisión vs Tiempo")
    axs[1, 1].grid(True)

    axs[1, 2].plot(losses, accuracies)
    axs[1, 2].set_xlabel("Pérdida")
    axs[1, 2].set_ylabel("Precisión")
    axs[1, 2].set_title("Precisión vs Pérdida")
    axs[1, 2].grid(True)

    # Ajustar espacio entre gráficos para evitar sobreposición
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    # plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_TRAIN_DIR, "metrics_plot.png"))
    logger.info("Gráficas de métricas guardadas.")
    # plt.show()
    plt.close()
    
    
def plot_solution(t_train, x_train, u_pred, u_train, filename="solution_comparison.png"):
    """
    Grafica la solución predicha y la solución real.
    
    Args:
        t_train: tensor de tiempo
        x_train: tensor de posición
        u_pred: solución predicha por el modelo
        u_train: solución real
        filename: nombre del archivo para guardar la gráfica
    """
    

    plt.figure(figsize=(15, 5))
    
    # Solución predicha
    plt.subplot(121)
    plt.pcolormesh(t_clean, x_clean, u_pred_clean, shading="auto", cmap="viridis")
    plt.colorbar(label="u(x,t)")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Solución PINN")
    
    # Solución exacta
    plt.subplot(122)
    plt.pcolormesh(t_clean, x_clean, u_train_clean, shading="auto", cmap="viridis")
    plt.colorbar(label="u(x,t)")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Solución exacta")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_TRAIN_DIR, filename))
    logger.info(f"Imagen de comparación guardada: {filename}")
    plt.close()

def prepare_tensor(tensor, target_dim=3):
    """
    Agrega dimensiones al final hasta alcanzar target_dim.
    Ej: si tensor.shape es (N, M) y target_dim=3 devuelve (N, M, 1).
    """
    while tensor.ndim < target_dim:
        tensor = tensor.unsqueeze(-1)
    return tensor

def load_data(file_name):
    """Carga datos de un archivo .mat"""
    data = loadmat(file_name)
    x_train = torch.tensor(data['x'].flatten(), dtype=torch.float32)
    t_train = torch.tensor(data['t'].flatten(), dtype=torch.float32)
    u0_train = torch.tensor(data['usol'], dtype=torch.float32)
    X, T = torch.meshgrid(x_train, t_train)
    return X, T, u0_train

def load_hyperparameters(file_name):
    """Carga hiperparámetros desde un archivo"""
    hyperparams = {}
    try:
        with open(file_name, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, value = line.split("=")
                hyperparams[key.strip()] = value.strip()
    except Exception as e:
        logger.error("No se pudo leer hyperparams.txt, usando valores por defecto.")
        hyperparams = {
            "num_epochs": "1000",
            "lr": "0.001",
            "nu": "0.01",
            "epochs_no_improve": "100",
            "min_loss_improvement": "0.001",
            "layers": "2,50,50,50,50,1",
            "seed": "1234"
        }
    return hyperparams

# Modificar carga de hiperparámetros: de .txt a hyperparams.json (archivo en la raíz del proyecto)
def load_hyperparams_json(file_name):
    with open(file_name, "r") as f:
        return json.load(f)

def save_environment_info(filename="training_environment.json"):
    env_info = {
        "python_version": platform.python_version(),
        "os": platform.system(),
        "os_version": platform.version(),
        "cpu": platform.processor(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "torch_version": torch.__version__,
        "installed_packages": subprocess.check_output([
            os.sys.executable, "-m", "pip", "freeze"
        ]).decode("utf-8").split("\n")
    }
    with open(filename, "w") as f:
        json.dump(env_info, f, indent=2)


if __name__ == '__main__':
    # Ya no es necesario crear el directorio aquí pues se hace al inicio

    hyperparams_path = os.path.join(os.path.abspath(TOTAL_PATH), "hyperparams.json")

    # Ruta del archivo JSON de hiperparámetros (ubicado en la raíz del proyecto)
    hyperparams = load_hyperparams_json(hyperparams_path)
    
    num_epochs = int(hyperparams["epochs"])
    lr = float(hyperparams["learning_rate"])
    epochs_no_improve = int(hyperparams["epochs_no_improve"])
    min_loss_improvement = float(hyperparams["min_loss_improvement"])
    layers = [int(layer) for layer in hyperparams["layers"]]
    seed = int(hyperparams["seed"])
    model_type = hyperparams["model_type"]
    batch_size = int(hyperparams["batch_size"])
    memory_limit_gb = float(hyperparams["memory_limit_gb"])

    OUTPUT_DIR = create_output_dir(model_type, num_epochs, lr, seed)
    OUTPUT_TRAIN_DIR = os.path.join(OUTPUT_DIR, "train_results")
    os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
    MODEL_DIR = os.path.join(OUTPUT_DIR, "model")
    
    LOG_DIR = os.path.join(OUTPUT_TRAIN_DIR, "train.log")
    fh = logging.FileHandler(LOG_DIR, mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Cargar datos usando DATA_DIR
    nu_input = 0.01
    data_loader = DataLoader(256, 100, nu_input)
    filename = os.path.join(os.path.abspath(DATA_DIR), "burgers_shock_mu_01_pi.mat")
    x_train, t_train, u_train = data_loader.load_data(filename)
    
    # Preparar tensores con dimensiones correctas
    x_train = prepare_tensor(x_train)
    t_train = prepare_tensor(t_train)
    u_train = prepare_tensor(u_train)
    
    # Seleccionar modelo desde nuestra estructura central
    model = get_pinn_model(model_type, layers)
    if torch.cuda.device_count() > 1:
        logger.info(f"Usando {torch.cuda.device_count()} GPUs con DataParallel.")
        model = nn.DataParallel(model.cuda())
    elif torch.cuda.device_count() == 1:
        logger.info("Usando GPU.")
        model = model.cuda()
    else:
        logger.info("Usando CPU.")
    logger.info(f"Arquitectura de la red ({model_type}):\n{model}")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(epochs_no_improve=epochs_no_improve, min_loss_improvement=min_loss_improvement, verbose=True)

    # Entrenamiento
    logger.info("Iniciando entrenamiento...")
    losses, accuracies, epoch_times, total_training_time = train_model(model, optimizer, 
                                num_epochs, early_stopping,x_train,
                                t_train, u_train, nu_input, batch_size,
                                memory_limit_gb)

    
    plot_metrics(losses, accuracies, epoch_times)
    
    # Generar predicción final y graficarla
    with torch.no_grad():
        u_pred = model(x_train, t_train).detach()
        
    # Guardar el modelo y sus pesos
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses.tolist(),
        'accuracies': accuracies.tolist(),
        'epoch_times': epoch_times.tolist(),
    }, os.path.join(MODEL_DIR, "weights_model.pth"))
    logger.info(f"Modelo guardado en: {MODEL_DIR}")
    
    # Crear hyperparams_train.json con parámetros y métricas
    hyperparams_train = hyperparams.copy()
    json_train_path = os.path.join(OUTPUT_TRAIN_DIR, "hyperparams_train.json")
    
    t_clean = t_train.squeeze().detach().cpu().numpy().tolist()
    x_clean = x_train.squeeze().detach().cpu().numpy().tolist()
    u_pred_clean = u_pred.squeeze().detach().cpu().numpy().tolist()
    u_train_clean = u_train.squeeze().detach().cpu().numpy().tolist()
    
    # Prepare training stats
    PRECISION = 4  # Define la cantidad de decimales que deseas conservar

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
    
    # Save training stats
    stats_path = os.path.join(OUTPUT_TRAIN_DIR, "training_stats.json")
    with open(stats_path, "w") as f:
        json.dump(training_stats, f, indent=4)
    logger.info(f"Training statistics saved to: {stats_path}")
    with open(json_train_path, "w") as f:
        json.dump(hyperparams_train, f, indent=4)
    logger.info(f"Hiperparámetros de entrenamiento guardados en: {json_train_path}")
    
    # Guardar la arquitectura (si no aparece ya en el log) en un JSON
    arch_path = os.path.join(MODEL_DIR, "model_architecture.txt")
    with open(arch_path, "w") as f:
        f.write(model.__repr__())
    logger.info(f"Estructura de la red guardada en: {arch_path}")
    
    # Graficar solución
    plot_solution(t_clean, x_clean, u_pred_clean, u_train_clean,
                 filename="solution_comparison.png")

    save_environment_info(os.path.join(OUTPUT_DIR, "environment.txt"))

    # Cerrar logger
    logger.info("Programa finalizado exitosamente.")
