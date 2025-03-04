import os
import time
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim
from scipy.io import loadmat
from tqdm import tqdm
from pinn_structure import PINN, EarlyStopping

# Definir rutas globales
TOTAL_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TOTAL_PATH, "data")

os.makedirs(DATA_DIR, exist_ok=True)

def create_output_dir(num_epochs, lr, seed):
    """Crea un directorio para guardar los resultados."""
    output_dir = os.path.join(TOTAL_PATH, "output", f"burgers_{num_epochs}_{lr}_{seed}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# Configurar logger con nombre dinámico
def create_log_file(num_epochs, lr, seed):
    log_name = f"burgers_{num_epochs}_{lr}_{seed}.log"  
    return os.path.join(OUTPUT_DIR, log_name)

# Configurar logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
if logger.hasHandlers():
    logger.handlers.clear()
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

class BurgersDataLoader:
    """Genera datos sintéticos para la ecuación de Burgers"""
    def __init__(self, spatial_points, time_points, nu):
        self.spatial_points = spatial_points
        self.time_points = time_points
        self.nu = nu
        
    def burgers_exact_solution(self, x, t):
        return -torch.tanh(x / (2 * self.nu * t + 1e-8))

    def generate_data(self):
        """Genera datos de entrenamiento para el modelo"""
        x = np.linspace(-1, 1, self.spatial_points)
        t = np.linspace(0, 1, self.time_points)
        X, T = torch.meshgrid(torch.tensor(x), torch.tensor(t))
        u0 = self.burgers_exact_solution(X, T)
        return X, T, u0
    
    def load_data(self, file_name):
        """Carga datos de un archivo"""
        data = loadmat(file_name)
        x_train = torch.tensor(data['x'].flatten(), dtype=torch.float32)
        t_train = torch.tensor(data['t'].flatten(), dtype=torch.float32)
        u0_train = torch.tensor(data['usol'], dtype=torch.float32)
        X, T = torch.meshgrid(x_train, t_train)
        return X, T, u0_train

    def plot(self, X, T, u0):
        """Grafica los datos"""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(X, T, u0, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u')
        
        ax = plt.subplot(122)
        plt.pcolormesh(X, T, u0, cmap='viridis')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('u(x, t)')
        plt.tight_layout()
        plt.show()

def calculate_accuracy(model, features, targets):
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
            targets = targets.reshape(predictions.shape)
            
        # Calcular error relativo elemento por elemento
        error = torch.abs(predictions - targets) / (torch.abs(targets) + 1e-8)
        # Tomar la media sobre todas las dimensiones
        accuracy = 1 - torch.mean(error)
        
    return accuracy.item() * 100

def burgers_pde_loss(model, x, t, nu):
    """Calcula la pérdida de la ecuación diferencial de Burgers."""
    x.requires_grad = True
    t.requires_grad = True
    u = model(x, t)
    
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    residual = u_t + u * u_x - nu * u_xx
    return torch.mean(residual ** 2)

def train_model(model, optimizer, num_epochs, early_stopping, x_train, t_train, u_train, nu):
    """Entrena el modelo usando la ecuación de Burgers."""
    start_time = time.time()
    loss_history = []
    accuracy_history = []
    
    with tqdm(total=num_epochs, desc="Entrenando la PINN", dynamic_ncols=True) as pbar:
        for epoch in range(num_epochs):
            epoch_start = time.time()
            optimizer.zero_grad()
            
            # Cálculo de pérdidas
            loss_data = torch.mean((model(x_train, t_train) - u_train) ** 2)
            loss_pde = burgers_pde_loss(model, x_train, t_train, nu)
            loss = loss_data + loss_pde
            
            loss.backward()
            optimizer.step()
            
            # Calcular y registrar métricas
            accuracy = calculate_accuracy(model, (x_train, t_train), u_train)
            accuracy_history.append(accuracy)
            loss_history.append(loss.item())
            
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
    logger.info(f"Pérdida final: {loss_history[-1]:.6f}")
    logger.info(f"Precisión final: {accuracy_history[-1]:.6f}%")
    
    return loss_history, accuracy_history

def plot_loss(losses, filename="loss_curve.png"):
    """Grafica la curva de pérdida durante el entrenamiento."""
    plt.figure(figsize=(10, 5))
    # if torch.is_tensor(losses):
    #     losses = losses.detach().cpu().numpy()
    plt.plot(losses)
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Curva de pérdida durante el entrenamiento')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    logger.info(f"Imagen de pérdida guardada: {filename}")
    plt.close()
    
def plot_accuracy(accuracy_history, filename="accuracy_curve.png"):
    """Grafica la curva de precisión durante el entrenamiento."""
    plt.figure(figsize=(10,5))
    plt.plot(accuracy_history)
    plt.xlabel("Época")
    plt.ylabel("Precisión (%)")
    plt.title("Precisión por épocas")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    logger.info(f"Imagen de precisión guardada: {filename}")
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
    t_clean = t_train.squeeze().detach().cpu().numpy()
    x_clean = x_train.squeeze().detach().cpu().numpy()
    u_pred_clean = u_pred.squeeze().detach().cpu().numpy()
    u_train_clean = u_train.squeeze().detach().cpu().numpy()

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
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
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
            "patience": "100",
            "min_delta": "0.001",
            "layers": "2,50,50,50,50,1",
            "seed": "1234"
        }
    return hyperparams
if __name__ == '__main__':
    # Ya no es necesario crear el directorio aquí pues se hace al inicio

    # Cargar hiperparámetros
    hyperparams = load_hyperparameters("hyperparams.txt")
    num_epochs = int(hyperparams["num_epochs"])
    lr = float(hyperparams["lr"])
    nu_input = float(hyperparams["nu"])
    patience = int(hyperparams["patience"])
    min_delta = float(hyperparams["min_delta"])
    layers = [int(layer) for layer in hyperparams["layers"].split(",")]
    seed = int(hyperparams["seed"])
    
    OUTPUT_DIR = create_output_dir(num_epochs, lr, seed)
    log_path = create_log_file(num_epochs, lr, seed)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Cargar datos usando DATA_DIR
    data_loader = BurgersDataLoader(256, 100, nu_input)
    filename = os.path.join(DATA_DIR, "burgers_shock_mu_01_pi.mat")
    x_train, t_train, u_train = data_loader.load_data(filename)
    
    # Preparar tensores con dimensiones correctas
    x_train = prepare_tensor(x_train)
    t_train = prepare_tensor(t_train)
    u_train = prepare_tensor(u_train)
    
    # Definir modelo y optimizador
    model = PINN(layers)
    logger.info(f"Arquitectura de la red:\n{model}")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, verbose=True)

    # Entrenamiento
    logger.info("Iniciando entrenamiento...")
    losses, accuracies = train_model(model, optimizer, num_epochs, early_stopping, 
                                   x_train, t_train, u_train, nu_input)
    
    # losses = losses.detach().numpy()
    # accuracies = accuracies.detach().numpy()
    # Generar gráficas
    plot_loss(losses, filename="loss_curve.png")
    plot_accuracy(accuracies, "accuracy_curve.png")
    
    # Generar predicción final y graficarla
    with torch.no_grad():
        u_pred = model(x_train, t_train).detach()
    plot_solution(t_train, x_train, u_pred, u_train, 
                 filename="solution_comparison.png")

    logger.info("Programa finalizado exitosamente.")
