import os
import sys
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.io import loadmat
import json
from datetime import datetime
from pinn_structure_v1 import PINN_V1
from pinn_structure_v2 import PINN_V2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# ...otras importaciones necesarias...

# Configurar logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Funciones de carga de arquitectura, datos de test, graficado y guardado de estadísticas
# ... (copiar la funcionalidad de model_evaluation.py adaptando las rutas) ...

def load_architecture(arch_path):
    with open(arch_path, "r") as file:
        data = file.read()
        activation_info = None
        if "(activation): " in data:
            idx = data.find("(activation): ")
            start = idx + len("(activation): ")
            end = data.find("()\n", start)
            activation_info = data[start:end].strip()
            logger.info(f"Función de activation cargada: {activation_info}")
        if "(linears)" in data:
            idx = data.find("(linears)")
            start = idx + len("(linears)")
            end = data.find("\n)\n", start)
            linears_block = data[start:end]
            layers_info = []
            pattern = re.compile(r"\(\s*[\d\-]+\s*\):\s*(?:(\d+)\s*x\s+)?Linear\(in_features=(\d+), out_features=(\d+)")
            matches = pattern.findall(linears_block)
            if matches:
                first_in = int(matches[0][1])
                layers_info.append(first_in)
                for mult, in_feat, out_feat in matches:
                    count = int(mult) if mult else 1
                    for _ in range(count):
                        layers_info.append(int(out_feat))
                logger.info(f"Arquitectura de capas cargada: {layers_info}")
            else:
                logger.warning("No se encontró información de linears en formato esperado.")
        if activation_info is None:
            sys.exit("Formato de arquitectura no reconocido.")
        return np.array(layers_info), activation_info

def load_data_test(data_dir):
    txt_file_path = os.path.join(data_dir, "test_file.txt")
    with open(txt_file_path, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    print("Archivos de test disponibles:")
    for idx, file_name in enumerate(files):
        print(f"{idx + 1}. {file_name}")
    seleccion = input("Seleccione el número del archivo de test que desea usar: ")
    try:
        index = int(seleccion) - 1
        selected_file = files[index]
    except:
        raise ValueError("Selección inválida.")
    latest_file = os.path.join(data_dir, selected_file) if not os.path.isabs(selected_file) else selected_file
    if not os.path.exists(latest_file):
        raise FileNotFoundError(f"Archivo {latest_file} no encontrado.")
    ext = os.path.splitext(latest_file)[1]
    if ext == ".mat":
        data = loadmat(latest_file)
    elif ext == ".npz":
        data = np.load(latest_file)
    else:
        raise ValueError("Formato no reconocido.")
    logger.info(f"Archivo de test seleccionado: {latest_file}")
    x = torch.tensor(data['x'].flatten(), dtype=torch.float32)
    t = torch.tensor(data['t'].flatten(), dtype=torch.float32)
    u0 = torch.tensor(data['usol'], dtype=torch.float32)
    X, T = torch.meshgrid(x, t, indexing='ij')
    if X.dim() < 3:
        X = X.unsqueeze(-1)
    if T.dim() < 3:
        T = T.unsqueeze(-1)
    if u0.dim() < 3:
        u0 = u0.unsqueeze(-1)
    return X, T, u0, selected_file

def plot_solution_evaluation(x, t, u_pred, u_real, output_dir, filename="solution_evaluation.png"):
    x_np = x.squeeze().detach().cpu().numpy()
    t_np = t.squeeze().detach().cpu().numpy()
    u_pred_np = u_pred.squeeze().detach().cpu().numpy()
    u_real_np = u_real.squeeze().detach().cpu().numpy()
    
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.pcolormesh(t_np, x_np, u_pred_np, shading="auto", cmap="viridis")
    plt.colorbar(label="u(x,t)")
    plt.title("Solución PINN (Predicción)")
    plt.xlabel("t")
    plt.ylabel("x")
    
    plt.subplot(122)
    plt.pcolormesh(t_np, x_np, u_real_np, shading="auto", cmap="viridis")
    plt.colorbar(label="u(x,t)")
    plt.title("Solución Real")
    plt.xlabel("t")
    plt.ylabel("x")
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, filename)
    plt.savefig(fig_path)
    plt.close()
    logger.info(f"Gráfica de solución guardada en: {fig_path}")

def save_evaluation_stats(stats, output_dir, model_name, data_name):
    stats_file = os.path.join(output_dir, "evaluation_history.json")
    if os.path.exists(stats_file):
        with open(stats_file, "r") as f:
            history = json.load(f)
    else:
        history = []
    evaluation_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "test_data": data_name,
        "metrics": stats
    }
    history.append(evaluation_entry)
    with open(stats_file, "w") as f:
        json.dump(history, f, indent=4)

def setup_logger(output_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    log_file = os.path.join(output_dir, "evaluation.log")
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, log_file

def main():
    OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "evaluation_results", "selected")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger, log_file = setup_logger(OUTPUT_DIR)
    logger.info(f"Iniciando evaluación. Logs en: {log_file}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_path = os.path.join(MODEL_DIR, "weights_model.pth")
    arch_path = os.path.join(MODEL_DIR, "model_architecture.txt")
    logger.info(f"Cargando modelo desde: {MODEL_DIR}")
    
    layers, activation = load_architecture(arch_path)
    model = PINN_V1(layers, activation)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    logger.info("Modelo cargado correctamente.")
    
    x_test, t_test, u_test, name_test = load_data_test(DATA_DIR)
    evaluate_regression(model, (x_test, t_test), u_test, device, OUTPUT_DIR, name_test)
    
if __name__ == '__main__':
    main()
