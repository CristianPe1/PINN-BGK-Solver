import os
import sys
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import re
from scipy.io import loadmat
import glob
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_curve, auc, confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score)
from pinn_structure_v1 import PINN_V1  # o PINN_V2 según corresponda
from pinn_structure_v2 import PINN_V2  # o PINN_V2 según corresponda
import json
from datetime import datetime
                             
# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


TOTAL_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TOTAL_PATH, "data/evaluation")
OUTPUT_ROOT = os.path.join(TOTAL_PATH, "output")
subdirs = [d for d in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, d))]
if not subdirs:
    raise Exception("No se encontraron carpetas en el directorio 'output'.")
print("Carpetas disponibles en 'output':")
for idx, folder in enumerate(subdirs, 1):
    print(f"{idx}. {folder}")
seleccion = input("Seleccione el número de la carpeta que desea usar: ")
try:
    index = int(seleccion) - 1
    if index < 0 or index >= len(subdirs):
        raise ValueError
    selected_folder = subdirs[index]
except ValueError:
    raise ValueError("Selección inválida.")
MODEL_DIR = os.path.join(OUTPUT_ROOT, selected_folder, "model")


def create_output_dir(selected_folder):
    output_dir = os.path.join(OUTPUT_ROOT, selected_folder, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# Función para evaluar modelo en clasificación
def evaluate_classification(model, X_test, y_test, device):
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(X_test).float().to(device)
        outputs = model(inputs)
        # Asumir que el modelo retorna logits; calcular probabilidades
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        y_pred = (probs >= 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)
    
    logger.info("=== Clasificación ===")
    logger.info(f"Exactitud: {acc:.4f}")
    logger.info(f"Precisión: {prec:.4f}")
    logger.info(f"Recall: {rec:.4f}")
    logger.info(f"F1-score: {f1:.4f}")
    logger.info(f"AUC: {roc_auc:.4f}")
    logger.info(f"Matriz de confusión:\n{cm}")
    
    # Guardar curva ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    roc_path = os.path.join(os.path.dirname(__file__), "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    logger.info(f"Curva ROC guardada en: {roc_path}")

# Función para evaluar modelo en regresión
def evaluate_regression(model, X_test, y_test, device, output_dir, data_name):
    # X_test is a tuple (x_test, t_test)
    model.eval()
    with torch.no_grad():
        x_tensor, t_tensor = X_test  # unpack the tuple
        x_tensor, t_tensor = x_tensor.to(device), t_tensor.to(device)
        outputs = model(x_tensor, t_tensor)
        u_pred = outputs.cpu().clone()  # Para graficar después
        outputs = outputs.cpu().numpy().flatten()

    mae = mean_absolute_error(y_test.numpy().flatten(), outputs)
    mse = mean_squared_error(y_test.numpy().flatten(), outputs)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test.numpy().flatten(), outputs)

    # Preparar estadísticas
    stats = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2)
    }
    
    # Obtener nombre del modelo desde la arquitectura
    model_name = f"PINN_{model.__class__.__name__}"
    
    # Guardar estadísticas
    save_evaluation_stats(stats, output_dir, model_name, data_name)
    
    # Mostrar resultados en consola
    logger.info("=== Métricas de Regresión ===")
    logger.info(f"MAE (Error Absoluto Medio): {mae:.4f} (menor es mejor)")
    logger.info(f"MSE (Error Cuadrático Medio): {mse:.4f} (menor es mejor)")
    logger.info(f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.4f} (menor es mejor)")
    logger.info(f"R² (Coeficiente de Determinación): {r2:.4f} (más cercano a 1 es mejor)")
    logger.info("\nInterpretación:")
    logger.info("- MAE: Error promedio en las mismas unidades que la variable objetivo")
    logger.info("- MSE: Error promedio al cuadrado, penaliza errores grandes")
    logger.info("- RMSE: Raíz cuadrada del MSE, interpretable en unidades originales")
    logger.info("- R²: Proporción de varianza explicada por el modelo (0 a 1)")

    # Si se desea graficar la solución (Burgers), llamamos a la nueva función
    file_label = os.path.splitext(data_name)[0]
    plot_solution_evaluation(
        x_tensor, t_tensor, u_pred, y_test, 
        output_dir, f"solution_evaluation_{file_label}.png"
    )

# Función para cargar arquitectura del modelo
def load_architecture(arch_path):
    print(arch_path)
    
    with open(arch_path, "r") as file:
        print("Archivo abierto")
        data = file.read()
        # Buscar la información inmediatamente después de "(activation)" hasta el siguiente salto de línea
        activation_info = None
        if "(activation): " in data:
            idx = data.find("(activation): ")
            start = idx + len("(activation): ")
            end = data.find("()\n", start)
            activation_info = data[start:end].strip()
            logger.info(f"Función de activation cargada: {activation_info}")
            
        # Buscar la información inmediatamente después de "(linears)" hasta el siguiente salto de línea
        linears_info = None
        if "(linears)" in data:
            idx = data.find("(linears)")
            start = idx + len("(linears)")
            end = data.find("\n)\n", start)
            linears_block = data[start:end]
            layers_info = []
            print(linears_block)
            # Regex explanation:
            # \(\s*[\d\-]+\s*\): matches the label (e.g., (0) or (1-3))
            # (?:(\d+)\s*x\s+)? optionally matches "3 x " and captures the multiplicity if present.
            # Linear\(in_features=(\d+), out_features=(\d+) matches the Linear layer with its in/out features.
            pattern = re.compile(r"\(\s*[\d\-]+\s*\):\s*(?:(\d+)\s*x\s+)?Linear\(in_features=(\d+), out_features=(\d+)")
            matches = pattern.findall(linears_block)
            if matches:
                # Use the in_features from the first match as the input dimension.
                first_in = int(matches[0][1])
                layers_info.append(first_in)
                for mult, in_feat, out_feat in matches:
                    count = int(mult) if mult else 1
                    for _ in range(count):
                        layers_info.append(int(out_feat))
                logger.info(f"Arquitectura de capas cargada: {layers_info}")
            else:
                logger.warning("No se encontró información de linears en formato esperado.")
        
        # Si no se encontró ninguna de las claves, se sale con un error.
        if activation_info is None and linears_info is None:
            sys.exit("Formato de arquitectura no reconocido.")
        
        linears_info = np.array(layers_info)    
        return linears_info, activation_info
    
def load_data_test(data_dir):  
    """Cargar datos de prueba para evaluación."""
    # Leer el nombre del archivo de test desde un archivo txt
    txt_file_path = os.path.join(data_dir, "test_file.txt")
    if not os.path.exists(txt_file_path):
        raise FileNotFoundError(f"No se encontró el archivo de texto {txt_file_path} que contiene los nombres de los archivos de test.")
    with open(txt_file_path, "r") as f:
        files = [line.strip() for line in f if line.strip()]

    if not files:
        raise ValueError("El archivo de test no contiene ningún nombre de archivo.")

    print("Archivos de test disponibles:")
    for idx, file_name in enumerate(files):
        print(f"{idx + 1}. {file_name}")

    seleccion = input("Seleccione el número del archivo de test que desea usar: ")
    try:
        index = int(seleccion) - 1
        if index < 0 or index >= len(files):
            raise ValueError
        selected_file = files[index]
    except ValueError:
        raise ValueError("Selección inválida.")

    latest_file = os.path.join(data_dir, selected_file) if not os.path.isabs(selected_file) else selected_file
    if not os.path.exists(latest_file):
        raise FileNotFoundError(f"El archivo de test especificado {latest_file} no existe.")

    file_ext = os.path.splitext(latest_file)[1]
    if file_ext == ".mat":
        data = loadmat(latest_file)
    elif file_ext == ".npz":
        data = np.load(latest_file)
    else:
        raise ValueError("Formato de archivo no reconocido.")
        
    logger.info(f"Archivo de test seleccionado: {latest_file}")
    x = torch.tensor(data['x'].flatten(), dtype=torch.float32)
    t = torch.tensor(data['t'].flatten(), dtype=torch.float32)
    u0 = torch.tensor(data['usol'], dtype=torch.float32)
    X, T = torch.meshgrid(x, t, indexing='ij')
    
    target_dim = 3

    # Asegurarse de que cada tensor tenga al menos target_dim dimensiones
    if X.dim() < target_dim:
        X = X.unsqueeze(-1)
    if T.dim() < target_dim:
        T = T.unsqueeze(-1)
    if u0.dim() < target_dim:
        u0 = u0.unsqueeze(-1)
        
    return X, T, u0, selected_file

def plot_solution_evaluation(x, t, u_pred, u_real, output_dir, filename="solution_evaluation.png"):
    """
    Grafica la solución predicha y la solución real para la ecuación de Burgers.
    """
    # Pasar a CPU y numpy
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
    """
    Guarda las estadísticas de evaluación en un archivo JSON acumulativo.
    """
    stats_file = os.path.join(output_dir, "evaluation_history.json")
    
    # Cargar historial existente o crear nuevo
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    # Agregar nueva evaluación
    evaluation_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "test_data": data_name,
        "metrics": stats
    }
    
    history.append(evaluation_entry)
    
    # Guardar historial actualizado
    with open(stats_file, 'w') as f:
        json.dump(history, f, indent=4)
    
    # # También crear un archivo de log más legible
    # log_file = os.path.join(output_dir, "evaluation_log.txt")
    # with open(log_file, 'a') as f:
    #     f.write(f"\n{'='*50}\n")
    #     f.write(f"Evaluación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    #     f.write(f"Modelo: {model_name}\n")
    #     f.write(f"Datos de prueba: {data_name}\n")
    #     f.write(f"Métricas:\n")
    #     f.write(f"  MAE: {stats['mae']:.6f}\n")
    #     f.write(f"  MSE: {stats['mse']:.6f}\n")
    #     f.write(f"  RMSE: {stats['rmse']:.6f}\n")
    #     f.write(f"  R²: {stats['r2']:.6f}\n")
    #     f.write(f"{'='*50}\n")

def setup_logger(output_dir):
    """Configura el logger para guardar en archivo y mostrar en consola"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Limpiar handlers existentes
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Crear formato para los logs
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para archivo
    log_file = os.path.join(output_dir, "evaluation.log")
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger, log_file

def main():
    OUTPUT_DIR = create_output_dir(selected_folder)
    
    # Configurar logger
    logger, log_file = setup_logger(OUTPUT_DIR)
    logger.info(f"Iniciando evaluación. Los logs se guardarán en: {log_file}")
    
    # Agregar ruta raíz del proyecto si es necesario
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ruta de los archivos
    weights_path = os.path.join(MODEL_DIR, "weights_model.pth")
    arch_path = os.path.join(MODEL_DIR, "model_architecture.txt")
    
    logger.info(f"Cargando modelo desde: {MODEL_DIR}")
    
    layers, activation = load_architecture(arch_path)
    
    # Crear el modelo e inicializar los pesos
    model = PINN_V1(layers, activation)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    logger.info("Modelo cargado correctamente.")

    # Cargar datasets de evaluación
    
        
    # sys.exit("Cierre de prueba")
    # Asumir que existen archivos .npz con arrays: X_test, y_test
  
    x_test, t_test, u_test, name_test = load_data_test(DATA_DIR)
    
    evaluate_regression(model, (x_test, t_test), u_test, device, OUTPUT_DIR, name_test)
    # Evaluación para clasificación
    # class_file = os.path.join(data_dir, "dataset_classification.npz")
    # if os.path.exists(class_file):
    #     data = np.load(class_file)
    #     X_test_clf = data["X_test"]
    #     y_test_clf = data["y_test"]
    #     evaluate_classification(model, X_test_clf, y_test_clf, device)
    # else:
    #     logger.warning(f"Dataset de clasificación no encontrado en {class_file}")
    
    # # Evaluación para regresión
    # reg_file = os.path.join(data_dir, "dataset_regression.npz")
    # if os.path.exists(reg_file):
    #     data = np.load(reg_file)
    #     X_test_reg = data["X_test"]
    #     y_test_reg = data["y_test"]
    #     evaluate_regression(model, X_test_reg, y_test_reg, device)
    # else:
    #     logger.warning(f"Dataset de regresión no encontrado en {reg_file}")

if __name__ == '__main__':
    main()
