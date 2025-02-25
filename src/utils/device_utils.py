import psutil
import torch
import logging
import time

logger = logging.getLogger(__name__)

def get_device_memory_info():
    """
    Obtiene información sobre el uso de memoria del dispositivo (GPU/CPU).
    
    Returns:
        dict: Información del dispositivo y su uso de memoria
    """
    if torch.cuda.is_available():
        # Obtener información de memoria GPU
        device = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device)
        memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)  # Convertir a GB
        memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)
        total_memory = gpu_properties.total_memory / (1024**3)
        
        return {
            "device": "GPU",
            "gpu_name": gpu_properties.name,
            "gpu_memory_total_gb": f"{total_memory:.2f} GB",
            "gpu_memory_allocated_gb": f"{memory_allocated:.2f} GB",
            "gpu_memory_reserved_gb": f"{memory_reserved:.2f} GB",
            "gpu_memory_free_gb": f"{total_memory - memory_allocated:.2f} GB"
        }
    else:
        # Obtener información de memoria RAM
        vm = psutil.virtual_memory()
        return {
            "device": "CPU",
            "memory_total_gb": f"{vm.total / (1024**3):.2f} GB",
            "memory_available_gb": f"{vm.available / (1024**3):.2f} GB",
            "memory_used_gb": f"{vm.used / (1024**3):.2f} GB",
            "memory_percent": f"{vm.percent:.1f}%"
        }

def get_new_batch_size(memory_limit_gb=2.0, batch_size=32, max_batch_size=1024):
    """
    Ajusta dinámicamente el tamaño del batch basado en la memoria disponible.
    
    Args:
        memory_limit_gb (float): Límite de memoria en GB
        batch_size (int): Tamaño base del batch
        max_batch_size (int): Tamaño máximo del batch
    
    Returns:
        int: Tamaño del batch ajustado
    """
    new_batch_size = batch_size
    while True:
        vm = psutil.virtual_memory()
        memory_available = vm.available / (1024**3)
        logger.info(f"Memoria CPU disponible: {memory_available:.2f}GB")
        if memory_available > memory_limit_gb * 1.1 and new_batch_size < max_batch_size:
            new_batch_size *= 2
            logger.info(f"Aumentando batch size a {new_batch_size}")
            time.sleep(0.5)
        else:
            logger.info(f"Memoria disponible cercana al límite: {memory_available:.2f}GB")
            break
    return new_batch_size

def get_dynamic_batch_size(device_load, base_batch_size=32, memory_limit_gb=2.0):
    """
    Ajusta dinámicamente el tamaño del batch basado en la memoria disponible.
    
    Args:
        device_load (dict): Información del dispositivo
        base_batch_size (int): Tamaño base del batch
        memory_limit_gb (float): Límite de memoria en GB
    
    Returns:
        int: Tamaño del batch ajustado
    """
    try:
        if device_load["device"] == "GPU":
            memory_allocated = float(device_load["gpu_memory_allocated_gb"].split()[0])
            if memory_allocated > memory_limit_gb:
                new_batch_size = max(1, base_batch_size // 2)
                logger.info(f"Reduciendo batch size a {new_batch_size} debido a uso de memoria GPU: {memory_allocated:.2f}GB")
                return new_batch_size
            else:
                new_batch_size = get_new_batch_size(memory_limit_gb, base_batch_size)
                logger.info(f"Nuevo batch size: {new_batch_size}")
                return new_batch_size
        elif device_load["device"] == "CPU":
            memory_available = float(device_load["memory_available_gb"].split()[0])
            if memory_available < memory_limit_gb:
                new_batch_size = max(1, base_batch_size // 2)
                logger.info(f"Reduciendo batch size a {new_batch_size} debido a memoria CPU disponible: {memory_available:.2f}GB")
                return new_batch_size
            else:
                new_batch_size = get_new_batch_size(memory_limit_gb, base_batch_size)
                logger.info(f"Nuevo batch size: {new_batch_size}")
                return new_batch_size
    except Exception as e:
        logger.warning(f"Error al ajustar batch size: {e}")
        return base_batch_size
    
    return base_batch_size

def monitor_device_usage():
    """
    Monitorea y registra el uso actual del dispositivo.
    
    Returns:
        dict: Información del dispositivo y métricas de uso
    """
    device_load = get_device_memory_info()
    
    if device_load["device"] == "GPU":
        logger.info(
            f"GPU Memory: "
            f"Total={device_load['gpu_memory_total_gb']}, "
            f"Used={device_load['gpu_memory_allocated_gb']}, "
            f"Reserved={device_load['gpu_memory_reserved_gb']}, "
            f"Free={device_load['gpu_memory_free_gb']}"
        )
    else:
        logger.info(
            f"CPU Memory: "
            f"Total={device_load['memory_total_gb']}, "
            f"Available={device_load['memory_available_gb']}, "
            f"Used={device_load['memory_used_gb']}, "
            f"Usage={device_load['memory_percent']}"
        )
    
    return device_load

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

