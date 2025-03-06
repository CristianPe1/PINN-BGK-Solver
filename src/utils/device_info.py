"""
Módulo para obtener y registrar información sobre dispositivos (CPU/GPU)
"""

import torch
import platform
import subprocess
import re
import os
import json
import logging
from datetime import datetime
import psutil

logger = logging.getLogger(__name__)

# Mapeo de algunos GPUs comunes de NVIDIA a su número de CUDA cores
# Este diccionario contiene algunos modelos comunes, pero no es exhaustivo
NVIDIA_GPU_CORES = {
    # Serie GeForce RTX 30xx
    'RTX 3090': 10496,
    'RTX 3080 Ti': 10240,
    'RTX 3080': 8704,
    'RTX 3070 Ti': 6144,
    'RTX 3070': 5888,
    'RTX 3060 Ti': 4864,
    'RTX 3060': 3584,
    'RTX 3050': 2560,
    
    # Serie GeForce RTX 20xx
    'RTX 2080 Ti': 4352,
    'RTX 2080 Super': 3072,
    'RTX 2080': 2944,
    'RTX 2070 Super': 2560,
    'RTX 2070': 2304,
    'RTX 2060 Super': 2176,
    'RTX 2060': 1920,
    
    # Serie GeForce GTX 16xx
    'GTX 1660 Ti': 1536,
    'GTX 1660 Super': 1408,
    'GTX 1660': 1408,
    'GTX 1650 Super': 1280,
    'GTX 1650': 896,
    
    # Serie GeForce GTX 10xx
    'GTX 1080 Ti': 3584,
    'GTX 1080': 2560,
    'GTX 1070 Ti': 2432,
    'GTX 1070': 1920,
    'GTX 1060': 1280,
    'GTX 1050 Ti': 768,
    'GTX 1050': 640,
    'GTX 1030': 384,
    
    # Serie Tesla
    'Tesla V100': 5120,
    'Tesla P100': 3584,
    'Tesla T4': 2560,
    'Tesla P4': 2560,
    'Tesla K80': 2496,
    'Tesla M60': 2048,
    'Tesla P40': 3840,
    
    # Serie Quadro
    'Quadro RTX 8000': 4608,
    'Quadro RTX 6000': 4608,
    'Quadro RTX 5000': 3072,
    'Quadro RTX 4000': 2304,
    'Quadro P6000': 3840,
    'Quadro P5000': 2560,
    'Quadro P4000': 1792,
    
    # Serie Titan
    'TITAN RTX': 4608,
    'TITAN V': 5120,
    'TITAN Xp': 3840,
    'TITAN X': 3072,
}

def get_cpu_info():
    """
    Obtiene información detallada sobre la CPU
    
    Returns:
        dict: Información de la CPU
    """
    cpu_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "architecture": platform.machine(),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "cpu_freq_mhz": psutil.cpu_freq().max if psutil.cpu_freq() else "N/A",
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2)
    }
    
    return cpu_info

def get_nvidia_smi_gpu_info():
    """
    Obtiene información detallada sobre las GPUs usando nvidia-smi
    Este método es más completo pero requiere que nvidia-smi esté disponible
    
    Returns:
        list: Lista de diccionarios con información de cada GPU
    """
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=gpu_name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'],
                              stdout=subprocess.PIPE, check=True, text=True)
        
        gpu_info = []
        for i, line in enumerate(result.stdout.strip().split('\n')):
            if line.strip():
                # Parsear cada línea de la salida
                name, mem_total, mem_free, mem_used, utilization, temp = [item.strip() for item in line.split(',')]
                
                # Buscar CUDA cores basado en el nombre de la GPU
                cuda_cores = None
                for gpu_model, cores in NVIDIA_GPU_CORES.items():
                    if gpu_model in name:
                        cuda_cores = cores
                        break
                
                info = {
                    "id": i,
                    "name": name,
                    "memory_total_mb": float(mem_total),
                    "memory_free_mb": float(mem_free),
                    "memory_used_mb": float(mem_used),
                    "utilization_percent": float(utilization),
                    "temperature_c": float(temp),
                    "cuda_cores": cuda_cores
                }
                gpu_info.append(info)
        
        return gpu_info
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("No se pudo ejecutar nvidia-smi o el comando falló")
        return []

def get_torch_gpu_info():
    """
    Obtiene información básica sobre las GPUs usando PyTorch
    Este método es menos detallado pero siempre disponible si CUDA está instalado
    
    Returns:
        list: Lista de diccionarios con información de cada GPU
    """
    gpu_info = []
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{i}')
            props = torch.cuda.get_device_properties(device)
            
            # Buscar CUDA cores basado en el nombre de la GPU
            cuda_cores = None
            for gpu_model, cores in NVIDIA_GPU_CORES.items():
                if gpu_model in props.name:
                    cuda_cores = cores
                    break
            
            info = {
                "id": i,
                "name": props.name,
                "memory_total_mb": round(props.total_memory / (1024**2), 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
                "cuda_cores_per_mp": cuda_cores // props.multi_processor_count if cuda_cores else None,
                "cuda_cores": cuda_cores
            }
            gpu_info.append(info)
    
    return gpu_info

def get_complete_device_info():
    """
    Recopila información completa sobre los dispositivos disponibles (CPU y GPU)
    
    Returns:
        dict: Información completa de dispositivos
    """
    # Información básica del sistema
    device_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cuda_available": torch.cuda.is_available(),
        "cpu_info": get_cpu_info(),
        "default_device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    # Intentar obtener información detallada de GPUs
    nvidia_gpu_info = get_nvidia_smi_gpu_info()
    if nvidia_gpu_info:
        device_info["gpu_info"] = nvidia_gpu_info
    else:
        torch_gpu_info = get_torch_gpu_info()
        if torch_gpu_info:
            device_info["gpu_info"] = torch_gpu_info
        else:
            device_info["gpu_info"] = []
    
    return device_info

def log_device_info(device_info=None, output_dir=None):
    """
    Registra información sobre los dispositivos disponibles
    
    Args:
        device_info (dict, optional): Información ya recopilada del dispositivo
        output_dir (str, optional): Directorio donde guardar el archivo
        
    Returns:
        dict: Información del dispositivo
    """
    if device_info is None:
        device_info = get_complete_device_info()
    
    # Registrar en el logger
    logger.info("=== Información del dispositivo ===")
    logger.info(f"Dispositivo predeterminado: {device_info['default_device']}")
    if torch.cuda.is_available():
        logger.info(f"CUDA disponible: {torch.cuda.is_available()}")
        logger.info(f"Número de GPUs: {torch.cuda.device_count()}")
        
        for i, gpu in enumerate(device_info.get("gpu_info", [])):
            logger.info(f"GPU {i}: {gpu.get('name')}")
            logger.info(f"  Memoria total: {gpu.get('memory_total_mb')} MB")
            
            cuda_cores = gpu.get("cuda_cores")
            if cuda_cores:
                logger.info(f"  CUDA cores: {cuda_cores}")
            else:
                logger.info(f"  CUDA cores: Desconocido para este modelo")
                
            logger.info(f"  Utilización: {gpu.get('utilization_percent', 'N/A')}%")
    else:
        logger.info("CUDA no disponible. Usando CPU.")
        logger.info(f"CPU: {device_info['cpu_info']['processor']}")
        logger.info(f"Núcleos físicos: {device_info['cpu_info']['cores_physical']}")
        logger.info(f"Núcleos lógicos: {device_info['cpu_info']['cores_logical']}")
    
    # Guardar en archivo si se proporciona un directorio de salida
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, "device_info.json")
        with open(file_path, "w") as f:
            json.dump(device_info, f, indent=4)
        logger.info(f"Información del dispositivo guardada en: {file_path}")
    
    return device_info

if __name__ == "__main__":
    # Configuración de logging para prueba independiente
    logging.basicConfig(level=logging.INFO)
    
    # Obtener y mostrar información del dispositivo
    device_info = get_complete_device_info()
    print(json.dumps(device_info, indent=2))
    
    log_device_info(device_info)
