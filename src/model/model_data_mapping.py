"""
Módulo para gestionar la relación entre tipos de modelos y conjuntos de datos adecuados.
Permite seleccionar automáticamente los datos correctos según el modelo elegido.
"""

import os
import logging
import glob
from pathlib import Path

logger = logging.getLogger(__name__)

# Mapeo de tipos de física/modelos a rutas de datos por defecto
DEFAULT_DATA_MAPPING = {
    # Formato: "tipo_modelo": {"path_pattern": "patrón/a/buscar/*.extensión", "default": "ruta/por_defecto.ext"}
    "burgers": {
        "path_pattern": "data/generated/burgers/burgers_nu*_*.mat",
        "default": "data/training/burgers_shock_mu_01_pi.mat"
    },
    "taylor_green": {
        "path_pattern": "data/generated/taylor_green/taylor_green_nu*_*.mat",
        "default": "data/generated/taylor_green/taylor_green_default.mat"
    },
    "kovasznay": {
        "path_pattern": "data/generated/kovasznay/kovasznay_re*_*.mat",
        "default": "data/generated/kovasznay/kovasznay_default.mat"
    },
    "cavity_flow": {
        "path_pattern": "data/generated/cavity_flow/cavity_flow_re*_*.mat",
        "default": "data/generated/cavity_flow/cavity_flow_default.mat"
    },
    "navier_stokes": {
        "path_pattern": "data/generated/*/navier_stokes_*.mat",
        "default": "data/generated/taylor_green/taylor_green_default.mat"
    },
    # Para tipos genéricos o no específicos
    "mlp": {
        "path_pattern": "data/training/*.mat",
        "default": "data/training/burgers_shock_mu_01_pi.mat"
    }
}

def find_matching_data_files(physics_type, base_dir=None):
    """
    Encuentra archivos de datos que coincidan con un tipo de física específico.
    
    Args:
        physics_type (str): Tipo de física (burgers, taylor_green, etc.)
        base_dir (str, optional): Directorio base para buscar. Si es None, usa el directorio actual.
        
    Returns:
        list: Lista de archivos que coinciden con el patrón de datos para ese tipo de física
    """
    if physics_type not in DEFAULT_DATA_MAPPING:
        logger.warning(f"No hay mapeo definido para el tipo de física '{physics_type}'. Usando tipo genérico 'mlp'.")
        physics_type = "mlp"
    
    # Obtener el patrón de búsqueda para este tipo de física
    path_pattern = DEFAULT_DATA_MAPPING[physics_type]["path_pattern"]
    
    # Si se proporciona un directorio base, ajustar la ruta
    if base_dir:
        path_pattern = os.path.join(base_dir, path_pattern)
    
    # Buscar archivos que coincidan con el patrón
    matching_files = glob.glob(path_pattern)
    
    # Ordenar por fecha de modificación (más reciente primero)
    matching_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return matching_files

def get_default_data_path(physics_type, base_dir=None):
    """
    Obtiene la ruta de datos por defecto para un tipo de física específico.
    
    Args:
        physics_type (str): Tipo de física (burgers, taylor_green, etc.)
        base_dir (str, optional): Directorio base para buscar. Si es None, usa el directorio actual.
        
    Returns:
        str: Ruta al archivo de datos por defecto
    """
    if physics_type not in DEFAULT_DATA_MAPPING:
        logger.warning(f"No hay mapeo definido para el tipo de física '{physics_type}'. Usando tipo genérico 'mlp'.")
        physics_type = "mlp"
    
    # Primero intentar encontrar archivos coincidentes
    matching_files = find_matching_data_files(physics_type, base_dir)
    
    if matching_files:
        # Usar el archivo más reciente
        logger.info(f"Usando el archivo de datos más reciente para '{physics_type}': {matching_files[0]}")
        return matching_files[0]
    
    # Si no se encuentran archivos, usar el valor por defecto
    default_path = DEFAULT_DATA_MAPPING[physics_type]["default"]
    
    # Si se proporciona un directorio base, ajustar la ruta
    if base_dir:
        default_path = os.path.join(base_dir, default_path)
    
    # Verificar si existe el archivo por defecto
    if os.path.exists(default_path):
        logger.info(f"Usando el archivo de datos por defecto para '{physics_type}': {default_path}")
        return default_path
    
    # Si no existe, advertir y devolver None
    logger.warning(f"No se encontró ningún archivo de datos para '{physics_type}'.")
    return None

def suggest_data_path_for_model(model_type=None, physics_type=None, config=None, base_dir=None):
    """
    Sugiere una ruta de datos para un modelo específico.
    
    Args:
        model_type (str, optional): Tipo de modelo (mlp, pinn_v1, etc.)
        physics_type (str, optional): Tipo de física (burgers, taylor_green, etc.)
        config (dict, optional): Configuración del modelo que puede incluir physics_type
        base_dir (str, optional): Directorio base para buscar
        
    Returns:
        str: Ruta sugerida al archivo de datos
    """
    # Determinar el tipo de física a partir de los parámetros disponibles
    if physics_type is None:
        if config and 'physics_type' in config:
            physics_type = config['physics_type']
        elif config and 'type' in config:
            physics_type = config['type']
        elif model_type:
            # Usar el tipo de modelo como último recurso
            physics_type = model_type
        else:
            physics_type = "mlp"  # Valor por defecto
    
    return get_default_data_path(physics_type, base_dir)

def verify_model_data_compatibility(model_config, data_path):
    """
    Verifica que un modelo y un conjunto de datos sean compatibles.
    
    Args:
        model_config (dict): Configuración del modelo
        data_path (str): Ruta al archivo de datos
        
    Returns:
        tuple: (is_compatible, message)
    """
    # Esta es una implementación básica. Se podría expandir para realizar
    # verificaciones más complejas basadas en las características del archivo de datos.
    
    # Verificar que el archivo existe
    if not os.path.exists(data_path):
        return False, f"El archivo de datos no existe: {data_path}"
    
    # Verificar extensión
    _, ext = os.path.splitext(data_path)
    if ext.lower() not in ['.mat', '.npz', '.h5', '.hdf5']:
        return False, f"Formato de archivo no soportado: {ext}"
    
    physics_type = model_config.get('physics_type', model_config.get('type', 'unknown'))
    
    # Verificación básica de compatibilidad basada en el nombre del archivo
    file_name = os.path.basename(data_path).lower()
    
    if physics_type == "burgers" and "burgers" not in file_name:
        return False, f"El archivo de datos no parece ser para el tipo 'burgers': {file_name}"
        
    if physics_type == "taylor_green" and "taylor_green" not in file_name:
        return False, f"El archivo de datos no parece ser para el tipo 'taylor_green': {file_name}"
        
    if physics_type == "kovasznay" and "kovasznay" not in file_name:
        return False, f"El archivo de datos no parece ser para el tipo 'kovasznay': {file_name}"
        
    if physics_type == "cavity_flow" and "cavity" not in file_name:
        return False, f"El archivo de datos no parece ser para el tipo 'cavity_flow': {file_name}"
    
    # Si llegamos aquí, asumimos que son compatibles
    return True, "El modelo y los datos parecen ser compatibles"

if __name__ == "__main__":
    # Configuración del logger para pruebas
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Probar la función para diferentes tipos de física
    for physics_type in ["burgers", "taylor_green", "kovasznay", "cavity_flow", "navier_stokes", "unknown_type"]:
        print(f"\nPara tipo de física '{physics_type}':")
        
        # Buscar archivos coincidentes
        matching_files = find_matching_data_files(physics_type)
        print(f"  Archivos coincidentes: {len(matching_files)}")
        for i, file in enumerate(matching_files[:3]):  # Mostrar solo los primeros 3
            print(f"    - {file}")
        if len(matching_files) > 3:
            print(f"    ... y {len(matching_files) - 3} más")
        
        # Obtener ruta por defecto
        default_path = get_default_data_path(physics_type)
        print(f"  Ruta por defecto: {default_path}")
