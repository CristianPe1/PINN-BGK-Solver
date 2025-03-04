"""
Script de diagnóstico para verificar la disponibilidad y estructura de los modelos
"""
import os
import sys
import importlib
import inspect
import logging

# Configurar logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def check_directory_structure():
    """Verifica la estructura de directorios del proyecto"""
    logger.info("Verificando estructura de directorios...")
    
    # Directorio raíz del proyecto
    root_dir = os.path.dirname(os.path.dirname(__file__))
    
    # Directorios importantes a verificar
    directories = [
        "config",
        "src",
        "src/structure_model",
        "src/structure_model/fluid_models",
        "src/model"
    ]
    
    # Verificar cada directorio
    for directory in directories:
        dir_path = os.path.join(root_dir, directory)
        exists = os.path.exists(dir_path)
        is_dir = os.path.isdir(dir_path) if exists else False
        
        logger.info(f"Directorio '{directory}': {'Existe' if exists else 'No existe'} - {'Es directorio' if is_dir else 'No es directorio'}")
        
        if exists and is_dir:
            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            logger.info(f"  Archivos: {files}")

def check_model_files():
    """Verifica la existencia de archivos de modelos importantes"""
    logger.info("Verificando archivos de modelos...")
    
    # Directorio raíz del proyecto
    root_dir = os.path.dirname(os.path.dirname(__file__))
    
    # Archivos importantes a verificar
    files = [
        "src/structure_model/base_model.py",
        "src/structure_model/pinn_small.py",
        "src/structure_model/fluid_models/taylor_green_pinn.py",
        "src/structure_model/fluid_models/navier_stokes_pinn.py",
        "src/structure_model/fluid_models/kovasznay_pinn.py",
        "src/structure_model/fluid_models/cavity_flow_pinn.py",
        "src/structure_model/model_adapter.py",
        "src/model/model_selector.py",
        "config/config.yaml"
    ]
    
    # Verificar cada archivo
    for file in files:
        file_path = os.path.join(root_dir, file)
        exists = os.path.exists(file_path)
        is_file = os.path.isfile(file_path) if exists else False
        
        logger.info(f"Archivo '{file}': {'Existe' if exists else 'No existe'} - {'Es archivo' if is_file else 'No es archivo'}")
        
        if not exists or not is_file:
            logger.error(f"El archivo {file} no existe o no es un archivo válido")

def try_import_model_files():
    """Intenta importar archivos de modelos para verificar si son válidos"""
    logger.info("Intentando importar archivos de modelos...")
    
    modules_to_check = [
        "structure_model.base_model",
        "structure_model.pinn_small",
        "structure_model.fluid_models.taylor_green_pinn",
        "structure_model.model_adapter",
        "model.model_selector"
    ]
    
    for module_name in modules_to_check:
        try:
            module = importlib.import_module(module_name)
            logger.info(f"Módulo '{module_name}' importado correctamente")
            
            # Listar clases definidas en el módulo
            classes = inspect.getmembers(module, inspect.isclass)
            class_names = [name for name, _ in classes if name[0] != '_']
            logger.info(f"  Clases definidas: {class_names}")
            
        except Exception as e:
            logger.error(f"Error al importar '{module_name}': {e}")

def check_fluid_models():
    """Verifica específicamente los modelos de fluidos"""
    logger.info("Verificando modelos de fluidos...")
    
    fluid_models_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "src", "structure_model", "fluid_models"
    )
    
    # Verificar si el directorio existe
    if not os.path.exists(fluid_models_dir) or not os.path.isdir(fluid_models_dir):
        logger.error(f"El directorio de modelos de fluidos no existe: {fluid_models_dir}")
        return
    
    # Listar archivos en el directorio
    files = os.listdir(fluid_models_dir)
    python_files = [f for f in files if f.endswith('.py')]
    logger.info(f"Archivos Python en el directorio de modelos de fluidos: {python_files}")
    
    # Verificar si hay un archivo __init__.py
    if "__init__.py" not in files:
        logger.warning("No se encontró __init__.py en el directorio de modelos de fluidos")
    
    # Importar y verificar el archivo taylor_green_pinn.py
    if "taylor_green_pinn.py" in python_files:
        try:
            from structure_model.fluid_models.taylor_green_pinn import TaylorGreenPINN
            logger.info("TaylorGreenPINN importado correctamente")
            
            # Verificar si la clase tiene los métodos esperados
            methods = [name for name, _ in inspect.getmembers(TaylorGreenPINN, inspect.isfunction)]
            logger.info(f"  Métodos en TaylorGreenPINN: {methods}")
            
        except Exception as e:
            logger.error(f"Error al importar TaylorGreenPINN: {e}")

def main():
    """Función principal que ejecuta todas las verificaciones"""
    logger.info("=== Comenzando diagnóstico de modelos ===")
    
    check_directory_structure()
    check_model_files()
    try_import_model_files()
    check_fluid_models()
    
    logger.info("=== Diagnóstico de modelos completado ===")

if __name__ == "__main__":
    main()
