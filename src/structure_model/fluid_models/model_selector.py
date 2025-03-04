import torch
import yaml
from pathlib import Path
import importlib
import logging

# Importamos los modelos disponibles
from .taylor_green_pinn import TaylorGreenPINN
from .cavity_flow_pinn import CavityFlowPINN
from . navier_stokes_pinn import NavierStokesPINN
from .kovasznay_pinn import KovasznayPINN

# Configurar logging
logger = logging.getLogger(__name__)

class FluidModelSelector:
    """
    Clase para seleccionar y cargar modelos de fluidos dinámicamente basándose
    en la configuración proporcionada.
    """
    
    def __init__(self):
        """Inicializa el selector de modelos de fluidos."""
        # Mapeo de nombres de modelos a sus clases correspondientes
        self.model_map = {
            "taylor_green": TaylorGreenPINN,
            # Agregar otros modelos aquí a medida que se implementen
            "navier_stokes": NavierStokesPINN,
            "kovasznay": KovasznayPINN,
            "cavity_flow": CavityFlowPINN,
        }
    
    def get_model(self, config_dict=None, config_file=None, model_name=None):
        """
        Carga y devuelve un modelo de fluido basado en la configuración.
        
        Args:
            config_dict (dict, optional): Diccionario con la configuración.
            config_file (str, optional): Ruta al archivo de configuración YAML.
            model_name (str, optional): Nombre del modelo a cargar (sobrescribe la configuración).
            
        Returns:
            BaseNeuralNetwork: Instancia del modelo de fluido seleccionado.
            
        Raises:
            ValueError: Si no se proporciona configuración o si el modelo no existe.
        """
        # Cargar configuración desde archivo si se proporciona
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error al cargar el archivo de configuración: {e}")
                raise
        # Usar el diccionario de configuración proporcionado
        elif config_dict:
            config = config_dict
        else:
            raise ValueError("Se debe proporcionar config_dict o config_file")
        
        # Determinar el modelo a usar
        if model_name:
            selected_model = model_name
        else:
            if "selected_model" in config:
                selected_model = config["selected_model"]
            else:
                raise ValueError("No se especificó un modelo en la configuración")
        
        # Buscar el modelo en la configuración
        if selected_model in config.get("models", {}):
            model_config = config["models"][selected_model]
            model_type = model_config.get("physics_type", "")
        else:
            raise ValueError(f"El modelo {selected_model} no está definido en la configuración")
        
        # Verificar si el modelo existe en nuestro mapeo
        if model_type in self.model_map:
            model_class = self.model_map[model_type]
            
            # Extraer parámetros relevantes para el modelo
            layers = model_config.get("layers", [])
            activation = model_config.get("activation_function", "Tanh")
            
            # Parámetros específicos para cada tipo de modelo
            kwargs = {}
            if model_type == "taylor_green":
                kwargs["nu"] = model_config.get("nu", config.get("physics", {}).get("nu", 0.01))
            # Agregar parámetros específicos para otros modelos aquí
            
            # Instanciar y retornar el modelo
            return model_class(layer_sizes=layers, activation_name=activation, **kwargs)
        else:
            raise ValueError(f"Modelo de tipo '{model_type}' no implementado")
    
    def list_available_models(self):
        """
        Lista todos los modelos de fluidos disponibles.
        
        Returns:
            list: Lista de nombres de modelos disponibles.
        """
        return list(self.model_map.keys())


# Función auxiliar para cargar un modelo desde la configuración
def load_fluid_model(config_path=None, config_dict=None, model_name=None):
    """
    Función de conveniencia para cargar un modelo de fluido.
    
    Args:
        config_path (str, optional): Ruta al archivo de configuración.
        config_dict (dict, optional): Diccionario con la configuración.
        model_name (str, optional): Nombre específico del modelo a cargar.
        
    Returns:
        BaseNeuralNetwork: Instancia del modelo seleccionado.
    """
    selector = FluidModelSelector()
    return selector.get_model(config_dict=config_dict, config_file=config_path, model_name=model_name)
