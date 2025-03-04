import torch
import torch.nn as nn
import numpy as np
import logging

from ..structure_model.pinn_structure_v1 import PINN_V1
from ..structure_model.burguers_models.pinn_structure_v2 import PINN_V2
from ..structure_model.pinn_small import PINN_Small
from ..structure_model.fluid_models import get_fluid_model, FLUID_MODELS

logger = logging.getLogger(__name__)

# Importar modelos de fluidos
try:
    
    FLUID_MODELS_AVAILABLE = True
except ImportError:
    logger.warning("Modelos de fluidos no disponibles. Algunas funcionalidades estarán limitadas.")
    FLUID_MODELS_AVAILABLE = False

class SwishActivation(nn.Module):
    """Implementación de la función de activación Swish: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResidualBlock(nn.Module):
    """Bloque residual para redes MLP con conexiones residuales"""
    def __init__(self, dim, activation="ReLU"):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        
        # Elegir activación
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "Tanh":
            self.activation = nn.Tanh()
        elif activation == "Swish":
            self.activation = SwishActivation()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x):
        identity = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        out = out + identity  # Conexión residual
        return self.activation(out)

class PINN_V2(nn.Module):
    """
    Versión mejorada de PINN con bloques residuales y activación Swish.
    """
    def __init__(self, layers, activation="Swish"):
        super().__init__()
        
        # Capas de la red
        self.net_layers = []
        for i in range(len(layers) - 2):
            self.net_layers.append(nn.Linear(layers[i], layers[i+1]))
            
            # Activación después de cada capa excepto la última
            if activation == "ReLU":
                self.net_layers.append(nn.ReLU())
            elif activation == "Tanh":
                self.net_layers.append(nn.Tanh())
            elif activation == "Swish":
                self.net_layers.append(SwishActivation())
            else:
                self.net_layers.append(nn.Tanh())
                
            # Añadir bloque residual cada 2 capas
            if i > 0 and i % 2 == 0 and layers[i] == layers[i+1]:
                self.net_layers.append(ResidualBlock(layers[i+1], activation))
        
        # Capa de salida
        self.net_layers.append(nn.Linear(layers[-2], layers[-1]))
        
        # Convertir a Sequential
        self.net = nn.Sequential(*self.net_layers)
        
        # Inicialización de pesos Xavier
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, input_tensor):
        """
        Forward pass del modelo.
        
        Args:
            input_tensor: Tensor de entrada [batch_size, 2] donde cada fila es (x, t)
                      o tupla (x, t) de tensores separados
            
        Returns:
            torch.Tensor: Salida de la red [batch_size, 1]
        """
        # Manejar ambos casos: cuando se pasan x, t por separado o como un tensor conjunto
        if isinstance(input_tensor, tuple) and len(input_tensor) == 2:
            x, t = input_tensor
            # Concatenar x y t para formar la entrada
            if x.dim() == 1:
                x = x.unsqueeze(1)
            if t.dim() == 1:
                t = t.unsqueeze(1)
            xt = torch.cat([x, t], dim=1)
        else:
            # Si ya viene como un tensor conjunto [batch_size, 2]
            xt = input_tensor
            
        return self.net(xt)

def create_model(config):
    """
    Crea un modelo según la configuración proporcionada.
    
    Args:
        config (dict): Diccionario con la configuración del modelo:
            - type: Tipo de modelo (mlp, mlp_residual, siren, etc.)
            - layers: Lista con tamaños de capas [entrada, capa1, ..., capaN, salida]
            - activation_function: Nombre de la función de activación
            
    Returns:
        nn.Module: Modelo creado según la configuración
    """
    # Extraer parámetros de configuración
    model_type = config.get("type", "mlp").lower()
    print(model_type)
    layers = config.get("layers", [2, 50, 50, 50, 50, 1])
    activation = config.get("activation_function", "Tanh")
    print(activation)
    
    # Verificar parámetros requeridos
    if not layers:
        raise ValueError("La lista de capas (layers) no puede estar vacía")
    
    logger.info(f"Creando modelo de tipo '{model_type}' con capas {layers} y activación '{activation}'")
    

    # Modelos de fluidos
    if model_type in ["navier_stokes", "kovasznay", "taylor_green", "cavity_flow"]:
        if not FLUID_MODELS_AVAILABLE:
            logger.error(f"Modelo de fluido {model_type} solicitado, pero los modelos de fluidos no están disponibles.")
            raise ImportError("Módulos de modelos de fluidos no disponibles")
            
        logger.info(f"Creando modelo de fluido {model_type} con capas {layers} y activación {activation}")
        
        # Parámetros adicionales específicos para fluidos
        kwargs = {
            "layer_sizes": layers,
            "activation_name": activation
        }
        
        # Añadir parámetros específicos según el modelo
        if model_type == "kovasznay":
            kwargs["re"] = config.get("re", 40.0)
        elif model_type == "taylor_green":
            kwargs["nu"] = config.get("nu", 0.01)
            
        return get_fluid_model(model_type, **kwargs)
        
    else:
        logger.warning(f"Tipo de modelo '{model_type}' no reconocido. Usando MLP por defecto.")
        return PINN_V1(layers, activation)

# Función para recuperar un modelo específico a partir del nombre
def get_model_by_name(name, config=None):
    """
    Devuelve una instancia de modelo por su nombre.
    
    Args:
        name (str): Nombre del modelo
        config (dict, optional): Configuración del modelo
    
    Returns:
        nn.Module: Instancia del modelo
    """
    if config is None:
        config = {}
    
    model_map = {
        "pinn_v1": PINN_V1,
        "pinn_v2": PINN_V2,
        "pinn_small": PINN_Small
    }
    
    if name.lower() not in model_map:
        logger.warning(f"Modelo '{name}' no encontrado. Usando PINN_V1 por defecto.")
        name = "pinn_v1"
    
    model_class = model_map[name.lower()]
    
    # Obtener parámetros de la configuración
    layers = config.get("layers", [2, 50, 50, 50, 50, 1])
    activation = config.get("activation_function", "Tanh")
    
    return model_class(layers, activation)
