import torch
import torch.nn as nn
import logging

from structure_model.pinn_structure_v1 import PINN_V1
from structure_model.burguers_models.pinn_structure_v2 import PINN_V2
from structure_model.pinn_small import PINN_Small

logger = logging.getLogger(__name__)

# Set fluid models flag - we'll handle missing imports more gracefully
FLUID_MODELS_AVAILABLE = False
try:
    from structure_model.fluid_models import (
        TaylorGreenPINN, NavierStokesPINN, KovasznayPINN, CavityFlowPINN
    )
    FLUID_MODELS_AVAILABLE = True
except ImportError:
    logger.warning("Modelos de fluidos no disponibles. Algunas funcionalidades estarán limitadas.")

# Define physics problem dimensions
PHYSICS_DIMENSIONS = {
    # Format: "physics_type": (input_dim, output_dim)
    "burgers": (2, 1),          # 1D Burgers: (x,t) -> u
    "poisson": (2, 1),          # 2D Poisson: (x,y) -> u
    "heat": (2, 1),             # 1D Heat: (x,t) -> u
    "wave": (2, 1),             # 1D Wave: (x,t) -> u
    "taylor_green": (3, 3),     # Taylor-Green: (x,y,t) -> (u,v,p)
    "navier_stokes": (3, 3),    # Navier-Stokes: (x,y,t) -> (u,v,p)
    "kovasznay": (2, 3),        # Kovasznay: (x,y) -> (u,v,p)
    "cavity_flow": (2, 3),      # Cavity flow: (x,y) -> (u,v,p)
    "mlp": (2, 1),              # Default MLP: generic 2->1 mapping
    "mlp_residual": (2, 1)      # Default residual MLP: generic 2->1 mapping
}

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

def create_model(config):
    """
    Crea un modelo según la configuración proporcionada.
    """
    # Extraer parámetros de configuración
    model_type = config.get("type", "mlp").lower()
    physics_type = config.get("physics_type", model_type)
    layers = config.get("layers", [])
    activation = config.get("activation_function", "Tanh")
    
    # Get required dimensions for this physics type
    if physics_type in PHYSICS_DIMENSIONS:
        input_dim, output_dim = PHYSICS_DIMENSIONS[physics_type]
        logger.info(f"Usando dimensiones para '{physics_type}': entrada={input_dim}, salida={output_dim}")
    else:
        # Default dimensions if physics type not recognized
        input_dim, output_dim = 2, 1
        logger.warning(f"Tipo de física '{physics_type}' no reconocido. Usando dimensiones por defecto: entrada={input_dim}, salida={output_dim}")
    
    # If layers not provided or empty, create default
    if not layers:
        layers = [input_dim, 50, 50, 50, output_dim]
    else:
        # Override first and last layer dimensions to match physics requirements
        if len(layers) > 0:
            layers[0] = input_dim  # Input dimension
        if len(layers) > 1:
            layers[-1] = output_dim  # Output dimension
            
    logger.info(f"Creando modelo de tipo '{model_type}' con capas {layers} y activación '{activation}'")

    # Modelos de fluidos
    if model_type in ["navier_stokes", "kovasznay", "taylor_green", "cavity_flow"]:
        if not FLUID_MODELS_AVAILABLE:
            logger.error(f"Modelo de fluido {model_type} solicitado, pero los modelos de fluidos no están disponibles.")
            raise ImportError("Módulos de modelos de fluidos no disponibles")
            
        # Validate dimensions for fluid models
        expected_input, expected_output = PHYSICS_DIMENSIONS.get(model_type, (3, 3))
        if layers[0] != expected_input or layers[-1] != expected_output:
            logger.warning(f"Ajustando dimensiones de capa para modelo {model_type}: {layers[0]}→{expected_input}, {layers[-1]}→{expected_output}")
            layers[0] = expected_input
            layers[-1] = expected_output
            
        # Return the appropriate fluid model
        if model_type == "navier_stokes":
            return NavierStokesPINN(layer_sizes=layers, activation_name=activation)
        elif model_type == "kovasznay":
            return KovasznayPINN(
                layer_sizes=layers, 
                activation_name=activation,
                re=config.get("re", 40.0)
            )
        elif model_type == "taylor_green":
            return TaylorGreenPINN(
                layer_sizes=layers, 
                activation_name=activation,
                nu=config.get("nu", 0.01)
            )
        elif model_type == "cavity_flow":
            return CavityFlowPINN(
                layer_sizes=layers, 
                activation_name=activation,
                nu=config.get("nu", 0.01)
            )
    else:
        # Default to PINN_V1 for non-fluid models
        # Also ensure proper dimensions for standard models
        expected_input, expected_output = PHYSICS_DIMENSIONS.get(physics_type, (2, 1))
        if layers[0] != expected_input or layers[-1] != expected_output:
            logger.warning(f"Ajustando dimensiones de capa para física {physics_type}: {layers[0]}→{expected_input}, {layers[-1]}→{expected_output}")
            layers[0] = expected_input
            layers[-1] = expected_output
            
        return PINN_V1(layers, activation)

def get_model_by_name(name, config=None):
    """
    Devuelve una instancia de modelo por su nombre.
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
    
    # Determine physics type to get dimensions
    physics_type = config.get("physics_type", "burgers")
    input_dim, output_dim = PHYSICS_DIMENSIONS.get(physics_type, (2, 1))
    
    # Get layers with proper dimensions
    layers = config.get("layers", [input_dim, 50, 50, 50, output_dim])
    if len(layers) > 0:
        layers[0] = input_dim
    if len(layers) > 1:
        layers[-1] = output_dim
        
    activation = config.get("activation_function", "Tanh")
    
    return model_class(layers, activation)
