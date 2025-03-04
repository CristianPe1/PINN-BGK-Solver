"""
Model package initialization
"""

"""
Módulo central para la gestión de modelos en PINN-BGK.
Proporciona acceso a las funciones de selección y carga de modelos.
"""

from .model_selector import (
    ModelSelector,
    load_model,
    load_fluid_model,
    load_saved_model
)

__all__ = [
    'ModelSelector',
    'load_model',
    'load_fluid_model',
    'load_saved_model',
]
