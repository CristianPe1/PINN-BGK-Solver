"""
Módulo de modelos de fluidos para Physics Informed Neural Networks.
Este módulo contiene implementaciones de modelos PINN para diferentes tipos de flujos:
- Taylor-Green Vortex
- Kovasznay Flow
- Cavity Flow
- Navier-Stokes general
"""

from .taylor_green_pinn import TaylorGreenPINN
from .model_selector import FluidModelSelector, load_fluid_model

# Exportar solo lo necesario
__all__ = [
    'TaylorGreenPINN',
    'FluidModelSelector',
    'load_fluid_model',
]
