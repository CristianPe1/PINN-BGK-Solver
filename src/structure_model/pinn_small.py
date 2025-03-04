import torch
import torch.nn as nn
from .base_model import BaseNeuralNetwork

class PINN_Small(BaseNeuralNetwork):
    """
    Implementación de una red neuronal PINN pequeña y rápida.
    Diseñada para cuando se requiere menor capacidad computacional
    pero manteniendo buena precisión.
    """
    def __init__(self, layer_sizes=[2, 20, 20, 1], activation_name="ReLU"):
        """
        Inicializa la PINN pequeña.
        
        Args:
            layer_sizes (list): Lista con tamaños de capas [entrada, capa1, ..., capaN, salida]
            activation_name (str): Nombre de la función de activación
        """
        super(PINN_Small, self).__init__(
            layer_sizes=layer_sizes, 
            activation_name=activation_name,
            name="PINN_Small"
        )
