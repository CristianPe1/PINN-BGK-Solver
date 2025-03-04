import torch
import torch.nn as nn
from .base_model import BaseNeuralNetwork

class PINN_V1(BaseNeuralNetwork):
    """
    Red neuronal informada por física (PINN) versión 1.
    Implementa una red neuronal MLP estándar para aproximar la solución de ecuaciones diferenciales.
    """
    def __init__(self, layer_sizes, activation_name="Tanh"):
        """
        Inicializa la PINN V1.
        
        Args:
            layer_sizes (list): Lista con tamaños de capas [entrada, capa1, ..., capaN, salida]
            activation_name (str): Nombre de la función de activación
        """
        super(PINN_V1, self).__init__(
            layer_sizes=layer_sizes, 
            activation_name=activation_name,
            name="PINN_V1"
        )
        
    # El método forward se hereda de BaseNeuralNetwork