import torch
import torch.nn as nn
from ..base_model import BaseNeuralNetwork

class NavierStokesPINN(BaseNeuralNetwork):
    """
    Red neuronal general para problemas de Navier-Stokes.
    Se puede usar como base para diferentes tipos de flujos.
    """
    def __init__(self, layer_sizes=[3, 64, 64, 64, 64, 3], activation_name="Tanh", nu=0.01):
        """
        Inicializa la red para Navier-Stokes.
        
        Args:
            layer_sizes (list): Lista con tamaños de capas.
                            Por defecto [3, 64, 64, 64, 64, 3]
                            donde entrada es (x, y, t) y salida es (u, v, p)
            activation_name (str): Nombre de la función de activación
            nu (float): Viscosidad cinemática
        """
        super(NavierStokesPINN, self).__init__(
            layer_sizes=layer_sizes, 
            activation_name=activation_name,
            name="NavierStokesPINN"
        )
        self.nu = nu
        
    def forward(self, x):
        """
        Propagación hacia adelante.
        
        Args:
            x (torch.Tensor): Tensor de entrada (x, y, t) o (x, y)
            
        Returns:
            torch.Tensor: Tensor de salida [batch_size, 3] donde cada fila es (u, v, p)
        """
        # Propagar a través de las primeras capas
        for i in range(len(self.linear_layers) - 1):
            x = self.activation(self.linear_layers[i](x))
        
        # Capa final sin activación
        output = self.linear_layers[-1](x)
        return output
