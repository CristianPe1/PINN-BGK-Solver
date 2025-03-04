import torch
import torch.nn as nn
from ..base_model import BaseNeuralNetwork

class CavityFlowPINN(BaseNeuralNetwork):
    """
    Red neuronal para el problema de flujo en cavidad, un problema clásico de fluidos.
    Modela el flujo en una cavidad cuadrada con una pared superior móvil.
    """
    def __init__(self, layer_sizes=[2, 50, 50, 50, 50, 3], activation_name="Tanh", nu=0.01, u0=1.0):
        """
        Inicializa la red para flujo en cavidad.
        
        Args:
            layer_sizes (list): Lista con tamaños de capas.
                            Por defecto [2, 50, 50, 50, 50, 3]
                            donde entrada es (x, y) y salida es (u, v, p)
            activation_name (str): Nombre de la función de activación
            nu (float): Viscosidad cinemática
            u0 (float): Velocidad de la pared superior
        """
        super(CavityFlowPINN, self).__init__(
            layer_sizes=layer_sizes, 
            activation_name=activation_name,
            name="CavityFlowPINN"
        )
        self.nu = nu
        self.u0 = u0
        
    def forward(self, x):
        """
        Propagación hacia adelante.
        
        Args:
            x (torch.Tensor): Tensor de entrada (x, y)
            
        Returns:
            torch.Tensor: Tensor de salida [batch_size, 3] donde cada fila es (u, v, p)
        """
        # Propagar a través de las primeras capas
        for i in range(len(self.linear_layers) - 1):
            x = self.activation(self.linear_layers[i](x))
        
        # Capa final sin activación
        output = self.linear_layers[-1](x)
        return output
        
    def apply_boundary_conditions(self, x, y):
        """
        Aplica condiciones de contorno para el problema de flujo en cavidad.
        
        Args:
            x (torch.Tensor): Coordenadas x de los puntos
            y (torch.Tensor): Coordenadas y de los puntos
            
        Returns:
            tuple: Tensores (u, v) con condiciones de contorno aplicadas
        """
        # Inicializar velocidades
        u = torch.zeros_like(x)
        v = torch.zeros_like(x)
        
        # Condición de pared superior móvil (y = 1)
        top_wall = (y >= 0.999)
        u[top_wall] = self.u0
        
        return u, v
