import torch
import torch.nn as nn
import math
from ..base_model import BaseNeuralNetwork

class KovasznayPINN(BaseNeuralNetwork):
    """
    Red neuronal para el flujo de Kovasznay, una solución analítica de las ecuaciones de Navier-Stokes.
    Útil para validar métodos numéricos para fluidos.
    """
    def __init__(self, layer_sizes=[2, 40, 40, 40, 3], activation_name="Tanh", re=40.0):
        """
        Inicializa la red para flujo de Kovasznay.
        
        Args:
            layer_sizes (list): Lista con tamaños de capas.
                            Por defecto [2, 40, 40, 40, 3]
                            donde entrada es (x, y) y salida es (u, v, p)
            activation_name (str): Nombre de la función de activación
            re (float): Número de Reynolds
        """
        super(KovasznayPINN, self).__init__(
            layer_sizes=layer_sizes, 
            activation_name=activation_name,
            name="KovasznayPINN"
        )
        self.re = re
        
        # Calcular lambda para la solución analítica
        self.lamb = 0.5 * re - math.sqrt(0.25 * (re ** 2) + 4 * (math.pi ** 2))
        
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
        
    def analytic_solution(self, x):
        """
        Calcula la solución analítica para el flujo de Kovasznay.
        
        Args:
            x (torch.Tensor): Tensor de entrada (x, y)
            
        Returns:
            torch.Tensor: Tensor con la solución analítica [batch_size, 3]
        """
        # Extraer coordenadas
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        
        # Calcular componentes de velocidad
        u = 1 - torch.exp(self.lamb * x_coord) * torch.cos(2 * math.pi * y_coord)
        v = self.lamb / (2 * math.pi) * torch.exp(self.lamb * x_coord) * torch.sin(2 * math.pi * y_coord)
        
        # Calcular presión
        p = 0.5 * (1 - torch.exp(2 * self.lamb * x_coord))
        
        return torch.cat([u, v, p], dim=1)
        
    def compute_pde_residual(self, x, u_pred):
        """
        Calcula el residual de las ecuaciones de Navier-Stokes para el flujo de Kovasznay.
        
        Args:
            x (torch.Tensor): Coordenadas (x, y)
            u_pred (torch.Tensor): Predicciones (u, v, p)
            
        Returns:
            torch.Tensor: Residual de las ecuaciones
        """
        # Esta función se implementa en la función de pérdida PDE específica
        pass
