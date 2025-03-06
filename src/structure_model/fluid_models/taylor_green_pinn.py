import torch
import torch.nn as nn
from ..base_model import BaseNeuralNetwork

class TaylorGreenPINN(BaseNeuralNetwork):
    """
    Red neuronal para vórtice de Taylor-Green, un flujo ideal que decae con el tiempo.
    Usa una arquitectura con capas más anchas para capturar la evolución temporal.
    """
    def __init__(self, layer_sizes=[5, 64, 64, 64, 64, 3], activation_name="Swish", nu=0.01):
        """
        Inicializa la red para vórtice de Taylor-Green.
        
        Args:
            layer_sizes (list): Lista con tamaños de capas.
                            Por defecto [5, 64, 64, 64, 64, 3]
                            donde entrada es (x, y, t, sin_x, cos_x) y salida es (u, v, p)
            activation_name (str): Nombre de la función de activación
            nu (float): Viscosidad cinemática
        """
        # El cambio clave está aquí: la primera dimensión es ahora 5 en lugar de 3
        super(TaylorGreenPINN, self).__init__(
            layer_sizes=layer_sizes, 
            activation_name=activation_name,
            name="TaylorGreenPINN"
        )
        self.nu = nu
        
        # Codificar periódicamente las entradas (importante para el flujo de Taylor-Green)
        self.freq_multiplier = nn.Parameter(torch.ones(1) * 2.0 * torch.pi)
        
    def forward(self, x):
        """
        Propagación hacia adelante.
        
        Args:
            x (torch.Tensor): Tensor de entrada [batch_size, 5]
            
        Returns:
            torch.Tensor: Tensor de salida [batch_size, 3] donde cada fila es (u, v, p)
        """
        # El tensor de entrada ya tiene 5 características, así que lo usamos directamente
        encoded_input = x
        
        # Propagar a través de las capas
        for i in range(len(self.linear_layers) - 1):
            encoded_input = self.activation(self.linear_layers[i](encoded_input))
        
        # Capa final sin activación
        output = self.linear_layers[-1](encoded_input)
        return output
    
    def analytic_solution(self, x):
        """
        Calcula la solución analítica para el vórtice de Taylor-Green.
        
        Args:
            x (torch.Tensor): Tensor de entrada (x, y, t)
            
        Returns:
            torch.Tensor: Tensor con la solución analítica [batch_size, 3]
        """
        # Extraer coordenadas (asume que las primeras 3 componentes son x, y, t)
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        t_coord = x[:, 2:3]
        
        # Calcular factor de decaimiento temporal
        decay = torch.exp(-2 * self.nu * t_coord)
        
        # Calcular componentes de velocidad
        u = -torch.cos(x_coord) * torch.sin(y_coord) * decay
        v = torch.sin(x_coord) * torch.cos(y_coord) * decay
        
        # Calcular presión
        p = -0.25 * (torch.cos(2*x_coord) + torch.cos(2*y_coord)) * decay * decay
        
        return torch.cat([u, v, p], dim=1)
