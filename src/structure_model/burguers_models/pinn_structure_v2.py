import torch
import torch.nn as nn
from .base_model import BaseNeuralNetwork

class PINN_V2(nn.Module):
    """
    Red neuronal informada por física (PINN) versión 2.
    Implementa una arquitectura con conexiones residuales para mejorar flujo de gradientes.
    """
    def __init__(self, layer_sizes, activation_name="Swish"):
        """
        Inicializa la PINN V2 con conexiones residuales.
        
        Args:
            layer_sizes (list): Lista con tamaños de capas [entrada, capa1, ..., capaN, salida]
            activation_name (str): Nombre de la función de activación
        """
        super(PINN_V2, self).__init__()
        self.layer_sizes = layer_sizes
        self.activation_name = activation_name
        self.name = "PINN_V2_Residual"
        
        # Mapeo de nombres de activaciones a funciones
        self.activation_map = {
            'Tanh': nn.Tanh(),
            'ReLU': nn.ReLU(),
            'Sigmoid': nn.Sigmoid(),
            'LeakyReLU': nn.LeakyReLU(0.2),
            'GELU': nn.GELU(),
            'Swish': lambda x: x * torch.sigmoid(x),  # Swish: x * sigmoid(x)
            'Sine': lambda x: torch.sin(x)  # Función seno para SIREN
        }
        
        # Verificar que la activación solicitada existe
        if activation_name not in self.activation_map:
            raise ValueError(f"Activación '{activation_name}' no soportada. Opciones: {list(self.activation_map.keys())}")
        
        # Inicializar función de activación
        self.activation = self.activation_map[activation_name]
        
        # Construir capas lineales con nombre estándar (linear_layers)
        self.linear_layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.linear_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
        # Inicialización de pesos
        self._initialize_weights()
        
        # Normalización por capas
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(layer_sizes[i+1]) 
            for i in range(len(layer_sizes) - 2)
        ])
    
    def _initialize_weights(self):
        """Inicializa los pesos de las capas lineales."""
        for layer in self.linear_layers:
            if self.activation_name == 'Tanh':
                nn.init.xavier_normal_(layer.weight)
            elif self.activation_name in ['ReLU', 'LeakyReLU']:
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            elif self.activation_name == 'Sine':
                omega_0 = 30.0
                with torch.no_grad():
                    if layer == self.linear_layers[0]:
                        layer.weight.uniform_(-1/layer.in_features, 1/layer.in_features)
                    else:
                        layer.weight.uniform_(-torch.sqrt(torch.tensor(6/layer.in_features))/omega_0, 
                                             torch.sqrt(torch.tensor(6/layer.in_features))/omega_0)
            else:
                nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """
        Propagación hacia adelante con conexiones residuales.
        
        Args:
            x (torch.Tensor): Tensor de entrada
            
        Returns:
            torch.Tensor: Salida de la red
        """
        # Primera capa siempre se aplica normalmente
        x = self.activation(self.linear_layers[0](x))
        
        # Capas intermedias con conexiones residuales
        for i in range(1, len(self.linear_layers) - 1):
            # Guardar entrada para conexión residual
            residual = x
            
            # Aplicar transformación lineal y activación
            x = self.linear_layers[i](x)
            x = self.layer_norms[i-1](x)  # Normalización
            x = self.activation(x)
            
            # Añadir conexión residual si las dimensiones coinciden
            if residual.shape == x.shape:
                x = x + residual
        
        # Última capa sin activación ni residual
        x = self.linear_layers[-1](x)
        return x
