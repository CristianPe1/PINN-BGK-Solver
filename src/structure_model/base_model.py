import torch
import torch.nn as nn
import numpy as np

class BaseNeuralNetwork(nn.Module):
    """
    Clase base para redes neuronales en el proyecto PINN-BGK.
    Define una estructura común y métodos estándar para todos los modelos,
    asegurando consistencia en los nombres de capas y compatibilidad.
    """
    def __init__(self, layer_sizes, activation_name="Tanh", name="BaseModel"):
        """
        Inicializa la red neuronal base.
        
        Args:
            layer_sizes (list): Lista con tamaños de capas [entrada, capa1, ..., capaN, salida]
            activation_name (str): Nombre de la función de activación
            name (str): Nombre del modelo para identificación
        """
        super(BaseNeuralNetwork, self).__init__()
        self.layer_sizes = layer_sizes
        self.activation_name = activation_name
        self.name = name
        
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
        
        # Construir capas lineales con nombre estándar
        self.linear_layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.linear_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
        # Inicialización de pesos según el tipo de activación
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializa pesos según el tipo de activación para mejorar el entrenamiento."""
        for layer in self.linear_layers:
            if self.activation_name == 'Tanh':
                # Xavier/Glorot para tanh
                nn.init.xavier_normal_(layer.weight)
            elif self.activation_name == 'ReLU' or self.activation_name == 'LeakyReLU':
                # He/Kaiming para ReLU
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            elif self.activation_name == 'Sine':
                # Inicialización especial para SIREN
                omega_0 = 30.0
                with torch.no_grad():
                    if layer == self.linear_layers[0]:
                        layer.weight.uniform_(-1/layer.in_features, 1/layer.in_features)
                    else:
                        layer.weight.uniform_(-np.sqrt(6/layer.in_features)/omega_0, 
                                            np.sqrt(6/layer.in_features)/omega_0)
            else:
                # Xavier por defecto
                nn.init.xavier_normal_(layer.weight)
            
            # Inicializar bias a cero
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """
        Propagación hacia adelante en la red.
        
        Args:
            x (torch.Tensor): Tensor de entrada
            
        Returns:
            torch.Tensor: Salida de la red
        """
        # Implementación genérica del forward pass
        for i in range(len(self.linear_layers) - 1):
            x = self.activation(self.linear_layers[i](x))
        
        # Última capa sin activación (típicamente para salida)
        x = self.linear_layers[-1](x)
        return x
    
    def __str__(self):
        """Representación en string del modelo."""
        return f"{self.name} con arquitectura {self.layer_sizes} y activación {self.activation_name}"
