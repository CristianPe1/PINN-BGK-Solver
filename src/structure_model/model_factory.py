import torch
import torch.nn as nn
import numpy as np
from .pinn_structure_v1 import PINN_V1

class SwishActivation(nn.Module):
    """Implementación de la función de activación Swish: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResidualBlock(nn.Module):
    """Bloque residual para redes MLP con conexiones residuales"""
    def __init__(self, dim, activation="ReLU"):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        
        # Elegir activación
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "Tanh":
            self.activation = nn.Tanh()
        elif activation == "Swish":
            self.activation = SwishActivation()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x):
        identity = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        out = out + identity  # Conexión residual
        return self.activation(out)

class PINN_V2(nn.Module):
    """
    Versión mejorada de PINN con bloques residuales y activación Swish.
    """
    def __init__(self, layers, activation="Swish"):
        super().__init__()
        
        # Capas de la red
        self.net_layers = []
        for i in range(len(layers) - 2):
            self.net_layers.append(nn.Linear(layers[i], layers[i+1]))
            
            # Activación después de cada capa excepto la última
            if activation == "ReLU":
                self.net_layers.append(nn.ReLU())
            elif activation == "Tanh":
                self.net_layers.append(nn.Tanh())
            elif activation == "Swish":
                self.net_layers.append(SwishActivation())
            else:
                self.net_layers.append(nn.Tanh())
                
            # Añadir bloque residual cada 2 capas
            if i > 0 and i % 2 == 0 and layers[i] == layers[i+1]:
                self.net_layers.append(ResidualBlock(layers[i+1], activation))
        
        # Capa de salida
        self.net_layers.append(nn.Linear(layers[-2], layers[-1]))
        
        # Convertir a Sequential
        self.net = nn.Sequential(*self.net_layers)
        
        # Inicialización de pesos Xavier
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, input_tensor):
        """
        Forward pass del modelo.
        
        Args:
            input_tensor: Tensor de entrada [batch_size, 2] donde cada fila es (x, t)
                      o tupla (x, t) de tensores separados
            
        Returns:
            torch.Tensor: Salida de la red [batch_size, 1]
        """
        # Manejar ambos casos: cuando se pasan x, t por separado o como un tensor conjunto
        if isinstance(input_tensor, tuple) and len(input_tensor) == 2:
            x, t = input_tensor
            # Concatenar x y t para formar la entrada
            if x.dim() == 1:
                x = x.unsqueeze(1)
            if t.dim() == 1:
                t = t.unsqueeze(1)
            xt = torch.cat([x, t], dim=1)
        else:
            # Si ya viene como un tensor conjunto [batch_size, 2]
            xt = input_tensor
            
        return self.net(xt)

def create_model(model_config):
    """
    Función factoría para crear diferentes modelos según la configuración.
    
    Args:
        model_config (dict): Configuración del modelo
        
    Returns:
        nn.Module: Instancia del modelo solicitado
    """
    model_type = model_config.get("type", "mlp")
    layers = model_config.get("layers", [2, 50, 50, 50, 50, 1])
    activation = model_config.get("activation_function", "Tanh")
    
    if model_type == "mlp":
        return PINN_V1(layers, activation)
    elif model_type == "mlp_residual":
        return PINN_V2(layers, activation)
    else:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}")
