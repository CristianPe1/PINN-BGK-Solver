import torch
import torch.nn as nn

import time
import torch
import torch.nn as nn

# -------- Versión V1 (por ejemplo, la usada en burguer_equation.py) --------
class PINN_V1(nn.Module):
    """
    Implementación simple de una red neuronal informada por la física (PINN)
    para resolver la ecuación de Burgers.
    """
    def __init__(self, layers, activation="Tanh"):
        """
        Inicializa la red neuronal con capas especificadas y función de activación.
        
        Args:
            layers (list): Lista con número de neuronas por capa (incluyendo entrada y salida)
            activation (str): Función de activación ("Tanh", "ReLU", "Sigmoid")
        """
        super(PINN_V1, self).__init__()
        
        # Verificar que hay al menos capas de entrada, oculta y salida
        if len(layers) < 3:
            raise ValueError("Se requieren al menos 3 capas (entrada, oculta, salida)")
            
        # Crear capas lineales
        self.linear_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linear_layers.append(nn.Linear(layers[i], layers[i+1]))
        
        # Definir función de activación
        if activation == "Tanh":
            self.activation = torch.tanh
        elif activation == "ReLU":
            self.activation = torch.relu
        elif activation == "Sigmoid":
            self.activation = torch.sigmoid
        else:
            self.activation = torch.tanh  # Default
            
        # Inicializar pesos con inicialización Xavier
        for m in self.linear_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, input_tensor):
        """
        Forward pass de la red. Acepta tanto un tensor único como argumentos separados.
        
        Args:
            input_tensor: Un tensor de forma [batch_size, 2] donde cada fila es (x, t)
                         o una tupla de tensores (x, t)
            
        Returns:
            torch.Tensor: Tensor de salida [batch_size, 1] con u(x, t)
        """
        # Manejar caso donde se proporcionan x, t por separado o como un tensor conjunto
        if isinstance(input_tensor, tuple) and len(input_tensor) == 2:
            x, t = input_tensor
            # Asegurar que tienen la dimensión correcta
            if x.dim() == 1:
                x = x.unsqueeze(1)  # [batch_size] -> [batch_size, 1]
            if t.dim() == 1:
                t = t.unsqueeze(1)  # [batch_size] -> [batch_size, 1]
            # Concatenar para formar un tensor [batch_size, 2]
            xt = torch.cat([x, t], dim=1)
        else:
            # Si ya viene como un tensor [batch_size, 2]
            xt = input_tensor
        
        # Forward pass a través de todas las capas
        out = xt
        n_layers = len(self.linear_layers)
        for i, linear in enumerate(self.linear_layers):
            out = linear(out)
            # Aplicar activación a todas las capas excepto la última
            if i < n_layers - 1:
                out = self.activation(out)
                
        return out

    def calculate_metrics(self, loss_val, test_fn):
        with torch.no_grad():
            error_vec, _ = test_fn()  # test_fn debe devolver (error, output)
            accuracy = 1 - error_vec.item()
            metrics = {
                'loss': loss_val.item(),
                'accuracy': accuracy * 100,
                'time_elapsed': time.time() - self.start_time
            }
            self.losses.append(metrics['loss'])
            self.accuracies.append(metrics['accuracy'])
            if len(self.losses) > 1 and abs(self.losses[-1] - self.losses[-2]) < 1e-6:
                self.convergence_time = metrics['time_elapsed']
            return metrics