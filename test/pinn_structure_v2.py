import torch
import torch.nn as nn

import time
import torch
import torch.nn as nn

class PINN_V2(nn.Module):
    def __init__(self, layers, activation=nn.Tanh()):
        super(PINN_V2, self).__init__()
        self.layers = layers
        self.batch_norms = nn.ModuleList()
        activation_function = getattr(nn, activation, None)
        if activation_function is not None:
            self.activation = activation_function()
        else:
            raise ValueError("Función de activación no soportada.")
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data)
            nn.init.zeros_(self.linears[i].bias.data)
        self.losses = []
        self.accuracies = []
        self.convergence_time = 0
        self.start_time = time.time()
        self.iter = 0

    def forward(self, x, t) :
        # Normalización Min-Max para los inputs x y t
        x_min, x_max = x.min(), x.max()
        t_min, t_max = t.min(), t.max()
        
        # Normalización al rango [0, 1]
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        t_norm = (t - t_min) / (t_max - t_min + 1e-8)

        # Concatenar los inputs normalizados
        a = torch.cat(tensors=(x_norm, t_norm), dim=len(x.shape) - 1)

        # Pasar a través de las capas lineales con Batch Normalization y activación
        for i, layer in enumerate(self.linears[:-1]):
            a = layer(a)  # Capa lineal
            
            if i < len(self.batch_norms):  # Evitar batch norm en la capa de salida
                a = self.batch_norms[i](a)  # Aplicar Batch Normalization
            
            a = self.activation(a)  # Función de activación
        
        # Capa de salida sin BatchNorm ni activación
        out = self.linears[-1](a)
        
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

    def closure(self, optimizer, loss_fn, test_fn, *loss_args):
        optimizer.zero_grad()
        loss_val = loss_fn(*loss_args)
        loss_val.backward()
        if self.iter % 100 == 0:
            metrics = self.calculate_metrics(loss_val, test_fn)
            # Se usa logging en el script principal para guardar métricas
            print(
                f"Iteración {self.iter}:\n"
                f"  Pérdida: {metrics['loss']:.6f}\n"
                f"  Precisión: {metrics['accuracy']:.2f}%\n"
                f"  Tiempo transcurrido: {metrics['time_elapsed']:.2f}s"
            )
            if self.convergence_time > 0:
                print(f"  ¡Convergencia alcanzada! Tiempo: {self.convergence_time:.2f}s")
        self.iter += 1
        return loss_val



