import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Configuración de la red neuronal
class PINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=5, neurons_per_layer=40):
        super(PINN, self).__init__()
        layers = [nn.Linear(input_dim, neurons_per_layer), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(neurons_per_layer, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Función de pérdida personalizada
def loss_function(pred_f_eq, pred_f_neq, true_f_eq, true_f_neq, res_eq, res_bc, res_ic):
    L_eq = torch.mean((res_eq) ** 2)
    L_fIC = torch.mean((pred_f_eq - true_f_eq) ** 2)
    L_mIC = torch.mean((pred_f_neq - true_f_neq) ** 2)
    L_bc = torch.mean((res_bc) ** 2)
    L_ic = torch.mean((res_ic) ** 2)
    return L_eq + L_fIC + L_mIC + L_bc + L_ic

# Generar datos sintéticos para pruebas
def generate_synthetic_data(num_samples=1000):
    x = np.random.uniform(0, 1, (num_samples, 2))
    f_eq = np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
    f_neq = np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
    res_eq = f_eq * f_neq
    res_bc = f_eq - f_neq
    res_ic = f_eq + f_neq
    return torch.Tensor(x), torch.Tensor(f_eq), torch.Tensor(f_neq), torch.Tensor(res_eq), torch.Tensor(res_bc), torch.Tensor(res_ic)

# Crear el modelo
input_dim = 2
output_dim = 1
model_eq = PINN(input_dim, output_dim)
model_neq = PINN(input_dim, output_dim)

# Generar datos de entrenamiento
x, f_eq, f_neq, res_eq, res_bc, res_ic = generate_synthetic_data()

# Configuración del optimizador
optimizer = optim.Adam(list(model_eq.parameters()) + list(model_neq.parameters()), lr=1e-3)

# Entrenamiento
num_epochs = 500
for epoch in range(num_epochs):
    optimizer.zero_grad()
    pred_f_eq = model_eq(x)
    pred_f_neq = model_neq(x)
    loss = loss_function(pred_f_eq, pred_f_neq, f_eq, f_neq, res_eq, res_bc, res_ic)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.6f}')

print("Entrenamiento completado.")
