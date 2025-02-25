import torch
import numpy as np
from diferential_equation_loss import burgers_pde_loss

def boundary_loss(model, x_boundary, u_boundary_true):
    """Calcula la pérdida en las condiciones de frontera utilizando los valores reales proporcionados"""
    t_boundary = torch.zeros_like(x_boundary)  # Si fuera necesario, se puede ajustar para incluir datos temporales específicos
    u_boundary_pred = model(x_boundary, t_boundary)
    return torch.mean((u_boundary_pred - u_boundary_true) ** 2)

class DynamicWeightedLoss:
    def __init__(self):
        self.w_data = torch.tensor(1.0, requires_grad=False)
        self.w_pde = torch.tensor(1.0, requires_grad=False)
        self.w_bc = torch.tensor(1.0, requires_grad=False)
        
        # Historial de pérdidas para análisis usando arreglos de numpy
        self.data_losses = np.array([], dtype=float)
        self.pde_losses = np.array([], dtype=float)
        self.bc_losses = np.array([], dtype=float)
        self.weights_history = np.empty((0, 3), dtype=float)

    def compute_loss(self, model, x_train, t_train, u_train, x_boundary, nu):
        """Calcula la pérdida total con pesos dinámicos"""
        # Pérdidas individuales
        loss_data = torch.mean((model(x_train, t_train) - u_train) ** 2)
        loss_pde = burgers_pde_loss(model, x_train, t_train, nu)
        loss_bc = boundary_loss(model, x_boundary, u_train)

        # Normalizar pérdidas y actualizar pesos
        with torch.no_grad():
            losses = torch.tensor([loss_data.item(), loss_pde.item(), loss_bc.item()])
            weights = 1.0 / (losses + 1e-8)
            weights = weights / torch.sum(weights)
            
            self.w_data, self.w_pde, self.w_bc = weights
            
            # Guardar historial usando numpy arrays
            self.data_losses = np.append(self.data_losses, loss_data.item())
            self.pde_losses = np.append(self.pde_losses, loss_pde.item())
            self.bc_losses = np.append(self.bc_losses, loss_bc.item())
            self.weights_history = np.vstack((self.weights_history, weights.cpu().numpy()))

        # Pérdida total ponderada
        # loss_total = self.w_data * loss_data + self.w_pde * loss_pde + self.w_bc * loss_bc
        loss_total = loss_data + loss_pde + loss_bc
        return loss_total, loss_data.item(), loss_pde.item(), loss_bc.item()

    def get_current_weights(self):
        """Retorna los pesos actuales de las pérdidas"""
        return {
            'data': self.w_data.item(),
            'pde': self.w_pde.item(),
            'bc': self.w_bc.item()
        }

    def get_loss_history(self):
        """Retorna el historial de pérdidas y pesos como arreglos de numpy"""
        return {
            'data_losses': self.data_losses,
            'pde_losses': self.pde_losses,
            'bc_losses': self.bc_losses,
            'weights_history': self.weights_history
        }
