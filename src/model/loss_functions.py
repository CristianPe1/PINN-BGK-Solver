import torch
import torch.nn as nn
import numpy as np

def mean_squared_error(pred, true, **kwargs):
    """
    Error cuadrático medio simple.
    
    Args:
        pred: Predicciones del modelo
        true: Valores reales
        **kwargs: Parámetros adicionales
        
    Returns:
        torch.Tensor: Valor de pérdida
    """
    weight = kwargs.get('weight', 1.0)
    return weight * torch.mean((pred - true) ** 2)

def weighted_mse(pred, true, **kwargs):
    """
    Error cuadrático medio ponderado que da diferentes pesos a puntos interiores y frontera.
    
    Args:
        pred: Predicciones del modelo
        true: Valores reales
        **kwargs: Parámetros adicionales, como interior_weight y boundary_weight
        
    Returns:
        torch.Tensor: Valor de pérdida ponderada
    """
    interior_weight = kwargs.get('interior_weight', 1.0)
    boundary_weight = kwargs.get('boundary_weight', 10.0)
    
    # Máscara para identificar puntos de frontera (asumiendo que x está en [-1, 1])
    x = kwargs.get('x', None)
    if x is None:
        return mean_squared_error(pred, true, weight=interior_weight)
    
    # Identificar puntos cercanos a la frontera
    is_boundary = (torch.abs(x) > 0.95).float()
    
    # Calcular pesos
    weights = interior_weight * (1 - is_boundary) + boundary_weight * is_boundary
    
    # Pérdida ponderada
    squared_error = (pred - true) ** 2
    loss = torch.mean(weights * squared_error)
    
    return loss

def physics_informed_loss(pred, true, **kwargs):
    """
    Función de pérdida que combina el error de datos y el residual de la ecuación diferencial.
    
    Args:
        pred: Predicciones del modelo
        true: Valores reales
        **kwargs: Parámetros adicionales, incluyendo modelo, x, t, nu, etc.
        
    Returns:
        torch.Tensor: Valor combinado de pérdida
    """
    data_weight = kwargs.get('data_weight', 1.0)
    physics_weight = kwargs.get('physics_weight', 0.5)
    
    # Pérdida de datos (MSE)
    data_loss = torch.mean((pred - true) ** 2)
    
    # Pérdida física (residual de la ecuación de Burgers)
    model = kwargs.get('model', None)
    x = kwargs.get('x', None)
    t = kwargs.get('t', None)
    nu = kwargs.get('nu', 0.01/np.pi)
    
    if model is None or x is None or t is None:
        return data_loss * data_weight
    
    # Calcular derivadas para el residual físico usando diferenciación automática
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    u = model(x, t)
    
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        retain_graph=True, create_graph=True
    )[0]
    
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x),
        retain_graph=True, create_graph=True
    )[0]
    
    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u),
        retain_graph=True, create_graph=True
    )[0]
    
    # Residual de la ecuación de Burgers: u_t + u*u_x - nu*u_xx = 0
    residual = u_t + u * u_x - nu * u_xx
    physics_loss = torch.mean(residual ** 2)
    
    # Combinar pérdidas
    total_loss = data_weight * data_loss + physics_weight * physics_loss
    
    return total_loss

def get_loss_function(loss_config):
    """
    Obtiene la función de pérdida según la configuración.
    
    Args:
        loss_config (dict): Configuración de la función de pérdida
        
    Returns:
        function: Función de pérdida configurada
    """
    loss_type = loss_config.get("type", "mean_squared_error")
    
    if loss_type == "mean_squared_error":
        return mean_squared_error
    elif loss_type == "weighted_mse":
        return weighted_mse
    elif loss_type == "physics_informed_loss":
        return physics_informed_loss
    else:
        raise ValueError(f"Tipo de función de pérdida desconocido: {loss_type}")
