import torch

def dirichlet_boundary_condition(model, x, t):
    """Implementa condición de frontera tipo Dirichlet"""
    # Frontera en x=0 y x=1
    x_boundary = torch.tensor([0.0, 1.0]).reshape(-1, 1)
    t_boundary = torch.linspace(0, 1, 100).reshape(-1, 1)
    X_b, T_b = torch.meshgrid(x_boundary.flatten(), t_boundary.flatten())
    boundary_points = torch.stack([X_b.flatten(), T_b.flatten()], dim=1)
    
    # Valor esperado en la frontera
    boundary_values = model(boundary_points)
    expected_values = torch.zeros_like(boundary_values)  # Para este caso específico
    
    return torch.mean((boundary_values - expected_values)**2)

def neumann_boundary_condition(model, x, t):
    """Implementa condición de frontera tipo Neumann"""
    x_boundary = torch.tensor([0.0, 1.0]).reshape(-1, 1)
    t_boundary = torch.linspace(0, 1, 100).reshape(-1, 1)
    X_b, T_b = torch.meshgrid(x_boundary.flatten(), t_boundary.flatten())
    boundary_points = torch.stack([X_b.flatten(), T_b.flatten()], dim=1)
    
    boundary_points.requires_grad_(True)
    u = model(boundary_points)
    du_dx = torch.autograd.grad(u.sum(), boundary_points, create_graph=True)[0][:, 0]
    
    return torch.mean(du_dx**2)

def periodic_boundary_condition(model, x, t):
    """Impone condiciones de frontera periódicas en x = 0 y x = 1."""
    x_left = torch.tensor([0.0], dtype=torch.float32).repeat(len(t))
    x_right = torch.tensor([1.0], dtype=torch.float32).repeat(len(t))
    t_boundary = t.repeat(2)

    u_left = model(torch.stack((x_left, t_boundary), dim=1))
    u_right = model(torch.stack((x_right, t_boundary), dim=1))

    u_x_left = torch.autograd.grad(u_left, x_left, grad_outputs=torch.ones_like(u_left), create_graph=True)[0]
    u_x_right = torch.autograd.grad(u_right, x_right, grad_outputs=torch.ones_like(u_right), create_graph=True)[0]

    loss_boundary = torch.mean((u_left - u_right)**2) + torch.mean((u_x_left - u_x_right)**2)

    return loss_boundary

