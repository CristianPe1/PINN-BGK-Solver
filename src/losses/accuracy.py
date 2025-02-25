import torch

def calculate_accuracy(model, features, targets):
    """
    Calcula la precisión del modelo como 1 - error medio relativo.
    
    Parámetros:
        model (torch.nn.Module): Modelo PyTorch.
        features (tuple[torch.Tensor] | torch.Tensor): Datos de entrada; se acepta una tupla si hay múltiples tensores.
        targets (torch.Tensor): Objetivos reales.
    
    Retorna:
        float: Precisión en porcentaje.
    """
    model.eval()
    with torch.no_grad():
        # Si features es una tupla, desempaquetarlo, sino usarlo directamente
        if isinstance(features, (tuple, list)):
            predictions = model(*features)
        else:
            predictions = model(features)
        # Calcular error relativo
        error = torch.abs(predictions - targets) / (torch.abs(targets) + 1e-8)
        accuracy = 1 - torch.mean(error)
    return accuracy.item() * 100  # Retorna precisión en porcentaje
