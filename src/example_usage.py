import os
import torch
from structure_model.fluid_models import load_fluid_model

def test_fluid_model_selection():
    """Ejemplo de cómo utilizar el selector de modelos de fluidos."""
    # Ruta al archivo de configuración
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             "config", "config.yaml")
    
    # Cargar un modelo usando la configuración por defecto
    model = load_fluid_model(config_path=config_path)
    print(f"Modelo cargado: {model.name}")
    
    # Probar el modelo con datos simulados
    batch_size = 10
    x = torch.rand(batch_size, 3)  # Entrada (x, y, t)
    output = model(x)
    print(f"Forma de salida: {output.shape}")
    
    # También podemos cargar un modelo específico
    taylor_green_model = load_fluid_model(config_path=config_path, model_name="taylor_green")
    print(f"Modelo específico cargado: {taylor_green_model.name}")

if __name__ == "__main__":
    test_fluid_model_selection()
