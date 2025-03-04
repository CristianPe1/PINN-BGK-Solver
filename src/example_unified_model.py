import os
import torch
from model import load_model, load_fluid_model, load_saved_model, ModelSelector

def test_unified_model_selector():
    """Ejemplo de uso del selector de modelos unificado."""
    # Ruta al archivo de configuración
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               "config", "config.yaml")
    
    print(f"Usando archivo de configuración: {config_path}")
    print(f"¿Existe el archivo? {os.path.exists(config_path)}")
    
    # 0. Verificar los modelos disponibles
    selector = ModelSelector()
    available_models = selector.list_available_models()
    print("\nModelos disponibles por categoría:")
    for category, models in available_models.items():
        print(f"- {category}: {list(models.keys())}")
    
    # 1. Cargar un modelo estándar
    try:
        standard_model = load_model(config_path=config_path, category="standard", model_type="pinn_small")
        print(f"Modelo estándar cargado: {standard_model.name}")
        
        # Para el modelo estándar (entrada 2D, salida 1D)
        x_standard = torch.rand(10, 2)  # [batch_size, input_dim]
        output_standard = standard_model(x_standard)
        print(f"Forma de salida del modelo estándar: {output_standard.shape}")
    except Exception as e:
        print(f"Error al cargar modelo estándar: {e}")
    
    # 2. Cargar un modelo de fluidos
    try:
        fluid_model = load_fluid_model(config_path=config_path, model_name="taylor_green")
        print(f"Modelo de fluidos cargado: {fluid_model.name}")
        
        # Para el modelo de fluidos (entrada 3D, salida 3D)
        x_fluid = torch.rand(10, 3)  # [batch_size, input_dim]
        output_fluid = fluid_model(x_fluid)
        print(f"Forma de salida del modelo de fluidos: {output_fluid.shape}")
    except Exception as e:
        print(f"Error al cargar modelo de fluidos: {e}")

if __name__ == "__main__":
    test_unified_model_selector()
