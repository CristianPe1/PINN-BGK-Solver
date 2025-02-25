import importlib
import torch
import torch.nn as nn
import yaml
from pathlib import Path
from .network_architecture import NetworkArchitecture

def load_config():
    """Carga la configuración desde el archivo YAML"""
    config_path = Path(__file__).parent.parent.parent / 'config' / 'model_config.yaml'
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

class ModelConstructor:
    def __init__(self):
        self.config = load_config()
        self.model_params = self.config['model']
        self.physics_params = self.config['physics']
        self.training_params = self.config['training']

    def get_structure_model(self):
        """
        Crea y configura el modelo completo con su arquitectura, función de pérdida y optimizador
        """
        # Crear la arquitectura de red
        model = NetworkArchitecture.create_network(
            network_type=self.model_params['type'].lower(),
            input_dim=self.model_params['input_dim'],
            hidden_dim=self.model_params['hidden_dim'],
            output_dim=self.model_params['output_dim']
        )
        
        # Configurar el optimizador
        optimizer = self.get_optimizer(model)
        
        # Agregar la función de pérdida y el optimizador como atributos del modelo
        model.loss_function = lambda x, t: self.get_total_loss_function(model, x, t)
        model.optimizer = optimizer
        
        return model

    def get_optimizer(self, model):
        """
        Configura el optimizador según el learning rate del YAML
        """
        return torch.optim.Adam(
            model.parameters(), 
            lr=self.model_params['learning_rate']
        )

    def get_boundary_loss(self, model, x, t, boundary_type):
        """
        Selecciona la función de pérdida de frontera según la configuración
        """
        try:
            module = importlib.import_module("src.losses.boundary_loss")
            loss_fn = getattr(module, f"{boundary_type}_boundary_condition")
            return loss_fn(model, x, t)
        except AttributeError:
            raise ValueError(f"Tipo de frontera '{boundary_type}' no implementado")

    def get_physics_loss(self, model, x, t, physics_type, nu):
        """
        Selecciona la función de pérdida física según la configuración
        """
        try:
            module = importlib.import_module("src.losses.differential_equation_loss")
            loss_fn = getattr(module, f"{physics_type}_residual")
            return loss_fn(model, x, t, nu)
        except AttributeError:
            raise ValueError(f"Tipo de ecuación diferencial '{physics_type}' no implementado")

    @staticmethod
    def get_mse_loss(model, x, t):
        """Calcula el MSE loss"""
        predictions = model(torch.stack((x, t), dim=1))
        return nn.MSELoss()(predictions, torch.zeros_like(predictions))


    def get_total_loss_function(self, model, x, t):
        """
        Configura la función de pérdida según los parámetros del YAML
        """
        nu = self.physics_params['nu']
        boundary_type = self.physics_params['boundary_type']
        physics_type = self.physics_params['physics_type']

        mse_loss = self.get_mse_loss(model, x, t)
        boundary_loss = self.get_boundary_loss(model, x, t, boundary_type)
        physics_loss = self.get_physics_loss(model, x, t, physics_type, nu)

        return mse_loss + boundary_loss + physics_loss
    
    
def create_model():
    constructor = ModelConstructor()
    model = constructor.get_structure_model()
    return model, constructor

