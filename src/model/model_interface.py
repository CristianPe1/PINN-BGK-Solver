import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class ModelAdapter:
    """
    Clase para adaptar modelos con diferentes estructuras de estado.
    Facilita la carga de modelos que pueden haber cambiado de estructura entre versiones.
    """
    
    @staticmethod
    def adapt_state_dict(model, state_dict):
        """
        Adapta un state_dict para que sea compatible con un modelo específico.
        
        Args:
            model (nn.Module): Modelo de destino
            state_dict (dict): Estado del modelo a adaptar
            
        Returns:
            dict: Estado adaptado para el modelo
        """
        # Crear dict de adaptaciones para diferentes versiones
        adaptations = {
            # Adaptación de versión con 'linears' a versión con 'linear_layers'
            'linears_to_linear_layers': lambda sd: {
                key.replace('linears.', 'linear_layers.'): value
                for key, value in sd.items()
            },
            # Adaptación inversa
            'linear_layers_to_linears': lambda sd: {
                key.replace('linear_layers.', 'linears.'): value
                for key, value in sd.items()
            }
        }
        
        # Determinar qué adaptación necesitamos
        model_keys = set(model.state_dict().keys())
        sd_keys = set(state_dict.keys())
        
        if any('linears.' in k for k in sd_keys) and all('linear_layers.' in k for k in model_keys):
            logger.info("Adaptando estado de 'linears' a 'linear_layers'...")
            return adaptations['linears_to_linear_layers'](state_dict)
            
        elif any('linear_layers.' in k for k in sd_keys) and all('linears.' in k for k in model_keys):
            logger.info("Adaptando estado de 'linear_layers' a 'linears'...")
            return adaptations['linear_layers_to_linears'](state_dict)
        
        # Si no se necesita adaptación, devolver el state_dict original
        return state_dict
    
    @staticmethod
    def load_model_with_adaptation(model, path, device=None):
        """
        Carga un modelo con adaptación automática del estado.
        
        Args:
            model (nn.Module): Modelo a cargar
            path (str): Ruta al archivo de estado
            device (torch.device, optional): Dispositivo donde cargar el modelo
            
        Returns:
            nn.Module: Modelo con estado cargado
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # Cargar checkpoint
        checkpoint = torch.load(path, map_location=device)
        
        # Obtener estado del modelo
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Adaptar estado y cargar en el modelo
        adapted_state = ModelAdapter.adapt_state_dict(model, state_dict)
        
        try:
            model.load_state_dict(adapted_state, strict=True)
            logger.info("Modelo cargado exitosamente con adaptación de estado.")
        except RuntimeError as e:
            logger.warning(f"Error al cargar estado con strict=True: {str(e)}")
            logger.info("Intentando cargar con strict=False...")
            
            missing, unexpected = model.load_state_dict(adapted_state, strict=False)
            
            if missing:
                logger.warning(f"Claves faltantes: {missing}")
            if unexpected:
                logger.warning(f"Claves inesperadas: {unexpected}")
            
        return model
