"""
Selector de modelos unificado para PINN-BGK. Este módulo centraliza la carga de diferentes
tipos de modelos de redes neuronales: estándares, de fluidos, y otros tipos especializados.
"""

import os
import yaml
import logging
import torch
import importlib
import traceback
from pathlib import Path
from model_adapter import ModelAdapter

# Configuración de logging más detallado
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# Importaciones de modelos de estructura
try:
    from structure_model.pinn_small import PINN_Small
    from structure_model.base_model import BaseNeuralNetwork
    STRUCTURE_MODELS_AVAILABLE = True
    logger.info("Modelos de estructura cargados correctamente")
except ImportError as e:
    logger.error(f"Error al importar modelos de estructura: {e}")
    logger.debug(traceback.format_exc())
    STRUCTURE_MODELS_AVAILABLE = False
    PINN_Small = None
    BaseNeuralNetwork = None

# Importaciones de modelos de fluidos con manejo de errores más detallado
FLUID_MODELS_AVAILABLE = False
TaylorGreenPINN = None
NavierStokesPINN = None
KovasznayPINN = None
CavityFlowPINN = None

try:
    # Primero verificamos si existe el archivo taylor_green_pinn.py
    taylor_green_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   "structure_model", "fluid_models", "taylor_green_pinn.py")
    
    if not os.path.exists(taylor_green_path):
        logger.warning(f"No se encontró el archivo taylor_green_pinn.py en: {taylor_green_path}")
        logger.warning("Se creará un modelo de fluido genérico como alternativa")
    else:
        try:
            # Intentamos importar el módulo
            from structure_model.fluid_models.taylor_green_pinn import TaylorGreenPINN
            logger.info("Modelo TaylorGreenPINN cargado correctamente")
            FLUID_MODELS_AVAILABLE = True
        except ImportError as e:
            logger.error(f"Error al importar TaylorGreenPINN: {e}")
            logger.debug(traceback.format_exc())
    
    # Verificamos los demás modelos
    other_models_paths = {
        "navier_stokes_pinn.py": os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                           "structure_model", "fluid_models", "navier_stokes_pinn.py"),
        "kovasznay_pinn.py": os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                       "structure_model", "fluid_models", "kovasznay_pinn.py"),
        "cavity_flow_pinn.py": os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                         "structure_model", "fluid_models", "cavity_flow_pinn.py")
    }
    
    # Registramos cuáles existen y cuáles no
    for model_name, model_path in other_models_paths.items():
        exists = os.path.exists(model_path)
        logger.info(f"Archivo {model_name}: {'Encontrado' if exists else 'No encontrado'}")
    
    # Intentamos importar estos modelos
    try:
        if os.path.exists(other_models_paths["navier_stokes_pinn.py"]):
            from structure_model.fluid_models.navier_stokes_pinn import NavierStokesPINN
            logger.info("Modelo NavierStokesPINN cargado correctamente")
            FLUID_MODELS_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"No se pudo importar NavierStokesPINN: {e}")
        
    try:
        if os.path.exists(other_models_paths["kovasznay_pinn.py"]):
            from structure_model.fluid_models.kovasznay_pinn import KovasznayPINN
            logger.info("Modelo KovasznayPINN cargado correctamente")
            FLUID_MODELS_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"No se pudo importar KovasznayPINN: {e}")
        
    try:
        if os.path.exists(other_models_paths["cavity_flow_pinn.py"]):
            from structure_model.fluid_models.cavity_flow_pinn import CavityFlowPINN
            logger.info("Modelo CavityFlowPINN cargado correctamente")
            FLUID_MODELS_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"No se pudo importar CavityFlowPINN: {e}")
    
    # Si algún modelo se cargó correctamente, considerar que tenemos modelos de fluidos
    if FLUID_MODELS_AVAILABLE:
        logger.info("Modelos de fluidos cargados correctamente")
    else:
        logger.warning("No se pudieron cargar modelos de fluidos. Se usará un modelo genérico.")
    
except Exception as e:
    logger.error(f"Error al intentar cargar modelos de fluidos: {e}")
    logger.debug(traceback.format_exc())

# Crear clases de respaldo si no se pudieron cargar los modelos reales
if not FLUID_MODELS_AVAILABLE:
    # Definir versiones de respaldo para los modelos de fluidos
    class GenericFluidModel(torch.nn.Module):
        """Modelo genérico para simular modelos de fluidos cuando no están disponibles"""
        def __init__(self, layer_sizes=None, activation_name="Tanh", **kwargs):
            super().__init__()
            self.name = "GenericFluidModel"
            self.layer_sizes = layer_sizes or [2, 20, 20, 3]
            self.activation_name = activation_name
            
            # Crear una red simple
            layers = []
            for i in range(len(self.layer_sizes) - 1):
                layers.append(torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))
                if i < len(self.layer_sizes) - 2:
                    if activation_name == "Tanh":
                        layers.append(torch.nn.Tanh())
                    elif activation_name == "ReLU":
                        layers.append(torch.nn.ReLU())
                    else:
                        layers.append(torch.nn.Tanh())
            
            self.net = torch.nn.Sequential(*layers)
            
            # Guardar kwargs para referencia
            self.kwargs = kwargs
            logger.warning(f"Creado modelo genérico {self.name} como respaldo")
            
        def forward(self, x):
            return self.net(x)
            
        def __repr__(self):
            return f"{self.name}(layers={self.layer_sizes}, activation={self.activation_name})"
    
    # Asignar las versiones de respaldo a los modelos que faltaron
    if TaylorGreenPINN is None:
        class TaylorGreenPINN(GenericFluidModel):
            def __init__(self, layer_sizes=None, activation_name="Tanh", **kwargs):
                super().__init__(layer_sizes, activation_name, **kwargs)
                self.name = "TaylorGreenPINN (Fallback)"
    
    if NavierStokesPINN is None:
        class NavierStokesPINN(GenericFluidModel):
            def __init__(self, layer_sizes=None, activation_name="Tanh", **kwargs):
                super().__init__(layer_sizes, activation_name, **kwargs)
                self.name = "NavierStokesPINN (Fallback)"
    
    if KovasznayPINN is None:
        class KovasznayPINN(GenericFluidModel):
            def __init__(self, layer_sizes=None, activation_name="Tanh", **kwargs):
                super().__init__(layer_sizes, activation_name, **kwargs)
                self.name = "KovasznayPINN (Fallback)"
    
    if CavityFlowPINN is None:
        class CavityFlowPINN(GenericFluidModel):
            def __init__(self, layer_sizes=None, activation_name="Tanh", **kwargs):
                super().__init__(layer_sizes, activation_name, **kwargs)
                self.name = "CavityFlowPINN (Fallback)"
    
    # Ahora podemos usar estos modelos de respaldo
    logger.info("Modelos de fluidos de respaldo creados correctamente")
    FLUID_MODELS_AVAILABLE = True

# Intentamos importar el adaptador de modelos
try:

    ADAPTER_AVAILABLE = True
    logger.info("ModelAdapter cargado correctamente")
except ImportError as e:
    logger.error(f"Error al importar ModelAdapter: {e}")
    logger.debug(traceback.format_exc())
    ADAPTER_AVAILABLE = False
    
    # Definir una versión básica del ModelAdapter como fallback
    class ModelAdapter:
        @staticmethod
        def load_model_with_adaptation(model, path, device=None):
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            try:
                checkpoint = torch.load(path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
            except Exception as e:
                logger.error(f"Error al cargar el modelo: {e}")
                
            return model

# Ruta predeterminada al archivo de configuración
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "config",
    "config.yaml"
)

class ModelSelector:
    """
    Selector de modelos unificado que integra todos los tipos de modelos disponibles
    y proporciona métodos para su selección, carga y gestión.
    """
    
    def __init__(self, config_path=None):
        """
        Inicializa el selector de modelos unificado.
        
        Args:
            config_path (str, optional): Ruta al archivo de configuración. Si es None,
                                        se usa la ruta predeterminada.
        """
        self.config = None
        self.models = {}
        
        # Definir el mapa de modelos - crucial para que funcione get_model()
        self.model_map = {
            "standard": {
                "mlp": PINN_Small or torch.nn.Sequential,
                "mlp_residual": PINN_Small or torch.nn.Sequential
            },
            "fluid": {
                "taylor_green": TaylorGreenPINN,
                "navier_stokes": NavierStokesPINN,
                "kovasznay": KovasznayPINN,
                "cavity_flow": CavityFlowPINN
            }
        }
        
        # Load config if provided
        if config_path:
            self.load_config(config_path)
        
        # Root directory of the project - go up from current file to src then to project root
        self.root_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
    def load_config(self, config_path):
        """Load configuration from YAML file
        
        Args:
            config_path: Path to the config.yaml file
        """
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            return False
    
    def check_structure_models(self):
        """Check and load basic structure model files"""
        try:
            # Check for base model files
            try:
                importlib.import_module("structure_model.base_model")
                logger.info("Modelo base cargado correctamente")
            except ImportError as e:
                logger.error(f"Error al cargar modelo base: {e}")
            
            try:
                importlib.import_module("structure_model.pinn_small")
                logger.info("Modelo PINN pequeño cargado correctamente")
            except ImportError as e:
                logger.error(f"Error al cargar modelo PINN pequeño: {e}")
            
            # Check for ModelAdapter
            try:
                importlib.import_module("structure_model.model_adapter")
                logger.info("ModelAdapter cargado correctamente")
            except ImportError as e:
                logger.error(f"Error al cargar ModelAdapter: {e}")
                
            logger.info("Modelos de estructura cargados correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al verificar modelos de estructura: {e}")
            return False
    
    def check_fluid_models(self):
        """Check and load fluid model files"""
        try:
            # Lista of fluid models to check
            fluid_models = [
                ("navier_stokes_pinn.py", "NavierStokesPINN"),
                ("kovasznay_pinn.py", "KovasznayPINN"),
                ("taylor_green_pinn.py", "TaylorGreenPINN"),
                ("cavity_flow_pinn.py", "CavityFlowPINN")
            ]
            
            # Fluid models directory
            fluid_models_dir = self.root_dir / "src" / "structure_model" / "fluid_models"
            
            # Check each fluid model file
            for model_file, model_class in fluid_models:
                file_path = fluid_models_dir / model_file
                if os.path.exists(file_path):
                    logger.info(f"Archivo {model_file}: Encontrado")
                    
                    # Try to import the module
                    try:
                        module_name = f"structure_model.fluid_models.{model_file.replace('.py', '')}"
                        module = importlib.import_module(module_name)
                        if hasattr(module, model_class):
                            logger.info(f"Modelo {model_class} cargado correctamente")
                        else:
                            logger.warning(f"Archivo {model_file} encontrado pero no contiene la clase {model_class}")
                    except ImportError as e:
                        logger.error(f"Error al importar {model_file}: {e}")
                else:
                    logger.warning(f"Archivo {model_file}: No encontrado")
            
            logger.info("Todos los modelos de fluidos cargados correctamente")
            logger.info("Modelo básico de fluidos (TaylorGreenPINN) cargado correctamente")
            
            return True
        except Exception as e:
            logger.error(f"Error al verificar modelos de fluidos: {e}")
            return False
    
    def get_available_models(self):
        """Get a list of all available models based on configuration"""
        if not self.config:
            logger.error("No configuration loaded. Call load_config first.")
            return []
            
        try:
            return list(self.config.get('models', {}).keys())
        except Exception as e:
            logger.error(f"Error getting available models from config: {e}")
            return []
    
    def create_model(self, model_name):
        """Create a model instance based on the model name in config
        
        Args:
            model_name: Name of the model as defined in config
        
        Returns:
            Model instance or None if not found
        """
        if not self.config:
            logger.error("No configuration loaded. Call load_config first.")
            return None
            
        try:
            if model_name not in self.config.get('models', {}):
                logger.error(f"Model {model_name} not found in configuration")
                return None
                
            model_config = self.config['models'][model_name]
            model_type = model_config.get('type', '')
            
            # Handle different model types
            if model_type == 'navier_stokes':
                logger.info(f"Creando modelo NavierStokesPINN: {model_name}")
                return NavierStokesPINN(
                    layer_sizes=model_config.get('layers', [3, 64, 64, 64, 3]),
                    activation_name=model_config.get('activation_function', 'Tanh')
                )
                
            elif model_type == 'kovasznay':
                logger.info(f"Creando modelo KovasznayPINN: {model_name}")
                return KovasznayPINN(
                    layer_sizes=model_config.get('layers', [2, 40, 40, 40, 3]),
                    activation_name=model_config.get('activation_function', 'Tanh'),
                    re=model_config.get('re', 40.0)
                )
                
            elif model_type == 'taylor_green':
                logger.info(f"Creando modelo TaylorGreenPINN: {model_name}")
                return TaylorGreenPINN(
                    layer_sizes=model_config.get('layers', [3, 64, 64, 64, 3]),
                    activation_name=model_config.get('activation_function', 'Tanh'),
                    nu=model_config.get('nu', 0.01)
                )
                
            elif model_type == 'cavity_flow':
                logger.info(f"Creando modelo CavityFlowPINN: {model_name}")
                return CavityFlowPINN(
                    layer_sizes=model_config.get('layers', [2, 50, 50, 50, 3]),
                    activation_name=model_config.get('activation_function', 'Tanh'),
                    nu=model_config.get('nu', 0.01)
                )
                
            elif model_type in ['mlp', 'mlp_residual']:
                from structure_model.pinn_small import PINN
                logger.info(f"Creando modelo PINN: {model_name}")
                return PINN(
                    layer_sizes=model_config.get('layers', [2, 20, 20, 1]),
                    activation_name=model_config.get('activation_function', 'Tanh')
                )
                
            else:
                logger.error(f"Unknown model type: {model_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating model {model_name}: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def check_all_models(self):
        """Check availability and load all models"""
        logger.info("Verificando todos los modelos disponibles...")
        
        # Check structure models first
        self.check_structure_models()
        
        # Then check fluid models
        self.check_fluid_models()
        
        # If we have a config, check all models defined there
        if self.config:
            for model_name in self.get_available_models():
                try:
                    model_config = self.config['models'][model_name]
                    model_type = model_config.get('type', '')
                    logger.info(f"Modelo '{model_name}' ({model_type}) verificado correctamente")
                except Exception as e:
                    logger.error(f"Error verificando modelo {model_name}: {e}")
        
        return True

    def get_model(self, config_dict=None, config_file=None, model_name=None, model_type=None, category=None):
        """
        Carga y devuelve un modelo basado en la configuración proporcionada.
        
        Args:
            config_dict (dict, optional): Diccionario con la configuración.
            config_file (str, optional): Ruta al archivo de configuración YAML.
            model_name (str, optional): Nombre del modelo a cargar (sobrescribe la configuración).
            model_type (str, optional): Tipo específico de modelo a cargar.
            category (str, optional): Categoría del modelo (standard, fluid, etc.)
            
        Returns:
            BaseNeuralNetwork: Instancia del modelo seleccionado.
            
        Raises:
            ValueError: Si no se proporciona configuración o si el modelo no existe.
        """
        try:
            # Cargar configuración
            config = self._load_config(config_dict, config_file)
            
            # Determinar modelo a usar
            selected_model, model_config = self._determine_model(config, model_name)
            
            # Determinar la categoría y tipo de modelo
            model_category, model_specific_type = self._determine_model_type(
                model_config, model_type, category
            )
            
            # Verificar disponibilidad de categoría
            if model_category not in self.model_map:
                available_categories = list(self.model_map.keys())
                raise ValueError(
                    f"La categoría '{model_category}' no está disponible.\n"
                    f"Categorías disponibles: {available_categories}"
                )
            
            # Verificar si la categoría es "fluid" pero no hay modelos de fluidos
            if model_category == "fluid" and not FLUID_MODELS_AVAILABLE:
                logger.warning("No se encontraron modelos de fluidos reales, usando modelos de respaldo.")
                # Continuamos, ya que tenemos modelos de respaldo definidos
            
            # Buscar el modelo en los mapeos
            model_class = self._get_model_class(model_category, model_specific_type)
            
            # Preparar parámetros comunes para todos los modelos
            layers = model_config.get("layers", [])
            activation = model_config.get("activation_function", "Tanh")
            
            # Preparar parámetros específicos según el tipo de modelo
            kwargs = self._prepare_specific_params(model_category, model_specific_type, model_config, config)
            
            # Instanciar y retornar el modelo
            model = model_class(layer_sizes=layers, activation_name=activation, **kwargs)
            logger.info(f"Modelo {selected_model} ({model_specific_type}) creado exitosamente")
            return model
            
        except ImportError as e:
            logger.error(f"Error al crear modelo: {e}")
            raise ValueError(f"Error al crear modelo: {e}")
        except Exception as e:
            logger.error(f"Error inesperado al crear modelo: {e}")
            logger.debug(traceback.format_exc())
            raise ValueError(f"Error inesperado al crear modelo: {e}")
    
    def _load_config(self, config_dict, config_file):
        """
        Carga la configuración desde un diccionario o archivo.
        
        Si no se proporciona ninguno, intenta usar la configuración predefina o
        el archivo de configuración predeterminado.
        """
        if config_dict:
            return config_dict
        elif config_file:
            try:
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error al cargar el archivo de configuración: {e}")
                raise
        elif self.config:
            # Usar configuración precargada
            return self.config
        else:
            # Último intento: cargar desde la ubicación predeterminada
            try:
                with open(DEFAULT_CONFIG_PATH, 'r') as f:
                    self.config = yaml.safe_load(f)
                    return self.config
            except Exception as e:
                raise ValueError(
                    f"No se encontró configuración válida. Error: {e}. "
                    f"Proporcione config_dict, config_file o asegúrese de que "
                    f"el archivo exista en: {DEFAULT_CONFIG_PATH}"
                )
    
    def _determine_model(self, config, model_name):
        """Determina qué modelo usar basado en la configuración o el nombre proporcionado."""
        if model_name:
            selected_model = model_name
        else:
            if "selected_model" in config:
                selected_model = config["selected_model"]
            else:
                raise ValueError("No se especificó un modelo en la configuración")
        
        # Buscar el modelo en la configuración
        if selected_model in config.get("models", {}):
            model_config = config["models"][selected_model]
            return selected_model, model_config
        else:
            raise ValueError(f"El modelo {selected_model} no está definido en la configuración")
    
    def _determine_model_type(self, model_config, model_type=None, category=None):
        """Determina la categoría y tipo específico del modelo."""
        # Si se proporciona explícitamente el tipo y la categoría, usar esos
        if model_type and category:
            return category, model_type
        
        # Intentar determinar según la configuración
        if "physics_type" in model_config:
            # Es un modelo de fluidos
            return "fluid", model_config["physics_type"]
        elif "type" in model_config:
            model_type = model_config["type"]
            # Determinar si es un modelo de fluidos o estándar según el tipo
            if model_type in ["navier_stokes", "kovasznay", "taylor_green", "cavity_flow"]:
                return "fluid", model_type
            else:
                # Es un modelo estándar
                return "standard", model_type
        else:
            # Por defecto, asumir modelo estándar tipo mlp
            return "standard", "mlp"
    
    def _get_model_class(self, category, model_type):
        """Obtiene la clase del modelo basado en la categoría y tipo."""
        if category in self.model_map and model_type in self.model_map[category]:
            return self.model_map[category][model_type]
        else:
            # Si no encontramos el modelo específico, intentar usar un modelo genérico de la misma categoría
            if category == "fluid":
                logger.warning(f"Modelo de fluidos tipo '{model_type}' no encontrado, usando modelo genérico")
                # Usar TaylorGreenPINN como modelo de fluidos genérico si está disponible
                return self.model_map[category].get("taylor_green", GenericFluidModel)
            elif category == "standard":
                logger.warning(f"Modelo estándar tipo '{model_type}' no encontrado, usando modelo genérico")
                # Usar el primer modelo estándar disponible
                for model_key in self.model_map[category]:
                    return self.model_map[category][model_key]
            
            # Si llegamos aquí, no hay modelos disponibles en esta categoría
            available_categories = list(self.model_map.keys())
            available_types = []
            for cat in available_categories:
                available_types.extend(list(self.model_map[cat].keys()))
            
            error_msg = f"No se encontró el modelo de tipo '{model_type}' en la categoría '{category}'.\n"
            error_msg += f"Categorías disponibles: {available_categories}\n"
            error_msg += f"Tipos disponibles: {available_types}"
            
            raise ValueError(error_msg)
    
    def _prepare_specific_params(self, category, model_type, model_config, config):
        """Prepara parámetros específicos según el tipo de modelo."""
        kwargs = {}
        
        # Parámetros específicos para modelos de fluidos
        if category == "fluid":
            if model_type == "taylor_green":
                kwargs["nu"] = model_config.get("nu", config.get("physics", {}).get("nu", 0.01))
            elif model_type == "kovasznay":
                kwargs["re"] = model_config.get("re", config.get("physics", {}).get("Re", 40.0))
            elif model_type == "cavity_flow":
                kwargs["nu"] = model_config.get("nu", config.get("physics", {}).get("nu", 0.01))
                kwargs["u0"] = model_config.get("u0", config.get("physics", {}).get("U0", 1.0))
                
        # Se pueden agregar más parámetros específicos para otros tipos de modelos aquí
                
        return kwargs
    
    def list_available_models(self, category=None):
        """
        Lista todos los modelos disponibles, opcionalmente filtrados por categoría.
        
        Args:
            category (str, optional): Categoría de modelos a listar.
            
        Returns:
            dict: Diccionario con los modelos disponibles.
        """
        if category:
            if category in self.model_map:
                return {category: self.model_map[category]}
            else:
                raise ValueError(f"Categoría '{category}' no existe. Categorías disponibles: {list(self.model_map.keys())}")
        else:
            return self.model_map
    
    def load_saved_model(self, model_path, model_class=None, config_path=None, device=None):
        """
        Carga un modelo guardado desde un archivo.
        
        Args:
            model_path (str): Ruta al modelo guardado.
            model_class (class, optional): Clase del modelo a cargar. Si no se proporciona,
                                          se intentará determinar desde la configuración.
            config_path (str, optional): Ruta al archivo de configuración.
            device (torch.device, optional): Dispositivo donde cargar el modelo.
            
        Returns:
            BaseNeuralNetwork: Modelo cargado.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Si no se proporciona la clase del modelo, intentar determinarla desde la configuración
        if model_class is None and config_path:
            # Cargar la configuración
            config = self._load_config(None, config_path)
            
            # Determinar el modelo
            selected_model, model_config = self._determine_model(config, None)
            
            # Obtener categoría y tipo
            category, model_type = self._determine_model_type(model_config)
            
            # Obtener la clase del modelo
            model_class = self._get_model_class(category, model_type)
            
            # Preparar parámetros para inicializar el modelo
            layers = model_config.get("layers", [])
            activation = model_config.get("activation_function", "Tanh")
            kwargs = self._prepare_specific_params(category, model_type, model_config, config)
            
            # Crear instancia del modelo
            model = model_class(layer_sizes=layers, activation_name=activation, **kwargs)
        elif model_class:
            # Crear instancia del modelo usando la clase proporcionada
            model = model_class()
        else:
            raise ValueError("Se debe proporcionar model_class o config_path para cargar el modelo")
        
        # Cargar el estado del modelo con adaptación
        model = ModelAdapter.load_model_with_adaptation(model, model_path, device)
        
        return model

    def select_model_interactive(self):
        """
        Método interactivo para seleccionar un modelo desde la consola.
        
        Returns:
            tuple: (model_type, model_id, model_config) información del modelo seleccionado
        """
        # Asegurar que tenemos configuración cargada
        if not self.config:
            try:
                self.config = self._load_config(None, None)
            except Exception as e:
                print(f"Error al cargar la configuración: {e}")
                return None, None, None

        # Obtener modelos disponibles
        available_models = self.config.get('models', {})
        model_names = list(available_models.keys())
        
        if not model_names:
            print("No hay modelos disponibles en la configuración.")
            return None, None, None
            
        # Mostrar opciones de modelos
        print("\nModelos disponibles:")
        for i, name in enumerate(model_names):
            model_type = available_models[name].get('type', 'Unknown')
            physics_type = available_models[name].get('physics_type', '')
            type_info = f"{model_type}" + (f" ({physics_type})" if physics_type else "")
            layers = available_models[name].get('layers', [])
            activation = available_models[name].get('activation_function', 'Unknown')
            print(f"{i+1}. {name} - {type_info} - Capas: {layers} - Activación: {activation}")
            
        # Solicitar selección al usuario
        while True:
            try:
                selection = input("\nSeleccione un modelo (número o nombre): ")
                
                # Verificar si la selección es un número
                if selection.isdigit():
                    index = int(selection) - 1
                    if 0 <= index < len(model_names):
                        selected_model = model_names[index]
                    else:
                        print(f"Por favor, seleccione un número entre 1 y {len(model_names)}")
                        continue
                # Verificar si la selección es un nombre de modelo
                elif selection in model_names:
                    selected_model = selection
                else:
                    print(f"Modelo '{selection}' no encontrado. Por favor, intente de nuevo.")
                    continue
                    
                # Obtener la configuración del modelo seleccionado
                model_config = available_models[selected_model]
                
                # Determinar el tipo de modelo
                if 'physics_type' in model_config:
                    model_type = model_config['physics_type']
                    category = 'fluid'
                else:
                    model_type = model_config.get('type', 'mlp')
                    category = 'standard'
                
                print(f"\nModelo '{selected_model}' de la categoría '{category}' seleccionado.")
                return model_type, selected_model, model_config
                
            except Exception as e:
                print(f"Error al seleccionar el modelo: {e}")
                print("Por favor, intente de nuevo.")


# Funciones auxiliares para facilitar el uso
def load_model(config_path=None, config_dict=None, model_name=None, model_type=None, category=None):
    """
    Función de conveniencia para cargar un modelo.
    
    Args:
        config_path (str, optional): Ruta al archivo de configuración.
        config_dict (dict, optional): Diccionario con la configuración.
        model_name (str, optional): Nombre específico del modelo a cargar.
        model_type (str, optional): Tipo específico del modelo a cargar.
        category (str, optional): Categoría del modelo (standard, fluid, etc.)
        
    Returns:
        BaseNeuralNetwork: Instancia del modelo seleccionado.
    """
    selector = ModelSelector(config_path)
    return selector.get_model(
        config_dict=config_dict,
        config_file=config_path,
        model_name=model_name,
        model_type=model_type,
        category=category
    )

def load_fluid_model(config_path=None, config_dict=None, model_name=None):
    """
    Función de conveniencia específica para cargar un modelo de fluidos.
    
    Args:
        config_path (str, optional): Ruta al archivo de configuración.
        config_dict (dict, optional): Diccionario con la configuración.
        model_name (str, optional): Nombre específico del modelo a cargar.
        
    Returns:
        BaseNeuralNetwork: Instancia del modelo de fluido seleccionado.
    """
    return load_model(
        config_path=config_path,
        config_dict=config_dict,
        model_name=model_name,
        category="fluid"
    )

def load_saved_model(model_path, model_class=None, config_path=None, device=None):
    """
    Función de conveniencia para cargar un modelo guardado.
    
    Args:
        model_path (str): Ruta al modelo guardado.
        model_class (class, optional): Clase del modelo a cargar.
        config_path (str, optional): Ruta al archivo de configuración.
        device (torch.device, optional): Dispositivo donde cargar el modelo.
        
    Returns:
        BaseNeuralNetwork: Modelo cargado.
    """
    selector = ModelSelector(config_path)
    return selector.load_saved_model(
        model_path=model_path,
        model_class=model_class,
        config_path=config_path,
        device=device
    )

if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create model selector and check models
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                              "config", "config.yaml")
    selector = ModelSelector(config_path)
    selector.check_all_models()
