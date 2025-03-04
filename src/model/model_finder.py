import os
import glob
import json
import logging

logger = logging.getLogger(__name__)

class ModelFinder:
    """
    Clase para buscar modelos entrenados en el sistema de archivos.
    Proporciona métodos para listar, filtrar y seleccionar modelos disponibles.
    """
    
    @staticmethod
    def find_model_files(search_paths=None, recursive=True):
        """
        Busca archivos de modelo (weights_model.pth) en las rutas especificadas.
        
        Args:
            search_paths (list): Lista de directorios donde buscar modelos.
                                Si es None, busca en ubicaciones predeterminadas.
            recursive (bool): Si es True, busca en subdirectorios.
            
        Returns:
            list: Lista de rutas a archivos de modelo encontrados
        """
        if search_paths is None:
            # Determinar directorio del proyecto
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            project_dir = os.path.dirname(current_dir)  # un nivel arriba de src/
            
            # Ubicaciones predeterminadas para buscar
            search_paths = [
                os.path.join(current_dir, "output"),  # /src/output
                os.path.join(project_dir, "output"),  # /output
                os.path.join(project_dir, "models"),  # /models
                os.path.dirname(current_dir),  # directorio padre de src/
                current_dir  # directorio src/
            ]
        
        model_files = []
        
        for path in search_paths:
            if not os.path.exists(path):
                continue
                
            # Buscar archivos de modelo en este directorio
            pattern = os.path.join(path, "**", "weights_model.pth") if recursive else os.path.join(path, "weights_model.pth")
            found_files = glob.glob(pattern, recursive=recursive)
            model_files.extend(found_files)
            
            # También buscar archivos de modelo con otros posibles nombres
            alt_patterns = [
                os.path.join(path, "**", "model.pth") if recursive else os.path.join(path, "model.pth"),
                os.path.join(path, "**", "*.pth") if recursive else os.path.join(path, "*.pth")
            ]
            
            for pattern in alt_patterns:
                found_files = glob.glob(pattern, recursive=recursive)
                model_files.extend([f for f in found_files if "weights_model.pth" not in f])
        
        return model_files
    
    @staticmethod
    def get_model_info(model_path):
        """
        Obtiene información sobre un modelo a partir de su archivo de pesos.
        
        Args:
            model_path (str): Ruta al archivo de modelo
            
        Returns:
            dict: Información del modelo o None si no se puede extraer
        """
        try:
            # Intentar identificar la carpeta del modelo
            model_dir = os.path.dirname(model_path)
            if model_dir.endswith("model"):
                model_dir = os.path.dirname(model_dir)
                
            # Buscar archivo de hiperparámetros
            hyperparams_path = None
            possible_paths = [
                os.path.join(model_dir, "train_results", "hyperparams_train.json"),
                os.path.join(model_dir, "hyperparams_train.json"),
                os.path.join(model_dir, "hyperparams.json")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    hyperparams_path = path
                    break
            
            # Buscar archivo de estadísticas
            stats_path = None
            possible_stats = [
                os.path.join(model_dir, "train_results", "training_stats.json"),
                os.path.join(model_dir, "training_stats.json"),
                os.path.join(model_dir, "stats.json")
            ]
            
            for path in possible_stats:
                if os.path.exists(path):
                    stats_path = path
                    break
            
            # Extraer nombre del modelo
            model_name = os.path.basename(model_dir)
            
            # Cargar hiperparámetros si existen
            hyperparams = None
            if hyperparams_path and os.path.exists(hyperparams_path):
                with open(hyperparams_path, 'r') as f:
                    hyperparams = json.load(f)
            
            # Cargar estadísticas si existen
            stats = None
            if stats_path and os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
            
            # Extraer tipo de problema físico
            physics_type = "unknown"
            if hyperparams and "physics" in hyperparams:
                physics_type = hyperparams.get("physics", {}).get("physics_type", "unknown")
            
            return {
                "path": model_path,
                "name": model_name,
                "hyperparams": hyperparams,
                "stats": stats,
                "physics_type": physics_type,
                "dir": model_dir
            }
            
        except Exception as e:
            logger.warning(f"Error al extraer información del modelo {model_path}: {str(e)}")
            # Devolver información básica
            return {
                "path": model_path,
                "name": os.path.basename(os.path.dirname(model_path)),
                "hyperparams": None,
                "stats": None,
                "physics_type": "unknown",
                "dir": os.path.dirname(model_path)
            }
    
    @staticmethod
    def find_models():
        """
        Busca y cataloga todos los modelos disponibles.
        
        Returns:
            dict: Diccionario con modelos organizados por tipo de física
        """
        # Buscar archivos de modelo
        model_files = ModelFinder.find_model_files()
        
        if not model_files:
            logger.warning("No se encontraron archivos de modelo.")
            return {}
        
        # Organizar modelos por tipo de física
        models_by_physics = {}
        
        for model_path in model_files:
            model_info = ModelFinder.get_model_info(model_path)
            physics_type = model_info.get("physics_type", "unknown")
            
            if physics_type not in models_by_physics:
                models_by_physics[physics_type] = []
                
            models_by_physics[physics_type].append(model_info)
        
        return models_by_physics
    
    @staticmethod
    def select_model_interactive():
        """
        Permite al usuario seleccionar un modelo interactivamente.
        
        Returns:
            str: Ruta al modelo seleccionado o None si no se seleccionó ninguno
        """
        models_by_physics = ModelFinder.find_models()
        
        if not models_by_physics:
            print("No se encontraron modelos para seleccionar.")
            return None
            
        # Mostrar modelos disponibles por tipo de física
        print("\n=== MODELOS DISPONIBLES ===")
        physics_types = list(models_by_physics.keys())
        
        for i, physics_type in enumerate(physics_types, 1):
            models = models_by_physics[physics_type]
            print(f"\n{i}. {physics_type.upper()} ({len(models)} modelos)")
            
            for j, model in enumerate(models, 1):
                name = model.get("name", "Sin nombre")
                
                # Mostrar estadísticas si están disponibles
                if model.get("stats"):
                    accuracy = model["stats"].get("final_accuracy", "N/A")
                    train_time = model["stats"].get("total_training_time_sec", "N/A")
                    print(f"   {j}. {name} (Precisión: {accuracy}, Tiempo: {train_time}s)")
                else:
                    print(f"   {j}. {name}")
        
        # Solicitar selección de tipo de física
        try:
            physics_choice = int(input("\nSeleccione tipo de problema físico (número): ")) - 1
            
            if physics_choice < 0 or physics_choice >= len(physics_types):
                print("Selección no válida.")
                return None
                
            selected_physics = physics_types[physics_choice]
            models = models_by_physics[selected_physics]
            
            # Solicitar selección de modelo
            model_choice = int(input(f"Seleccione modelo de {selected_physics} (número): ")) - 1
            
            if model_choice < 0 or model_choice >= len(models):
                print("Selección no válida.")
                return None
                
            selected_model = models[model_choice]
            return selected_model["path"]
            
        except ValueError:
            print("Error: Ingrese un número válido.")
            return None
        except Exception as e:
            print(f"Error al seleccionar modelo: {str(e)}")
            return None
