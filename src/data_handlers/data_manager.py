import os
import json
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from datetime import datetime

try:
    import fenics as fe
    import mshr
    FENICS_AVAILABLE = True
except ImportError:
    print("FEniCS no está disponible. Algunas funcionalidades estarán limitadas.")
    FENICS_AVAILABLE = False

# Configuración del logger
logger = logging.getLogger("data_manager")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class DataManager:
    """
    Clase unificada para carga y generación de datos para simulación de fluidos.
    Incluye funcionalidades de carga de datos, preparación de tensores y
    generación de datos sintéticos para diversas ecuaciones físicas.
    """
    def __init__(self, spatial_points=256, time_points=100, output_dir=None):
        """
        Inicializa el manager con parámetros específicos.
        
        Args:
            spatial_points (int): Número de puntos en la malla espacial
            time_points (int): Número de puntos en la malla temporal
            output_dir (str, optional): Directorio donde guardar los resultados.
                                      Por defecto es "data/generated"
        """
        self.spatial_points = spatial_points
        self.time_points = time_points
        
        if output_dir is None:
            # Directorio por defecto para datos generados
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.output_dir = os.path.join(root_dir, "data", "generated")
        else:
            self.output_dir = output_dir
            
        # Crear directorio si no existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Crear subdirectorio de logs
        self.logs_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Configurar el FileHandler para el logger
        log_file = os.path.join(self.logs_dir, f"data_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(fh)
            # Añadir también un StreamHandler para ver logs en la consola
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        
        # Generar un identificador único para esta sesión
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Inicializar lista para registrar todas las simulaciones generadas
        self.generated_simulations = []
        
        # Verificar si FEniCS está disponible
        if not FENICS_AVAILABLE and output_dir is not None:  # Solo mostrar si es un objeto de generación
            logger.warning("FEniCS no está disponible. Las simulaciones numéricas no funcionarán.")
        
        logger.info(f"DataManager inicializado: spatial_points={spatial_points}, time_points={time_points}")
        if output_dir is not None:
            logger.info(f"Directorio de salida: {self.output_dir}")
    
    # ============== MÉTODOS DE CARGA Y PREPARACIÓN DE DATOS ==============
        
    def load_data_from_file(self, file_path):
        """
        Carga datos desde un archivo .mat y los devuelve como tensores de entrada y salida.

        Args:
            file_path (str): Ruta al archivo .mat

        Returns:
            tuple: (input_tensor, output_tensor) donde:
                   - input_tensor: Tensor que representa las coordenadas (x, t) [batch_size, 2]
                   - output_tensor: Tensor que representa la solución u(x, t) [batch_size, 1]
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
        
        try:
            data = loadmat(file_path)
            logger.info(f"Archivo .mat cargado correctamente: {file_path}")
            
            # Intentar extraer las claves esperadas
            x_data = None
            t_data = None
            u_data = None
            
            if 'x' in data and 't' in data and 'usol' in data:
                # Formato estándar con x, t, usol
                x_data = np.array(data['x']).flatten()
                t_data = np.array(data['t']).flatten()
                u_data = np.array(data['usol'])
                logger.info(f"Formato estándar detectado: x:{x_data.shape}, t:{t_data.shape}, usol:{u_data.shape}")
                
                # Crear malla de coordenadas (x, t)
                X, T = np.meshgrid(x_data, t_data, indexing='ij')
                
                # Reshape para crear tensor de entrada (x, t) y tensor de salida u(x, t)
                input_points = np.vstack([X.flatten(), T.flatten()]).T  # Shape: [n_points, 2]
                output_values = u_data.flatten().reshape(-1, 1)  # Shape: [n_points, 1]
                
            else:
                # Intentar detectar formato alternativo
                available_keys = [k for k in data.keys() if not k.startswith('__')]
                logger.warning(f"Formato estándar no detectado. Claves disponibles: {available_keys}")
                
                # Buscar cualquier matriz numérica y usarla como datos
                matrices = []
                for key in available_keys:
                    if isinstance(data[key], np.ndarray) and data[key].size > 1:
                        matrices.append((key, data[key]))
                
                if len(matrices) >= 2:
                    # Usar la primera matriz como entrada y la segunda como salida
                    input_key, input_data = matrices[0]
                    output_key, output_data = matrices[1]
                    
                    logger.info(f"Usando matriz '{input_key}' como entrada y '{output_key}' como salida")
                    
                    # Asegurar que la forma es adecuada
                    input_points = input_data.reshape(-1, input_data.shape[-1] if input_data.ndim > 1 else 1)
                    output_values = output_data.reshape(-1, 1)
                    
                    # Si input_points tiene solo 1 dimensión, crear una segunda
                    if input_points.shape[1] == 1:
                        input_points = np.hstack([input_points, np.zeros_like(input_points)])
                else:
                    raise ValueError("No se encontraron suficientes matrices numéricas en el archivo")
            
            # Asegurarse de que ambos tensores tengan el mismo número de muestras (filas)
            n_samples = min(input_points.shape[0], output_values.shape[0])
            input_points = input_points[:n_samples]
            output_values = output_values[:n_samples]
            
            # Convertir a tensores PyTorch
            input_tensor = torch.tensor(input_points, dtype=torch.float32)
            output_tensor = torch.tensor(output_values, dtype=torch.float32)
            
            logger.info(f"Datos procesados con éxito. Formas finales: Input:{input_tensor.shape}, Output:{output_tensor.shape}")
            return input_tensor, output_tensor
                
        except Exception as e:
            logger.error(f"Error al procesar el archivo .mat: {str(e)}")
            raise

    def prepare_data(self, data, target_dim=3):
        """
        Asegura que el tensor tenga al menos 'target_dim' dimensiones.
        Por ejemplo, si tensor.shape es (N, M) y target_dim=3 devuelve (N, M, 1).

        Args:
            tensor (torch.Tensor): Tensor a preparar.
            target_dim (int): Número de dimensiones deseadas.
        
        Returns:
            torch.Tensor: Tensor preparado.
        """
        tensor = torch.tensor(data, dtype=torch.float32)
        while tensor.ndim < target_dim:
            tensor = tensor.unsqueeze(-1)
        return tensor
    
    # ============== MÉTODOS DE GENERACIÓN DE DATOS ==============
    
    def load_and_process_data(self, file_path, data_source="real", nu=0.01/np.pi):
        """
        Carga los datos desde el archivo especificado, ya sean reales (archivo .mat)
        o sintéticos (llamando a create_synthetic_data). Luego prepara los tensores
        para que sean compatibles (agrega dimensiones si es necesario), grafica la solución 
        y registra en log las dimensiones de los datos cargados.
        
        Se interpreta que el primer tensor representa el conjunto de entrada (x) y el 
        segundo la salida (solución).
        
        Args:
            file_path (str): Ruta del archivo a cargar (para datos reales).
            data_source (str): "real" o "synthetic"
            nu (float): Coeficiente de viscosidad (solo para datos sintéticos)
        
        Returns:
            tuple: (input_tensor, solution_tensor) procesados como tensores.
        """
        if data_source == "real":
            logger.info(f"Cargando datos reales desde {file_path}...")
            # Se ignora el segundo tensor (por ejemplo, malla temporal) y se usa el primero como entrada
            x_tensor, solution_tensor = self.load_data_from_file(file_path)
        elif data_source == "synthetic":
            logger.info("Cargando datos sintéticos utilizando la función interna...")
            x_tensor, solution_tensor = self.create_synthetic_data(nu)
        else:
            raise ValueError("El parámetro data_source debe ser 'real' o 'synthetic'")
        
        # Asegurar que los tensores tengan al menos 3 dimensiones
        x_tensor = self.prepare_tensor(x_tensor)
        solution_tensor = self.prepare_tensor(solution_tensor)
        
        logger.info(f"Datos cargados: input shape: {x_tensor.shape}, solution shape: {solution_tensor.shape}")
        
        # Grafica rápida de la solución
        plot_file = self.plot_solution(x_tensor, solution_tensor, filename=f"quick_plot_{self.session_id}.png")
        
        # Registrar la carga de datos en un log
        log_info = {
            "file_loaded": file_path,
            "data_source": data_source,
            "input_shape": list(x_tensor.shape),
            "solution_shape": list(solution_tensor.shape),
            "visualization_file": plot_file,
            "load_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        log_file = os.path.join(self.logs_dir, f"data_loaded_{self.session_id}.json")
        with open(log_file, 'w') as f:
            json.dump(log_info, f, indent=2)
        logger.info(f"Información de la carga de datos registrada en: {log_file}")
        
        return x_tensor, solution_tensor
    
    def create_synthetic_data(self, nu):
        """
        Genera datos sintéticos para la ecuación de Burgers y los devuelve como un par
        de tensores: un tensor de entrada (x, t) y un tensor de salida u(x, t).

        Args:
            nu (float): Coeficiente de viscosidad

        Returns:
            tuple: (input_tensor, output_tensor) donde:
                   - input_tensor: Tensor que representa las coordenadas (x, t) [batch_size, 2]
                   - output_tensor: Tensor que representa la solución u(x, t) [batch_size, 1]
        """
        x = np.linspace(-1, 1, self.spatial_points)
        t = np.linspace(0, 1, self.time_points)
        
        # Crear malla de coordenadas
        X, T = np.meshgrid(x, t, indexing='ij')
        
        # Calcular solución analítica de Burgers
        usol = np.zeros((self.spatial_points, self.time_points))
        
        # Condición inicial en t=0: -sin(pi*x)
        usol[:, 0] = -np.sin(np.pi * x)
        
        # Para cada tiempo t>0, calcular la solución exacta
        for j in range(1, self.time_points):
            t_val = t[j]
            for i in range(self.spatial_points):
                x_val = x[i]
                # Solución exacta de Burgers
                denominator = 1 + (1 - np.exp(-nu*np.pi**2*t_val)) * np.cos(np.pi * x_val) / (np.sin(np.pi * x_val) + 1e-8)
                usol[i, j] = -np.sin(np.pi * x_val) * np.exp(-nu*np.pi**2*t_val) / denominator
        
        # Reshape para crear tensor de entrada (x, t) y tensor de salida u(x, t)
        input_points = np.column_stack((X.flatten(), T.flatten()))
        output_values = usol.flatten().reshape(-1, 1)
        
        # Convertir a tensores PyTorch
        input_tensor = torch.tensor(input_points, dtype=torch.float32)
        output_tensor = torch.tensor(output_values, dtype=torch.float32)
        
        logger.info(f"Datos sintéticos generados: {len(input_tensor)} puntos.")
        logger.info(f"Shape tensor entrada: {input_tensor.shape}, shape tensor salida: {output_tensor.shape}")
        
        return input_tensor, output_tensor

    def burgers_synthetic_solution(self, nu=0.01/np.pi, save=True):
        """
        Genera una solución sintética para la ecuación de Burgers 1D.
        
        Args:
            nu (float): Coeficiente de viscosidad
            save (bool): Si True, guarda los resultados
                
        Returns:
            tuple: (x, t, usol) para uso directo, donde:
                   - x: Array de coordenadas espaciales
                   - t: Array de coordenadas temporales
                   - usol: Matriz de solución [nx, nt]
        """
        logger.info(f"Generando solución sintética de Burgers con nu={nu}...")
        start_time = datetime.now()
        
        # Crear mallas
        x = np.linspace(-1, 1, self.spatial_points)
        t = np.linspace(0, 1, self.time_points)
        X, T = np.meshgrid(x, t)
        usol = np.zeros((self.spatial_points, self.time_points))
        
        # Condición inicial en t=0: -sin(pi*x)
        usol[:, 0] = -np.sin(np.pi * x)
        
        # Para cada tiempo t>0, calcular la solución exacta
        for j in range(1, self.time_points):
            t_val = t[j]
            for i in range(self.spatial_points):
                x_val = x[i]
                # Solución exacta de Burgers para la condición inicial -sin(pi*x)
                denominator = 1 + (1 - np.exp(-nu*np.pi**2*t_val)) * np.cos(np.pi * x_val) / (np.sin(np.pi * x_val) + 1e-8)
                usol[i, j] = -np.sin(np.pi * x_val) * np.exp(-nu*np.pi**2*t_val) / denominator
        
        # Crear también los tensores de entrada y salida para el modelo
        X_flat = X.transpose().flatten()
        T_flat = T.transpose().flatten()
        input_points = np.column_stack((X_flat, T_flat))
        output_values = usol.transpose().flatten().reshape(-1, 1)
        
        # Convertir a tensores PyTorch para uso futuro
        input_tensor = torch.tensor(input_points, dtype=torch.float32)
        output_tensor = torch.tensor(output_values, dtype=torch.float32)
        
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        
        # Guardar resultados
        output_file = None
        if save:
            output_file = os.path.join(self.output_dir, f"burgers_shock_nu_{nu:.5f}.mat")
            savemat(output_file, {
                'x': x.reshape(-1, 1), 
                't': t.reshape(1, -1), 
                'usol': usol,
                'input_tensor': input_points,
                'output_tensor': output_values,
                'parameters': {
                    'nu': nu,
                    'spatial_points': self.spatial_points,
                    'time_points': self.time_points
                }
            })
            logger.info(f"Datos guardados en {output_file}")
        
        # Calcular estadísticas y registrar la generación
        # ... código existente para estadísticas y registro ...
        
        logger.info(f"Datos sintéticos generados: Input:{input_tensor.shape}, Output:{output_tensor.shape}")
        return x, t, usol

    def _log_simulation(self, simulation_type, parameters, output_file, metadata=None):
        """
        Registra información sobre una simulación generada.
        
        Args:
            simulation_type (str): Tipo de simulación (ej: "kovasznay", "burgers")
            parameters (dict): Parámetros usados
            output_file (str): Ruta del archivo generado
            metadata (dict, optional): Metadatos adicionales
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        simulation_info = {
            "timestamp": timestamp,
            "simulation_type": simulation_type,
            "parameters": parameters,
            "output_file": output_file,
            "metadata": metadata or {}
        }
        
        # Añadir a la lista de simulaciones
        self.generated_simulations.append(simulation_info)
        
        # Guardar como archivo JSON con timestamped filename
        metadata_file = os.path.join(
            self.logs_dir, 
            f"{simulation_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(metadata_file, 'w') as f:
            json.dump(simulation_info, f, indent=2)
        
        # Log info
        logger.info(f"Simulación generada: {simulation_type}")
        logger.info(f"Parámetros: {parameters}")
        logger.info(f"Archivo de salida: {output_file}")
        logger.info(f"Metadatos guardados en: {metadata_file}")
        
        return metadata_file
    
    def save_generation_report(self):
        """
        Guarda un informe de todas las simulaciones generadas en esta sesión.
        
        Returns:
            str: Ruta al archivo de informe generado
        """
        if not self.generated_simulations:
            logger.warning("No hay simulaciones registradas para generar un informe.")
            return None
            
        report = {
            "session_id": self.session_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_simulations": len(self.generated_simulations),
            "simulations": self.generated_simulations
        }
        
        report_file = os.path.join(self.logs_dir, f"generation_report_{self.session_id}.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Informe de generación guardado en: {report_file}")
        logger.info(f"Total de simulaciones generadas: {len(self.generated_simulations)}")
        
        return report_file
    
    # Aquí se añadirían los otros métodos para generación de datos de fluidos
    # generate_kovasznay_flow, generate_taylor_green_vortex, generate_lid_driven_cavity, etc.
    # que por brevedad no se incluyen aquí

    # ==================== MÉTODO UNIFICADO DE CARGA Y PROCESAMIENTO ====================
    

    def plot_solution(self, x_tensor, solution_tensor, filename="solution_plot.png"):
        """
        Visualiza la solución en un gráfico.
        
        Se asume que x_tensor representa el conjunto de entrada y 
        solution_tensor la salida correspondiente.
        """
        plt.figure(figsize=(8, 6))
        # Para tensores 1D se realiza un gráfico de línea
        if x_tensor.ndim == 1:
            plt.plot(x_tensor.detach().cpu().numpy(), 
                     solution_tensor.detach().cpu().numpy())
            plt.xlabel("Entrada (x)")
            plt.ylabel("Solución")
        # Si son tensores 2D se usa un mapa de calor
        elif x_tensor.ndim == 2 and solution_tensor.ndim == 2:
            plt.pcolormesh(x_tensor.squeeze().detach().cpu().numpy(),
                           solution_tensor.squeeze().detach().cpu().numpy(),
                           shading='auto', cmap='viridis')
            plt.xlabel("Entrada (x)")
            plt.ylabel("Solución")
            plt.colorbar(label="Valor")
        else:
            # Caso general: se muestra la solución como imagen
            plt.imshow(solution_tensor.squeeze().detach().cpu().numpy(),
                       aspect='auto', cmap='viridis')
            plt.xlabel("Índice de columna")
            plt.ylabel("Índice de fila")
            plt.colorbar(label="Valor")
        plt.title("Visualización de la solución")
        plot_file = os.path.join(self.output_dir, f"data_visualization_{self.session_id}.png")
        plt.savefig(plot_file, dpi=300)
        plt.close()
        logger.info(f"Gráfico de la solución guardado en: {plot_file}")
        
        return plot_file

# Ejemplo de uso simple para pruebas
if __name__ == "__main__":
    # Configuración personalizada del logger para este bloque
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(console_handler)
    
    # Crear DataManager
    data_manager = DataManager(spatial_points=256, time_points=100)
    
    # Generar solución sintética de Burgers 1D
    x, t, usol = data_manager.burgers_synthetic_solution(nu=0.01/np.pi)
    print(f"Solución de Burgers generada con forma: {usol.shape}")

    # Cargar y procesar datos sintéticos
    X, T, U = data_manager.load_and_process_data(file_path="dummy_path.mat", data_source="synthetic", nu=0.01/np.pi)
    print(f"Datos procesados: X shape: {X.shape}, T shape: {T.shape}, U shape: {U.shape}")
