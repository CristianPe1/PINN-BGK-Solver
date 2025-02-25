import os
import numpy as np
import torch
from scipy.io import loadmat

class DataLoader:
    """
    Clase para gestionar la generación y carga de datos para la ecuación de Burgers.
    Tiene métodos separados para generar datos sintéticos y cargar datos desde archivo.
    """
    def __init__(self, spatial_points, time_points):
        """
        Args:
            spatial_points (int): Número de puntos espaciales.
            time_points (int): Número de puntos en el tiempo.
            nu (float): Coeficiente de viscosidad.
        """
        self.spatial_points = spatial_points
        self.time_points = time_points


    def create_synthetic_data(self, nu):
        """
        Genera datos sintéticos usando una solución exacta simplificada.

        Returns:
            tuple: Tensores (X, T, u0) generados de forma sintética.
        """
        x = np.linspace(-1, 1, self.spatial_points)
        t = np.linspace(0, 1, self.time_points)
        # Convertir a tensores
        x_tensor = torch.tensor(x, dtype=torch.float32)
        t_tensor = torch.tensor(t, dtype=torch.float32)
        # Crear mallas de coordenadas
        X, T = torch.meshgrid(x_tensor, t_tensor, indexing='ij')
        u0 = -torch.tanh(X / (2 * nu * T + 1e-8))
        return X, T, u0

    def load_data_from_file(self, file_path):
        """
        Carga datos desde un archivo .mat.
        
        Args:
            file_path (str): Ruta al archivo .mat.
        
        Returns:
            tuple: Tensores (X, T, u0) cargados del archivo.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
        data = loadmat(file_path)
        try:
            x_data = data['x'].flatten()
            t_data = data['t'].flatten()
            u0_data = data['usol']
        except KeyError as e:
            raise KeyError("El archivo .mat no contiene las claves esperadas: 'x', 't' o 'usol'") from e

        # Convertir a tensores
        x_tensor = torch.tensor(x_data, dtype=torch.float32)
        t_tensor = torch.tensor(t_data, dtype=torch.float32)
        u0_tensor = torch.tensor(u0_data, dtype=torch.float32)
        # Crear mallas
        X, T = torch.meshgrid(x_tensor, t_tensor, indexing='ij')
        return X, T, u0_tensor

    @staticmethod
    def prepare_tensor(tensor, target_dim=3):
        """
        Asegura que el tensor tenga al menos 'target_dim' dimensiones.
        Por ejemplo, si tensor.shape es (N, M) y target_dim=3 devuelve (N, M, 1).

        Args:
            tensor (torch.Tensor): Tensor a preparar.
            target_dim (int): Número de dimensiones deseadas.
        
        Returns:
            torch.Tensor: Tensor preparado.
        """
        while tensor.ndim < target_dim:
            tensor = tensor.unsqueeze(-1)
        return tensor