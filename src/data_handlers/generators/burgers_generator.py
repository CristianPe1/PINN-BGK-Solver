import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.io
from scipy.integrate import solve_ivp

# Agregamos el directorio raíz al path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class BurgersGenerator:
    """
    Generador de datos sintéticos para la ecuación de Burgers.
    u_t + u*u_x = nu*u_xx
    """
    def __init__(self, nu=0.01/np.pi, nx=256, nt=100, tmax=1.0, xmin=-1.0, xmax=1.0):
        """
        Inicializa el generador con parámetros específicos.
        
        Args:
            nu (float): Coeficiente de viscosidad
            nx (int): Número de puntos en la malla espacial
            nt (int): Número de puntos en la malla temporal
            tmax (float): Tiempo máximo de simulación
            xmin (float): Posición mínima en x
            xmax (float): Posición máxima en x
        """
        self.nu = nu
        self.nx = nx
        self.nt = nt
        self.tmax = tmax
        self.xmin = xmin
        self.xmax = xmax
        self.x = np.linspace(xmin, xmax, nx)
        self.t = np.linspace(0, tmax, nt)
    
    def initial_condition_shock(self, x):
        """Condición inicial para generar una onda de choque."""
        return -np.sin(np.pi * x)
    
    def initial_condition_smooth(self, x):
        """Condición inicial suave."""
        return np.exp(-10 * x**2)
    
    def generate_exact_solution(self, ic_type="shock"):
        """
        Genera la solución exacta de la ecuación de Burgers.
        Para la ecuación de Burgers con condiciones iniciales específicas,
        se conoce la solución exacta en ciertos casos.
        """
        x = self.x
        t = self.t
        nu = self.nu
        
        # Crear matrices para la solución
        usol = np.zeros((self.nx, self.nt))
        
        # Para cada tiempo t, calculamos la solución exacta
        for i, t_val in enumerate(t):
            if t_val == 0:
                if ic_type == "shock":
                    usol[:, i] = self.initial_condition_shock(x)
                else:
                    usol[:, i] = self.initial_condition_smooth(x)
            else:
                # Solución exacta para condición inicial de onda de choque
                if ic_type == "shock":
                    usol[:, i] = -np.sin(np.pi * x) * np.exp(-nu*np.pi**2*t_val) / \
                                 (1 + (1 - np.exp(-nu*np.pi**2*t_val)) * np.cos(np.pi * x) / np.sin(np.pi * x))
                else:
                    # Para otro tipo de condiciones iniciales, se necesitaría resolver numéricamente
                    pass
        
        return x[:, np.newaxis], t[np.newaxis, :], usol
    
    def generate_numeric_solution(self, ic_type="shock"):
        """
        Genera una solución numérica para la ecuación de Burgers usando
        diferencias finitas con un esquema upwind para el término advectivo.
        """
        # Implementar aquí el método de diferencias finitas
        # ...
        pass
    
    def save_solution(self, file_path, x, t, usol):
        """
        Guarda la solución en un archivo .mat compatible con MATLAB/SciPy.
        
        Args:
            file_path (str): Ruta donde guardar el archivo
            x (ndarray): Malla espacial
            t (ndarray): Malla temporal
            usol (ndarray): Matriz solución [nx, nt]
        """
        data = {
            'x': x,
            't': t,
            'usol': usol
        }
        scipy.io.savemat(file_path, data)
        print(f"Datos guardados en {file_path}")
    
    def plot_solution(self, x, t, usol):
        """
        Visualiza la solución generada.
        
        Args:
            x (ndarray): Malla espacial
            t (ndarray): Malla temporal
            usol (ndarray): Matriz solución
        """
        X, T = np.meshgrid(x.flatten(), t.flatten())
        
        fig = plt.figure(figsize=(15, 5))
        
        # Gráfico 3D
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(T.T, X.T, usol, cmap='viridis')
        ax1.set_xlabel('t')
        ax1.set_ylabel('x')
        ax1.set_zlabel('u')
        ax1.set_title('Solución de la ecuación de Burgers')
        
        # Mapa de calor
        ax2 = fig.add_subplot(122)
        im = ax2.pcolormesh(T.T, X.T, usol, shading='auto', cmap='viridis')
        fig.colorbar(im, ax=ax2)
        ax2.set_xlabel('t')
        ax2.set_ylabel('x')
        ax2.set_title('Mapa de calor')
        
        plt.tight_layout()
        plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Crear directorio para datos sintéticos si no existe
    output_dir = os.path.join(PROJECT_ROOT, "data", "synthetic")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar datos
    bg = BurgersGenerator(nu=0.01/np.pi)
    x, t, usol = bg.generate_exact_solution(ic_type="shock")
    
    # Visualizar
    bg.plot_solution(x, t, usol)
    
    # Guardar
    output_file = os.path.join(output_dir, "burgers_shock_synthetic.mat")
    bg.save_solution(output_file, x, t, usol)
