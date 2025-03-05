"""
Generadores de datos analíticos para problemas de fluidos sin necesidad de FEniCS
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import logging
from scipy.io import savemat

# Configurar el logger
logger = logging.getLogger(__name__)

class AnalyticalFluidGenerator:
    """
    Clase base para generar soluciones analíticas de problemas de fluidos
    utilizando únicamente NumPy y PyTorch.
    """
    
    def __init__(self, output_dir=None):
        """
        Inicializa el generador de soluciones analíticas
        
        Args:
            output_dir (str, opcional): Directorio donde guardar las soluciones
        """
        self.output_dir = output_dir or "data/generated"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_taylor_green(self, nu=0.01, nx=64, ny=64, nt=10, t_range=(0, 1)):
        """
        Genera datos sintéticos para el vórtice de Taylor-Green
        
        Args:
            nu (float): Viscosidad cinemática
            nx (int): Número de puntos en dirección x
            ny (int): Número de puntos en dirección y
            nt (int): Número de puntos en dirección t
            t_range (tuple): Rango de tiempo
            
        Returns:
            dict: Datos generados incluyendo campos de velocidad y presión
        """
        logger.info(f"Generando vórtice de Taylor-Green analítico: nx={nx}, ny={ny}, nt={nt}, nu={nu}")
        
        # Crear mallas (dominio estándar: [0, 2π] x [0, 2π])
        x = np.linspace(0, 2*np.pi, nx)
        y = np.linspace(0, 2*np.pi, ny)
        t = np.linspace(t_range[0], t_range[1], nt)
        
        # Crear grids 3D (meshgrid) para cálculos vectorizados
        # FIX: Use 'xy' indexing for better compatibility with matplotlib
        X, Y, T = np.meshgrid(x, y, t, indexing='xy')
        
        # Calcular la solución analítica
        decay = np.exp(-2 * nu * T)
        u = -np.cos(X) * np.sin(Y) * decay
        v = np.sin(X) * np.cos(Y) * decay
        p = -0.25 * (np.cos(2*X) + np.cos(2*Y)) * decay * decay
        
        # Crear tensores de entrada para pytorch
        # Need to flatten the arrays correctly due to different indexing
        x_flat = X.reshape(-1)
        y_flat = Y.reshape(-1)
        t_flat = T.reshape(-1)
        
        inputs = np.column_stack([x_flat, y_flat, t_flat])
        outputs = np.column_stack([
            u.reshape(-1), v.reshape(-1), p.reshape(-1)
        ])
        
        # Crear datos de entrenamiento/prueba (80/20 split)
        indices = np.random.permutation(len(inputs))
        train_size = int(0.8 * len(inputs))
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        train_inputs = inputs[train_indices]
        train_outputs = outputs[train_indices]
        test_inputs = inputs[test_indices]
        test_outputs = outputs[test_indices]
        
        # Convertir a tensores PyTorch para compatibilidad
        result = {
            'inputs': torch.tensor(inputs, dtype=torch.float32),
            'outputs': torch.tensor(outputs, dtype=torch.float32),
            'train_inputs': torch.tensor(train_inputs, dtype=torch.float32),
            'train_outputs': torch.tensor(train_outputs, dtype=torch.float32),
            'test_inputs': torch.tensor(test_inputs, dtype=torch.float32),
            'test_outputs': torch.tensor(test_outputs, dtype=torch.float32),
            'grid': {
                'x': x,
                'y': y,
                't': t,
                'X': X,
                'Y': Y,
                'T': T,
                'u': u,
                'v': v,
                'p': p
            },
            'params': {
                'nu': nu,
                'nx': nx,
                'ny': ny,
                'nt': nt
            }
        }
        
        return result
    
    def generate_kovasznay(self, re=40, nx=100, ny=50, x_range=(-0.5, 1.0), y_range=(-0.5, 1.5)):
        """
        Genera datos sintéticos para el flujo de Kovasznay
        
        Args:
            re (float): Número de Reynolds
            nx (int): Número de puntos en dirección x
            ny (int): Número de puntos en dirección y
            x_range (tuple): Rango de coordenada x
            y_range (tuple): Rango de coordenada y
            
        Returns:
            dict: Datos generados incluyendo campos de velocidad y presión
        """
        logger.info(f"Generando flujo de Kovasznay analítico: nx={nx}, ny={ny}, Re={re}")
        
        # Calcular lambda para el flujo de Kovasznay
        lmbd = 0.5 * re - np.sqrt(0.25 * (re**2) + 4 * (np.pi**2))
        
        # Crear mallas
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        
        # Crear grid 2D
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Calcular solución analítica
        u = 1 - np.exp(lmbd * X) * np.cos(2 * np.pi * Y)
        v = (lmbd / (2 * np.pi)) * np.exp(lmbd * X) * np.sin(2 * np.pi * Y)
        p = 0.5 * (1 - np.exp(2 * lmbd * X))
        
        # Crear tensores de entrada para pytorch
        x_flat = X.flatten()
        y_flat = Y.flatten()
        
        inputs = np.column_stack([x_flat, y_flat])
        outputs = np.column_stack([
            u.flatten(), v.flatten(), p.flatten()
        ])
        
        # Crear datos de entrenamiento/prueba (80/20 split)
        indices = np.random.permutation(len(inputs))
        train_size = int(0.8 * len(inputs))
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        train_inputs = inputs[train_indices]
        train_outputs = outputs[train_indices]
        test_inputs = inputs[test_indices]
        test_outputs = outputs[test_indices]
        
        # Convertir a tensores PyTorch para compatibilidad
        result = {
            'inputs': torch.tensor(inputs, dtype=torch.float32),
            'outputs': torch.tensor(outputs, dtype=torch.float32),
            'train_inputs': torch.tensor(train_inputs, dtype=torch.float32),
            'train_outputs': torch.tensor(train_outputs, dtype=torch.float32),
            'test_inputs': torch.tensor(test_inputs, dtype=torch.float32),
            'test_outputs': torch.tensor(test_outputs, dtype=torch.float32),
            'grid': {
                'x': x,
                'y': y,
                'X': X,
                'Y': Y,
                'u': u,
                'v': v,
                'p': p
            },
            'params': {
                're': re,
                'lambda': lmbd,
                'nx': nx,
                'ny': ny
            }
        }
        
        return result
    
    def generate_cavity_flow(self, re=100, n=64):
        """
        Genera datos sintéticos para el flujo en cavidad
        Usa un modelo aproximado ya que no hay solución analítica exacta
        
        Args:
            re (float): Número de Reynolds
            n (int): Número de puntos en cada dirección
            
        Returns:
            dict: Datos generados incluyendo campos de velocidad y presión
        """
        logger.info(f"Generando flujo de cavidad analítico: n={n}, Re={re}")
        
        # Crear mallas
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        
        # Crear grid 2D
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Crear aproximaciones analíticas al flujo de cavidad
        # Estas no son soluciones exactas de Navier-Stokes pero capturan
        # el comportamiento general de un flujo de cavidad

        # Velocidad en x:
        # - Debe ser 1 en la tapa superior (y=1)
        # - Debe ser 0 en los bordes laterales y fondo
        # - Interior: aproximación suave que cumple estas condiciones
        u = 16 * X * (1-X) * Y**2 * (2*Y - 3)
        
        # Velocidad en y:
        # - Debe ser 0 en todos los bordes
        # - Interior: vórtice principal que satisface continuidad
        v = -16 * Y * (1-Y) * X**2 * (1-X)
        
        # Presión:
        # - Aproximación básica del campo de presión
        # - El factor 1/re simula cómo la presión varía con Reynolds
        p = (np.sin(np.pi*X) * np.sin(np.pi*Y)) / re
        
        # Añadir perturbaciones para el vórtice secundario en la esquina inferior derecha
        # para Reynolds altos (re > 500)
        if re > 500:
            x_sec = X - 0.75
            y_sec = Y - 0.25
            r_sec = np.sqrt(x_sec**2 + y_sec**2)
            vortex_sec = np.exp(-30 * r_sec) * 0.1
            u = u - y_sec * vortex_sec
            v = v + x_sec * vortex_sec
        
        # Corregir para cumplir las condiciones de contorno exactamente
        # Tapa superior: u=1, v=0
        u[:-1, -1] = 1.0
        v[:-1, -1] = 0.0
        
        # Otros bordes: u=0, v=0
        u[0, :-1] = 0.0  # izquierda
        v[0, :-1] = 0.0
        u[-1, :-1] = 0.0  # derecha
        v[-1, :-1] = 0.0
        u[1:-1, 0] = 0.0  # fondo
        v[1:-1, 0] = 0.0
        
        # Crear tensores de entrada para pytorch
        x_flat = X.flatten()
        y_flat = Y.flatten()
        
        inputs = np.column_stack([x_flat, y_flat])
        outputs = np.column_stack([
            u.flatten(), v.flatten(), p.flatten()
        ])
        
        # Separar puntos de frontera
        is_boundary = (
            (np.isclose(x_flat, 0.0)) | 
            (np.isclose(x_flat, 1.0)) | 
            (np.isclose(y_flat, 0.0)) | 
            (np.isclose(y_flat, 1.0))
        )
        boundary_inputs = inputs[is_boundary]
        boundary_outputs = outputs[is_boundary]
        
        # Crear datos de entrenamiento/prueba (80/20 split)
        # Usar solo puntos interiores para entrenamiento/prueba
        interior_mask = ~is_boundary
        interior_inputs = inputs[interior_mask]
        interior_outputs = outputs[interior_mask]
        
        indices = np.random.permutation(len(interior_inputs))
        train_size = int(0.8 * len(interior_inputs))
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        train_inputs = interior_inputs[train_indices]
        train_outputs = interior_outputs[train_indices]
        test_inputs = interior_inputs[test_indices]
        test_outputs = interior_outputs[test_indices]
        
        # Convertir a tensores PyTorch para compatibilidad
        result = {
            'inputs': torch.tensor(inputs, dtype=torch.float32),
            'outputs': torch.tensor(outputs, dtype=torch.float32),
            'train_inputs': torch.tensor(train_inputs, dtype=torch.float32),
            'train_outputs': torch.tensor(train_outputs, dtype=torch.float32),
            'test_inputs': torch.tensor(test_inputs, dtype=torch.float32),
            'test_outputs': torch.tensor(test_outputs, dtype=torch.float32),
            'boundary_inputs': torch.tensor(boundary_inputs, dtype=torch.float32),
            'boundary_outputs': torch.tensor(boundary_outputs, dtype=torch.float32),
            'grid': {
                'x': x,
                'y': y,
                'X': X,
                'Y': Y,
                'u': u,
                'v': v,
                'p': p
            },
            'params': {
                're': re,
                'nx': n,
                'ny': n
            }
        }
        
        return result
    
    def generate_burgers(self, nu=0.01/np.pi, nx=256, nt=100, x_range=(-1, 1), t_range=(0, 1)):
        """
        Genera datos sintéticos para la ecuación de Burgers 1D
        
        Args:
            nu (float): Viscosidad
            nx (int): Número de puntos en espacio
            nt (int): Número de puntos en tiempo
            x_range (tuple): Rango espacial
            t_range (tuple): Rango temporal
            
        Returns:
            dict: Datos generados
        """
        logger.info(f"Generando solución analítica de Burgers: nx={nx}, nt={nt}, nu={nu}")
        
        # Crear mallas
        x = np.linspace(x_range[0], x_range[1], nx)
        t = np.linspace(t_range[0], t_range[1], nt)
        
        # Crear grid 2D
        X, T = np.meshgrid(x, t, indexing='ij')
        
        # Solución para condición inicial u(x,0) = -sin(pi*x)
        u = np.zeros_like(X)
        
        # Condición inicial
        u[:, 0] = -np.sin(np.pi * x)
        
        # Para cada tiempo t>0, calcular la solución exacta
        for j in range(1, nt):
            t_val = t[j]
            for i in range(nx):
                x_val = x[i]
                # Solución exacta de Burgers
                denominator = 1 + (1 - np.exp(-nu*np.pi**2*t_val)) * np.cos(np.pi * x_val) / (np.sin(np.pi * x_val) + 1e-8)
                u[i, j] = -np.sin(np.pi * x_val) * np.exp(-nu*np.pi**2*t_val) / denominator
        
        # Crear tensores de entrada para pytorch
        x_flat = X.flatten()
        t_flat = T.flatten()
        
        inputs = np.column_stack([x_flat, t_flat])
        outputs = u.flatten().reshape(-1, 1)
        
        # Crear datos de entrenamiento/prueba (80/20 split)
        indices = np.random.permutation(len(inputs))
        train_size = int(0.8 * len(inputs))
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        train_inputs = inputs[train_indices]
        train_outputs = outputs[train_indices]
        test_inputs = inputs[test_indices]
        test_outputs = outputs[test_indices]
        
        # Convertir a tensores PyTorch para compatibilidad
        result = {
            'inputs': torch.tensor(inputs, dtype=torch.float32),
            'outputs': torch.tensor(outputs, dtype=torch.float32),
            'train_inputs': torch.tensor(train_inputs, dtype=torch.float32),
            'train_outputs': torch.tensor(train_outputs, dtype=torch.float32),
            'test_inputs': torch.tensor(test_inputs, dtype=torch.float32),
            'test_outputs': torch.tensor(test_outputs, dtype=torch.float32),
            'grid': {
                'x': x,
                't': t,
                'X': X,
                'T': T,
                'u': u
            },
            'params': {
                'nu': nu,
                'nx': nx,
                'nt': nt
            }
        }
        
        return result
    
    def save_data(self, data, path, format='mat'):
        """
        Guarda los datos generados en un archivo.
        
        Args:
            data (dict): Datos generados
            path (str): Ruta donde guardar el archivo
            format (str): Formato de archivo ('mat', 'npz')
            
        Returns:
            str: Ruta del archivo guardado
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if format == 'mat':
            # Convert PyTorch tensors to NumPy arrays for mat file
            save_dict = {
                'inputs': data['inputs'].numpy(),
                'outputs': data['outputs'].numpy(),
                'train_inputs': data['train_inputs'].numpy(),
                'train_outputs': data['train_outputs'].numpy(),
                'test_inputs': data['test_inputs'].numpy(),
                'test_outputs': data['test_outputs'].numpy()
            }
            
            # Add grid data
            for key, value in data['grid'].items():
                save_dict[key] = value
                
            # Add parameters
            save_dict['params'] = data['params']
            
            # Save boundary data if available
            if 'boundary_inputs' in data:
                save_dict['boundary_inputs'] = data['boundary_inputs'].numpy()
                save_dict['boundary_outputs'] = data['boundary_outputs'].numpy()
                
            savemat(path, save_dict)
            logger.info(f"Datos guardados en formato MAT: {path}")
            
        elif format == 'npz':
            # Prepare dictionary for npz
            save_dict = {}
            
            # Convert PyTorch tensors to NumPy
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    save_dict[key] = value.numpy()
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        save_dict[f"{key}_{subkey}"] = subvalue
                else:
                    save_dict[key] = value
                    
            np.savez(path, **save_dict)
            logger.info(f"Datos guardados en formato NPZ: {path}")
            
        else:
            logger.error(f"Formato no soportado: {format}")
            return None
            
        return path
    
    def plot_solution(self, data, output_dir, problem_type):
        """
        Genera visualizaciones de las soluciones
        
        Args:
            data (dict): Datos generados
            output_dir (str): Directorio donde guardar las imágenes
            problem_type (str): Tipo de problema ('taylor_green', 'kovasznay', etc.)
            
        Returns:
            list: Rutas de las imágenes generadas
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if problem_type == 'burgers':
            return self._plot_burgers(data, output_dir)
        elif problem_type == 'taylor_green':
            return self._plot_taylor_green(data, output_dir)
        elif problem_type == 'kovasznay':
            return self._plot_kovasznay(data, output_dir)
        elif problem_type == 'cavity_flow':
            return self._plot_cavity_flow(data, output_dir)
        else:
            logger.error(f"Tipo de problema desconocido: {problem_type}")
            return []
    
    def _plot_burgers(self, data, output_dir):
        """Genera visualizaciones para la ecuación de Burgers"""
        files = []
        
        # Extraer datos
        X = data['grid']['X']
        T = data['grid']['T']
        u = data['grid']['u']
        
        # Plot de evolución espaciotemporal
        plt.figure(figsize=(12, 8))
        contour = plt.contourf(X, T, u, 50, cmap='viridis')
        plt.colorbar(contour, label='Velocity (u)')
        plt.xlabel('Position (x)')
        plt.ylabel('Time (t)')
        plt.title('Burgers Equation Solution')
        
        space_time_file = os.path.join(output_dir, 'burgers_spacetime.png')
        plt.savefig(space_time_file, dpi=300, bbox_inches='tight')
        plt.close()
        files.append(space_time_file)
        
        # Plots de perfiles de velocidad en diferentes tiempos
        plt.figure(figsize=(12, 8))
        nt = T.shape[1]
        times_to_plot = [0, nt//4, nt//2, 3*nt//4, -1]
        
        for i, t_idx in enumerate(times_to_plot):
            plt.plot(X[:, t_idx], u[:, t_idx], label=f't = {T[0, t_idx]:.3f}', linewidth=2)
            
        plt.xlabel('Position (x)')
        plt.ylabel('Velocity (u)')
        plt.title('Burgers Equation: Velocity Profiles')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        profiles_file = os.path.join(output_dir, 'burgers_profiles.png')
        plt.savefig(profiles_file, dpi=300, bbox_inches='tight')
        plt.close()
        files.append(profiles_file)
        
        return files
    
    def _plot_taylor_green(self, data, output_dir):
        """Genera visualizaciones para el vórtice de Taylor-Green"""
        files = []
        
        # Extraer datos
        X = data['grid']['X']
        Y = data['grid']['Y']
        T = data['grid']['T']
        u = data['grid']['u']
        v = data['grid']['v']
        p = data['grid']['p']
        
        # Seleccionar tiempos para visualizar
        nt = T.shape[2]
        times_to_plot = [0, nt//2, -1]
        
        for t_idx in times_to_plot:
            # Crear gráfico de velocidad
            plt.figure(figsize=(10, 8))
            speed = np.sqrt(u[:,:,t_idx]**2 + v[:,:,t_idx]**2)
            contour = plt.contourf(X[:,:,t_idx], Y[:,:,t_idx], speed, 50, cmap='viridis')
            plt.colorbar(contour, label='Velocity Magnitude')
            
            # Añadir vectores de velocidad
            stride = max(1, min(X.shape[0], X.shape[1]) // 20)
            plt.quiver(X[::stride,::stride,t_idx], Y[::stride,::stride,t_idx], 
                      u[::stride,::stride,t_idx], v[::stride,::stride,t_idx],
                      color='white', scale=20)
            
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Taylor-Green Velocity at t = {T[0,0,t_idx]:.3f}')
            
            vel_file = os.path.join(output_dir, f'taylor_green_vel_t{t_idx}.png')
            plt.savefig(vel_file, dpi=300, bbox_inches='tight')
            plt.close()
            files.append(vel_file)
            
            # Crear gráfico de presión
            plt.figure(figsize=(10, 8))
            contour = plt.contourf(X[:,:,t_idx], Y[:,:,t_idx], p[:,:,t_idx], 50, cmap='coolwarm')
            plt.colorbar(contour, label='Pressure')
            
            # FIX: Use pcolormesh instead of streamplot for simplicity
            # This avoids the formatting requirements of streamplot
            plt.pcolormesh(X[:,:,t_idx], Y[:,:,t_idx], p[:,:,t_idx], cmap='coolwarm', alpha=0.3, shading='auto')
            
            # Add just the quiver plot with smaller arrows for streamlines effect
            plt.quiver(X[::stride,::stride,t_idx], Y[::stride,::stride,t_idx], 
                      u[::stride,::stride,t_idx], v[::stride,::stride,t_idx],
                      color='white', scale=40, width=0.001)
            
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Taylor-Green Pressure at t = {T[0,0,t_idx]:.3f}')
            
            press_file = os.path.join(output_dir, f'taylor_green_press_t{t_idx}.png')
            plt.savefig(press_file, dpi=300, bbox_inches='tight')
            plt.close()
            files.append(press_file)
        
        return files
    
    def _plot_kovasznay(self, data, output_dir):
        """Genera visualizaciones para el flujo de Kovasznay"""
        files = []
        
        # Extraer datos
        X = data['grid']['X']
        Y = data['grid']['Y']
        u = data['grid']['u']
        v = data['grid']['v']
        p = data['grid']['p']
        
        # Crear gráfico de velocidad
        plt.figure(figsize=(12, 8))
        speed = np.sqrt(u**2 + v**2)
        contour = plt.contourf(X, Y, speed, 50, cmap='viridis')
        plt.colorbar(contour, label='Velocity Magnitude')
        
        # Añadir vectores de velocidad
        stride = max(1, min(X.shape[0], Y.shape[0]) // 25)
        plt.quiver(X[::stride,::stride], Y[::stride,::stride], 
                  u[::stride,::stride], v[::stride,::stride],
                  color='white', scale=10, alpha=0.7)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Kovasznay Flow Velocity (Re = {data["params"]["re"]})')
        
        vel_file = os.path.join(output_dir, 'kovasznay_velocity.png')
        plt.savefig(vel_file, dpi=300, bbox_inches='tight')
        plt.close()
        files.append(vel_file)
        
        # Crear gráfico de presión
        plt.figure(figsize=(12, 8))
        contour = plt.contourf(X, Y, p, 50, cmap='coolwarm')
        plt.colorbar(contour, label='Pressure')
        
        # Añadir líneas de corriente
        plt.streamplot(X, Y, u, v, color='white', linewidth=0.7, density=1.5)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Kovasznay Flow Pressure (Re = {data["params"]["re"]})')
        
        press_file = os.path.join(output_dir, 'kovasznay_pressure.png')
        plt.savefig(press_file, dpi=300, bbox_inches='tight')
        plt.close()
        files.append(press_file)
        
        return files
        
    def _plot_cavity_flow(self, data, output_dir):
        """Genera visualizaciones para el flujo en cavidad"""
        files = []
        
        # Extraer datos
        X = data['grid']['X']
        Y = data['grid']['Y']
        u = data['grid']['u']
        v = data['grid']['v']
        p = data['grid']['p']
        
        # Crear gráfico de velocidad
        plt.figure(figsize=(10, 10))
        speed = np.sqrt(u**2 + v**2)
        contour = plt.contourf(X, Y, speed, 50, cmap='viridis')
        plt.colorbar(contour, label='Velocity Magnitude')
        
        # Añadir vectores de velocidad
        stride = max(1, min(X.shape[0], Y.shape[0]) // 20)
        plt.quiver(X[::stride,::stride], Y[::stride,::stride], 
                  u[::stride,::stride], v[::stride,::stride],
                  color='white', scale=2)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Lid-Driven Cavity Flow Velocity (Re = {data["params"]["re"]})')
        
        vel_file = os.path.join(output_dir, 'cavity_flow_velocity.png')
        plt.savefig(vel_file, dpi=300, bbox_inches='tight')
        plt.close()
        files.append(vel_file)
        
        # Crear gráfico de presión
        plt.figure(figsize=(10, 10))
        contour = plt.contourf(X, Y, p, 50, cmap='coolwarm')
        plt.colorbar(contour, label='Pressure')
        
        # Añadir líneas de corriente
        plt.streamplot(X, Y, u, v, color='white', linewidth=0.7, density=1.5, arrowsize=0.8)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Lid-Driven Cavity Flow Pressure (Re = {data["params"]["re"]})')
        
        press_file = os.path.join(output_dir, 'cavity_flow_pressure.png')
        plt.savefig(press_file, dpi=300, bbox_inches='tight')
        plt.close()
        files.append(press_file)
        
        # Crear gráfico de vórtices con líneas de corriente coloreadas por vorticidad
        plt.figure(figsize=(10, 10))
        
        # Calcular vorticidad (dv/dx - du/dy) usando diferencias finitas
        dx = X[1, 0] - X[0, 0]
        dy = Y[0, 1] - Y[0, 0]
        dudy, dudx = np.gradient(u, dy, dx)
        dvdy, dvdx = np.gradient(v, dy, dx)
        vorticity = dvdx - dudy
        
        # Colorear por vorticidad
        contour = plt.contourf(X, Y, vorticity, 50, cmap='RdBu_r')
        plt.colorbar(contour, label='Vorticity')
        
        # Añadir líneas de corriente
        plt.streamplot(X, Y, u, v, color='white', linewidth=0.7, density=1.5)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Lid-Driven Cavity Flow Vorticity (Re = {data["params"]["re"]})')
        
        vort_file = os.path.join(output_dir, 'cavity_flow_vorticity.png')
        plt.savefig(vort_file, dpi=300, bbox_inches='tight')
        plt.close()
        files.append(vort_file)
        
        return files


# Entry point para usar la clase desde línea de comandos
if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Generador analítico de datos de fluidos')
    parser.add_argument('--problem', type=str, required=True, choices=['taylor_green', 'kovasznay', 'cavity_flow', 'burgers'],
                        help='Tipo de problema de fluidos')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directorio donde guardar los resultados')
    parser.add_argument('--format', type=str, default='mat', choices=['mat', 'npz'],
                        help='Formato para guardar los datos')
    
    # Parámetros específicos para cada problema
    parser.add_argument('--nu', type=float, default=0.01,
                        help='Viscosidad (para taylor_green y burgers)')
    parser.add_argument('--re', type=float, default=40,
                        help='Número de Reynolds (para kovasznay y cavity_flow)')
    parser.add_argument('--nx', type=int, default=64,
                        help='Número de puntos en dirección x')
    parser.add_argument('--ny', type=int, default=64,
                        help='Número de puntos en dirección y')
    parser.add_argument('--nt', type=int, default=20,
                        help='Número de puntos en dirección temporal')
    
    args = parser.parse_args()
    
    # Crear generador
    generator = AnalyticalFluidGenerator(output_dir=args.output_dir)
    
    # Determinar directorio de salida específico para visualizaciones
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        vis_dir = os.path.join(args.output_dir, args.problem, "visualizaciones", timestamp)
    else:
        vis_dir = os.path.join(generator.output_dir, args.problem, "visualizaciones", timestamp)
    
    os.makedirs(vis_dir, exist_ok=True)
    
    # Generar datos según el tipo de problema
    if args.problem == 'taylor_green':
        data = generator.generate_taylor_green(nu=args.nu, nx=args.nx, ny=args.ny, nt=args.nt)
        filename = f"taylor_green_nu{args.nu}_nx{args.nx}_ny{args.ny}_nt{args.nt}_{timestamp}.{args.format}"
    elif args.problem == 'kovasznay':
        data = generator.generate_kovasznay(re=args.re, nx=args.nx, ny=args.ny)
        filename = f"kovasznay_re{args.re}_nx{args.nx}_ny{args.ny}_{timestamp}.{args.format}"
    elif args.problem == 'cavity_flow':
        data = generator.generate_cavity_flow(re=args.re, n=args.nx)
        filename = f"cavity_flow_re{args.re}_n{args.nx}_{timestamp}.{args.format}"
    elif args.problem == 'burgers':
        data = generator.generate_burgers(nu=args.nu, nx=args.nx, nt=args.nt)
        filename = f"burgers_nu{args.nu}_nx{args.nx}_nt{args.nt}_{timestamp}.{args.format}"
    
    # Guardar datos
    if args.output_dir:
        output_path = os.path.join(args.output_dir, args.problem, filename)
    else:
        output_path = os.path.join(generator.output_dir, args.problem, filename)
        
    saved_path = generator.save_data(data, output_path, format=args.format)
    
    # Generar visualizaciones
    vis_files = generator.plot_solution(data, vis_dir, args.problem)
    
    print(f"Datos generados y guardados en: {saved_path}")
    print(f"Visualizaciones guardadas en: {vis_dir}")