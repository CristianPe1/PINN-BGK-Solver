import numpy as np
import torch
import os
import logging
import h5py
from pathlib import Path
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class TaylorGreenDataGenerator:
    """Generador de datos sintéticos para el flujo de Taylor-Green"""
    
    def __init__(self, nu=0.01):
        """
        Inicializa el generador de datos para Taylor-Green
        
        Args:
            nu (float): Viscosidad cinemática
        """
        self.nu = nu
        
    def generate_data(self, nx=50, ny=50, nt=20, x_range=(0, 2*np.pi), y_range=(0, 2*np.pi), t_range=(0, 1)):
        """
        Genera datos sintéticos para el vórtice de Taylor-Green
        
        Args:
            nx (int): Número de puntos en dirección x
            ny (int): Número de puntos en dirección y
            nt (int): Número de puntos en dirección temporal
            x_range (tuple): Rango de coordenada x
            y_range (tuple): Rango de coordenada y
            t_range (tuple): Rango de tiempo
            
        Returns:
            dict: Diccionario con datos sintéticos
        """
        logger.info(f"Generando {nx}×{ny}×{nt} puntos para flujo de Taylor-Green")
        
        # Generar mallas de coordenadas
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        t = np.linspace(t_range[0], t_range[1], nt)
        
        X, Y, T = np.meshgrid(x, y, t, indexing='ij')
        
        # Vectorizar para cálculos eficientes
        x_flat = X.flatten()
        y_flat = Y.flatten()
        t_flat = T.flatten()
        
        # Calcular solución analítica
        decay = np.exp(-2 * self.nu * t_flat)
        u = -np.cos(x_flat) * np.sin(y_flat) * decay
        v = np.sin(x_flat) * np.cos(y_flat) * decay
        p = -0.25 * (np.cos(2*x_flat) + np.cos(2*y_flat)) * decay * decay
        
        # Crear tensores de entrada y salida
        inputs = np.column_stack([x_flat, y_flat, t_flat])
        outputs = np.column_stack([u, v, p])
        
        # Dividir en conjuntos de entrenamiento y prueba (80/20)
        num_samples = inputs.shape[0]
        indices = np.random.permutation(num_samples)
        train_size = int(0.8 * num_samples)
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        train_inputs = inputs[train_indices]
        train_outputs = outputs[train_indices]
        test_inputs = inputs[test_indices]
        test_outputs = outputs[test_indices]
        
        # Crear diccionario con todos los datos
        data = {
            'inputs': inputs,
            'outputs': outputs,
            'train_inputs': train_inputs,
            'train_outputs': train_outputs,
            'test_inputs': test_inputs,
            'test_outputs': test_outputs,
            'grid': {
                'x': x,
                'y': y,
                't': t,
                'X': X,
                'Y': Y,
                'T': T
            },
            'meta': {
                'nu': self.nu,
                'nx': nx,
                'ny': ny,
                'nt': nt
            }
        }
        
        return data
    
    def save_data(self, data, path, format='hdf5'):
        """
        Guarda los datos generados en un archivo
        
        Args:
            data (dict): Datos generados
            path (str): Ruta donde guardar los datos
            format (str): Formato ('hdf5', 'npz', 'csv')
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if format == 'hdf5':
            with h5py.File(path, 'w') as f:
                # Guardar inputs/outputs
                f.create_dataset('inputs', data=data['inputs'])
                f.create_dataset('outputs', data=data['outputs'])
                f.create_dataset('train_inputs', data=data['train_inputs'])
                f.create_dataset('train_outputs', data=data['train_outputs'])
                f.create_dataset('test_inputs', data=data['test_inputs'])
                f.create_dataset('test_outputs', data=data['test_outputs'])
                
                # Guardar grid
                grid = f.create_group('grid')
                for key, value in data['grid'].items():
                    grid.create_dataset(key, data=value)
                
                # Guardar metadatos
                meta = f.create_group('meta')
                for key, value in data['meta'].items():
                    meta.attrs[key] = value
                    
            logger.info(f"Datos guardados en formato HDF5: {path}")
                
        elif format == 'npz':
            np.savez(path, **data)
            logger.info(f"Datos guardados en formato NPZ: {path}")
            
        elif format == 'csv':
            base_path = os.path.splitext(path)[0]
            np.savetxt(f"{base_path}_inputs.csv", data['inputs'], delimiter=',', header='x,y,t')
            np.savetxt(f"{base_path}_outputs.csv", data['outputs'], delimiter=',', header='u,v,p')
            np.savetxt(f"{base_path}_train_inputs.csv", data['train_inputs'], delimiter=',', header='x,y,t')
            np.savetxt(f"{base_path}_train_outputs.csv", data['train_outputs'], delimiter=',', header='u,v,p')
            np.savetxt(f"{base_path}_test_inputs.csv", data['test_inputs'], delimiter=',', header='x,y,t')
            np.savetxt(f"{base_path}_test_outputs.csv", data['test_outputs'], delimiter=',', header='u,v,p')
            logger.info(f"Datos guardados en formato CSV: {base_path}_*.csv")
            
        else:
            logger.error(f"Formato desconocido: {format}")
            
    def plot_data(self, data, output_dir, time_idx=0):
        """
        Visualiza los datos generados
        
        Args:
            data (dict): Datos generados
            output_dir (str): Directorio para guardar visualizaciones
            time_idx (int): Índice de tiempo a visualizar
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Extraer datos para un tiempo específico
        nx = data['meta']['nx']
        ny = data['meta']['ny']
        X = data['grid']['X'][:, :, time_idx]
        Y = data['grid']['Y'][:, :, time_idx]
        
        # Reconstruir campos para el tiempo seleccionado
        t_selected = data['grid']['t'][time_idx]
        decay = np.exp(-2 * self.nu * t_selected)
        
        u = -np.cos(X) * np.sin(Y) * decay
        v = np.sin(X) * np.cos(Y) * decay
        p = -0.25 * (np.cos(2*X) + np.cos(2*Y)) * decay * decay
        
        # Magnitud de velocidad
        vel_mag = np.sqrt(u**2 + v**2)
        
        # Plot velocity magnitude
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(X, Y, vel_mag, 50, cmap='viridis')
        plt.colorbar(contour, label='Velocity Magnitude')
        plt.quiver(X[::3, ::3], Y[::3, ::3], u[::3, ::3], v[::3, ::3], 
                  scale=25, color='white', alpha=0.7)
        plt.title(f'Taylor-Green Vortex at t={t_selected:.3f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'{output_dir}/taylor_green_velocity_t{time_idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot pressure
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(X, Y, p, 50, cmap='coolwarm')
        plt.colorbar(contour, label='Pressure')
        plt.streamplot(X, Y, u, v, color='white', linewidth=0.5, density=1.5, arrowsize=0.5)
        plt.title(f'Taylor-Green Pressure at t={t_selected:.3f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'{output_dir}/taylor_green_pressure_t{time_idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizaciones guardadas en {output_dir}")


class KovasznayDataGenerator:
    """Generador de datos sintéticos para el flujo de Kovasznay"""
    
    def __init__(self, re=40):
        """
        Inicializa el generador de datos para Kovasznay
        
        Args:
            re (float): Número de Reynolds
        """
        self.re = re
        
    def generate_data(self, nx=100, ny=50, x_range=(-0.5, 1.0), y_range=(-0.5, 1.5)):
        """
        Genera datos sintéticos para el flujo de Kovasznay
        
        Args:
            nx (int): Número de puntos en dirección x
            ny (int): Número de puntos en dirección y
            x_range (tuple): Rango de coordenada x
            y_range (tuple): Rango de coordenada y
            
        Returns:
            dict: Diccionario con datos sintéticos
        """
        logger.info(f"Generando {nx}×{ny} puntos para flujo de Kovasznay")
        
        # Calcular lambda
        lmbd = 0.5 * self.re - np.sqrt(0.25 * (self.re**2) + 4 * (np.pi**2))
        
        # Generar mallas de coordenadas
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Vectorizar para cálculos eficientes
        x_flat = X.flatten()
        y_flat = Y.flatten()
        
        # Calcular solución analítica
        u = 1 - np.exp(lmbd * x_flat) * np.cos(2 * np.pi * y_flat)
        v = (lmbd / (2 * np.pi)) * np.exp(lmbd * x_flat) * np.sin(2 * np.pi * y_flat)
        p = 0.5 * (1 - np.exp(2 * lmbd * x_flat))
        
        # Crear tensores de entrada y salida
        inputs = np.column_stack([x_flat, y_flat])
        outputs = np.column_stack([u, v, p])
        
        # Dividir en conjuntos de entrenamiento y prueba (80/20)
        num_samples = inputs.shape[0]
        indices = np.random.permutation(num_samples)
        train_size = int(0.8 * num_samples)
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        train_inputs = inputs[train_indices]
        train_outputs = outputs[train_indices]
        test_inputs = inputs[test_indices]
        test_outputs = outputs[test_indices]
        
        # Crear diccionario con todos los datos
        data = {
            'inputs': inputs,
            'outputs': outputs,
            'train_inputs': train_inputs,
            'train_outputs': train_outputs,
            'test_inputs': test_inputs,
            'test_outputs': test_outputs,
            'grid': {
                'x': x,
                'y': y,
                'X': X,
                'Y': Y,
            },
            'meta': {
                're': self.re,
                'lambda': lmbd,
                'nx': nx,
                'ny': ny
            }
        }
        
        return data
    
    def save_data(self, data, path, format='hdf5'):
        """
        Guarda los datos generados en un archivo
        
        Args:
            data (dict): Datos generados
            path (str): Ruta donde guardar los datos
            format (str): Formato ('hdf5', 'npz', 'csv')
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if format == 'hdf5':
            with h5py.File(path, 'w') as f:
                # Guardar inputs/outputs
                f.create_dataset('inputs', data=data['inputs'])
                f.create_dataset('outputs', data=data['outputs'])
                f.create_dataset('train_inputs', data=data['train_inputs'])
                f.create_dataset('train_outputs', data=data['train_outputs'])
                f.create_dataset('test_inputs', data=data['test_inputs'])
                f.create_dataset('test_outputs', data=data['test_outputs'])
                
                # Guardar grid
                grid = f.create_group('grid')
                for key, value in data['grid'].items():
                    grid.create_dataset(key, data=value)
                
                # Guardar metadatos
                meta = f.create_group('meta')
                for key, value in data['meta'].items():
                    meta.attrs[key] = value
                    
            logger.info(f"Datos guardados en formato HDF5: {path}")
                
        elif format == 'npz':
            np.savez(path, **data)
            logger.info(f"Datos guardados en formato NPZ: {path}")
            
        elif format == 'csv':
            base_path = os.path.splitext(path)[0]
            np.savetxt(f"{base_path}_inputs.csv", data['inputs'], delimiter=',', header='x,y')
            np.savetxt(f"{base_path}_outputs.csv", data['outputs'], delimiter=',', header='u,v,p')
            np.savetxt(f"{base_path}_train_inputs.csv", data['train_inputs'], delimiter=',', header='x,y')
            np.savetxt(f"{base_path}_train_outputs.csv", data['train_outputs'], delimiter=',', header='u,v,p')
            np.savetxt(f"{base_path}_test_inputs.csv", data['test_inputs'], delimiter=',', header='x,y')
            np.savetxt(f"{base_path}_test_outputs.csv", data['test_outputs'], delimiter=',', header='u,v,p')
            logger.info(f"Datos guardados en formato CSV: {base_path}_*.csv")
            
        else:
            logger.error(f"Formato desconocido: {format}")
            
    def plot_data(self, data, output_dir):
        """
        Visualiza los datos generados
        
        Args:
            data (dict): Datos generados
            output_dir (str): Directorio para guardar visualizaciones
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Extraer datos
        X = data['grid']['X']
        Y = data['grid']['Y']
        lmbd = data['meta']['lambda']
        
        # Reconstruir campos
        u = 1 - np.exp(lmbd * X) * np.cos(2 * np.pi * Y)
        v = (lmbd / (2 * np.pi)) * np.exp(lmbd * X) * np.sin(2 * np.pi * Y)
        p = 0.5 * (1 - np.exp(2 * lmbd * X))
        
        # Magnitud de velocidad
        vel_mag = np.sqrt(u**2 + v**2)
        
        # Plot velocity magnitude
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(X, Y, vel_mag, 50, cmap='viridis')
        plt.colorbar(contour, label='Velocity Magnitude')
        plt.quiver(X[::5, ::5], Y[::5, ::5], u[::5, ::5], v[::5, ::5], 
                  scale=10, color='white', alpha=0.7)
        plt.title(f'Kovasznay Flow (Re={self.re})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'{output_dir}/kovasznay_velocity.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot pressure
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(X, Y, p, 50, cmap='coolwarm')
        plt.colorbar(contour, label='Pressure')
        plt.streamplot(X, Y, u, v, color='white', linewidth=0.5, density=1.5, arrowsize=0.5)
        plt.title(f'Kovasznay Pressure (Re={self.re})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'{output_dir}/kovasznay_pressure.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizaciones guardadas en {output_dir}")


class CavityFlowDataGenerator:
    """Generador de datos sintéticos para el flujo en cavidad"""
    
    def __init__(self, re=100):
        """
        Inicializa el generador de datos para flujo en cavidad
        
        Args:
            re (float): Número de Reynolds
        """
        self.re = re
        
    def generate_data(self, n=50):
        """
        Genera datos sintéticos para el flujo en cavidad
        En lugar de una solución analítica (que no existe para este problema),
        usamos una solución aproximada basada en series de polinomios.
        
        Args:
            n (int): Número de puntos en cada dirección
            
        Returns:
            dict: Diccionario con datos sintéticos
        """
        logger.info(f"Generando {n}×{n} puntos para flujo en cavidad")
        
        # Generar malla uniforme
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Vectorizar para cálculos eficientes
        x_flat = X.flatten()
        y_flat = Y.flatten()
        
        # Crear aproximación para u
        def u_approx(x, y):
            # Aproximación polinómica que satisface u=1 en y=1 y u=0 en bordes
            return x * (1-x)**2 * y**2 * (2*y-1) * 16
        
        # Crear aproximación para v
        def v_approx(x, y):
            # Aproximación polinómica que satisface v=0 en todos los bordes
            return -y * (1-y)**2 * x**2 * (1-x) * 16
        
        # Aproximación para presión (gradiente consistente con u y v)
        def p_approx(x, y, re=self.re):
            # Presión normalizada entre 0 y 1
            p_center = np.sin(np.pi * x) * np.sin(np.pi * y)
            return p_center / re
        
        # Calcular campos
        u = u_approx(x_flat, y_flat)
        v = v_approx(x_flat, y_flat)
        p = p_approx(x_flat, y_flat)
        
        # Crear tensores de entrada y salida
        inputs = np.column_stack([x_flat, y_flat])
        outputs = np.column_stack([u, v, p])
        
        # Dividir en conjuntos de entrenamiento y prueba (80/20)
        num_samples = inputs.shape[0]
        indices = np.random.permutation(num_samples)
        train_size = int(0.8 * num_samples)
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        train_inputs = inputs[train_indices]
        train_outputs = outputs[train_indices]
        test_inputs = inputs[test_indices]
        test_outputs = outputs[test_indices]
        
        # Crear datos para condiciones de frontera (importantes para PINN)
        # Borde superior: y=1 con velocidad u=1, v=0
        top_x = np.linspace(0, 1, n)
        top_y = np.ones_like(top_x)
        top_u = np.ones_like(top_x)
        top_v = np.zeros_like(top_x)
        top_p = p_approx(top_x, top_y)
        top_inputs = np.column_stack([top_x, top_y])
        top_outputs = np.column_stack([top_u, top_v, top_p])
        
        # Otros bordes: u=0, v=0
        # Borde inferior: y=0
        bottom_x = np.linspace(0, 1, n)
        bottom_y = np.zeros_like(bottom_x)
        bottom_inputs = np.column_stack([bottom_x, bottom_y])
        bottom_outputs = np.column_stack([
            np.zeros_like(bottom_x), 
            np.zeros_like(bottom_x),
            p_approx(bottom_x, bottom_y)
        ])
        
        # Borde izquierdo: x=0
        left_y = np.linspace(0, 1, n)
        left_x = np.zeros_like(left_y)
        left_inputs = np.column_stack([left_x, left_y])
        left_outputs = np.column_stack([
            np.zeros_like(left_y), 
            np.zeros_like(left_y),
            p_approx(left_x, left_y)
        ])
        
        # Borde derecho: x=1
        right_y = np.linspace(0, 1, n)
        right_x = np.ones_like(right_y)
        right_inputs = np.column_stack([right_x, right_y])
        right_outputs = np.column_stack([
            np.zeros_like(right_y), 
            np.zeros_like(right_y),
            p_approx(right_x, right_y)
        ])
        
        # Crear diccionario con todos los datos
        data = {
            'inputs': inputs,
            'outputs': outputs,
            'train_inputs': train_inputs,
            'train_outputs': train_outputs,
            'test_inputs': test_inputs,
            'test_outputs': test_outputs,
            'boundary_inputs': np.vstack([top_inputs, bottom_inputs, left_inputs, right_inputs]),
            'boundary_outputs': np.vstack([top_outputs, bottom_outputs, left_outputs, right_outputs]),
            'grid': {
                'x': x,
                'y': y,
                'X': X,
                'Y': Y,
            },
            'meta': {
                're': self.re,
                'n': n
            }
        }
        
        return data
    
    def save_data(self, data, path, format='hdf5'):
        """
        Guarda los datos generados en un archivo
        
        Args:
            data (dict): Datos generados
            path (str): Ruta donde guardar los datos
            format (str): Formato ('hdf5', 'npz', 'csv')
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if format == 'hdf5':
            with h5py.File(path, 'w') as f:
                # Guardar inputs/outputs
                f.create_dataset('inputs', data=data['inputs'])
                f.create_dataset('outputs', data=data['outputs'])
                f.create_dataset('train_inputs', data=data['train_inputs'])
                f.create_dataset('train_outputs', data=data['train_outputs'])
                f.create_dataset('test_inputs', data=data['test_inputs'])
                f.create_dataset('test_outputs', data=data['test_outputs'])
                f.create_dataset('boundary_inputs', data=data['boundary_inputs'])
                f.create_dataset('boundary_outputs', data=data['boundary_outputs'])
                
                # Guardar grid
                grid = f.create_group('grid')
                for key, value in data['grid'].items():
                    grid.create_dataset(key, data=value)
                
                # Guardar metadatos
                meta = f.create_group('meta')
                for key, value in data['meta'].items():
                    meta.attrs[key] = value
                    
            logger.info(f"Datos guardados en formato HDF5: {path}")
                
        elif format == 'npz':
            np.savez(path, **data)
            logger.info(f"Datos guardados en formato NPZ: {path}")
            
        elif format == 'csv':
            base_path = os.path.splitext(path)[0]
            np.savetxt(f"{base_path}_inputs.csv", data['inputs'], delimiter=',', header='x,y')
            np.savetxt(f"{base_path}_outputs.csv", data['outputs'], delimiter=',', header='u,v,p')
            np.savetxt(f"{base_path}_train_inputs.csv", data['train_inputs'], delimiter=',', header='x,y')
            np.savetxt(f"{base_path}_train_outputs.csv", data['train_outputs'], delimiter=',', header='u,v,p')
            np.savetxt(f"{base_path}_test_inputs.csv", data['test_inputs'], delimiter=',', header='x,y')
            np.savetxt(f"{base_path}_test_outputs.csv", data['test_outputs'], delimiter=',', header='u,v,p')
            np.savetxt(f"{base_path}_boundary_inputs.csv", data['boundary_inputs'], delimiter=',', header='x,y')
            np.savetxt(f"{base_path}_boundary_outputs.csv", data['boundary_outputs'], delimiter=',', header='u,v,p')
            logger.info(f"Datos guardados en formato CSV: {base_path}_*.csv")
            
        else:
            logger.error(f"Formato desconocido: {format}")
            
    def plot_data(self, data, output_dir):
        """
        Visualiza los datos generados
        
        Args:
            data (dict): Datos generados
            output_dir (str): Directorio para guardar visualizaciones
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Extraer datos de la malla
        X = data['grid']['X']
        Y = data['grid']['Y']
        n = data['meta']['n']
        
        # Reconstruir campos
        u = np.reshape(data['outputs'][:, 0], (n, n))
        v = np.reshape(data['outputs'][:, 1], (n, n))
        p = np.reshape(data['outputs'][:, 2], (n, n))
        
        # Magnitud de velocidad
        vel_mag = np.sqrt(u**2 + v**2)
        
        # Plot velocity magnitude
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(X, Y, vel_mag, 50, cmap='viridis')
        plt.colorbar(contour, label='Velocity Magnitude')
        plt.quiver(X[::3, ::3], Y[::3, ::3], u[::3, ::3], v[::3, ::3], 
                  scale=5, color='white', alpha=0.7)
        plt.title(f'Cavity Flow (Re={self.re})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'{output_dir}/cavity_flow_velocity.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot pressure
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(X, Y, p, 50, cmap='coolwarm')
        plt