import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import logging
from scipy.interpolate import griddata

class FluidVisualizer:
    """
    Clase especializada para visualizar soluciones de fluidos, con manejo especial
    para cada tipo de flujo (Taylor-Green, Kovasznay, Cavity Flow).
    """
    
    def __init__(self, output_dir=None, logger=None):
        """
        Inicializa el visualizador de fluidos.
        
        Args:
            output_dir (str): Directorio para guardar visualizaciones
            logger (logging.Logger): Logger para mensajes
        """
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Directorio creado: {output_dir}")
    
    def _to_numpy(self, tensor):
        """Convierte tensor a numpy si es necesario"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor if tensor is not None else None
    
    def plot_taylor_green_flow(self, X, Y, u_pred, v_pred, p_pred=None, 
                              u_true=None, v_true=None, p_true=None,
                              filename="taylor_green_solution.png"):
        """
        Visualiza un flujo Taylor-Green que normalmente está definido en un dominio periódico.
        
        Args:
            X, Y: Mallas de coordenadas (2D)
            u_pred, v_pred, p_pred: Componentes de velocidad y presión predichas
            u_true, v_true, p_true: Componentes de velocidad y presión reales (opcional)
            filename: Nombre del archivo para guardar
        """
        self.logger.info("Generando visualización específica para flujo Taylor-Green...")
        
        # Convertir todos los tensores a numpy
        X = self._to_numpy(X)
        Y = self._to_numpy(Y)
        u_pred = self._to_numpy(u_pred)
        v_pred = self._to_numpy(v_pred)
        p_pred = self._to_numpy(p_pred)
        u_true = self._to_numpy(u_true)
        v_true = self._to_numpy(v_true)
        p_true = self._to_numpy(p_true)
        
        # Determinar si tenemos datos reales para comparación
        has_true = u_true is not None and v_true is not None
        
        # Verificar la forma de los arrays y prepararlos para visualización
        if X.ndim == 1 or Y.ndim == 1:
            self.logger.info("Creando malla 2D a partir de arrays 1D")
            X_grid, Y_grid = np.meshgrid(X.flatten(), Y.flatten())
        else:
            X_grid, Y_grid = X, Y
        
        # Verificar si u y v necesitan reshaping
        if u_pred.ndim == 1:
            n = int(np.sqrt(len(u_pred)))
            u_pred = u_pred.reshape(n, n)
            v_pred = v_pred.reshape(n, n)
            if p_pred is not None:
                p_pred = p_pred.reshape(n, n)
            if has_true:
                if u_true.ndim == 1:
                    u_true = u_true.reshape(n, n)
                    v_true = v_true.reshape(n, n)
                    if p_true is not None:
                        p_true = p_true.reshape(n, n)
        
        # Asegurarse de que las formas coincidan para evitar errores
        if X_grid.shape != u_pred.shape:
            self.logger.warning(f"Las dimensiones no coinciden: X:{X_grid.shape}, u:{u_pred.shape}")
            # Interpolar a una malla uniforme
            x_unique = np.linspace(X_grid.min(), X_grid.max(), 100)
            y_unique = np.linspace(Y_grid.min(), Y_grid.max(), 100)
            X_uniform, Y_uniform = np.meshgrid(x_unique, y_unique)
            
            # Interpolar campos
            points = np.column_stack((X_grid.flatten(), Y_grid.flatten()))
            u_pred_uniform = griddata(points, u_pred.flatten(), (X_uniform, Y_uniform), method='cubic')
            v_pred_uniform = griddata(points, v_pred.flatten(), (X_uniform, Y_uniform), method='cubic')
            
            if p_pred is not None:
                p_pred_uniform = griddata(points, p_pred.flatten(), (X_uniform, Y_uniform), method='cubic')
            else:
                p_pred_uniform = None
            
            if has_true:
                u_true_uniform = griddata(points, u_true.flatten(), (X_uniform, Y_uniform), method='cubic')
                v_true_uniform = griddata(points, v_true.flatten(), (X_uniform, Y_uniform), method='cubic')
                if p_true is not None:
                    p_true_uniform = griddata(points, p_true.flatten(), (X_uniform, Y_uniform), method='cubic')
                else:
                    p_true_uniform = None
            
            # Usar los campos interpolados
            X_grid, Y_grid = X_uniform, Y_uniform
            u_pred, v_pred = u_pred_uniform, v_pred_uniform
            p_pred = p_pred_uniform
            if has_true:
                u_true, v_true = u_true_uniform, v_true_uniform
                p_true = p_true_uniform
        
        # Configurar tamaño de figura
        ncols = 3 if p_pred is not None else 2
        nrows = 2 if has_true else 1
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5))
        
        # Si solo hay una fila, convertir axs a un array 2D
        if nrows == 1:
            axs = np.array([axs])
        
        # Visualizar campos de velocidad predichos
        # Velocidad U
        im0 = axs[0, 0].pcolormesh(X_grid, Y_grid, u_pred, cmap='viridis', shading='auto')
        axs[0, 0].set_title('Velocidad U - Predicción')
        axs[0, 0].set_xlabel('x')
        axs[0, 0].set_ylabel('y')
        fig.colorbar(im0, ax=axs[0, 0])
        
        # Velocidad V
        im1 = axs[0, 1].pcolormesh(X_grid, Y_grid, v_pred, cmap='viridis', shading='auto')
        axs[0, 1].set_title('Velocidad V - Predicción')
        axs[0, 1].set_xlabel('x')
        axs[0, 1].set_ylabel('y')
        fig.colorbar(im1, ax=axs[0, 1])
        
        # Campo de velocidad (magnitud y líneas de corriente)
        if ncols > 2:
            # Presión
            im2 = axs[0, 2].pcolormesh(X_grid, Y_grid, p_pred, cmap='coolwarm', shading='auto')
            axs[0, 2].set_title('Presión - Predicción')
            axs[0, 2].set_xlabel('x')
            axs[0, 2].set_ylabel('y')
            fig.colorbar(im2, ax=axs[0, 2])
        
        # Visualizar campos de velocidad reales si están disponibles
        if has_true:
            # Velocidad U
            im3 = axs[1, 0].pcolormesh(X_grid, Y_grid, u_true, cmap='viridis', shading='auto')
            axs[1, 0].set_title('Velocidad U - Real')
            axs[1, 0].set_xlabel('x')
            axs[1, 0].set_ylabel('y')
            fig.colorbar(im3, ax=axs[1, 0])
            
            # Velocidad V
            im4 = axs[1, 1].pcolormesh(X_grid, Y_grid, v_true, cmap='viridis', shading='auto')
            axs[1, 1].set_title('Velocidad V - Real')
            axs[1, 1].set_xlabel('x')
            axs[1, 1].set_ylabel('y')
            fig.colorbar(im4, ax=axs[1, 1])
            
            # Presión o campo de velocidad
            if ncols > 2 and p_true is not None:
                im5 = axs[1, 2].pcolormesh(X_grid, Y_grid, p_true, cmap='coolwarm', shading='auto')
                axs[1, 2].set_title('Presión - Real')
                axs[1, 2].set_xlabel('x')
                axs[1, 2].set_ylabel('y')
                fig.colorbar(im5, ax=axs[1, 2])
        
        # Añadir líneas de corriente en subplots adicionales
        fig2, axs2 = plt.subplots(1, 2 if has_true else 1, figsize=(10, 5))
        
        # Si solo hay una columna, convertir axs2 a array para indexación consistente
        if not has_true:
            axs2 = np.array([axs2])
        
        # Líneas de corriente para predicción
        try:
            # Crear una malla regular para streamplot (requisito de Matplotlib)
            nx = 50  # Número de puntos para visualización
            ny = 50
            x_reg = np.linspace(X_grid.min(), X_grid.max(), nx)
            y_reg = np.linspace(Y_grid.min(), Y_grid.max(), ny)
            X_reg, Y_reg = np.meshgrid(x_reg, y_reg)
            
            # Interpolar campos de velocidad a esta malla
            points = np.column_stack((X_grid.flatten(), Y_grid.flatten()))
            u_reg = griddata(points, u_pred.flatten(), (X_reg, Y_reg), method='linear')
            v_reg = griddata(points, v_pred.flatten(), (X_reg, Y_reg), method='linear')
            
            # Velocidad combinada para predicción
            speed = np.sqrt(u_reg**2 + v_reg**2)
            axs2[0].streamplot(X_reg, Y_reg, u_reg, v_reg, density=1.2, color='black')
            im6 = axs2[0].pcolormesh(X_reg, Y_reg, speed, cmap='viridis', alpha=0.7, shading='auto')
            axs2[0].set_title('Líneas de Corriente - Predicción')
            fig2.colorbar(im6, ax=axs2[0])
            
            # Velocidad combinada para datos reales
            if has_true:
                u_true_reg = griddata(points, u_true.flatten(), (X_reg, Y_reg), method='linear')
                v_true_reg = griddata(points, v_true.flatten(), (X_reg, Y_reg), method='linear')
                speed_true = np.sqrt(u_true_reg**2 + v_true_reg**2)
                axs2[1].streamplot(X_reg, Y_reg, u_true_reg, v_true_reg, density=1.2, color='black')
                im7 = axs2[1].pcolormesh(X_reg, Y_reg, speed_true, cmap='viridis', alpha=0.7, shading='auto')
                axs2[1].set_title('Líneas de Corriente - Real')
                fig2.colorbar(im7, ax=axs2[1])
        except Exception as e:
            self.logger.error(f"Error al generar líneas de corriente: {str(e)}")
            # En caso de error, mostrar un mensaje
            axs2[0].text(0.5, 0.5, 'Error al generar líneas de corriente', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=axs2[0].transAxes)
        
        # Ajustar layout y guardar
        fig.tight_layout()
        fig2.tight_layout()
        
        # Guardar figuras
        if self.output_dir:
            fig_path = os.path.join(self.output_dir, filename)
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            
            # Guardar la segunda figura con líneas de corriente
            streamlines_path = os.path.join(self.output_dir, f"streamlines_{filename}")
            fig2.savefig(streamlines_path, dpi=300, bbox_inches='tight')
            
            self.logger.info(f"Visualizaciones guardadas en:\n- {fig_path}\n- {streamlines_path}")
        else:
            # Si no hay directorio de salida, mostrar las figuras
            plt.show()
        
        # Cerrar figuras para liberar memoria
        plt.close(fig)
        plt.close(fig2)
        
        return True

    def visualize_flow(self, flow_type, X, Y, u_pred, v_pred, p_pred=None,
                      u_true=None, v_true=None, p_true=None, filename=None):
        """
        Método principal que selecciona la visualización apropiada según el tipo de flujo.
        
        Args:
            flow_type: Tipo de flujo ('taylor_green', 'kovasznay', 'cavity_flow')
            X, Y: Coordenadas
            u_pred, v_pred, p_pred: Campos predichos
            u_true, v_true, p_true: Campos reales (opcional)
            filename: Nombre base para guardar los archivos
        
        Returns:
            bool: True si la visualización se realizó correctamente
        """
        if filename is None:
            filename = f"{flow_type}_solution.png"
            
        if flow_type == 'taylor_green':
            return self.plot_taylor_green_flow(X, Y, u_pred, v_pred, p_pred,
                                             u_true, v_true, p_true, filename)
        else:
            # Para otros tipos de flujo, usar la visualización estándar
            from .visualization import Visualizer
            visualizer = Visualizer(self.output_dir, self.logger)
            visualizer.plot_fluid_solution(X, Y, u_pred, v_pred, p_pred,
                                          u_true, v_true, p_true, filename)
            return True
