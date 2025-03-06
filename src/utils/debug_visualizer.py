"""
Módulo para depuración visual - Genera visualizaciones adicionales para resolución de problemas
"""
import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging.getLogger(__name__)

class DebugVisualizer:
    """
    Clase para crear visualizaciones de depuración para problemas con los modelos
    o las predicciones. Genera gráficos más detallados que la clase Visualizer estándar.
    """
    
    def __init__(self, output_dir):
        """
        Inicializa el visualizador de depuración
        
        Args:
            output_dir (str): Directorio donde se guardarán las visualizaciones
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Directorio creado: {output_dir}")
    
    def tensor_to_numpy(self, tensor):
        """Convierte un tensor PyTorch a NumPy array"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def generate_detailed_visualization(self, input_data, prediction, ground_truth=None, 
                                       physics_type="unknown", filename="detailed_viz.png"):
        """
        Genera visualizaciones detalladas para un problema específico
        
        Args:
            input_data (tensor): Datos de entrada al modelo
            prediction (tensor): Predicción del modelo
            ground_truth (tensor, opcional): Datos reales/verdaderos
            physics_type (str): Tipo de problema físico
            filename (str): Nombre del archivo para guardar
            
        Returns:
            str: Ruta del archivo guardado
        """
        logger.info(f"Generando visualización detallada para {physics_type}...")
        
        # Convertir datos a numpy
        input_np = self.tensor_to_numpy(input_data)
        pred_np = self.tensor_to_numpy(prediction)
        gt_np = self.tensor_to_numpy(ground_truth) if ground_truth is not None else None
        
        # Crear una figura principal
        plt.figure(figsize=(16, 12))
        plt.suptitle(f"Visualización Detallada - {physics_type.upper()}", fontsize=16)
        
        # Mostrar información sobre las formas
        info_text = f"Formas:\n"
        info_text += f"- Entrada: {input_np.shape}\n"
        info_text += f"- Predicción: {pred_np.shape}\n"
        if gt_np is not None:
            info_text += f"- Ground Truth: {gt_np.shape}\n"
        plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                   bbox=dict(facecolor='yellow', alpha=0.2))
        
        # Dependiendo del problema, generar visualización específica
        if physics_type in ["taylor_green", "kovasznay", "cavity_flow"]:
            # Para problemas de fluidos 2D (asumimos entrada como X, Y)
            if input_np.shape[1] >= 2:
                # Extraer coordenadas X, Y
                if input_np.ndim == 2:
                    x = input_np[:, 0]
                    y = input_np[:, 1]
                    
                    # Determinar si necesitamos reconstruir la malla 2D
                    n_points = len(x)
                    nx = int(np.sqrt(n_points))
                    ny = n_points // nx
                    
                    # Si no es un cuadrado perfecto, usar una malla regular
                    if nx * ny != n_points:
                        nx = ny = int(np.sqrt(n_points))
                        
                    logger.info(f"Reconstruyendo malla 2D: {nx} x {ny}")
                    
                    # Crear malla uniformemente espaciada
                    x_unique = np.linspace(x.min(), x.max(), nx)
                    y_unique = np.linspace(y.min(), y.max(), ny)
                    X, Y = np.meshgrid(x_unique, y_unique)
                    
                    # Comprobar cuántas componentes tiene la predicción
                    comp_count = pred_np.shape[1] if pred_np.ndim > 1 else 1
                    
                    if comp_count >= 2:
                        # Flujo 2D con componentes u, v
                        u_pred = pred_np[:, 0].reshape(nx, ny)
                        v_pred = pred_np[:, 1].reshape(nx, ny)
                        
                        # Ground truth si está disponible
                        if gt_np is not None:
                            gt_comp = gt_np.shape[1] if gt_np.ndim > 1 else 1
                            if gt_comp >= 2:
                                u_true = gt_np[:, 0].reshape(nx, ny)
                                v_true = gt_np[:, 1].reshape(nx, ny)
                            else:
                                u_true = gt_np.reshape(nx, ny)
                                v_true = None
                        else:
                            u_true = v_true = None
                        
                        # Visualizar velocidades
                        ax1 = plt.subplot(2, 3, 1)
                        im1 = ax1.pcolormesh(X, Y, u_pred, shading='auto', cmap='viridis')
                        plt.colorbar(im1, ax=ax1, label='Velocidad U - Predicción')
                        ax1.set_title("Componente U - Predicción")
                        
                        ax2 = plt.subplot(2, 3, 2)
                        im2 = ax2.pcolormesh(X, Y, v_pred, shading='auto', cmap='viridis')
                        plt.colorbar(im2, ax=ax2, label='Velocidad V - Predicción')
                        ax2.set_title("Componente V - Predicción")
                        
                        # Magnitud y líneas de corriente
                        speed = np.sqrt(u_pred**2 + v_pred**2)
                        ax3 = plt.subplot(2, 3, 3)
                        im3 = ax3.pcolormesh(X, Y, speed, shading='auto', cmap='viridis')
                        plt.colorbar(im3, ax=ax3, label='|V|')
                        ax3.set_title("Magnitud Velocidad - Predicción")
                        
                        try:
                            # Intentar añadir líneas de corriente
                            ax3.streamplot(X, Y, u_pred, v_pred, density=1.5, color='black')
                        except Exception as e:
                            logger.warning(f"No se pudo generar streamplot: {e}")
                            # Usar quiver en su lugar
                            skip = max(1, min(nx, ny) // 15)
                            ax3.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                                     u_pred[::skip, ::skip], v_pred[::skip, ::skip],
                                     scale=25, color='black')
                        
                        # Visualizar verdad si disponible
                        if u_true is not None:
                            ax4 = plt.subplot(2, 3, 4)
                            im4 = ax4.pcolormesh(X, Y, u_true, shading='auto', cmap='viridis')
                            plt.colorbar(im4, ax=ax4, label='Velocidad U - Real')
                            ax4.set_title("Componente U - Real")
                            
                            if v_true is not None:
                                ax5 = plt.subplot(2, 3, 5)
                                im5 = ax5.pcolormesh(X, Y, v_true, shading='auto', cmap='viridis')
                                plt.colorbar(im5, ax=ax5, label='Velocidad V - Real')
                                ax5.set_title("Componente V - Real")
                                
                                # Magnitud y líneas de corriente
                                speed_true = np.sqrt(u_true**2 + v_true**2)
                                ax6 = plt.subplot(2, 3, 6)
                                im6 = ax6.pcolormesh(X, Y, speed_true, shading='auto', cmap='viridis')
                                plt.colorbar(im6, ax=ax6, label='|V|')
                                ax6.set_title("Magnitud Velocidad - Real")
                                
                                try:
                                    # Intentar añadir líneas de corriente
                                    ax6.streamplot(X, Y, u_true, v_true, density=1.5, color='black')
                                except Exception as e:
                                    logger.warning(f"No se pudo generar streamplot para datos reales: {e}")
                                    # Usar quiver en su lugar
                                    skip = max(1, min(nx, ny) // 15)
                                    ax6.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                                            u_true[::skip, ::skip], v_true[::skip, ::skip],
                                            scale=25, color='black')
                    else:
                        # Solo una componente
                        pred_reshaped = pred_np.reshape(nx, ny)
                        
                        ax1 = plt.subplot(1, 2, 1)
                        im1 = ax1.pcolormesh(X, Y, pred_reshaped, shading='auto', cmap='viridis')
                        plt.colorbar(im1, ax=ax1)
                        ax1.set_title("Predicción")
                        
                        if gt_np is not None:
                            gt_reshaped = gt_np.reshape(nx, ny)
                            ax2 = plt.subplot(1, 2, 2)
                            im2 = ax2.pcolormesh(X, Y, gt_reshaped, shading='auto', cmap='viridis')
                            plt.colorbar(im2, ax=ax2)
                            ax2.set_title("Real")
            else:
                # No podemos manejar este caso específicamente
                plt.text(0.5, 0.5, f"No se pudo visualizar datos de entrada con forma {input_np.shape}",
                        ha='center', va='center', fontsize=14)
                
        # Para problemas 1D como Burgers
        elif physics_type == "burgers":
            # En Burgers típicamente input es (x, t) y output es u(x, t)
            if input_np.shape[1] == 2:  # (x, t)
                x = input_np[:, 0]
                t = input_np[:, 1]
                
                # Determinar dimensiones de la malla
                x_unique = np.unique(x.round(decimals=8))
                t_unique = np.unique(t.round(decimals=8))
                nx = len(x_unique)
                nt = len(t_unique)
                
                logger.info(f"Burgers: Detectados {nx} puntos x y {nt} puntos t")
                
                if nx * nt == len(x):  # Malla regular
                    # Reshape predictions to 2D grid
                    u_pred = pred_np.reshape(nx, nt)
                    u_true = gt_np.reshape(nx, nt) if gt_np is not None else None
                    
                    # Crear malla 2D
                    X, T = np.meshgrid(x_unique, t_unique, indexing='ij')
                    
                    plt.subplot(1, 2, 1)
                    plt.pcolormesh(T, X, u_pred, shading='auto', cmap='viridis')
                    plt.colorbar(label='u(x,t)')
                    plt.title('Solución Predicha')
                    plt.xlabel('t')
                    plt.ylabel('x')
                    
                    if u_true is not None:
                        plt.subplot(1, 2, 2)
                        plt.pcolormesh(T, X, u_true, shading='auto', cmap='viridis')
                        plt.colorbar(label='u(x,t)')
                        plt.title('Solución Real')
                        plt.xlabel('t')
                        plt.ylabel('x')
                else:
                    # Datos no están en malla regular
                    plt.subplot(1, 2, 1)
                    plt.scatter(t, x, c=pred_np, cmap='viridis', s=10)
                    plt.colorbar(label='u(x,t)')
                    plt.title('Predicción (Puntos Dispersos)')
                    plt.xlabel('t')
                    plt.ylabel('x')
                    
                    if gt_np is not None:
                        plt.subplot(1, 2, 2)
                        plt.scatter(t, x, c=gt_np, cmap='viridis', s=10)
                        plt.colorbar(label='u(x,t)')
                        plt.title('Real (Puntos Dispersos)')
                        plt.xlabel('t')
                        plt.ylabel('x')
        else:
            # Caso genérico - mostrar histogramas y dispersión
            plt.subplot(2, 2, 1)
            plt.hist(pred_np.flatten(), bins=50, alpha=0.7, color='blue')
            plt.title("Histograma de Predicciones")
            plt.grid(True)
            
            if gt_np is not None:
                plt.subplot(2, 2, 2)
                plt.hist(gt_np.flatten(), bins=50, alpha=0.7, color='green')
                plt.title("Histograma de Valores Reales")
                plt.grid(True)
                
                # Scatter plot de predicción vs real
                plt.subplot(2, 2, 3)
                plt.scatter(gt_np.flatten(), pred_np.flatten(), s=10, alpha=0.5)
                plt.plot([gt_np.min(), gt_np.max()], [gt_np.min(), gt_np.max()], 'r--')
                plt.title("Predicción vs Real")
                plt.xlabel("Real")
                plt.ylabel("Predicción")
                plt.grid(True)
                
                # Error
                if gt_np.shape == pred_np.shape:
                    plt.subplot(2, 2, 4)
                    error = np.abs(gt_np - pred_np)
                    plt.hist(error.flatten(), bins=50, alpha=0.7, color='red')
                    plt.title(f"Histograma de Error (MAE={error.mean():.4f})")
                    plt.grid(True)
        
        # Ajustar layout y guardar
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Visualización detallada guardada en: {save_path}")
        return save_path
    
    def debug_taylor_green(self, X, Y, u_pred, v_pred, p_pred=None, 
                           u_true=None, v_true=None, p_true=None,
                           filename="taylor_green_debug.png"):
        """
        Visualización específica para el flujo de Taylor-Green con verificaciones detalladas
        
        Args:
            X, Y: Mallas de coordenadas
            u_pred, v_pred: Componentes de velocidad predichas
            p_pred: Componente de presión predicha (opcional)
            u_true, v_true: Componentes de velocidad reales (opcional) 
            p_true: Componente de presión real (opcional)
            filename: Nombre del archivo a guardar
            
        Returns:
            str: Ruta al archivo guardado
        """
        logger.info("Generando visualización específica para Taylor-Green...")
        
        # Convertir entradas a numpy
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return x
        
        X = to_numpy(X)
        Y = to_numpy(Y)
        u_pred = to_numpy(u_pred)
        v_pred = to_numpy(v_pred)
        p_pred = to_numpy(p_pred)
        u_true = to_numpy(u_true)
        v_true = to_numpy(v_true)
        p_true = to_numpy(p_true)
        
        # Verificar dimensiones
        logger.info(f"Shapes - X:{X.shape}, Y:{Y.shape}, u_pred:{u_pred.shape}, v_pred:{v_pred.shape}")
        if u_true is not None:
            logger.info(f"Shapes - u_true:{u_true.shape}, v_true:{v_true.shape}")
        
        # Crear figura
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle("Visualización de Depuración - Taylor-Green", fontsize=16)
        
        # Agregar información sobre formas
        shapes_text = (f"X: {X.shape}, Y: {Y.shape}\n"
                      f"u_pred: {u_pred.shape}, v_pred: {v_pred.shape}\n")
        
        if u_true is not None:
            shapes_text += f"u_true: {u_true.shape}, v_true: {v_true.shape}\n"
        
        if p_pred is not None:
            shapes_text += f"p_pred: {p_pred.shape}"
            if p_true is not None:
                shapes_text += f", p_true: {p_true.shape}"
        
        plt.figtext(0.02, 0.02, shapes_text, fontsize=10, 
                   bbox=dict(facecolor='yellow', alpha=0.2))
        
        # Determinar distribución de subplots
        has_true = u_true is not None
        has_pressure = p_pred is not None
        
        # Primera fila - componentes de velocidad predichas
        ax1 = plt.subplot(2 if has_true else 1, 3, 1)
        im1 = ax1.pcolormesh(X, Y, u_pred, shading='auto', cmap='viridis')
        plt.colorbar(im1, ax=ax1)
        ax1.set_title("U - Predicción")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        
        ax2 = plt.subplot(2 if has_true else 1, 3, 2)
        im2 = ax2.pcolormesh(X, Y, v_pred, shading='auto', cmap='viridis')
        plt.colorbar(im2, ax=ax2)
        ax2.set_title("V - Predicción")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        
        # Tercer panel - magnitud de velocidad y líneas de corriente
        ax3 = plt.subplot(2 if has_true else 1, 3, 3)
        speed = np.sqrt(u_pred**2 + v_pred**2)
        im3 = ax3.pcolormesh(X, Y, speed, shading='auto', cmap='viridis')
        plt.colorbar(im3, ax=ax3, label='|V|')
        ax3.set_title("Magnitud Velocidad - Predicción")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        
        # Intentar añadir líneas de corriente
        try:
            # Verificar que X e Y tengan la forma apropiada para streamplot
            if X.ndim == 2 and Y.ndim == 2:
                # Crear una malla regular para streamplot
                rows_equal = np.allclose(X[0], X[-1], rtol=1e-5)
                cols_equal = np.allclose(Y[:,0], Y[:,-1], rtol=1e-5)
                
                if not rows_equal or not cols_equal:
                    # Si no tiene filas/cols iguales, crear malla regular
                    x_1d = np.linspace(np.min(X), np.max(X), 50)
                    y_1d = np.linspace(np.min(Y), np.max(Y), 50)
                    X_reg, Y_reg = np.meshgrid(x_1d, y_1d)
                    
                    # Interpolar velocidades a la malla regular
                    from scipy.interpolate import griddata
                    points = np.column_stack((X.flatten(), Y.flatten()))
                    
                    u_reg = griddata(points, u_pred.flatten(), (X_reg, Y_reg), method='linear')
                    v_reg = griddata(points, v_pred.flatten(), (X_reg, Y_reg), method='linear')
                    
                    # Streamplot en la malla regular
                    ax3.streamplot(X_reg, Y_reg, u_reg, v_reg, density=1.5, color='black')
                    logger.info("Streamplot creado con malla regularizada")
                else:
                    # Si ya es regular, usar directamente
                    ax3.streamplot(X, Y, u_pred, v_pred, density=1.5, color='black')
                    logger.info("Streamplot creado con malla original")
            else:
                logger.warning(f"X e Y deben ser 2D para streamplot. Shapes: X={X.shape}, Y={Y.shape}")
        except Exception as e:
            logger.error(f"Error al crear streamplot: {e}")
            # Usar quiver en su lugar
            try:
                skip = max(1, min(X.shape) // 15)
                ax3.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                         u_pred[::skip, ::skip], v_pred[::skip, ::skip],
                         scale=25, color='black')
                logger.info(f"Quiver creado como alternativa con skip={skip}")
            except Exception as e2:
                logger.error(f"Error al crear quiver: {e2}")
        
        # Segunda fila - datos reales si están disponibles
        if has_true:
            ax4 = plt.subplot(2, 3, 4)
            im4 = ax4.pcolormesh(X, Y, u_true, shading='auto', cmap='viridis')
            plt.colorbar(im4, ax=ax4)
            ax4.set_title("U - Real")
            ax4.set_xlabel("x")
            ax4.set_ylabel("y")
            
            ax5 = plt.subplot(2, 3, 5)
            im5 = ax5.pcolormesh(X, Y, v_true, shading='auto', cmap='viridis')
            plt.colorbar(im5, ax=ax5)
            ax5.set_title("V - Real")
            ax5.set_xlabel("x")
            ax5.set_ylabel("y")
            
            ax6 = plt.subplot(2, 3, 6)
            speed_true = np.sqrt(u_true**2 + v_true**2)
            im6 = ax6.pcolormesh(X, Y, speed_true, shading='auto', cmap='viridis')
            plt.colorbar(im6, ax=ax6, label='|V|')
            ax6.set_title("Magnitud Velocidad - Real")
            ax6.set_xlabel("x")
            ax6.set_ylabel("y")
            
            # Intentar añadir líneas de corriente
            try:
                if X.ndim == 2 and Y.ndim == 2:
                    # Usar la misma lógica que antes
                    rows_equal = np.allclose(X[0], X[-1], rtol=1e-5)
                    cols_equal = np.allclose(Y[:,0], Y[:,-1], rtol=1e-5)
                    
                    if not rows_equal or not cols_equal:
                        # Si no tiene filas/cols iguales, usar la malla regular previa
                        x_1d = np.linspace(np.min(X), np.max(X), 50)
                        y_1d = np.linspace(np.min(Y), np.max(Y), 50)
                        X_reg, Y_reg = np.meshgrid(x_1d, y_1d)
                        
                        # Interpolar velocidades a la malla regular
                        from scipy.interpolate import griddata
                        points = np.column_stack((X.flatten(), Y.flatten()))
                        
                        u_reg_true = griddata(points, u_true.flatten(), (X_reg, Y_reg), method='linear')
                        v_reg_true = griddata(points, v_true.flatten(), (X_reg, Y_reg), method='linear')
                        
                        # Streamplot en la malla regular
                        ax6.streamplot(X_reg, Y_reg, u_reg_true, v_reg_true, density=1.5, color='black')
                    else:
                        # Si ya es regular, usar directamente
                        ax6.streamplot(X, Y, u_true, v_true, density=1.5, color='black')
            except Exception as e:
                logger.error(f"Error al crear streamplot para datos reales: {e}")
                try:
                    skip = max(1, min(X.shape) // 15)
                    ax6.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                             u_true[::skip, ::skip], v_true[::skip, ::skip],
                             scale=25, color='black')
                except Exception as e2:
                    logger.error(f"Error al crear quiver para datos reales: {e2}")
        
        # Ajustar layout y guardar
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Crear archivo adicional con visualización alterna
        self._create_alternative_visualization(X, Y, u_pred, v_pred, u_true, v_true,
                                            os.path.join(self.output_dir, f"alt_{filename}"))
        
        logger.info(f"Visualización de depuración guardada en: {save_path}")
        return save_path
    
    def _create_alternative_visualization(self, X, Y, u_pred, v_pred, u_true=None, v_true=None, filename="alt_viz.png"):
        """Crea visualización adicional usando contorno y quiver"""
        plt.figure(figsize=(12, 10))
        
        # Simplificar la malla
        if X.size > 1000:
            skip = max(1, X.size // 1000)
            X_simple = X[::skip, ::skip]
            Y_simple = Y[::skip, ::skip]
            u_pred_simple = u_pred[::skip, ::skip]
            v_pred_simple = v_pred[::skip, ::skip]
            if u_true is not None:
                u_true_simple = u_true[::skip, ::skip]
                v_true_simple = v_true[::skip, ::skip]
        else:
            X_simple, Y_simple = X, Y
            u_pred_simple, v_pred_simple = u_pred, v_pred
            if u_true is not None:
                u_true_simple, v_true_simple = u_true, v_true
        
        # Velocidad con contornos
        plt.subplot(2, 2, 1)
        speed = np.sqrt(u_pred**2 + v_pred**2)
        plt.contourf(X, Y, speed, levels=20, cmap='viridis')
        plt.colorbar(label='|V|')
        plt.title("Magnitud de Velocidad - Predicción")
        plt.xlabel("x")
        plt.ylabel("y")
        
        # Velocidad con quiver
        plt.subplot(2, 2, 2)
        plt.quiver(X_simple, Y_simple, u_pred_simple, v_pred_simple, speed[::skip, ::skip], 
                  scale=25, cmap='viridis')
        plt.colorbar(label='|V|')
        plt.title("Campo de Velocidad - Predicción")
        plt.xlabel("x")
        plt.ylabel("y")
        
        # Si hay datos reales, visualizarlos también
        if u_true is not None:
            plt.subplot(2, 2, 3)
            speed_true = np.sqrt(u_true**2 + v_true**2)
            plt.contourf(X, Y, speed_true, levels=20, cmap='viridis')
            plt.colorbar(label='|V|')
            plt.title("Magnitud de Velocidad - Real")
            plt.xlabel("x")
            plt.ylabel("y")
            
            plt.subplot(2, 2, 4)
            plt.quiver(X_simple, Y_simple, u_true_simple, v_true_simple, speed_true[::skip, ::skip], 
                      scale=25, cmap='viridis')
            plt.colorbar(label='|V|')
            plt.title("Campo de Velocidad - Real")
            plt.xlabel("x")
            plt.ylabel("y")
        
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Visualización alternativa guardada en: {filename}")
