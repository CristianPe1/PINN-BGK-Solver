import matplotlib.pyplot as plt
import torch
import numpy as np
import os

import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, output_dir, logger):
        self.output_dir = output_dir
        self.logger = logger

    def plot_metrics(self, losses, accuracies, epoch_times, learning_rates=None):
        """
        Visualiza las métricas de entrenamiento.
        
        Args:
            losses: Lista de pérdidas por época
            accuracies: Lista de precisiones por época
            epoch_times: Lista de tiempos por época
            learning_rates: Lista de tasas de aprendizaje por época (opcional)
        """
        if learning_rates is not None:
            # Crear gráfico con 3x2 subplots
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        else:
            # Usar diseño original de 2x3
            fig, axs = plt.subplots(2, 3, figsize=(15, 7))

        # Primera fila de gráficas
        axs[0, 0].plot(losses, label="Pérdida")
        axs[0, 0].set_xlabel("Época")
        axs[0, 0].set_ylabel("Pérdida")
        axs[0, 0].set_title("Pérdida vs Época")
        axs[0, 0].grid(True)

        axs[0, 1].plot(accuracies, label="Precisión")
        axs[0, 1].set_xlabel("Época")
        axs[0, 1].set_ylabel("Precisión (%)")
        axs[0, 1].set_title("Precisión vs Época")
        axs[0, 1].grid(True)

        axs[0, 2].plot(epoch_times, label="Tiempo")
        axs[0, 2].set_xlabel("Época")
        axs[0, 2].set_ylabel("Tiempo (s)")
        axs[0, 2].set_title("Tiempo vs Época")
        axs[0, 2].grid(True)

        accumulated_time = np.cumsum(epoch_times)

        # Segunda fila de gráficas
        axs[1, 0].plot(accumulated_time, losses)
        axs[1, 0].set_xlabel("Tiempo (s)")
        axs[1, 0].set_ylabel("Pérdida")
        axs[1, 0].set_title("Pérdida vs Tiempo")
        axs[1, 0].grid(True)

        axs[1, 1].plot(accumulated_time, accuracies)
        axs[1, 1].set_xlabel("Tiempo (s)")
        axs[1, 1].set_ylabel("Precisión")
        axs[1, 1].set_title("Precisión vs Tiempo")
        axs[1, 1].grid(True)

        if learning_rates is not None:
            # Gráfica de learning rate
            axs[1, 2].plot(learning_rates, marker='o', markersize=3)
            axs[1, 2].set_xlabel("Época")
            axs[1, 2].set_ylabel("Learning Rate")
            axs[1, 2].set_title("Learning Rate vs Época")
            axs[1, 2].set_yscale('log')  # Escala logarítmica para LR
            axs[1, 2].grid(True)
        else:
            # Original: Precisión vs Pérdida
            axs[1, 2].plot(losses, accuracies)
            axs[1, 2].set_xlabel("Pérdida")
            axs[1, 2].set_ylabel("Precisión")
            axs[1, 2].set_title("Precisión vs Pérdida")
            axs[1, 2].grid(True)

        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        plt.savefig(os.path.join(self.output_dir, "metrics_plot.png"))
        self.logger.info("Gráficas de métricas guardadas.")
        plt.close()

    def plot_solution(self, t_train, x_train, u_pred, u_train, filename="solution_comparison.png"):
        """
        Grafica la solución predicha y la solución real.

        Args:
            t_train: tensor de tiempo
            x_train: tensor de posición
            u_pred: solución predicha por el modelo
            u_train: solución real
            filename: nombre del archivo para guardar la gráfica
        """

        

        plt.figure(figsize=(15, 5))

        # Solución predicha
        plt.subplot(121)
        plt.pcolormesh(t_train, x_train, u_pred, shading="auto", cmap="viridis")
        plt.colorbar(label="u(x,t)")
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title("Solución PINN")

        # Solución exacta
        plt.subplot(122)
        plt.pcolormesh(t_train, x_train, u_train, shading="auto", cmap="viridis")
        plt.colorbar(label="u(x,t)")
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title("Solución exacta")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        self.logger.info(f"Imagen de comparación guardada: {filename}")
        plt.close()

    def plot_error_heatmap(self, t, x, error, filename="error_heatmap.png"):
        """
        Visualiza el error entre la predicción y el valor real como un mapa de calor.
        
        Args:
            t: Array de valores de tiempo
            x: Array de valores de posición
            error: Matriz de error (diferencia absoluta entre predicción y valor real)
            filename: Nombre del archivo para guardar la visualización
        """
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(t, x, error, shading='auto', cmap='hot')
        plt.colorbar(label="Error absoluto")
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title("Mapa de calor del error")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        self.logger.info(f"Gráfico de error guardado como {filename}")
        plt.close()

    def plot_prediction_vs_true(self, y_pred, y_true, filename="prediction_vs_true.png"):
        """
        Grafica una comparación simple entre los valores predichos y reales.
        
        Args:
            y_pred (array): Valores predichos por el modelo
            y_true (array): Valores reales
            filename (str): Nombre del archivo para guardar la visualización
        """
        plt.figure(figsize=(10, 8))
        
        # Graficar valores reales vs. predichos
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Graficar línea de referencia y=x
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel("Valores reales")
        plt.ylabel("Valores predichos")
        plt.title("Comparación de valores reales vs. predichos")
        plt.grid(True)
        
        # Guardar figura
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, filename)
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        self.logger.info(f"Gráfico de comparación guardado en: {output_file}")
        return output_file

    def plot_fluid_solution(self, X, Y, u_pred, v_pred, p_pred=None, 
                       u_true=None, v_true=None, p_true=None, 
                       filename="fluid_solution.png"):
        """
        Visualiza la solución de un problema de fluidos 2D con manejo mejorado para las líneas de corriente.
        
        Args:
            X, Y: Mallas de coordenadas
            u_pred, v_pred, p_pred: Campos predichos (velocidad x, velocidad y, presión)
            u_true, v_true, p_true: Campos reales (opcional)
            filename: Nombre del archivo para guardar
        """
        self.logger.info("Generando visualización de fluido...")
        
        # Convertir tensores a numpy si es necesario
        def to_numpy(tensor):
            if isinstance(tensor, torch.Tensor):
                return tensor.detach().cpu().numpy()
            return tensor if tensor is not None else None
        
        X = to_numpy(X)
        Y = to_numpy(Y)
        u_pred = to_numpy(u_pred)
        v_pred = to_numpy(v_pred)
        p_pred = to_numpy(p_pred)
        u_true = to_numpy(u_true)
        v_true = to_numpy(v_true)
        p_true = to_numpy(p_true)
        
        # Determinar el número de filas (1 si no hay ground truth, 2 si lo hay)
        has_true = u_true is not None and v_true is not None
        rows = 2 if has_true else 1
        cols = 3 if p_pred is not None else 2  # Columnas: u, v, y opcionalmente p
        
        # Crear figura con tamaño proporcional al número de subplots
        plt.figure(figsize=(cols * 5, rows * 5))
        
        # Preparar malla adecuada para streamplot
        # Streamplot requiere que X varíe a lo largo de columnas e Y a lo largo de filas
        # y que los valores de X en cada fila sean idénticos, y los de Y en cada columna sean idénticos
        
        # Extraer valores únicos de X e Y
        try:
            x_unique = np.unique(X.flatten())
            y_unique = np.unique(Y.flatten())
            X_grid, Y_grid = np.meshgrid(x_unique, y_unique)
            
            # Crear interpoladores para todos los campos
            from scipy.interpolate import griddata
            points = np.column_stack((X.flatten(), Y.flatten()))
            
            # Interpolar campos de predicción
            u_pred_grid = griddata(points, u_pred.flatten(), (X_grid, Y_grid), method='linear')
            v_pred_grid = griddata(points, v_pred.flatten(), (X_grid, Y_grid), method='linear')
            
            if p_pred is not None:
                p_pred_grid = griddata(points, p_pred.flatten(), (X_grid, Y_grid), method='linear')
            
            # Interpolar campos reales si existen
            if has_true:
                u_true_grid = griddata(points, u_true.flatten(), (X_grid, Y_grid), method='linear')
                v_true_grid = griddata(points, v_true.flatten(), (X_grid, Y_grid), method='linear')
                if p_true is not None:
                    p_true_grid = griddata(points, p_true.flatten(), (X_grid, Y_grid), method='linear')
            
            # Informar éxito
            self.logger.info(f"Interpolación de malla exitosa, nueva forma: {X_grid.shape}")
            
        except Exception as e:
            # Si la interpolación falla, usar los arrays originales
            self.logger.warning(f"Error en la interpolación: {e}. Usando malla original.")
            X_grid, Y_grid = X, Y
            u_pred_grid, v_pred_grid = u_pred, v_pred
            p_pred_grid = p_pred
            if has_true:
                u_true_grid, v_true_grid = u_true, v_true
                p_true_grid = p_true if p_true is not None else None
        
        # Predicción - Velocidad U
        plt.subplot(rows, cols, 1)
        plt.pcolormesh(X_grid, Y_grid, u_pred_grid, cmap='viridis', shading='auto')
        plt.colorbar(label='u')
        plt.title('Velocidad U - Predicción')
        plt.xlabel('x')
        plt.ylabel('y')
        
        # Predicción - Velocidad V
        plt.subplot(rows, cols, 2)
        plt.pcolormesh(X_grid, Y_grid, v_pred_grid, cmap='viridis', shading='auto')
        plt.colorbar(label='v')
        plt.title('Velocidad V - Predicción')
        plt.xlabel('x')
        plt.ylabel('y')
        
        # Predicción - Presión P (si está disponible)
        if p_pred is not None:
            plt.subplot(rows, cols, 3)
            plt.pcolormesh(X_grid, Y_grid, p_pred_grid, cmap='coolwarm', shading='auto')
            plt.colorbar(label='p')
            plt.title('Presión - Predicción')
            plt.xlabel('x')
            plt.ylabel('y')
        
        # Campo de velocidad combinado (predicción)
        plt.subplot(rows, cols, cols)
        
        try:
            # Verificar que no hay valores NaN
            if np.isnan(u_pred_grid).any() or np.isnan(v_pred_grid).any():
                raise ValueError("Hay valores NaN en los campos de velocidad")
            
            # Calcular magnitud para el fondo
            speed = np.sqrt(u_pred_grid**2 + v_pred_grid**2)
            
            # Crear pcolormesh con la magnitud de velocidad
            plt.pcolormesh(X_grid, Y_grid, speed, cmap='viridis', shading='auto')
            plt.colorbar(label='|V|')
            
            # Añadir líneas de corriente
            # Streamplot es muy sensible a la forma de los datos - usar skip para reducir densidad si hay errores
            skip = max(1, min(X_grid.shape) // 25)  # Skip para reducir tamaño si la malla es muy densa
            plt.streamplot(
                X_grid[::skip, ::skip], 
                Y_grid[::skip, ::skip], 
                u_pred_grid[::skip, ::skip], 
                v_pred_grid[::skip, ::skip], 
                density=1.0, 
                color='black', 
                linewidth=1.0, 
                arrowsize=1.0
            )
            
        except Exception as e:
            self.logger.warning(f"Error al generar streamplot: {e}. Usando quiver (flechas) como alternativa.")
            # Usar quiver (flechas) como alternativa segura
            skip = max(1, min(X_grid.shape) // 15)
            plt.quiver(
                X_grid[::skip, ::skip], 
                Y_grid[::skip, ::skip],
                u_pred_grid[::skip, ::skip], 
                v_pred_grid[::skip, ::skip],
                scale=25, 
                color='black', 
                width=0.002
            )
            # También mostrar la magnitud de velocidad
            speed = np.sqrt(u_pred_grid**2 + v_pred_grid**2)
            plt.pcolormesh(X_grid, Y_grid, speed, cmap='viridis', shading='auto')
            plt.colorbar(label='|V|')
        
        plt.title('Campo de Velocidad - Predicción')
        plt.xlabel('x')
        plt.ylabel('y')
        
        # Si hay datos reales, graficarlos en la segunda fila
        if has_true:
            # Real - Velocidad U
            plt.subplot(rows, cols, cols + 1)
            plt.pcolormesh(X_grid, Y_grid, u_true_grid, cmap='viridis', shading='auto')
            plt.colorbar(label='u')
            plt.title('Velocidad U - Real')
            plt.xlabel('x')
            plt.ylabel('y')
            
            # Real - Velocidad V
            plt.subplot(rows, cols, cols + 2)
            plt.pcolormesh(X_grid, Y_grid, v_true_grid, cmap='viridis', shading='auto')
            plt.colorbar(label='v')
            plt.title('Velocidad V - Real')
            plt.xlabel('x')
            plt.ylabel('y')
            
            # Real - Presión P (si está disponible)
            if p_true is not None and cols > 2:
                plt.subplot(rows, cols, cols + 3)
                plt.pcolormesh(X_grid, Y_grid, p_true_grid, cmap='coolwarm', shading='auto')
                plt.colorbar(label='p')
                plt.title('Presión - Real')
                plt.xlabel('x')
                plt.ylabel('y')
            
            # Campo de velocidad combinado (real)
            plt.subplot(rows, cols, 2*cols)
            try:
                # Verificar que no hay valores NaN
                if np.isnan(u_true_grid).any() or np.isnan(v_true_grid).any():
                    raise ValueError("Hay valores NaN en los campos de velocidad reales")
                
                # Calcular magnitud para el fondo
                speed_true = np.sqrt(u_true_grid**2 + v_true_grid**2)
                plt.pcolormesh(X_grid, Y_grid, speed_true, cmap='viridis', shading='auto')
                plt.colorbar(label='|V|')
                
                # Añadir líneas de corriente con misma configuración que en la predicción
                skip = max(1, min(X_grid.shape) // 25)
                plt.streamplot(
                    X_grid[::skip, ::skip], 
                    Y_grid[::skip, ::skip], 
                    u_true_grid[::skip, ::skip], 
                    v_true_grid[::skip, ::skip], 
                    density=1.0, 
                    color='black', 
                    linewidth=1.0, 
                    arrowsize=1.0
                )
                
            except Exception as e:
                self.logger.warning(f"Error al generar streamplot para datos reales: {e}. Usando quiver como alternativa.")
                # Usar quiver como alternativa
                skip = max(1, min(X_grid.shape) // 15)
                plt.quiver(
                    X_grid[::skip, ::skip], 
                    Y_grid[::skip, ::skip],
                    u_true_grid[::skip, ::skip], 
                    v_true_grid[::skip, ::skip],
                    scale=25, 
                    color='black', 
                    width=0.002
                )
                # También mostrar la magnitud de velocidad
                speed_true = np.sqrt(u_true_grid**2 + v_true_grid**2)
                plt.pcolormesh(X_grid, Y_grid, speed_true, cmap='viridis', shading='auto')
                plt.colorbar(label='|V|')
            
            plt.title('Campo de Velocidad - Real')
            plt.xlabel('x')
            plt.ylabel('y')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        self.logger.info(f"Visualización de fluido guardada en: {save_path}")
