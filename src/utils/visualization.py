import matplotlib.pyplot as plt
import torch
import numpy as np
import os

import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, output_train_dir, logger):
        self.OUTPUT_TRAIN_DIR = output_train_dir
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
        plt.savefig(os.path.join(self.OUTPUT_TRAIN_DIR, "metrics_plot.png"))
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
        plt.savefig(os.path.join(self.OUTPUT_TRAIN_DIR, filename))
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
        plt.savefig(os.path.join(self.OUTPUT_TRAIN_DIR, filename), dpi=300)
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
        output_file = os.path.join(self.OUTPUT_TRAIN_DIR, filename)
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        self.logger.info(f"Gráfico de comparación guardado en: {output_file}")
        return output_file
