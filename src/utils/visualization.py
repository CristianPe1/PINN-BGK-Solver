import matplotlib.pyplot as plt
import torch
import numpy as np
import os

import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, output_train_dir, logger):
        self.OUTPUT_TRAIN_DIR = output_train_dir
        self.logger = logger

    def plot_metrics(self, losses, accuracies, epoch_times):
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

        # Se asume que t_clean, x_clean, u_pred_clean y u_train_clean son preprocesados a partir de los datos de entrada.
        # Si es necesario, este paso debería implementarse antes de utilizar esta función.
        t_clean = t_train.cpu().detach().numpy() if torch.is_tensor(t_train) else t_train
        x_clean = x_train.cpu().detach().numpy() if torch.is_tensor(x_train) else x_train
        u_pred_clean = u_pred.cpu().detach().numpy() if torch.is_tensor(u_pred) else u_pred
        u_train_clean = u_train.cpu().detach().numpy() if torch.is_tensor(u_train) else u_train

        plt.figure(figsize=(15, 5))

        # Solución predicha
        plt.subplot(121)
        plt.pcolormesh(t_clean, x_clean, u_pred_clean, shading="auto", cmap="viridis")
        plt.colorbar(label="u(x,t)")
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title("Solución PINN")

        # Solución exacta
        plt.subplot(122)
        plt.pcolormesh(t_clean, x_clean, u_train_clean, shading="auto", cmap="viridis")
        plt.colorbar(label="u(x,t)")
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title("Solución exacta")

        plt.tight_layout()
        plt.savefig(os.path.join(self.OUTPUT_TRAIN_DIR, filename))
        self.logger.info(f"Imagen de comparación guardada: {filename}")
        plt.close()
