import os
import sys
# Aseguramos que la ruta raíz "code" esté en sys.path
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

import time
import torch
import logging

import numpy as np
from tqdm import tqdm
from losses.diferential_equation_loss import burgers_pde_loss
from utils.device_utils import get_dynamic_batch_size, monitor_device_usage
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
from tqdm import tqdm
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
if logger.hasHandlers():
    logger.handlers.clear()
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

class Trainer:
    def __init__(self, model, optimizer, num_epochs, batch_size, memory_limit_gb,
                 early_stop_patience=50, min_loss_improvement=0.001, verbose=False):
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.initial_batch_size = batch_size
        self.memory_limit_gb = memory_limit_gb
        self.verbose = verbose
        
        # Variables para early stopping
        self.early_stop_patience = early_stop_patience
        self.min_loss_improvement = min_loss_improvement
        self.best_loss = float("inf")
        self.no_improve_count = 0

        # Historial
        self.loss_history = None
        self.accuracy_history = None
        self.epoch_time_history = None

    def _check_early_stop(self, current_loss):
        if current_loss < self.best_loss - self.min_loss_improvement:
            self.best_loss = current_loss
            self.no_improve_count = 0
            if self.verbose:
                print(f"Mejora detectada: nuevo loss {self.best_loss:.4f}. Reiniciando contador.")
        else:
            self.no_improve_count += 1
            if self.verbose:
                print(f"Sin mejora: contador {self.no_improve_count}/{self.early_stop_patience}.")
        return self.no_improve_count >= self.early_stop_patience

    def calculate_accuracy(self, features, targets) -> float:
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(*features) if isinstance(features, (tuple, list)) else self.model(features)
            if predictions.shape != targets.shape:
                targets = targets.view_as(predictions)
            denominator = torch.clamp(torch.abs(targets), min=1e-8)
            error = torch.abs(predictions - targets) / denominator
            error = torch.clamp(error, min=0.0, max=1.0)
            accuracy = 1 - torch.mean(error)
        return max(0.0, accuracy.item())

    def train(self, x_train, t_train, u_train, nu):
        start_time = time.time()
        self.loss_history = np.zeros(self.num_epochs)
        self.accuracy_history = np.zeros(self.num_epochs)
        self.epoch_time_history = np.zeros(self.num_epochs)

        logger.info(f"Usando batch size: {self.initial_batch_size}")
        device_load = monitor_device_usage()
        batch_size = get_dynamic_batch_size(device_load, self.initial_batch_size, self.memory_limit_gb)

        with tqdm(total=self.num_epochs, desc="Entrenando la PINN", dynamic_ncols=True) as pbar:
            for epoch in range(self.num_epochs):
                epoch_start = time.time()
                inputs = torch.cat((x_train.reshape(-1, 1), t_train.reshape(-1, 1)), dim=1)
                targets = u_train.reshape(-1, 1)
                permutation = torch.randperm(inputs.size(0))
                batch_loss = 0.0
                num_batches = 0
                for i in range(0, inputs.size(0), batch_size):
                    indices = permutation[i:i+batch_size]
                    batch_inputs = inputs[indices]
                    batch_x = batch_inputs[:, 0].unsqueeze(1).clone().detach().requires_grad_(True)
                    batch_t = batch_inputs[:, 1].unsqueeze(1).clone().detach().requires_grad_(True)
                    batch_targets = targets[indices]

                    self.optimizer.zero_grad()
                    pred = self.model(batch_x, batch_t)
                    loss_data = torch.mean((pred - batch_targets) ** 2)
                    loss_pde = burgers_pde_loss(self.model, batch_x, batch_t, nu)
                    loss = loss_data + loss_pde
                    loss.backward()
                    self.optimizer.step()

                    batch_loss += loss.item()
                    num_batches += 1

                avg_loss = batch_loss / num_batches
                self.loss_history[epoch] = avg_loss
                accuracy = self.calculate_accuracy((x_train, t_train), u_train)
                self.accuracy_history[epoch] = accuracy
                self.epoch_time_history[epoch] = time.time() - epoch_start

                epoch_time = time.time() - epoch_start
                estimated_time = (epoch_time * (self.num_epochs - epoch)) / 60
                pbar.set_postfix({
                    "Loss": f"{avg_loss:.6f}",
                    "Acc": f"{accuracy:.6f}",
                    "ETC": f"{estimated_time:.2f}m"
                })
                pbar.update(1)

                if self._check_early_stop(avg_loss):
                    logger.info(f"Early stopping activado en la época {epoch}")
                    break

        total_time = time.time() - start_time
        logger.info(f"Entrenamiento finalizado en {total_time/60:.2f} minutos")
        final_loss = self.loss_history[np.nonzero(self.loss_history)[0][-1]] if np.any(self.loss_history != 0) else 0.0
        final_accuracy = self.accuracy_history[np.nonzero(self.accuracy_history)[0][-1]] if np.any(self.accuracy_history != 0) else 0.0
        logger.info(f"Pérdida final: {final_loss:.4f}")
        logger.info(f"Precisión final: {final_accuracy:.4f}")
        return self.loss_history, self.accuracy_history, self.epoch_time_history, total_time


