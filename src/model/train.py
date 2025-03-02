import torch
import numpy as np
import time
import logging
import sys
from tqdm import tqdm

# Configuración del logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Trainer:
    """
    Clase para entrenamiento de modelos PINN con soporte para early stopping,
    monitoreo de memoria y visualización de progreso.
    """
    def __init__(self, model, optimizer, epochs=1000, batch_size=32, 
                memory_limit_gb=4, scheduler=None, early_stop_params=None,
                loss_fn=None, loss_kwargs=None, verbose=False):
        """
        Inicializa el entrenador con modelo, optimizador y parámetros de entrenamiento.
        
        Args:
            model: Modelo PyTorch a entrenar
            optimizer: Optimizador PyTorch
            epochs: Número máximo de épocas
            batch_size: Tamaño del batch para entrenamiento
            memory_limit_gb: Límite de memoria en GB
            scheduler: Learning rate scheduler (opcional)
            early_stop_params: Diccionario con parámetros de early stopping
                {
                    "enabled": True/False,
                    "patience": épocas a esperar,
                    "min_improvement": mejora mínima requerida
                }
            loss_fn: Función de pérdida
            loss_kwargs: Parámetros adicionales para la función de pérdida
            verbose: Si debe mostrar progreso detallado
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.batch_size = batch_size
        self.memory_limit_gb = memory_limit_gb
        
        # Configurar early stopping
        if early_stop_params is None:
            early_stop_params = {"enabled": True, "patience": 50, "min_improvement": 1e-5}
            
        self.early_stop_enabled = early_stop_params.get("enabled", True)
        self.early_stop_patience = early_stop_params.get("patience", 50)
        self.min_loss_improvement = early_stop_params.get("min_improvement", 1e-5)
        
        # Función de pérdida
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.MSELoss()
        self.loss_kwargs = loss_kwargs if loss_kwargs is not None else {}
        
        self.verbose = verbose
        self.device = next(model.parameters()).device
        
        # Configurar el logger si es necesario
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
            
    def train(self, input_tensor, output_tensor, **loss_kwargs):
        """
        Entrena el modelo usando los tensores de entrada y salida proporcionados.
        
        Args:
            input_tensor: Tensor de entrada [batch_size, input_dim]
            output_tensor: Tensor objetivo [batch_size, output_dim]
            **loss_kwargs: Parámetros adicionales para la función de pérdida
            
        Returns:
            tuple: (losses, accuracies, epoch_times, total_time)
        """
        logger.info(f"Iniciando entrenamiento con {self.epochs} épocas, batch_size={self.batch_size}")
        
        # Verificar que los tensores estén en el dispositivo correcto
        if input_tensor.device != self.device:
            input_tensor = input_tensor.to(self.device)
        if output_tensor.device != self.device:
            output_tensor = output_tensor.to(self.device)
            
        # Inicializar listas para métricas
        losses = []
        accuracies = []
        epoch_times = []
        learning_rates = []  # Para seguimiento de learning rates
        best_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # Combinar loss kwargs con los proporcionados en el init
        all_loss_kwargs = {**self.loss_kwargs, **loss_kwargs}
        
        # Iniciar cronómetro
        start_time = time.time()
        
        # Crear barra de progreso si verbose es True
        pbar = tqdm(range(self.epochs)) if self.verbose else range(self.epochs)
        
        # Bucle de épocas
        for epoch in pbar:
            epoch_start = time.time()
            
            # Early stopping check
            if self.early_stop_enabled and patience_counter >= self.early_stop_patience:
                logger.info(f"Early stopping en época {epoch}: sin mejora en {self.early_stop_patience} épocas")
                # Restaurar el mejor modelo si se guardó
                if best_model_state is not None:
                    self.model.load_state_dict(best_model_state)
                break
                
            # Entrenamiento de esta época
            self.model.train()
            epoch_loss = 0.0
            
            # Número total de muestras
            num_samples = input_tensor.shape[0]
            
            if self.batch_size is None or self.batch_size >= num_samples:
                # Entrenar con todos los datos
                self.optimizer.zero_grad()
                
                # Forward pass directo con el tensor de entrada
                outputs = self.model(input_tensor)
                
                # Calcular pérdida
                loss = self.loss_fn(outputs, output_tensor, **all_loss_kwargs)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss = loss.item()
            else:
                # Entrenamiento por batches
                num_batches = int(np.ceil(num_samples / self.batch_size))
                
                # Crear índices para shuffling
                indices = torch.randperm(num_samples).to(self.device)
                
                # Iterar sobre batches
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min((batch_idx + 1) * self.batch_size, num_samples)
                    batch_indices = indices[start_idx:end_idx]
                    
                    # Preparar batch
                    batch_inputs = input_tensor[batch_indices]
                    batch_targets = output_tensor[batch_indices]
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    batch_outputs = self.model(batch_inputs)
                    
                    # Calcular pérdida
                    batch_loss = self.loss_fn(batch_outputs, batch_targets, **all_loss_kwargs)
                    
                    # Backward pass
                    batch_loss.backward()
                    self.optimizer.step()
                    
                    # Acumular pérdida
                    epoch_loss += batch_loss.item() * len(batch_indices) / num_samples
            
            # Evaluar accuracy (para seguimiento)
            with torch.no_grad():
                self.model.eval()
                outputs = self.model(input_tensor)
                    
                mse = torch.mean((outputs - output_tensor) ** 2)
                accuracy = 1.0 / (1.0 + mse)  # Convertir MSE a una métrica tipo accuracy
            
            # Registrar métricas
            losses.append(epoch_loss)
            accuracies.append(accuracy.item())
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            # Registrar learning rate actual
            current_lr = self.optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            # Aplicar scheduler si existe
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(epoch_loss)
                else:
                    self.scheduler.step()
            
            # Comprobar mejora para early stopping
            if self.early_stop_enabled:
                if epoch_loss < best_loss - self.min_loss_improvement:
                    best_loss = epoch_loss
                    patience_counter = 0
                    # Guardar mejor modelo
                    best_model_state = {k: v.cpu().detach() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
            
            # Actualizar barra de progreso con información
            if self.verbose:
                desc = f"Época {epoch+1}/{self.epochs} | Pérdida: {epoch_loss:.6f} | Acc: {accuracy.item()*100:.2f}% | LR: {current_lr:.6f}"
                if isinstance(pbar, tqdm):
                    pbar.set_description(desc)
            # También mostrar en log periódicamente
            if self.verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                logger.info(f"Época {epoch+1}/{self.epochs}, Pérdida: {epoch_loss:.6f}, "
                           f"Accuracy: {accuracy.item()*100:.2f}%, LR: {current_lr:.6f}, Tiempo: {epoch_time:.2f}s")
        
        # Calcular tiempo total
        total_time = time.time() - start_time
        logger.info(f"Entrenamiento completado en {total_time:.2f}s")
        logger.info(f"Mejor pérdida: {best_loss:.6f}")
        
        # Convertir listas a tensores para facilitar su uso
        losses = torch.tensor(losses)
        accuracies = torch.tensor(accuracies)
        epoch_times = torch.tensor(epoch_times)
        learning_rates = torch.tensor(learning_rates)
        
        return losses, accuracies, epoch_times, total_time, learning_rates


