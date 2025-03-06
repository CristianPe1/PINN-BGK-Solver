import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..base_model import BaseNeuralNetwork

class LatticeBoltzmannBase(BaseNeuralNetwork):
    """
    Clase base para modelos Physics-Informed de Lattice Boltzmann.
    Extiende la funcionalidad de BaseNeuralNetwork para operadores de colisión en LBM.
    """
    def __init__(self, layer_sizes=None, activation_name="ReLU", name="LatticeBoltzmannBase"):
        """
        Inicializa un modelo de LatticeBoltzmann.
        
        Args:
            layer_sizes (list): Lista de tamaños de capas. Si es None, se usa [9, 50, 50, 9] por defecto
            activation_name (str): Nombre de función de activación
            name (str): Nombre del modelo
        """
        # Configurar dimensiones por defecto si no se proporciona
        if layer_sizes is None:
            layer_sizes = [9, 50, 50, 9]  # Por defecto para LBM D2Q9
            
        super(LatticeBoltzmannBase, self).__init__(
            layer_sizes=layer_sizes,
            activation_name=activation_name,
            name=name
        )
        
        # Velocidades discretas para LBM D2Q9
        self.velocities = torch.tensor([
            [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], 
            [1, 1], [-1, 1], [-1, -1], [1, -1]
        ], dtype=torch.float32)
        
        # Pesos de equilibrio para D2Q9
        self.weights = torch.tensor([
            4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36
        ], dtype=torch.float32)
        
    def forward(self, x):
        """
        Propagación hacia adelante básica (sin características especiales).
        
        Args:
            x (torch.Tensor): Función de distribución f_i de entrada [batch, 9]
            
        Returns:
            torch.Tensor: Función de distribución post-colisión
        """
        # Implementación básica
        for i, layer in enumerate(self.linear_layers):
            if i < len(self.linear_layers) - 1:
                x = self.activation(layer(x))
            else:
                x = layer(x)
        return x
    
    def compute_macroscopic(self, f):
        """
        Calcula variables macroscópicas (densidad, velocidad) a partir de las distribuciones.
        
        Args:
            f (torch.Tensor): Distribuciones [batch, 9]
            
        Returns:
            tuple: (densidad, velocidad_x, velocidad_y)
        """
        # Densidad: suma de todas las distribuciones
        rho = torch.sum(f, dim=1, keepdim=True)
        
        # Momento: suma de distribuciones ponderadas por velocidades
        momentum_x = torch.sum(f * self.velocities[:, 0], dim=1, keepdim=True)
        momentum_y = torch.sum(f * self.velocities[:, 1], dim=1, keepdim=True)
        
        # Velocidad: momento dividido por densidad
        u_x = momentum_x / (rho + 1e-10)
        u_y = momentum_y / (rho + 1e-10)
        
        return rho, u_x, u_y
    
    def feq(self, rho, u_x, u_y):
        """
        Calcula la distribución de equilibrio.
        
        Args:
            rho (torch.Tensor): Densidad [batch, 1]
            u_x (torch.Tensor): Velocidad en x [batch, 1]
            u_y (torch.Tensor): Velocidad en y [batch, 1]
            
        Returns:
            torch.Tensor: Distribución de equilibrio [batch, 9]
        """
        # Velocidades al cuadrado
        usqr = 3.0 * (u_x**2 + u_y**2)
        
        # Producto escalar entre velocidades discretas y velocidad macroscópica
        cu = 3.0 * (self.velocities[:, 0].view(1, -1) * u_x + 
                    self.velocities[:, 1].view(1, -1) * u_y)
        
        # Distribución de equilibrio
        feq = rho * self.weights.view(1, -1) * (1.0 + cu + 0.5 * cu**2 - 0.5 * usqr.view(-1, 1))
        
        return feq


class NNNaivePINN(LatticeBoltzmannBase):
    """
    Implementación naive del operador de colisión basado en redes neuronales.
    """
    def __init__(self, layer_sizes=None, activation_name="ReLU"):
        super(NNNaivePINN, self).__init__(
            layer_sizes=layer_sizes,
            activation_name=activation_name,
            name="NNNaive"
        )


class NNSymPINN(LatticeBoltzmannBase):
    """
    Operador de colisión con simetría D8 incorporada.
    """
    def __init__(self, layer_sizes=None, activation_name="ReLU"):
        super(NNSymPINN, self).__init__(
            layer_sizes=layer_sizes,
            activation_name=activation_name,
            name="NNSym"
        )
        self.d8_group = self.generate_d8_group()
        
    def generate_d8_group(self):
        """Genera las 8 transformaciones del grupo D8 para D2Q9."""
        # Rotaciones y reflexiones para D2Q9
        # Orden de velocidades: [0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [-1,-1], [1,-1]
        
        # Matriz de permutación para rotación de 90°
        rot90 = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],  # centro permanece igual
            [0, 0, 1, 0, 0, 0, 0, 0, 0],  # [1,0] -> [0,1]
            [0, 0, 0, 0, 1, 0, 0, 0, 0],  # [0,1] -> [0,-1]
            [0, 1, 0, 0, 0, 0, 0, 0, 0],  # [-1,0] -> [1,0]
            [0, 0, 0, 1, 0, 0, 0, 0, 0],  # [0,-1] -> [-1,0]
            [0, 0, 0, 0, 0, 0, 0, 0, 1],  # [1,1] -> [1,-1]
            [0, 0, 0, 0, 0, 1, 0, 0, 0],  # [-1,1] -> [1,1]
            [0, 0, 0, 0, 0, 0, 1, 0, 0],  # [-1,-1] -> [-1,1]
            [0, 0, 0, 0, 0, 0, 0, 1, 0]   # [1,-1] -> [-1,-1]
        ], dtype=torch.float32)
        
        # Matriz de permutación para reflexión en el eje x
        reflect_x = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],  # centro permanece igual
            [0, 1, 0, 0, 0, 0, 0, 0, 0],  # [1,0] -> [1,0]
            [0, 0, 0, 0, 1, 0, 0, 0, 0],  # [0,1] -> [0,-1]
            [0, 0, 0, 1, 0, 0, 0, 0, 0],  # [-1,0] -> [-1,0]
            [0, 0, 1, 0, 0, 0, 0, 0, 0],  # [0,-1] -> [0,1]
            [0, 0, 0, 0, 0, 0, 0, 0, 1],  # [1,1] -> [1,-1]
            [0, 0, 0, 0, 0, 0, 0, 1, 0],  # [-1,1] -> [-1,-1]
            [0, 0, 0, 0, 0, 0, 1, 0, 0],  # [-1,-1] -> [-1,1]
            [0, 0, 0, 0, 0, 1, 0, 0, 0]   # [1,-1] -> [1,1]
        ], dtype=torch.float32)
        
        # Generar todas las transformaciones del grupo D8
        transforms = []
        current = torch.eye(9)
        
        # Rotaciones (0°, 90°, 180°, 270°)
        for _ in range(4):
            transforms.append(current.clone())
            current = torch.matmul(rot90, current)
        
        # Reflexiones (aplicadas a cada rotación)
        for i in range(4):
            transforms.append(torch.matmul(reflect_x, transforms[i]))
            
        return transforms
        
    def forward(self, x):
        """
        Forward con equivarianza D8: aplica el modelo a cada transformación 
        del input y promedia los resultados tras aplicar la transformación inversa.
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Asegurar que las matrices de transformación estén en el mismo dispositivo que los datos
        transforms = [t.to(device) for t in self.d8_group]
        
        # Aplicar el modelo a cada transformación y promediar los resultados
        outputs = []
        for transform in transforms:
            # Aplicar transformación a la entrada
            x_transformed = torch.matmul(x, transform)
            
            # Propagar a través del modelo
            temp = x_transformed
            for i, layer in enumerate(self.linear_layers):
                if i < len(self.linear_layers) - 1:
                    temp = self.activation(layer(temp))
                else:
                    temp = layer(temp)
            
            # Aplicar transformación inversa (transpuesta) al resultado
            output_transformed = torch.matmul(temp, transform.T)
            outputs.append(output_transformed)
        
        # Promediar todos los outputs
        return torch.mean(torch.stack(outputs), dim=0)


class NNConsPINN(LatticeBoltzmannBase):
    """
    Operador de colisión con conservación de masa y momento incorporada.
    """
    def __init__(self, layer_sizes=None, activation_name="ReLU"):
        super(NNConsPINN, self).__init__(
            layer_sizes=layer_sizes,
            activation_name=activation_name,
            name="NNCons"
        )
        self.C = self.build_transformation_matrix()
        
    def build_transformation_matrix(self):
        """Construye la matriz de transformación para conservación."""
        # Matriz que transforma entre momentos y distribuciones
        # Las primeras filas corresponden a los momentos conservados:
        # - Densidad (suma de todas las distribuciones)
        # - Momento en x
        # - Momento en y
        C = torch.zeros(9, 9, dtype=torch.float32)
        
        # Fila para densidad: suma de todas las distribuciones
        C[0, :] = 1.0
        
        # Filas para momentos: producto con velocidades
        for i in range(9):
            C[1, i] = self.velocities[i, 0]  # Momento en x
            C[2, i] = self.velocities[i, 1]  # Momento en y
        
        # Completar con base para el resto del espacio
        # (Estos valores pueden ser optimizados según el problema específico)
        C[3, :] = torch.tensor([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=torch.float32)  # Estrés xx
        C[4, :] = torch.tensor([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=torch.float32)  # Estrés yy
        C[5, :] = torch.tensor([0, 0, 0, 0, 0, 1, -1, 1, -1], dtype=torch.float32)  # Estrés xy
        C[6, :] = torch.tensor([0, 1, -1, 1, -1, 0, 0, 0, 0], dtype=torch.float32)  # Flujo q_x
        C[7, :] = torch.tensor([0, 0, 0, 0, 0, 1, 1, -1, -1], dtype=torch.float32)  # Flujo q_y
        C[8, :] = torch.tensor([1, -2, -2, -2, -2, 4, 4, 4, 4], dtype=torch.float32)  # Energía ε
        
        # Normalizar filas para mejor estabilidad numérica
        for i in range(9):
            norm = torch.norm(C[i, :])
            if norm > 1e-10:
                C[i, :] /= norm
        
        return C
        
    def forward(self, x):
        """
        Forward con conservación: separa los momentos conservados y no conservados,
        aplica la red solo a los momentos no conservados.
        """
        device = x.device
        C = self.C.to(device)
        C_inv = torch.inverse(C).to(device)
        
        # Transformar a espacio de momentos
        moments = torch.matmul(x, C.T)
        
        # Separar momentos conservados (densidad, momento x, momento y) y no conservados
        conserved = moments[:, :3]
        non_conserved = moments[:, 3:]
        
        # Aplicar la red neuronal solo a momentos no conservados
        temp = non_conserved
        for i, layer in enumerate(self.linear_layers):
            if i < len(self.linear_layers) - 1:
                temp = self.activation(layer(temp))
            else:
                temp = layer(temp)[:, :6]  # Asegurar que dimensiones sean correctas
                
        # Recombinar momentos y transformar de vuelta a distribuciones
        recombined_moments = torch.cat([conserved, temp], dim=1)
        return torch.matmul(recombined_moments, C_inv)


class NNSymConsPINN(LatticeBoltzmannBase):
    """
    Operador de colisión que combina simetría D8 y conservación de cantidades físicas.
    """
    def __init__(self, layer_sizes=None, activation_name="ReLU"):
        super(NNSymConsPINN, self).__init__(
            layer_sizes=layer_sizes,
            activation_name=activation_name,
            name="NNSymCons"
        )
        self.sym_model = NNSymPINN(layer_sizes, activation_name)
        self.cons_model = NNConsPINN(layer_sizes, activation_name)
        
    def forward(self, x):
        """
        Forward combinado que aplica tanto simetría como conservación.
        """
        # Calcular variables macroscópicas iniciales
        rho, u_x, u_y = self.compute_macroscopic(x)
        
        # Obtener colisión con simetría D8
        f_sym = self.sym_model(x)
        
        # Calcular nuevas variables macroscópicas después de la colisión con simetría
        rho_new, u_x_new, u_y_new = self.compute_macroscopic(f_sym)
        
        # Corregir para conservar masa y momento
        correction_factor = rho / (rho_new + 1e-10)
        f_corrected = f_sym * correction_factor
        
        # Aplicar el modelo de conservación
        return self.cons_model(f_corrected)

# Función factory para crear modelos de Lattice Boltzmann
def create_lattice_boltzmann_model(config):
    """
    Crea una instancia de modelo de Lattice Boltzmann según la configuración.
    
    Args:
        config (dict): Configuración del modelo
        
    Returns:
        LatticeBoltzmannBase: Instancia del modelo solicitado
    """
    model_type = config.get("variant", "naive").lower()
    layer_sizes = config.get("layers", [9, 50, 50, 9])
    activation = config.get("activation_function", "ReLU")
    
    if model_type == "naive":
        return NNNaivePINN(layer_sizes, activation)
    elif model_type == "sym":
        return NNSymPINN(layer_sizes, activation)
    elif model_type == "cons":
        return NNConsPINN(layer_sizes, activation)
    elif model_type == "symcons":
        return NNSymConsPINN(layer_sizes, activation)
    else:
        raise ValueError(f"Tipo de modelo LBM desconocido: {model_type}")
