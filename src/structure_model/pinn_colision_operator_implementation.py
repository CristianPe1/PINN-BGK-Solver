
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Definición de la red neuronal Naive
class NNNaive(nn.Module):
    def __init__(self, input_dim=9, output_dim=9, hidden_dim=50):
        super(NNNaive, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Definición de la red neuronal Sym con equivarianza D8
class NNSym(NNNaive):
    def __init__(self, *args, **kwargs):
        super(NNSym, self).__init__(*args, **kwargs)
        self.d8_group = self.generate_d8_group()

    def generate_d8_group(self):
        # Genera las 8 transformaciones del grupo D8
        rotations = [np.roll(np.eye(9), i, axis=0) for i in range(4)]
        reflections = [np.flip(rot, axis=0) for rot in rotations]
        return rotations + reflections

    def forward(self, x):
        outputs = [super().forward(torch.matmul(x, torch.Tensor(g))) for g in self.d8_group]
        return torch.mean(torch.stack(outputs), dim=0)

# Definición de la red neuronal Cons con conservación de masa y momento
class NNCons(NNNaive):
    def __init__(self, *args, **kwargs):
        super(NNCons, self).__init__(*args, **kwargs)
        self.C = self.build_transformation_matrix()

    def build_transformation_matrix(self):
        # Construcción de la matriz de transformación C
        C = np.eye(9)
        C[0, :] = 1  # Conservación de la masa
        C[1, :] = np.arange(9)  # Ejemplo para u_x
        C[2, :] = np.arange(9)[::-1]  # Ejemplo para u_y
        return torch.Tensor(C)

    def forward(self, x):
        pre_collision = x
        nn_output = super().forward(x)
        A = torch.matmul(torch.inverse(self.C), torch.matmul(torch.diag(torch.Tensor([1, 1, 1] + [0]*6)), self.C))
        B = torch.matmul(torch.inverse(self.C), torch.matmul(torch.diag(torch.Tensor([0, 0, 0] + [1]*6)), self.C))
        return torch.matmul(A, pre_collision) + torch.matmul(B, nn_output)

# Definición de la red neuronal Sym+Cons
class NNSymCons(NNCons, NNSym):
    def __init__(self, *args, **kwargs):
        super(NNSymCons, self).__init__(*args, **kwargs)

    def forward(self, x):
        outputs = [super(NNSym, self).forward(torch.matmul(x, torch.Tensor(g))) for g in self.d8_group]
        averaged_output = torch.mean(torch.stack(outputs), dim=0)
        return super(NNCons, self).forward(averaged_output)

# Ejemplo de uso
if __name__ == '__main__':
    input_data = torch.rand((1, 9))
    model_naive = NNNaive()
    model_sym = NNSym()
    model_cons = NNCons()
    model_symcons = NNSymCons()

    print("Naive NN Output:", model_naive(input_data))
    print("Sym NN Output:", model_sym(input_data))
    print("Cons NN Output:", model_cons(input_data))
    print("Sym+Cons NN Output:", model_symcons(input_data))
