import torch
import torch.nn as nn

# =====================================================================
# ARQUITECTURA 1: Red Base / Ultra-Simple (Pequeña)
# =====================================================================
class MLP_Small(nn.Module):
    def __init__(self, input_dim):
        super(MLP_Small, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),  # Capa oculta única
            nn.ReLU(),                 # Función de activación
            nn.Linear(16, 1)           # Capa de salida (1 logit para clasificación binaria)
        )
        
    def forward(self, x):
        return self.network(x)

# =====================================================================
# ARQUITECTURA 2: Red Estándar de Capas Progresivas (Mediana)
# =====================================================================
class MLP_Medium(nn.Module):
    def __init__(self, input_dim):
        super(MLP_Medium, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),  # Primera capa oculta
            nn.ReLU(),
            nn.Linear(32, 16),         # Segunda capa oculta (reducción gradual)
            nn.ReLU(),
            nn.Linear(16, 1)           # Capa de salida
        )
        
    def forward(self, x):
        return self.network(x)

# =====================================================================
# ARQUITECTURA 3: Red Ancha con Regularización contra Overfitting (Compleja)
# =====================================================================
class MLP_Large(nn.Module):
    def __init__(self, input_dim):
        super(MLP_Large, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),  # Capa oculta ancha
            nn.ReLU(),
            nn.Dropout(p=0.3),         # Regularización por Dropout al 30%
            nn.Linear(64, 32),         # Segunda capa oculta
            nn.ReLU(),
            nn.Linear(32, 1)           # Capa de salida
        )
        
    def forward(self, x):
        return self.network(x)