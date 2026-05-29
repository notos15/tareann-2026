import torch
import torch.nn as nn


class MLP_Simple(nn.Module):
    """
    Arquitectura 1: MLP simple con una capa oculta.
    - 1 capa oculta de 32 neuronas con activación ReLU
    - Salida con 1 neurona (clasificación binaria)
    
    Justificación: Es la arquitectura más simple posible para el problema.
    Sirve como línea base (baseline) para comparar con modelos más complejos.
    Con tan pocas neuronas, puede que no capture toda la complejidad del
    problema pero es rápida de entrenar y difícil de sobreajustar.
    """
    def __init__(self, input_dim):
        super(MLP_Simple, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class MLP_Medium(nn.Module):
    """
    Arquitectura 2: MLP con dos capas ocultas y Dropout.
    - Capa oculta 1: 64 neuronas, ReLU
    - Dropout(0.3) para regularización
    - Capa oculta 2: 32 neuronas, ReLU
    - Dropout(0.3) para regularización
    - Salida: 1 neurona con Sigmoid

    Justificación: Añadir una segunda capa oculta permite al modelo aprender
    representaciones más abstractas. El Dropout actúa como regularización,
    reduciendo el sobreajuste, lo cual es importante dado el desbalance de
    clases (735 no-legendarios vs 65 legendarios). Las activaciones ReLU
    son preferidas sobre sigmoid/tanh en capas ocultas por mitigar el
    problema del gradiente desvaneciente.
    """
    def __init__(self, input_dim):
        super(MLP_Medium, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class MLP_Deep(nn.Module):
    """
    Arquitectura 3: MLP más profunda con tres capas ocultas y BatchNorm.
    - Capa oculta 1: 128 neuronas, BatchNorm, ReLU, Dropout(0.4)
    - Capa oculta 2: 64 neuronas,  BatchNorm, ReLU, Dropout(0.3)
    - Capa oculta 3: 32 neuronas,  BatchNorm, ReLU
    - Salida: 1 neurona con Sigmoid

    Justificación: La arquitectura más compleja de las tres. El BatchNorm
    normaliza las activaciones de cada capa, estabilizando el entrenamiento
    y permitiendo tasas de aprendizaje más altas. El Dropout más agresivo
    en la primera capa ayuda a prevenir el sobreajuste dado el tamaño
    relativamente pequeño del dataset (800 muestras). Esta arquitectura
    puede capturar relaciones no lineales más complejas entre los atributos,
    aunque corre mayor riesgo de sobreajuste.
    """
    def __init__(self, input_dim):
        super(MLP_Deep, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
