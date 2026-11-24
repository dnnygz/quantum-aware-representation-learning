"""One frozen circuit definition for A and B."""
import numpy as np
import pennylane as qml
import torch
from torch import nn


def angles(z):
    return np.pi * torch.tanh(z / 2)


class Encoder(nn.Sequential):
    def __init__(self, inputs):
        super().__init__(nn.Linear(inputs, 8), nn.Tanh(), nn.Linear(8, 4))


class VQC(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(0.1 * torch.randn(2, 4, 2, dtype=torch.float64))
        self.device = qml.device('default.qubit', wires=4, shots=None)
        self.calls = 0
        self.samples = 0
        @qml.qnode(self.device, interface='torch', diff_method='backprop')
        def circuit(x, weights):
            for i in range(4):
                qml.RY(x[..., i], wires=i)
            for layer in weights:
                for i in range(4):
                    qml.RY(layer[i, 0], wires=i)
                    qml.RZ(layer[i, 1], wires=i)
                for i in range(3):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(3))
        self.circuit = circuit

    def forward(self, z):
        self.calls += 1
        self.samples += len(z)
        expectation = self.circuit(angles(z), self.weights)
        # Fixed link, no trainable calibration: ER+ probability.
        return ((1 + expectation) / 2).clamp(1e-7, 1 - 1e-7)


class Model(nn.Module):
    def __init__(self, inputs, encoder=False, quantum=True):
        super().__init__()
        # Keep the quantum RNG stream identical to A while retaining the
        # encoder's seed-specific initialization. CPU-only protocol.
        with torch.random.fork_rng(devices=[]):
            self.encoder = Encoder(inputs) if encoder else nn.Identity()
        self.head = VQC() if quantum else nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())
        self.quantum = quantum
        self.double()

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z).reshape(-1)
