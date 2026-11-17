import numpy as np

class MLP:
    def __init__(self, input_dim=4, hidden_dim=8, output_dim=2, seed=None):
        self.rng = np.random.default_rng(seed)
        # He initialization used
        self.W1 = self.rng.normal(0, np.sqrt(2 / input_dim), size=(input_dim, hidden_dim))
        self.W2 = self.rng.normal(0, np.sqrt(2 / hidden_dim), size=(hidden_dim, output_dim))
        self.b1 = np.zeros(hidden_dim)
        self.b2 = np.zeros(output_dim)

    def ReLu(self, Z):
        A = np.maximum(0, Z)
        return A

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = self.ReLu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        return Z2
    