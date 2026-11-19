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
        self.X = X
        self.Z1 = self.X @ self.W1 + self.b1
        self.A1 = self.ReLu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        return self.Z2
    
    def ReLu_derivative(self, Z):
        return (Z > 0).astype(float)
    
    def back_propagation(self, dQ):
        dW2 = self.A1.T @ dQ
        db2 = np.sum(dQ, axis=0)
        dA1 = dQ @ self.W2.T
        dZ1 = dA1 * self.ReLu_derivative(self.Z1)
        dW1 = self.X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)