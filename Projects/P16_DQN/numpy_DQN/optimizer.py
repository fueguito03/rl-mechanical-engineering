import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def step(self, model):
        params_and_grads = [
        (model.W1, model.dW1),
        (model.W2, model.dW2),
        (model.b1, model.db1),
        (model.b2, model.db2)
        ]

        for param, grad in params_and_grads:
            param -= self.learning_rate * grad

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, model):
        self.t += 1

        params_and_grads = [
        ("W1", model.W1, model.dW1),
        ("W2", model.W2, model.dW2),
        ("b1", model.b1, model.db1),
        ("b2", model.b2, model.db2)
        ]

        for name, param, grad in params_and_grads:
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)

            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad)**2

            m_hat = self.m[name] / (1 - self.beta1**self.t)
            v_hat = self.v[name] / (1 - self.beta2**self.t)

            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)