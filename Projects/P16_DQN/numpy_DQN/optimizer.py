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
