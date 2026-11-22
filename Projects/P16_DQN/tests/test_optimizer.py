import numpy as np
from numpy_DQN.optimizer import SGD
from numpy_DQN.network import MLP

def test_SGD():
    model = MLP(input_dim=2, hidden_dim=2, output_dim=2)
    
    X = np.array([[1,2], [2, 1]])
    model.W1 = np.array([[1, -1], [2, 1]], dtype=float)
    model.W2 = np.array([[1, 2], [-1, 3]], dtype=float)
    model.b1 = np.array([0, 0], dtype=float)
    model.b2 = np.array([0, 0], dtype=float)
    Y = np.array([[2, 10], [1, 3]])

    Q = model.forward(X)
    model.back_propagation(dQ=2 * (Q - Y))

    grad_descent = SGD(learning_rate=0.01)
    grad_descent.step(model)

    assert np.allclose(model.W1, np.array([[0.32, -1.14], [1.42, 0.72]]))
    assert np.allclose(model.W2, np.array([[0.56, 1.30], [-1.04, 2.94]]))
    assert np.allclose(model.b1, np.array([-0.42, -0.14]))
    assert np.allclose(model.b2, np.array([-0.10, -0.16]))
