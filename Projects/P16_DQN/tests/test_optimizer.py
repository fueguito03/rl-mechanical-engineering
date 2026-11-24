import numpy as np
from numpy_DQN.optimizer import SGD
from numpy_DQN.optimizer import Adam
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

def test_ADAM_step():
    model = MLP(input_dim=2, hidden_dim=2, output_dim=2)

    model.W1 = np.array([[1, -1], [2, 1]], dtype=float)
    model.W2 = np.array([[1, 2], [-1, 3]], dtype=float)
    model.b1 = np.array([0, 0], dtype=float)
    model.b2 = np.array([0, 0], dtype=float)

    model.dW1 = np.full_like(model.W1, 4.0)
    model.dW2 = np.full_like(model.W2, 4.0)
    model.db1 = np.full_like(model.b1, 4.0)
    model.db2 = np.full_like(model.b2, 4.0)

    grad_descent = Adam()
    grad_descent.step(model)

    assert np.allclose(model.W1, np.array([[0.999, -1.001], [1.999, 0.999]]))
    assert np.allclose(model.W2, np.array([[0.999, 1.999], [-1.001, 2.999]]))
    assert np.allclose(model.b1, np.array([-0.001, -0.001]))
    assert np.allclose(model.b2, np.array([-0.001, -0.001]))

    model.dW1.fill(-2.0)
    model.dW2.fill(-2.0)
    model.db1.fill(-2.0)
    model.db2.fill(-2.0)

    grad_descent.step(model)
    delta = 0.001 * (0.16 / 0.19) / (np.sqrt(0.019984 / 0.001999) + 1e-8)

    assert np.allclose(model.W1, np.array([[0.999, -1.001], [1.999, 0.999]]) - delta)
    assert np.allclose(model.W2, np.array([[0.999, 1.999], [-1.001, 2.999]]) - delta)
    assert np.allclose(model.b1, np.array([-0.001, -0.001]) - delta)
    assert np.allclose(model.b2, np.array([-0.001, -0.001]) - delta)