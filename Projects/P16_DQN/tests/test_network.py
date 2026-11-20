import numpy as np
from numpy_DQN.network import MLP

def test_forward_shape():
    model = MLP(seed=42)
    X = np.random.default_rng(123).normal(size=(32,4))

    Q = model.forward(X)

    assert Q.shape == (32, 2)


def test_forward_propagation():
    model = MLP(input_dim=2, hidden_dim=2, output_dim=1)
    X = np.array([1, 2])
    model.W1 = np.array([
        [1, -1], 
        [2, 1]
        ])
    model.W2 = np.array([
        [1],
        [3]
    ])

    Q = model.forward(X)

    assert np.allclose(Q, [8.0])


def test_back_propagation():
    model = MLP(input_dim=2, hidden_dim=2, output_dim=2)
    X = np.array([[1, 2]])
    model.W1 = np.array([
        [1, -1], 
        [2, 1]
        ])
    model.W2 = np.array([
        [1, 2],
        [-1, 3]
    ])

    Q = model.forward(X)
    model.back_propagation(dQ=np.array([[4, 6]]))

    assert np.allclose(model.dW2, np.array([[20, 30], [4, 6]]))
    assert np.allclose(model.db2, np.array([4, 6]))
    assert np.allclose(model.dW1, np.array([[16, 14], [32, 28]]))
    assert np.allclose(model.db1, np.array([16, 14]))

    assert np.allclose(Q, np.array([[4, 13]]))
    