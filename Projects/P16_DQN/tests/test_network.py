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
    pass