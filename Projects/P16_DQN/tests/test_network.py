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
    

def test_gradient_checking():
    model = MLP(input_dim=2, hidden_dim=2, output_dim=2)
    X = np.array([[1,2], [2, 1]])
    model.W1 = np.array([[1, -1], [2, 1]], dtype=float)
    model.W2 = np.array([[1, 2], [-1, 3]], dtype=float)
    model.b1 = np.array([0, 0], dtype=float)
    model.b2 = np.array([0, 0], dtype=float)
    Y = np.array([[2, 10], [1, 3]])

    Q = model.forward(X)
    # SSE used for Loss
    L = np.sum((Q - Y)**2)
    model.back_propagation(dQ=2 * (Q - Y))

    h = 1e-5

    params_and_grads = [
        (model.W1, model.dW1),
        (model.W2, model.dW2),
        (model.b1, model.db1),
        (model.b2, model.db2)
    ]

    for param, grad in params_and_grads:
        for idx in np.ndindex(param.shape):
            original_weight = param[idx]

            param[idx] = original_weight + h
            Q = model.forward(X)
            L_plus = np.sum((Q - Y)**2)

            param[idx] = original_weight - h
            Q = model.forward(X)
            L_minus = np.sum((Q - Y)**2)

            param[idx] = original_weight

            dL_numerical_gradient = (L_plus - L_minus) / (2 * h)

            assert np.allclose(grad[idx] , dL_numerical_gradient)
