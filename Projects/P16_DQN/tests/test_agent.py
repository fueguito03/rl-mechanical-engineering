import numpy as np
from numpy_DQN.agent import DQNAgent

def test_update_target():
    agent = DQNAgent(seed=42)

    params = [
        (agent.online.W1, agent.target.W1),
        (agent.online.W2, agent.target.W2),
        (agent.online.b1, agent.target.b1),
        (agent.online.b2, agent.target.b2)
    ]

    for online_param, target_param in params:
        assert np.array_equal(online_param, target_param)

    agent.online.W1[0, 0] += 10
    assert not np.array_equal(agent.online.W1, agent.target.W1)
    agent.update_target()
    assert np.array_equal(agent.online.W1, agent.target.W1)

def test_select_action_greedy():
    agent = DQNAgent(seed=42)

    agent.online.W1.fill(0)
    agent.online.W2.fill(0)
    agent.online.b1.fill(0)
    agent.online.b2 = np.array([1.0, 3.0])

    state = np.array([0.1, 0.2, -0.1, 0.3])

    action = agent.select_action(state, epsilon=0.0)

    assert action == 1

def test_select_action_random():
    agent = DQNAgent(seed=42)
    state = np.array([0.1, 0.2, -0.1, 0.3])

    for _ in range(100):
        action = agent.select_action(state, epsilon=1.0)
        assert action in (0, 1)
        assert isinstance(action, int)

def test_learn_insufficient_buffer():
    agent = DQNAgent(seed=42)

    loss = agent.learn(batch_size=32)

    assert loss is None

def test_learn():
    state_1 = np.array([0.0, 0.1, 0.02, -0.1])
    next_state_1 = np.array([0.01, 0.15, 0.018, -0.08])

    state_2 = np.array([0.1, -0.2, 0.15, 0.3])
    next_state_2 = np.array([0.08, -0.25, 0.22, 0.4])

    agent = DQNAgent(seed=42)

    # Normal Transition
    agent.store_transition(state_1, 0, 1.0, next_state_1, False, False)

    # Terminal Transition
    agent.store_transition(state_2, 1, 1.0, next_state_2, True, False)

    loss = agent.learn(batch_size=2)

    assert loss is not None
    assert np.isfinite(loss)