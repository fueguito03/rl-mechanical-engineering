import numpy as np
from numpy_DQN.replay_buffer import ReplayBuffer

def test_capacity():
    buffer = ReplayBuffer(capacity=3, seed=42)

    # add 5 experiences
    next_state = np.array([0.2, 0.3, 0.4, 0.5])

    for i in range(5):
        state = np.array([i, 0.2, 0.3, 0.4])
        buffer.push(
        state=state,
        action=0,
        reward=1.0,
        next_state=next_state,
        terminated=False,
        truncated=False
    )

    assert len(buffer) == 3

    for i in range(3):
        assert buffer.buffer[i][0][0] == i + 2

# Testing if batch is picked without being overlapped
def test_sample():
    buffer = ReplayBuffer(capacity=10, seed=42)

    # add 5 experiences
    next_state = np.array([0.2, 0.3, 0.4, 0.5])
    for i in range(5):
        state = np.array([i, 0.2, 0.3, 0.4])
        buffer.push(
        state=state,
        action=0,
        reward=1.0,
        next_state=next_state,
        terminated=False,
        truncated=False
    )
               
    batch = buffer.sample(3)

    batch_ids = [transition[0][0] for transition in batch]

    assert len(batch_ids) == 3
    assert len(set(batch_ids)) == 3

# Testing that same seed gives same results
def test_seed():
    buffer_1 = ReplayBuffer(capacity=10, seed=42)
    buffer_2 = ReplayBuffer(capacity=10, seed=42)

    # Buffer_1
    next_state = np.array([0.2, 0.3, 0.4, 0.5])
    for i in range(5):
        state = np.array([i, 0.2, 0.3, 0.4])
        buffer_1.push(
        state=state,
        action=0,
        reward=1.0,
        next_state=next_state,
        terminated=False,
        truncated=False
    )
    # Buffer_2        
    for i in range(5):
        state = np.array([i, 0.2, 0.3, 0.4])
        buffer_2.push(
        state=state,
        action=0,
        reward=1.0,
        next_state=next_state,
        terminated=False,
        truncated=False
    )

    batch_1 = buffer_1.sample(3)
    batch_2 = buffer_2.sample(3)

    ids_1 = [transition[0][0] for transition in batch_1]
    ids_2 = [transition[0][0] for transition in batch_2]

    assert ids_1 == ids_2
