import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, seed=None):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.buffer = []

    def push(self, state, action, reward,
             next_state, terminated, truncated):
        transition = (state.copy(), action, reward, next_state.copy(), terminated, truncated)
        self.buffer.append(transition)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough transitions in buffer")
        
        if batch_size < 1:
            raise ValueError("Batch size is below 0")

        indices = self.rng.choice(
            len(self.buffer),
            size=batch_size,
            replace=False
            )
        batch = [self.buffer[i] for i in indices]
        return batch

    def __len__(self):
        return len(self.buffer)
