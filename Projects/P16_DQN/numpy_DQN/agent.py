import numpy as np
from numpy_DQN.network import MLP
from numpy_DQN.replay_buffer import ReplayBuffer
from numpy_DQN.optimizer import SGD, Adam

class DQNAgent:
    def __init__(self, state_dim=4, hidden_dim=64, action_dim=2, gamma=0.99, seed=None):
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.replay_buffer = ReplayBuffer(capacity=10000, seed=seed)
        self.optimizer = SGD()

        self.online = MLP(input_dim=state_dim, hidden_dim=hidden_dim, output_dim=action_dim, seed=seed)
        self.target = MLP(input_dim=state_dim, hidden_dim=hidden_dim, output_dim=action_dim, seed=seed)

    def store_transition(self, state, action, reward,
             next_state, terminated, truncated):
        self.replay_buffer.push(state, action, reward, next_state, terminated, truncated)

    def update_target(self):
        self.target.W1 = self.online.W1.copy()
        self.target.W2 = self.online.W2.copy()
        self.target.b1 = self.online.b1.copy()
        self.target.b2 = self.online.b2.copy()

    def select_action(self,state,epsilon):
        rand = self.rng.random()
        
        if rand < epsilon:
            return int(self.rng.integers(0, 2))
        
        X = state.reshape(1, -1)
        Q = self.online.forward(X)

        return int(np.argmax(Q))
    
    def learn(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return None
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, terminated, truncated = zip(*batch)

        states = np.stack(states)
        actions = np.asarray(actions, dtype=int)
        rewards = np.asarray(rewards, dtype=float)
        next_states = np.stack(next_states)
        terminated = np.asarray(terminated, dtype=bool)
        truncated = np.asarray(truncated, dtype=bool)

        Q = self.online.forward(X=states)
        Q_hat = self.target.forward(X=next_states)
        maxQ_hat = np.max(Q_hat, axis=1)

        y_TD = rewards + self.gamma * (~terminated) * maxQ_hat
        Q_selected = Q[np.arange(len(actions)), actions]

        Loss = (1 / batch_size) * np.sum((y_TD - Q_selected)**2)

        dQ = np.zeros_like(Q)
        dQ[np.arange(batch_size), actions] = (-2 / batch_size) * (y_TD - Q_selected)

        self.online.back_propagation(dQ)
        self.optimizer.step(self.online)

        return float(Loss)
