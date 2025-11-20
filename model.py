import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from collections import deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = 4
n_actions = 5 # move -x, x, -y, y, or no action

gamma = 0.99
alpha = 0.001
epsilon = 0.1
num_episodes = 200
batch_size = 64
replay_buffer_size = 50000

# Define Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize Q-network and optimizer
q_net = QNetwork(state_dim, n_actions).to(device)
optimizer = optim.Adam(q_net.parameters(), lr=alpha)
loss_fn = nn.MSELoss()
replay_buffer = deque(maxlen=replay_buffer_size)

def train_dqn():
    """Train the DQN using experience replay."""
    if len(replay_buffer) < batch_size:
        return
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(dones).to(device)

    q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = q_net(next_states).max(1)[0].detach()
    targets = rewards + gamma * next_q_values * (1 - dones)

    loss = loss_fn(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def predict(state, epsilon=0):
    if random.random() < epsilon:
        return random.randint(0, n_actions-1)
    else:
        state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
        with torch.no_grad():
            q_values = q_net(state_tensor).cpu()
            return int(np.argmax(q_values))
    