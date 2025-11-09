import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from collections import deque
import random

SEQ_LEN = 40

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = 8 # targetx, targety, ballx, bally, ballxvel, ballyvel, floorrotx, floorroty
n_actions = 5 # move -x, x, -y, y, or no action

gamma = 1
alpha = 0.01
epsilon = 0.15 #chance that a random action is taken
num_episodes = 200
batch_size = 64
replay_buffer_size = 80000

# Define Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(QNetwork, self).__init__()
        self.hidden_size = 128
        self.num_layers = 2

        self.lstm = nn.LSTM(state_dim, self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, n_actions)

        self.hidden = (torch.zeros(self.num_layers, 1, self.hidden_size),
                       torch.zeros(self.num_layers, 1, self.hidden_size))

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
                0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(device)

        output = torch.relu(self.lstm(x, (h0, c0))[0][:, -1, :])
        x = torch.relu(self.fc1(output))
        return self.fc2(x)

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
        state = np.reshape(state, (1, SEQ_LEN, state_dim))
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            q_values = q_net(state_tensor).cpu()
            return int(np.argmax(q_values))
    