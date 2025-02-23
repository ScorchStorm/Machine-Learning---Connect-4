import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from pettingzoo.classic import connect_four_v3

# Hyperparameters
alpha = 0.9  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate
batch_size = 64
memory_size = 10000
train_start = 1000  # Start training after this many experiences
num_episodes = 1000

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize environment
env = connect_four_v3.env(render_mode=None)
state_size = np.prod(env.observation_space(env.possible_agents[0])['observation'].shape)
action_size = env.action_space(env.possible_agents[0]).n

# Initialize DQN and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
memory = deque(maxlen=memory_size)

def choose_action(state, mask):
    global epsilon
    if np.random.rand() < epsilon:
        return np.random.choice([i for i in range(len(mask)) if mask[i] == 1])
    state = torch.FloatTensor(state).to(device).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(state).cpu().numpy()[0]
    valid_q_values = [q_values[i] if mask[i] == 1 else -np.inf for i in range(len(mask))]
    return np.argmax(valid_q_values)

def train_model():
    if len(memory) < train_start:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)
    
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0].detach()
    target_q_values = rewards + (gamma * next_q_values * (1 - dones))
    
    loss = nn.MSELoss()(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for episode in range(num_episodes):
    env.reset(seed=42)
    prev_state = None
    prev_action = None
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        state = observation["observation"].flatten()
        mask = observation["action_mask"]

        if termination or truncation:
            action = None
            memory.append((prev_state, prev_action, reward, state, True))
        else:
            action = choose_action(state, mask)
            if prev_state is not None:
                memory.append((prev_state, prev_action, reward, state, False))

        prev_state, prev_action = state, action
        env.step(action)
        train_model()
    
    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

env.close()