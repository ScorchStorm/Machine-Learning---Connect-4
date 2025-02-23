# This code was made by giving the example code from the website https://pettingzoo.farama.org/environments/classic/connect_four/ to ChatGPT, and asking it to complete the policy.
# Here is the link to the conversation with ChatGPT: https://chatgpt.com/c/67ba581d-b5cc-8002-a4e6-0d1e3db206e5
import numpy as np
import pickle
from pettingzoo.classic import connect_four_v3

# Parameters
alpha = 0.9  # Learning rate
gamma = 0.05  # Discount factor
epsilon = 0.05  # Exploration rate


# Load or initialize Q-table
try:
    with open("q_table.pkl", "rb") as f:
        Q_table = pickle.load(f)
except FileNotFoundError:
    Q_table = {}

def get_q_value(state, action):
    return Q_table.get((state, action), 0.0)

def choose_action(observation, agent):
    state = tuple(observation["observation"].flatten())
    mask = observation["action_mask"]
    valid_actions = [a for a in range(len(mask)) if mask[a] == 1]
    
    if np.random.rand() < epsilon:  # Explore
        return np.random.choice(valid_actions)
    
    # Exploit: Choose best known action
    q_values = [get_q_value(state, a) for a in valid_actions]
    return valid_actions[np.argmax(q_values)]

def update_q_table(prev_state, action, reward, new_state):
    best_future_q = max([get_q_value(new_state, a) for a in range(7)], default=0.0)
    old_q = get_q_value(prev_state, action)
    Q_table[(prev_state, action)] = old_q + alpha * (reward + gamma * best_future_q - old_q)

# Training loop
env = connect_four_v3.env(render_mode="human")
for episode in range(1000):  # Number of episodes to train
    env.reset(seed=42)
    prev_state = None
    prev_action = None
    
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        state = tuple(observation["observation"].flatten())

        if termination or truncation:
            action = None
        else:
            action = choose_action(observation, agent)

        if prev_state is not None:
            update_q_table(prev_state, prev_action, reward, state)
        
        prev_state, prev_action = state, action
        env.step(action)

    if episode%10 == 0: # Save Q-table every 10 episodes
        with open("q_table.pkl", "wb") as f:
            print(f'Saving updated Q-table on episode {episode}')
            pickle.dump(Q_table, f)

env.close()