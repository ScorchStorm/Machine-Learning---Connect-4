import numpy as np
import random
import pickle
from pettingzoo.classic import connect_four_v3

SHOW_GAME = True  # Set to False to disable game visualization

class Connect4:
    def __init__(self):
        self.env = connect_four_v3.env(render_mode="human" if SHOW_GAME else None)
        self.env.reset()
        self.episode_memory = []

    def reset(self):
        self.env.reset()
        self.episode_memory = []

    def get_valid_moves(self):
        observation, _, _, _, _ = self.env.last()
        return [i for i, valid in enumerate(observation["action_mask"]) if valid]

    def make_move(self, action):
        observation, _, _, _, _ = self.env.last()  # Get the current state
        state = observation["observation"]
        self.env.step(action)
        self.episode_memory.append((state, action))  # Store state-action pair


    def check_winner(self):
        observation, reward, termination, _, _ = self.env.last()
        if termination:
            return 1 if reward == 1 else 2 if reward == -1 else 0
        return 0

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((tuple(state.flatten()), action), 0.0)

    def choose_action(self, state, valid_moves):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_moves)
        
        q_values = [self.get_q_value(state, a) for a in valid_moves]
        max_q = max(q_values)
        return random.choice([a for a, q in zip(valid_moves, q_values) if q == max_q])

    def update_q_values(self, episode_memory, reward):
        discounted_reward = reward
        for state, action in reversed(episode_memory):
            prev_q = self.get_q_value(state, action)
            self.q_table[(tuple(state.flatten()), action)] = prev_q + self.alpha * (discounted_reward - prev_q)
            discounted_reward *= self.gamma

    def save_q_table(self, filename="q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename="q_table.pkl"):
        try:
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            self.q_table = {}

def train_agent(episodes=10000):
    env = Connect4()
    agent = QLearningAgent()
    
    for _ in range(episodes):
        env.reset()
        done = False
        
        while not done:
            observation, _, _, _, _ = env.env.last()  # Get the current observation
            state = observation["observation"]  # Extract state properly
            valid_moves = env.get_valid_moves()
            
            action = agent.choose_action(state, valid_moves)
            env.make_move(action)
            
            winner = env.check_winner()
            if winner != 0 or not valid_moves:
                reward = 1 if winner == 1 else -1 if winner == 2 else 0
                agent.update_q_values(env.episode_memory, reward)
                done = True
    
    agent.save_q_table()
    return agent

agent = train_agent(100)
print("training Complete! =D")