# This code seemed to work well, but I couldn't see the agents playing against each other, so I had ChatGPT make V2.2
import numpy as np
import random
import pickle

class Connect4:
    ROWS = 6
    COLS = 7

    def __init__(self):
        self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
        self.current_player = 1
        self.episode_memory = []  # Store full episode history

    def reset(self):
        self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
        self.current_player = 1
        self.episode_memory = []

    def get_valid_moves(self):
        return [c for c in range(self.COLS) if self.board[0][c] == 0]

    def make_move(self, col):
        for row in range(self.ROWS-1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                self.episode_memory.append((self.board.copy(), col, self.current_player))
                self.current_player = 3 - self.current_player
                return True
        return False

    def check_winner(self):
        for c in range(self.COLS - 3):
            for r in range(self.ROWS):
                if self.board[r][c] == self.board[r][c+1] == self.board[r][c+2] == self.board[r][c+3] != 0:
                    return self.board[r][c]

        for c in range(self.COLS):
            for r in range(self.ROWS - 3):
                if self.board[r][c] == self.board[r+1][c] == self.board[r+2][c] == self.board[r+3][c] != 0:
                    return self.board[r][c]

        for c in range(self.COLS - 3):
            for r in range(self.ROWS - 3):
                if self.board[r][c] == self.board[r+1][c+1] == self.board[r+2][c+2] == self.board[r+3][c+3] != 0:
                    return self.board[r][c]

        for c in range(self.COLS - 3):
            for r in range(3, self.ROWS):
                if self.board[r][c] == self.board[r-1][c+1] == self.board[r-2][c+2] == self.board[r-3][c+3] != 0:
                    return self.board[r][c]
        
        return 0

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((tuple(map(tuple, state)), action), 0.0)

    def choose_action(self, state, valid_moves):
        if not valid_moves:  # If no valid moves, return a random column (fallback)
            return random.randint(0, Connect4.COLS - 1)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_moves)
        q_values = [self.get_q_value(state, a) for a in valid_moves]
        if not q_values:  # Ensure q_values is not empty before calling max()
            return random.choice(valid_moves)
        max_q = max(q_values)
        return random.choice([a for a, q in zip(valid_moves, q_values) if q == max_q])

    def update_q_values(self, episode_memory, reward):
        discounted_reward = reward
        for state, action, _ in reversed(episode_memory):
            prev_q = self.get_q_value(state, action)
            self.q_table[(tuple(map(tuple, state)), action)] = prev_q + self.alpha * (discounted_reward - prev_q)
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
            state = env.board.copy()
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

agent = train_agent(100000)
print("We made it here!")