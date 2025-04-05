import numpy as np
from tqdm import tqdm
import pickle
import os
import gymnasium as gym
import core.helper as helper
from models.custom.baseModel import BaseModel

"""
MonteCarlo is a model free tabular RL algorithm which uses a Q-function to compute the expected reward and an epsilon-greedy policy 
to balance between exploration and exploitation.
It is an off-policy algorithm by selecting the full episode during the update.
The model expects a discrete action and observation space.

-------------------------------------------------------------
Example how to use this model:

Option 1: Create from terminal

env = ... # create environment 
model = MonteCarlo.create_model(env)

Option 2: Create manually
env = ... # create environment 
learning_rate = ...
gamma = ...
epsilon = ...

model = MonteCarlo(env, learning_rate, gamma, epsilon)

-------------------------------------------------------------
"""

class MonteCarlo(BaseModel):
    """
    -------------------------------------------------------------
    Attributes:

    env: gym.Env
        The environment the agent interacts with

    learning_rate: float
        The learning rate for Q-learning updates

    gamma: float
        The discount factor for future rewards

    epsilon: float
        The exploration rate for epsilon-greedy policy

    qtable: np.ndarray
        The qtable for storing the expected rewards for state-action pairs

    -------------------------------------------------------------

    Methods:

    update:
        Updates the Q-table using the full episode 

    reset:
        Resets the Q-table to zero

    predict:
        Selects an action based on epsilon-greedy policy

    learn;
        Trains the agent using the environment 

    save:
        Saves the agent's state to a file

    load:
        Loads a saved agent from a file

    create_model (static):
        Create a model from terminal

    """
    def __init__(self, env:gym.Env, learning_rate: float, gamma:float, epsilon: float):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.returns = { (s, a): [] for s in range(self.env.observation_space.n) for a in range(self.env.action_space.n) }
        self.reset()

    def reset(self):
        """
        Resets the Q-table and the saved returns to zero
        """
        self.qtable = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.returns = { (s, a): [] for s in range(self.env.observation_space.n) for a in range(self.env.action_space.n) }
   
    def predict(self, state:int, is_training:bool=False):
        """
        Selects an action based on epsilon-greedy policy

        Parameters:

        state:
            last state

        is_training:
            is set to true, if model is trained to enable exploration
        """

        # exploration
        if np.random.random() < self.epsilon and is_training:
            action = self.env.action_space.sample()

        # exploitation
        else:
            max_ids = np.where(self.qtable[state, :] == max(self.qtable[state, :]))[0]
            action = np.random.choice(max_ids) # choose random action if tie break (size of max_ids >1)
        return action
    
    def update(self, episode_data):
        """
        Updates the Q-table using the full episode 

        Parameters:

        episode_data:
            episode data saved as (state, action, reward) tuples
        """
        G = 0
        for t in reversed(range(len(episode_data))): # go through episode in reversed order
            state, action, reward = episode_data[t] # get tuple at time t
            G = reward + self.gamma * G # sum up return with discount factor gamma

            # check if (state, action) pair occured the first time in the episode
            if not any([(state == x[0] and action == x[1]) for x in episode_data[:t]]): 
                self.returns[(state, action)].append(G) # save return for the pair
                self.qtable[state, action] = np.mean(self.returns[(state, action)]) # update qvalue for the pair by taking the mean of the saved returns

    def learn(self, num_episodes: int):
        """
        Trains the agent using the environment 

        arguments:
        
        num_episodes:
            number of episodes the model is trained
        
        """
        for i in tqdm(range(num_episodes)):
            state = self.env.reset()[0]  # reset environment after each episode
            done = False
            episode_data = [] 

            while not done: # while episode is not over
                action = self.predict(state, is_training=True) # choose action for current state
                new_state, reward, terminated, truncated, info = self.env.step(action)
                episode_data.append((state, action, reward)) # save (state, action, reward) tuple
                done = terminated or truncated
                state = new_state # update state
            
            self.update(episode_data) # update qtable after episode was over

    def save(self, file_path: str):
            self.env = None
            with open(file_path, 'wb') as file:
                pickle.dump(self, file)
            print(f"Model saved to {file_path}")

    @staticmethod
    def load(file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        with open(file_path, 'rb') as file:
            agent = pickle.load(file)
        print(f"Model loaded from {file_path}")
        return agent
    
    @staticmethod
    def create_model(env:gym.Env):
        print("Specify parameters for the montecarlo model\n")

        learning_rate = helper.valid_parameter("Learning rate", float, [0,1])
        gamma = helper.valid_parameter("Discount factor", float, [0,1])
        epsilon = helper.valid_parameter("Exploration rate", float, [0,1])
    

        return MonteCarlo(env, learning_rate, gamma, epsilon)
        