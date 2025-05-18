import numpy as np
from tqdm import tqdm
import pickle
import os
import gymnasium as gym
import core.helper as helper
from models.baseModel import BaseModel

"""
Sarsa is a model free tabular RL algorithm which uses a Q-function to compute the expected reward and an epsilon-greedy policy 
to balance between exploration and exploitation.
It is an on-policy algorithm by selecting the actual action taken in the next state during the update.

The model expects a discrete action and observation space.

-------------------------------------------------------------
Example how to use this model:

Option 1: Create from terminal

env = ... # create environment 
model = Sarsa.create_model(env)

Option 2: Create manually
env = ... # create environment 
learning_rate = ...
gamma = ...
epsilon = ...

model = Sarsa(env, learning_rate, gamma, epsilon)

-------------------------------------------------------------
"""

class Sarsa(BaseModel): 
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
        Updates the Q-table using the Sarsa update rule (on-policy temporal difference)

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
    def __init__(self, env: gym.Env, learning_rate: float, gamma: float, eps_initial: float, eps_min: float=0.01, eps_decay: float=0.999):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps_initial = eps_initial
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.eps = eps_initial

        self.reset()

    def update(self, state: int, action: int, reward: float, new_state: int, new_action: int):
        """
        Updates the Q-table using the Sarsa-update rule (on-policy temporal difference)

        parameters: 
        
        state:
            last state

        action:
            action performed on last state

        reward: 
            reward received for taking action at last state

        new state: 
            new state after action was performed
        """

        # prediction error: on-policy temporal difference
        prediction_error = (
            reward
            + self.gamma * self.qtable[new_state, new_action]
            - self.qtable[state, action]
        )

        # update q-value for (state, action) pair
        self.qtable[state, action] += self.learning_rate * prediction_error

    def reset(self):
        """
        Resets the Q-table to zero
        """
        self.qtable = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.eps = self.eps_initial

    def predict(self, state: int, is_training: bool=False):
        """
        Selects an action based on epsilon-greedy policy

        Parameters:

        state:
            last state

        is_training:
            is set to true, if model is trained to enable exploration
        """

        # exploration
        if np.random.random() < self.eps and is_training:
            action = self.env.action_space.sample()

        # exploitation
        else:
            max_ids = np.where(self.qtable[state, :] == max(self.qtable[state, :]))[0]
            action = np.random.choice(max_ids) # choose random action if tie break (size of max_ids >1)
        return action

    def learn(self, num_episodes: int):
        """
        Trains the agent using the environment 

        arguments:
        
        num_episodes:
            number of episodes the model is trained
        
        """
        for i in tqdm(range(num_episodes)):
            state = self.env.reset()[0] # reset environment after each episode
            # receive best action for first state. Needs to be outside the while loop to perform on-policy update
            action = self.predict(state=state, is_training=True) 
            done = False

            while not done: # while episode is not over
                new_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                new_action = self.predict(state=new_state, is_training=True) # on-policy: choose best action for new state

                self.update(state, action, reward, new_state, new_action) # update qtable

                state, action = new_state, new_action # update state and action

            self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def save(self, file_path: str):
        """
        Saves the agent's state to a file

        Arguments:

        save:
            path of file to be saved to
        """
        self.env = None
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
        print(f"Model saved to {file_path}")

    @staticmethod
    def load(file_path: str):
        """
        Loads a saved agent from a file

        Arguments:

        save:
            path of file to be load from 
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        with open(file_path, 'rb') as file:
            agent = pickle.load(file)
        print(f"Model loaded from {file_path}")
        return agent
    
    @staticmethod
    def create_model(env:gym.Env):
        """
        Create a model from terminal

        Parameters:
            env:
                environment (custom or open ai environmnent)  
        """
        print("Specify parameters for the Sarsa model")

        learning_rate = helper.valid_parameter("Learning rate", float, [0,1])
        gamma = helper.valid_parameter("Discount factor", float, [0,1])
        eps_min = helper.valid_parameter("Minimum Exploration rate", float, [0,1])
        eps_decay = helper.valid_parameter("Exploration decay rate", float, [0,1])
        eps_initial = helper.valid_parameter("Initial Exploration rate", float, [0,1])
    
        return Sarsa(env, learning_rate, gamma, eps_initial, eps_min, eps_decay)