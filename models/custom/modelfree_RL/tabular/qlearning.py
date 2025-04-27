import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os
import gymnasium as gym
import core.helper as helper
from models.custom.baseModel import BaseModel

"""
QLearning is a model free tabular RL algorithm which uses a Q-function to compute the expected reward and an epsilon-greedy policy 
to balance between exploration and exploitation.
It is an off-policy algorithm by selecting the maximum reward for the best action taken in the next state during the update.

The model expects a discrete action and observation space.

-------------------------------------------------------------
Example how to use this model:

Option 1: Create from terminal

env = ... # create environment 
model = Qlearning.create_model(env)

Option 2: Create manually
env = ... # create environment 
learning_rate = ...
gamma = ...
epsilon = ...

model = Qlearning(env, learning_rate, gamma, epsilon)

-------------------------------------------------------------
"""

class Qlearning(BaseModel):
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
        Updates the Q-table using the Q-learning update rule (off-policy temporal difference)

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
    def __init__(self, env, learning_rate: float, gamma: float, epsilon: float):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.reset()

    def update(self, state: int, action: int, reward: float, new_state: int):
        """
        Updates the Q-table using the Q-learning update rule (off-policy temporal difference)

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

        # prediction error: off-policy temporal difference
        prediction_error = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        ) 
        
        # update q-value for (state, action) pair
        self.qtable[state, action]= self.qtable[state, action] + self.learning_rate * prediction_error

    def reset(self):
        """
        Resets the Q-table to zero
        """
        print(f"observation space: {self.env.observation_space.n}")
        self.qtable = np.zeros((self.env.observation_space.n, self.env.action_space.n))

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
        if np.random.random() < self.epsilon and is_training:
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
            done = False
            
            #print("run")
            while not done: # while episode is not over
                action = self.predict(state=state, is_training=True) # receive best action for current state

                new_state, reward, terminated, truncated, info = self.env.step(action) # perform action and receive result (new state, reward, etc.)

                done = terminated or truncated 

                self.update(state, action, reward, new_state) # update qtable

                state = new_state # update state 
            #print("done")

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

        df_params = pd.DataFrame({
            "Parameter": ["Learning Rate", "Discount Factor (Gamma)", "Exploration Rate (Epsilon)"],
            "Value": [self.learning_rate, self.gamma, self.epsilon]
        })

        # Create DataFrame for Q-table
        df_qtable = pd.DataFrame(self.qtable)

        # Save parameters to CSV (overwrite if file exists)
        df_params.to_csv(file_path+"ss", index=False)

        # Append Q-table to the same file
        df_qtable.to_csv(file_path+"ss", mode='a', index=False, header=False)

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

        print("Specify parameters for the qlearning model\n")

        learning_rate = helper.valid_parameter("Learning rate", float, [0,1])
        gamma = helper.valid_parameter("Discount factor", float, [0,1])
        epsilon = helper.valid_parameter("Exploration rate", float, [0,1])

        return Qlearning(env, learning_rate, gamma, epsilon)   