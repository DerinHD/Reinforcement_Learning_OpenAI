import random
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm
import pickle
import os
import gymnasium as gym
from tqdm import tqdm
import core.helper as helper

actions = [0,10,20,30,40,50]
states = [0,1,2]

"""
DynaQ is a model based tabular RL algorithm which uses a Q-function to compute the expected reward and an epsilon-greedy policy 
to balance between exploration and exploitation. To simulate the environment (:=use model based approach), we save observed pairs (state, action)
with the transitions (reward, newstate) in a dictionairy and sample pairs during simulation (:=planning) to update the qfunction
It is an off-policy algorithm by selecting the maximum reward for the best action taken in the next state during the update.

The model expects a discrete action and observation space.

-------------------------------------------------------------
Example how to use this model:

Option 1: Create from terminal

env = ... # create environment 
model = DynaQ.create_model(env)

Option 2: Create manually
env = ... # create environment 
learning_rate = ...
gamma = ...
epsilon = ...

model = DynaQ(env, learning_rate, gamma, epsilon)

-------------------------------------------------------------
"""

class DynaQ:
    def __init__(self, env:gym.Env, learning_rate=0.1, gamma= 0.0, epsilon= 0.1, num_planning_steps=5):
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

        num_planning_steps:
            Number of planning steps during simulation

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
        self.rng = np.random.default_rng(10)
        self.env = env
        self.state_size = env.observation_space.n
        self.action_size = env.action_space.n
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_planning_steps = num_planning_steps
        self.reset()

    def update(self, state, action, reward, new_state):
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
                
        prediction_error = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        ) 
        
        self.qtable[state, action]= self.qtable[state, action] + self.learning_rate * prediction_error

    def reset(self):
        """
        Resets the Q-table to zero and clears the model
        """
        self.qtable = np.zeros((self.state_size, self.action_size))
        self.model = {}

    def predict(self, state, is_training=False):
        """
        Selects an action based on epsilon-greedy policy

        Parameters:

        state:
            last state

        is_training:
            is set to true, if model is trained to enable exploration
        """

        if np.random.random() < self.epsilon and is_training:
            action = self.env.action_space.sample()

        else:
            max_ids = np.where(self.qtable[state, :] == max(self.qtable[state, :]))[0]
            action = self.rng.choice(max_ids)
        return action
    
    def learn(self, num_episodes):
        """
        Trains the agent using the environment 

        arguments:
        
        num_episodes:
            number of episodes the model is trained
        
        """
        for episode in tqdm(range(num_episodes)):
            state = self.env.reset()[0]
            done = False

            while not done: 
                action = self.predict(state, is_training=True)

                next_state, reward, terminated, truncated , info = self.env.step(action)
                done = terminated or truncated

                self.update(state, action, reward, next_state)

                self.model[(state, action)] = (reward, next_state)

                for _ in range(self.num_planning_steps): # planning 

                    plan_state, plan_action_idx = random.choice(list(self.model.keys()))
                    plan_reward, plan_next_state = self.model[(plan_state, plan_action_idx)]

                    self.update(plan_state, plan_action_idx, plan_reward, plan_next_state)

                state = next_state


    def save(self, file_path):
        """
        Saves the agent's state to a file

        Arguments:

        save:
            path of file to be saved to
        """
        self.env = None
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
        print(f"Agent saved to {file_path}")

    @staticmethod
    def load(file_path):
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
        print(f"Agent loaded from {file_path}")
        return agent
    
    @staticmethod
    def create_model(env:gym.Env):
        """
        Create a model from terminal

        Parameters:
            env:
                environment (custom or open ai environmnent)  
        """
        print("Specify parameters for the dynaQ model\n")

        learning_rate = helper.valid_parameter("Learning rate", float, [0,1])
        gamma = helper.valid_parameter("Discount factor", float, [0,1])
        epsilon = helper.valid_parameter("Exploration rate", float, [0,1])
        num_planning_steps = helper.valid_parameter("Number of planning steps", int, [1,np.inf])
    
        return DynaQ(env, learning_rate, gamma, epsilon, num_planning_steps)
    
