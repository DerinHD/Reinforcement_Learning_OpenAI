import numpy as np
import gymnasium as gym
from collections import defaultdict
from tqdm import tqdm
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from models.baseModel import BaseModel

"""
This file contains the approx IRL algorithm.
"""

def approx_IRL(model: BaseModel, demonstration_data, num_iterations = 1000, alpha = 0.1, num_episodes_train = 1000, num_episodes_simulate = 100):
    """
    Approximate Inverse Reinforcement Learning (IRL) algorithm.
    Args:
        model (BaseModel): The model to be used for IRL.
        demonstration_data (list): The demonstration data to be used for IRL.
        num_iterations (int): The number of iterations to run the algorithm.
        alpha (float): The learning rate for the algorithm.
        num_episodes_train (int): The number of episodes to train the model.
        num_episodes_simulate (int): The number of episodes to simulate the model.
    Returns:
        theta (np.ndarray): The learned reward function.
        model (BaseModel): The trained model.
    """
    mu_agent_feature = np.zeros(model.env.feature_size)

    for e in range(len(demonstration_data)):
        for key, value in demonstration_data[e].items():
            state = value[0]
            action = value[1]
            feature = model.env.transform_state_action_to_feature(state, action)
            mu_agent_feature += feature

    mu_agent_feature /=  len(demonstration_data)

    theta = np.random.uniform(-1, 1, model.env.feature_size)  
    model.env.approx_IRL_configure(theta)
    model.env.reward_Function = "Approx_IRL"

    for _ in tqdm(range(num_iterations)):
        model.reset()
        model.learn(num_episodes_train)

        mu_policy_feature = np.zeros(model.env.feature_size)
        i=0
        # simulate policy
        for _ in range(num_episodes_simulate): 
            state, _ = model.env.reset()
            done = False
            while not done:
                action = model.predict(state=state)
                next_state, reward, terminated, truncated, _ = model.env.step(action)
                mu_policy_feature += model.env.transform_state_action_to_feature(state, action)
                done = terminated or truncated
                state = next_state
            i+=1

        mu_policy_feature /= i

        # update theta
        theta += alpha * (mu_agent_feature - mu_policy_feature)
        
        model.env.approx_IRL_configure(theta)
    
    return theta, model