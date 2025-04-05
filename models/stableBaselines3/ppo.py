
from stable_baselines3 import PPO

import gymnasium as gym
import core.helper as helper
import os

"""
Create Proximal Policy Optimization (PPO) via terminal. For more details, view https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html.
"""

@staticmethod
def create_ppo_model(env:gym.Env):
    print("Specify parameters for the dqn model")

    prompt = "Which of the following policys? \n"
    list_of_policies = ["MlpPolicy", "CnnPolicy", "MultiInputPolicy"]

    pairs, single_choice_prompt = helper.create_single_choice(list_of_policies)

    input_single_choice_idx = helper.get_valid_input(prompt + single_choice_prompt,  pairs.keys()) # get index of environment 
    input_policy = pairs[input_single_choice_idx]
    
    learning_rate = helper.valid_parameter("Learning rate", float, [0,1])
    gamma = helper.valid_parameter("Discount factor", float, [0,1])

    return PPO(policy=input_policy, learning_rate=learning_rate, gamma=gamma, env=env, verbose=1)

@staticmethod
def load_ppo_model(file_path):
    if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'rb') as file:
            model = PPO.load(file)
            
    print(f"Model loaded from {file_path}")
    return model

    
