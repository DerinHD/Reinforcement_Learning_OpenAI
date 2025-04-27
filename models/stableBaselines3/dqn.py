import os
import pickle
from stable_baselines3 import DQN
import gymnasium as gym
import core.helper as helper

"""
Create Deep Q Network (DQN) via terminal. For more details, view https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html. 
"""

@staticmethod
def create_dqn_model(env:gym.Env):
    print("Specify parameters for the dqn model")

    prompt = "Which of the following policys? \n"

    list_of_policies = ["MlpPolicy", "CnnPolicy", "MultiInputPolicy"]

    pairs, single_choice_prompt = helper.create_single_choice(list_of_policies)

    input_single_choice_idx = helper.get_valid_input(prompt + single_choice_prompt,  pairs.keys()) # get index of environment 
    input_policy = pairs[input_single_choice_idx]
    
    learning_rate = helper.valid_parameter("Learning rate", float, [0,1])
    gamma = helper.valid_parameter("Discount factor", float, [0,1])
    epsilon_init = helper.valid_parameter("Initial Exploration rate", float, [0,1])
    epsilon_fraction = helper.valid_parameter("exploration_fraction", float, [0,1])
    epsilon_end = helper.valid_parameter("Final Exploration rate", float, [0,1])

    return DQN(policy=input_policy, learning_rate=learning_rate, gamma= gamma,exploration_initial_eps= epsilon_init, exploration_fraction=epsilon_fraction,exploration_final_eps=epsilon_end,env=env, verbose=1)


@staticmethod
def load_dqn_model(file_path):
    if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'rb') as file:
            model = DQN.load(file)
            
    print(f"Model loaded from {file_path}")
    return model

    
