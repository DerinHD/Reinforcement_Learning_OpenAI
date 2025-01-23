from sb3_contrib.ars import ARS

from stable_baselines3 import DQN
import gymnasium as gym
import core.helper as helper
import os

"""
Create Augmented Random Search (ARS) via terminal. 
For more details, view https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/tree/master/sb3_contrib/ars and
https://arxiv.org/abs/1803.07055.
"""

@staticmethod
def create_ars_model(env:gym.Env):
    print("Specify parameters for the ars model")

    prompt = "Which of the following policys? \n"
    list_of_policies = ["MlpPolicy","LinearPolicy"]

    pairs, single_choice_prompt = helper.create_single_choice(list_of_policies)

    input_single_choice_idx = helper.get_valid_input(prompt + single_choice_prompt,  pairs.keys()) # get index of environment 
    input_policy = pairs[input_single_choice_idx]
    
    learning_rate = helper.valid_parameter("Learning rate", float, [0,1])

    return ARS(policy=input_policy, learning_rate= learning_rate, env=env, verbose=1)

@staticmethod
def load_ars_model(file_path):
    if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'rb') as file:
            model = ARS.load(file)
            
    print(f"Model loaded from {file_path}")
    return model
