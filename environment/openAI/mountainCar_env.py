import os
import pickle
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import core.helper as helper

"""
Create, save and load Open AI Mountain Car environment.
More details about environment can be found here: https://gymnasium.farama.org/environments/classic_control/mountain_car/.

Example how to use:

env = create_mountain_car_environment()
model = create_dqn_model(env)

"""
def create_mountain_car_environment():
    """
    Create Open AI Frozenlake environment by configuring its parameters via terminal.
    """
        
    print("Specify parameters for the mountain car environment\n")

    parameters = []
    
    print("\n")
    seed = helper.valid_parameter("Environment seed", int)

    parameters = {'seed': seed}

    return gym.make(
        id="MountainCar-v0",
        render_mode = "rgb_array"
        
    ), parameters


def load_mountain_car_env(file_path):
    """
    Load Open AI Mountain Car environment by providing the name of the environment file which contains parameter configuration

    parameters:

    file_path:
        path of file to be load to

    returns:
        environment parameters
    """
        
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'rb') as file:
        parameters = pickle.load(file)
            
    return gym.make(
        id="MountainCar-v0",
        render_mode = "human"
    )

def save_mountain_car_env(file_path, parameters):
    """
    Save Open AI Mountain Car environment by saving its parameters

    parameters:

    file_path:
        path of file to be saved at
        
    parameters:
        environment parameters
    """
    with open(file_path, 'wb') as file:
        pickle.dump(parameters, file)