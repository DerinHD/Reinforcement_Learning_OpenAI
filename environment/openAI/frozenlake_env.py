import os
import pickle
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import core.helper as helper

"""
Create, save and load Open AI Frozenlake environment.
More details about environment can be found here: https://gymnasium.farama.org/environments/toy_text/frozen_lake/.

Example how to use:

env = create_frozen_lake_environment()
model = Qlearning.create_model(env)
"""

    
def create_frozen_lake_environment():
    """
    Create Open AI Frozenlake environment by configuring its parameters via terminal.
    """
    print("Specify parameters for the frozenlake environment\n")

    parameters = []

    is_slippery_idx = helper.get_valid_input(f"Environment is slippery \n1)True \n2)False", ["1", "2"])
    is_slippery = False
    
    if is_slippery_idx == "1":
        is_slippery = True

    map_size = helper.valid_parameter("Map size", int, [2, 20])
    
    seed = helper.valid_parameter("Environment seed", int)

    prob_frozen = helper.valid_parameter("Probability that tile is frozen", float, [0.0,1.0])

    parameters = {"is_slippery": is_slippery, 
                  "map_size": map_size, 
                  'seed': seed, 
                  'prob_frozen': prob_frozen,
                  }

    return gym.make(
        "FrozenLake-v1",
        is_slippery=is_slippery,
        render_mode="rgb_array",
        desc=generate_random_map(
            size=map_size, seed=seed, p=prob_frozen
        ),
    ), parameters


def load_frozen_lake_env(file_path):
    """
    Load Open AI Frozenlake environment by providing the name of the environment file which contains parameter configuration


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
        "FrozenLake-v1",
        is_slippery=parameters["is_slippery"],
        render_mode="human",
        desc=generate_random_map(
            size=parameters["map_size"], seed=parameters["seed"], p=parameters["prob_frozen"]
        )
    )

def save_frozen_lake_env(file_path, parameters):
    """
    Save Open AI Frozenlake environment by saving its parameters

    parameters:

    file_path:
        path of file to be saved at
        
    parameters:
        environment parameters
    """
    with open(file_path, 'wb') as file:
        pickle.dump(parameters, file)

def get_action_names_frozen_lake_environment():
    """
    Get the list of names for the actions 
    """
    action_names = {
        0: "Move left",
        1: "Move down",
        2: "Move right",
        3: "Move up"
    }

    return action_names