import pickle
import numpy as np
import os

# import environments
from environment.openAI.frozenlake_env import FrozenLake
from environment.custom.trustgame_env import TrustGameEnv
from environment.openAI.mountainCarDiscrete_env import MountainCarDiscreteState
from environment.openAI.mountainCarContinuous_env import MountainCarContinuousState

# import rl models
# custom
from models.custom.modelfree_RL.tabular.qlearning import Qlearning
from models.custom.modelfree_RL.tabular.sarsa import Sarsa
from models.custom.modelfree_RL.tabular.montecarlo import MonteCarlo
from models.custom.modelbased_RL.dynaQ import DynaQ

# Stablebaselines3
from models.stableBaselines3.dqn import DQNModel
from models.stableBaselines3.a2c import A2CModel
from models.stableBaselines3.ppo import PPOModel


"""
This file contains settings for the project.
"""

# List of environments
frozenlake = "FrozenLake-v1"
mountainCarDiscreteState = "MountainCarDiscreteState"
mountainCarContinuousState = "MountainCarContinuousState"
trustgame = "Trustgame"

# List of models
qlearning = "QLearning"
sarsa = "Sarsa"
montecarlo = "Montecarlo"
dynaQ = "DynaQ"
dqn = "DQN"
a2c = "A2C"
ppo = "PPO"

list_of_environments = {
    frozenlake: ["Discrete", "Discrete"],
    mountainCarDiscreteState: ["Discrete", "Discrete"],
    mountainCarContinuousState: ["Box", "Discrete"],
    trustgame: ["Discrete", "Discrete"],
}

models_compatibility_action_space = {
    "Box": [ppo, a2c],
    "Discrete": [qlearning,sarsa, montecarlo,dqn,ppo,dynaQ,a2c],
    "MultiDiscrete": [ppo,a2c],
    "MultiBinary": [ppo, a2c],
}

models_compatibility_observation_space = {
    "Box": [dqn, a2c, ppo, a2c],
    "Discrete": [qlearning,a2c,sarsa,montecarlo, dqn,ppo,dynaQ, a2c],
    "MultiDiscrete": [dqn,a2c,ppo],
    "MultiBinary": [dqn,ppo, a2c],
}

def create_environment(environment_name, folder_path):
    """
    Creates environment and saves it in the specified folder
    parameters:
    environment_name:
        name of the environment
    folder_path:
        path to the folder where the environment should be saved
    
    returns:
        env:
            environment
        parameters:
            parameters of the environment
    """         
    env = None
    parameters = None

    if environment_name == trustgame:
        env = TrustGameEnv.create_environment(render_mode="rgb_array")
    elif environment_name == frozenlake:
        env = FrozenLake.create_environment(render_mode="rgb_array")
    elif environment_name == mountainCarDiscreteState:
        env = MountainCarDiscreteState.create_environment(render_mode="rgb_array")
    elif environment_name == mountainCarContinuousState:   
        env = MountainCarContinuousState.create_environment(render_mode="rgb_array")
        # specify more environments
    else:
        raise Exception("Environment not found")

    env.save_environment(f"{folder_path}/{environment_name}.environment")
    parameters = env.parameters
    
    return env, parameters

def create_model(model_name, env):
    """
    Creates model and saves it in the specified folder
    parameters:
    model_name:
        name of the model
    env:
        environment
    folder_path:
        path to the folder where the model should be saved
    returns:
        model:
            model
    """
    model = None 

    if model_name == qlearning:
        model = Qlearning.create_model(env)
    elif model_name == sarsa:
        model = Sarsa.create_model(env)
    elif model_name == montecarlo:
        model = MonteCarlo.create_model(env)
    elif model_name == dqn:
        model = DQNModel.create_model(env)
    elif model_name == a2c:
        model = A2CModel.create_model(env)
    elif model_name == ppo:
        model = PPOModel.create_model(env)
    elif model_name == DynaQ:
        model = DynaQ.create_model(env)
    else:
        raise Exception("Model not found")
    

    return model

def create_model_and_learn(model_name, folder_name, num_episodes, env):
    """
    Creates model and learns it in the specified folder
    parameters:
    model_name:
        name of the model
    folder_name:
        name of the folder where the model should be saved
    num_episodes:
        number of episodes the model should be trained
    env:
        environment
    """
    # 1. Create model
    model = create_model(model_name, env)

    # 2. Create folder for trained model
    model.learn(num_episodes)
    
    # 3. Save model
    model.save(f"../data/trainedModels/{folder_name}/{model_name}.model")

def load_environment(folder_path, render_mode ="human"):
    """
    Loads environment from the trained model folder

    parameters:

    folder_name:
        name of the folder where the model was trained
    
    render_mode:
        render mode of the environment

    returns:
        env:
            environment
        parameters:
            parameters of the environment
        environment_name:
            name of the environment
    """

    content = os.listdir(folder_path)

    idx = None
    for i, c in enumerate(content):
        if len(c.split(".")) >=2: # check if file has extension
            if c.split(".")[1] == "environment":
                idx = i

    environment_name = content[idx].split(".")[0]

    env = None
    path = f"{folder_path}/{content[idx]}"

    if environment_name == frozenlake:
        env = FrozenLake.load_environment(path, render_mode)
    elif environment_name == trustgame:
        env = TrustGameEnv.load_environment(path, render_mode)
    elif environment_name == mountainCarDiscreteState:
        env = MountainCarContinuousState.load_environment(path, render_mode)
    elif environment_name == mountainCarContinuousState:
        env = MountainCarContinuousState.load_environment(path, render_mode)

    return env, env.parameters, environment_name

def load_model(folder_path):
    """
    Loads model from the trained model folder

    parameters:

    folder_name:
        name of the folder where the model was trained

    returns:
        model:
            model
    """
    content = os.listdir(folder_path)
    idx = None
    for i, c in enumerate(content):
        if len(c.split(".")) >=2: # check if file has extension
            if c.split(".")[1] == "model" or c.split(".")[1] == "zip": # zip is for stable baselines3
                idx = i

    model = None
    model_name = content[idx].split(".")[0]
    path = f"{folder_path}/{content[idx]}"

    if model_name == qlearning:
        model = Qlearning.load(path)
    elif model_name == sarsa:
        model = Sarsa.load(path)
    if model_name == montecarlo:
        model = MonteCarlo.load(path)
    if model_name == dqn:
        model = DQNModel.load(path)
    if model_name == a2c:
        model = A2CModel.load(path)
    if model_name == ppo:
        model = PPOModel.load(path)
    if model_name == dynaQ:
        model = DynaQ.load(path)
    
    return model
    
def load_demonstration_data(folder_path):
    """
    Loads demonstration data from the demonstrationData folder

    parameters:

    folder_name:
        name of the folder where the model was trained
    
    returns:
        demonstration_data:
            demonstration data
    """

    content = os.listdir(folder_path)

    idx = None
    for i, c in enumerate(content):
        if len(c.split(".")) >=2: # check if file has extension
            if c.split(".")[1] == "data":
                idx = i

    path = f"{folder_path}/{content[idx]}"

    demonstration_data = None

    with open(path, 'rb') as file:
        demonstration_data = pickle.load(file)
        
    return demonstration_data

def get_action_names(env_name):
    """
    Get the list of names for the actions 

    Parameters:

    env_name:
        name of the environment

    Returns:
        list of action names
    """
    if env_name == frozenlake:
        return FrozenLake.get_action_names()
    elif env_name == mountainCarDiscreteState:
        return MountainCarDiscreteState.get_action_names()
    elif env_name == mountainCarContinuousState:
        return MountainCarContinuousState.get_action_names()
    elif env_name == trustgame:
        return TrustGameEnv.get_action_names()
    else:
        pass
