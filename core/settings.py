import numpy as np
import os

# import environments
from environment.openAI.frozenlake_env import create_frozen_lake_environment, save_frozen_lake_env, load_frozen_lake_env, get_action_names_frozen_lake_environment
from environment.custom.trustgame_env import create_trustgame_environment, save_trustgame_env, load_trustgame_env
from environment.openAI.mountainCar_env import create_mountain_car_environment, save_mountain_car_env, load_mountain_car_env, get_action_names_mountain_car_environment

# import rl models
# custom
from models.custom.modelfree_RL.tabular.qlearning import Qlearning
from models.custom.modelfree_RL.tabular.sarsa import Sarsa
from models.custom.modelfree_RL.tabular.montecarlo import MonteCarlo
from models.custom.modelbased_RL.dynaQ import DynaQ

# Stablebaselines3
from models.openAI.dqn import create_dqn_model, load_dqn_model
from models.openAI.ars import create_ars_model, load_ars_model
from models.openAI.ppo import create_ppo_model, load_ppo_model

list_of_openAI_environments = {
    "FrozenLake-v1": ["Discrete", "Discrete"],
    "MountainCar-v0": ["Box", "Discrete"]
}

list_of_custom_environments = {
    "Trustgame": ["Discrete", "Discrete"],
}


models_compatibility_action_space = {
    "Box": ["ARS","PPO"],
    "Discrete": ["QLearning", "Sarsa", "Montecarlo","DQN", "ARS","PPO", "DynaQ"],
    "MultiDiscrete": ["PPO"],
    "MultiBinary": ["PPO"],
}

models_compatibility_observation_space = {
    "Box": ["DQN", "ARS", "PPO"],
    "Discrete": ["QLearning", "Sarsa", "Montecarlo", "DQN", "ARS","PPO", "DynaQ"],
    "MultiDiscrete": ["DQN", "ARS","PPO"],
    "MultiBinary": ["DQN", "ARS","PPO"],
}

def create_custom_environment(environment_name, folder_path):
    env = None
    parameters = None
    if environment_name == "Trustgame":
        env, parameters = create_trustgame_environment()
        save_trustgame_env(f"{folder_path}/{environment_name}.environment", parameters)
    else:
        pass

    # specify more environments
    return env, parameters

def create_openAI_environment(environment_name, folder_path):
    env = None
    parameters = None
    
    if environment_name == "FrozenLake-v1":
        env, parameters = create_frozen_lake_environment()
        save_frozen_lake_env(f"{folder_path}//{environment_name}.environment", parameters)
    elif environment_name == "MountainCar-v0":
        env, parameters = create_mountain_car_environment()
        save_mountain_car_env(f"{folder_path}//{environment_name}.environment", parameters)
    else:
        pass

    # specify more environments
    return env, parameters

def create_model_and_learn(model_name, folder_name, num_episodes, env):
    model = None
    if model_name == "QLearning":
        model = Qlearning.create_model(env)
        model.learn(num_episodes)
        model.save(f"../trained_models/{folder_name}/{model_name}.model")

    elif model_name == "Sarsa":
        model = Sarsa.create_model(env)
        model.learn(num_episodes)
        model.save(f"../trained_models/{folder_name}/{model_name}.model")

    elif model_name == "Montecarlo":
        model = MonteCarlo.create_model(env)
        model.learn(num_episodes)
        model.save(f"../trained_models/{folder_name}/{model_name}.model")

    elif model_name == "DQN":
        model = create_dqn_model(env)
        model.learn(total_timesteps=num_episodes)
        model.save(f"../trained_models/{folder_name}/{model_name}")

    elif model_name == "ARS":
        model = create_ars_model(env)
        model.learn(total_timesteps=num_episodes)
        model.save(f"../trained_models/{folder_name}/{model_name}")

    elif model_name == "PPO":
        model = create_ppo_model(env)
        model.learn(total_timesteps=num_episodes, reset_num_timesteps= False)
        model.save(f"../trained_models/{folder_name}/{model_name}")

    elif model_name == "DynaQ":
        model = DynaQ.create_model(env)
        model.learn(num_episodes)
        model.save(f"../trained_models/{folder_name}/{model_name}.model")

def load_environment(folder_path):
    """
    Loads environment from the trained model folder

    parameters:

    folder_name:
        name of the folder where the model was trained
    """

    content = os.listdir(folder_path)

    idx = None
    for i, c in enumerate(content):
        if len(c.split(".")) >=2:
            if c.split(".")[1] == "environment":
                idx = i

    environment_name = content[idx].split(".")[0]

    env = None
    path = f"{folder_path}/{content[idx]}"

    if environment_name == "FrozenLake-v1":
        env = load_frozen_lake_env(path)
    elif environment_name == "Trustgame":
        env = load_trustgame_env(path)
    elif environment_name == "MountainCar-v0":
        env = load_mountain_car_env(path)

    return env

def load_model(folder_name):
    """
    Loads model from the trained model folder

    parameters:

    folder_name:
        name of the folder where the model was trained
    """
    directory = f"../trained_models/{folder_name}"
    content = os.listdir(directory)
    idx = None
    for i, c in enumerate(content):
        if len(c.split(".")) >=2:
            if c.split(".")[1] == "model" or c.split(".")[1] == "zip":
                idx = i

    model = None
    model_name = content[idx].split(".")[0]
    path = f"{directory}/{content[idx]}"

    if model_name == "QLearning":
        model = Qlearning.load(path)
    elif model_name == "Sarsa":
        model = Sarsa.load(path)
    if model_name == "Montecarlo":
        model = MonteCarlo.load(path)
    if model_name == "DQN":
        model = load_dqn_model(path)
    if model_name == "ARS":
        model = load_ars_model(path)
    if model_name == "PPO":
        model = load_ppo_model(path)
    if model_name == "DynaQ":
        model = DynaQ.load(path)
    
    return model
    
def get_action_names(env_name):
    """
    Get the list of names for the actions 

    Parameters:

    env_name:
        name of the environment
    """
    if env_name ==  "FrozenLake-v1":
        return get_action_names_frozen_lake_environment()
    elif env_name == "MountainCar-v0":
        return get_action_names_mountain_car_environment()
    else:
        pass