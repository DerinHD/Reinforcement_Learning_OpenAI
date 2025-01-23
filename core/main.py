import pickle
import settings
import numpy as np
import os
from pathvalidate import is_valid_filename

# OpenAi wrappers
from gymnasium.wrappers import RecordVideo # used to record training as video
from stable_baselines3.common.monitor import Monitor # used to monitor rewards during training

import helper



"""
This file allows the user to train or evaluate a model on a specific environment via terminal.
The model as well as the environment can be custom or from the OpenAI gymnasium/stablebaselines3 library.

-------------------------------------------------------------------
Option 1: User chooses to train a model:

Input:
- Enter folder name of trained model
- Choose environment and configure its parameters
- Choose compatible model and configure its parameters 
- Choose how many episodes you want to train the model on the environment
- Choose if you want to record the training (will be saved as video or text file)

Note:  
    - You can't enter a name which already exists in the trained_models/ folder. Just delete the existing folder if you want to use the name)
    
Output:
- The trained model will be saved in the folder trained_models/.
- The folder contains
    - <environment_name>.environment: parameters of the environment
    - <model_name>.model: saved rl model
    - rewards.data: rewards for each episode in the training process

-------------------------------------------------------------------
Option 2: User choose to evaluate a model:

Input:
- Enter folder name of trained mode

Output: 
- An episode will be shown with the trained model
-------------------------------------------------------------------

"""

def episode_trigger(episode_id):
    return episode_id % 10 == 0


def train():
    """
    Trains a model on an environment and saves it in the directory trained_models
    """
    # 1. Enter folder name of trained model
    input_foldername = None
    while True:
        input_foldername= input("How do you want to name the trained model? Ensure that the folder name is valid and does not exist already:\nAnswer:")
        if is_valid_filename(input_foldername) and not os.path.exists(f"../trained_models/{input_foldername}"):
            os.makedirs(f"../trained_models/{input_foldername}")
            print("\n")
            break
        else:
            print("Invalid input. Please try again\n")

    # 2. User decides if he want to train on custom or Open AI environment
    env = None
    env_spaces = None

    input_env_custom_or_OpenAI = helper.get_valid_input("On which environment do you want to train a model? \n1) Custom environment \n2) Open AI environment", ["1","2"])
    
    env_list = None
    if input_env_custom_or_OpenAI == "1": # Custom environment was chosen
        env_list = settings.list_of_custom_environments
    else: # Open AI environment was chosen
        env_list = settings.list_of_openAI_environments

    
    # 3 User decides which environment from the given list he wants to use
    prompt = "Which of the following environments do you want to use? \n"

    pairs, single_choice_prompt = helper.create_single_choice(env_list)

    input_single_choice_idx = helper.get_valid_input(prompt + single_choice_prompt,  pairs.keys()) # get index of environment 
    environment_name = pairs[input_single_choice_idx]
     
    # 4 User configures environment parameters
    if input_env_custom_or_OpenAI == "1": # Custom environment was chosen
        env_spaces = settings.list_of_custom_environments[environment_name] # get observation and action space 
        env, env_parameters = settings.create_custom_environment(environment_name, input_foldername)
    else: # Open AI environment was chosen
        env_spaces = settings.list_of_openAI_environments[environment_name] # get observation and action space 
        env, env_parameters = settings.create_openAI_environment(environment_name, input_foldername)
    

    # get a list of models which are compatible with the chosen environment by looking at the observation and action spaces 
    list_of_compatible_models = np.intersect1d(settings.models_compatibility_observation_space[env_spaces[0]], settings.models_compatibility_action_space[env_spaces[1]])

    # 5. User decides which of the compatible models he wants to use
    prompt ="Which of the following compatible models do you want to use?\n"
    pairs, single_choice_prompt = helper.create_single_choice(list_of_compatible_models)

    input_model_idx = helper.get_valid_input(prompt + single_choice_prompt, pairs.keys())
    model_name = pairs[input_model_idx]

    # 6. Choose number of episodes to train
    num_episodes = helper.valid_parameter("Number of episodes", int, [1,np.inf])

    # 7. Choose if you want to record the training (will be saved as video or text file)
    input_record_or_not = helper.get_valid_input("Do you want to record the model during training?\n1)Yes \n2)No", ["1", "2"])

    if input_record_or_not == "1":
        episodes_to_record = helper.valid_parameter_list("episode to record", int, [1, num_episodes])

        input_video_or_not = helper.get_valid_input("\nDoes your environment supports video?\n1)Yes \n2)No ",  ["1","2"])
        if input_video_or_not =="1":
            try:
                env = RecordVideo(env, f"../trained_models/{input_foldername}/recordings", episode_trigger=lambda e: e in episodes_to_record)
            except AttributeError:
                print("Video not supported")
        else:
            pass
            #TODO


    print("Train_model...\n")

    env = Monitor(env=env) # wrap with Monitor class from stable-baselines3 library which monitors the records during training
    
    # 8. User configures model parameters and model gets trained
    settings.create_model_and_learn(model_name, input_foldername, num_episodes, env)

    # 9. Save records during training
    path_rewards = f"../trained_models/{input_foldername}/rewards.data"
    with open(path_rewards, 'wb') as file:
        pickle.dump(env.get_episode_rewards(), file)


def evaluate():

    # 1. Enter folder name of trained model
    input_trained_model = None
    while True:
        input_trained_model = input("Name the folder of the trained model: \n")
        if os.path.exists(f"../trained_models/{input_trained_model}"):
            print("\n")
            break
        else:
            print("Invalid file name. Try again\n")

    # 2. Load environment and model
    env = settings.load_environment(input_trained_model)
    model = settings.load_model(input_trained_model)

    print("Run an episode")

    obs,_ = env.reset()

    episode_over = False
    while not episode_over:
        action = None
        result = model.predict(obs)
        if isinstance(result, tuple):
            action, _ = result
        else:
            action = result

        action = action.item()

        observation, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated
        env.render()
        obs = observation

    print("Finish")

def main():
    logo = """
  _____  _         ____                              _____    _____          _      _                    
 |  __ \| |       / __ \                       /\   |_   _|  / ____|        | |    | |                   
 | |__) | |      | |  | |_ __   ___ _ __      /  \    | |   | |     ___   __| | ___| |__   __ _ ___  ___ 
 |  _  /| |      | |  | | '_ \ / _ \ '_ \    / /\ \   | |   | |    / _ \ / _` |/ _ \ '_ \ / _` / __|/ _ \\
 | | \ \| |____  | |__| | |_) |  __/ | | |  / ____ \ _| |_  | |___| (_) | (_| |  __/ |_) | (_| \__ \  __/
 |_|  \_\______|  \____/| .__/ \___|_| |_| /_/    \_\_____|  \_____\___/ \__,_|\___|_.__/ \__,_|___/\___|
                        | |                                                                              
                        |_|                                                                                                                         
        """

    welcomeText = "\nWelcome to my reinforcement learning Open AI codebase. \
\nThis program allows the user to train or evaluate a model on a specific environment via terminal. \
\nThe model as well as the environment can be custom or from the OpenAI gymnasium/stablebaselines3 library.\
\nTo visualize results of the trained model(e.g. rewards during training), run the file visualizations.py.\n"
    
    print(logo+welcomeText)

    input_train_or_evaluate = helper.get_valid_input("Do you want to train or evaluate a model? \n1)Train \n2)Evaluate", ["1","2"])
    
    if input_train_or_evaluate == "1":
        train() 
    else:
        evaluate() 

if __name__ == "__main__":
    main()






