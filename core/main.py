import pickle
import settings
import numpy as np
import os
from pathvalidate import is_valid_filename
from collections import defaultdict

# OpenAi wrappers
from gymnasium.wrappers import RecordVideo # used to record training as video
from stable_baselines3.common.monitor import Monitor # used to monitor rewards during training
from inverseRL.max_entropy import max_entropy

import helper
import pygame

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
        if is_valid_filename(input_foldername) and not os.path.exists(f"../data/trainedModels/{input_foldername}"):
            os.makedirs(f"../data/trainedModels/{input_foldername}")
            print("\n")
            break
        else:
            print("Invalid input. Please try again\n")

    # 2. User decides if he want to train on custom or Open AI environment
    env_list = settings.list_of_environments

    # 3 User decides which environment from the given list he wants to use
    prompt = "Which of the following environments do you want to use? \n"

    pairs, single_choice_prompt = helper.create_single_choice(env_list)

    input_single_choice_idx = helper.get_valid_input(prompt + single_choice_prompt,  pairs.keys()) # get index of environment 
    environment_name = pairs[input_single_choice_idx]
     
    # 4 User configures environment parameters
    env_spaces = settings.list_of_environments[environment_name] # get observation and action space 
    env, _ = settings.create_environment(environment_name, f"../data/trainedModels/{input_foldername}")

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
                env = RecordVideo(env, f"../data/trainedModels/{input_foldername}/recordings", episode_trigger=lambda e: e in episodes_to_record)
            except AttributeError:
                print("Video not supported")
        else:
            pass
            #TODO
    print(env)

    print("Train_model...\n")

    env = Monitor(env=env) # wrap with Monitor class from stable-baselines3 library which monitors the records during training
    
    # 8. User configures model parameters and model gets trained
    settings.create_model_and_learn(model_name, input_foldername, num_episodes, env)

    # 9. Save records during training
    path_rewards = f"../data/trainedModels/{input_foldername}/rewards.data"
    with open(path_rewards, 'wb') as file:
        pickle.dump(env.get_episode_rewards(), file)


def evaluate():
    # 1. Enter folder ncreate_demonstration_dataame of trained model
    input_trained_model = None
    while True:
        input_trained_model = input("Name the folder of the trained model: \n")
        if os.path.exists(f"../data/trainedModels/{input_trained_model}"):
            print("\n")
            break
        else:
            print("Invalid file name. Try again\n")

    # 2. Load environment and model
    env, _, _ = settings.load_environment(f"../data/trainedModels/{input_trained_model}")
    model = settings.load_model(f"../data/trainedModels/{input_trained_model}")

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

def show_inverse_RL():
    input_trained_model = None
    while True:
        input_trained_model = input("Name the folder of the reward function model: \n")
        if os.path.exists(f"../data/rewardFunctionData/{input_trained_model}"):
            print("\n")
            break
        else:
            print("Invalid file name. Try again\n")

    env, _, _ = settings.load_environment(f"../data/rewardFunctionData/{input_trained_model}", render_mode="human")
    model = settings.load_model(f"../data/rewardFunctionData/{input_trained_model}")

    theta = None

    with open(f"../data/rewardFunctionData/{input_trained_model}/reward_function.data", 'rb') as file:
        theta = pickle.load(file)

    env.reward_function = "max_entropy"
    env.max_entropy_configure(theta)

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

def create_demonstration_data():
    """
    Creates demonstration data for a specific environment which can be used for inverse reinforcement learning to generate an own reward function.
    """

    # 1. Enter folder name to save demonstration data and environment parameters
    input_foldername = None
    while True:
        input_foldername= input("How do you want to name the demonstration model? Ensure that the folder name is valid and does not exist already:\nAnswer:")
        if is_valid_filename(input_foldername) and not os.path.exists(f"../data/demonstrationData/{input_foldername}"):
            os.makedirs(f"../data/demonstrationData/{input_foldername}")
            print("\n")
            break
        else:
            print("Invalid input. Please try again\n")

    # 2. Choose environment
    env_with_discrete_action_space = defaultdict(list)

    for item in settings.list_of_environments.items():
        env_name, spaces = item
        action_space = spaces[1]
        if action_space == "Discrete":
            env_with_discrete_action_space[env_name] = spaces

    prompt = "Which of the following environments do you want to use? \n"

    pairs, single_choice_prompt = helper.create_single_choice(env_with_discrete_action_space)

    input_single_choice_idx = helper.get_valid_input(prompt + single_choice_prompt,  pairs.keys()) # get index of environment 
    environment_name = pairs[input_single_choice_idx]

    # Save environment parameters
    env, _ = settings.create_environment(environment_name, f"../data/demonstrationData/{input_foldername}")

    # Load environment with GUI visualization
    env, _, _ = settings.load_environment(f"../data/demonstrationData/{input_foldername}")

    # key list
    key_to_action = {
        pygame.K_0: 0, 
        pygame.K_1: 1, 
        pygame.K_2: 2, 
        pygame.K_3: 3, 
        pygame.K_4: 4, 
        pygame.K_5: 5, 
        pygame.K_6: 6, 
        pygame.K_7: 7, 
        pygame.K_8: 8,
        pygame.K_9: 9,
        #...
    }


    # Run a specific number of episodes to generate demonstration data
    demonstration_data = []
    data_for_df = []
    done = False
    while not done:
        print("Actions available:")
        action_names = settings.get_action_names(environment_name)
        for key, value in action_names.items():
            print(f"Action index: {key}, value: {value}")

        obs,_ = env.reset()
        transitions = {}
        i = 0
        episode_over = False
        while not episode_over:
            env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    episode_over = True
                if event.type == pygame.KEYDOWN:
                    if event.key in key_to_action and key_to_action[event.key] in action_names.keys():
                        action = key_to_action[event.key]  

                        observation, reward, terminated, truncated, info = env.step(action)

                        transitions[i] = [obs, action]

                        episode_over = terminated or truncated

                        obs = observation
                        i +=1
        
        demonstration_data.append(transitions)
        input_done = helper.get_valid_input("\nDo you want to play another episode? \n1)Yes\n2)No", ["1", "2"])

        if input_done == "2":
            done = True


    #with open(f"../data/demonstrationData/{input_foldername}/demonstration.data", 'wb') as file:
    #    pickle.dump(demonstration_data, file)

    print("Finish")

def show_demonstration_data():
    input_folder_name = None
    while True:
        input_folder_name = input("Name the folder which contains the demonstration data: \n")
        if os.path.exists(f"../data/demonstrationData/{input_folder_name}"):
            print("\n")
            break
        else:
            print("Invalid file name. Try again\n")

    #1. Show environment parameters
    env, parameters, _ = settings.load_environment(f"../data/demonstrationData/{input_folder_name}")

    print(f"Parameters of the environment: {parameters}\n")

    demonstration_data = settings.load_demonstration_data(f"../data/demonstrationData/{input_folder_name}")

    for i in range(len(demonstration_data)):
        print(f"Episode {i}:\n")
        for key, value in demonstration_data[i].items():
            print(f"step: {key} , (state, action): {value}")

def inverse_RL():
    input_folder_name = None
    while True:
        input_folder_name = input("Name the folder which contains the demonstration data: \n")
        if os.path.exists(f"../data/demonstrationData/{input_folder_name}"):
            print("\n")
            break
        else:
            print("Invalid file name. Try again\n")

    save_folder_name = None
    while True:
        save_folder_name = input("Name the folder in which the reward function data should be saved: \n")
        if is_valid_filename(save_folder_name) and not os.path.exists(f"../data/rewardFunctionData/{save_folder_name}"):
            os.makedirs(f"../data/rewardFunctionData/{save_folder_name}")
            break
        else:
            print("Invalid file name. Try again\n")

    #1. Show environment parameters
    env, parameters, environment_name = settings.load_environment(f"../data/demonstrationData/{input_folder_name}", render_mode="rgb_array")
    demonstration_data = settings.load_demonstration_data(f"../data/demonstrationData/{input_folder_name}")

    env_spaces = settings.list_of_environments[environment_name] 

    # get a list of models which are compatible with the chosen environment by looking at the observation and action spaces 
    list_of_compatible_models = np.intersect1d(settings.models_compatibility_observation_space[env_spaces[0]], settings.models_compatibility_action_space[env_spaces[1]])

    prompt ="Which of the following compatible models do you want to use?\n"
    pairs, single_choice_prompt = helper.create_single_choice(list_of_compatible_models)

    input_model_idx = helper.get_valid_input(prompt + single_choice_prompt, pairs.keys())
    model_name = pairs[input_model_idx]
    model = settings.create_model(model_name, env)

    theta, model = max_entropy(model, demonstration_data, num_iterations = 300, alpha = 0.1)

    with open(f"../data/rewardFunctionData/{save_folder_name}/reward_function.data", 'wb') as file:
        pickle.dump(theta, file)

    with open(f"../data/rewardFunctionData/{save_folder_name}/{environment_name}.environment", 'wb') as file:
        pickle.dump(parameters, file)

    model.save(f"../data/rewardFunctionData/{save_folder_name}/{model_name}.model")

def showRewardFunction():
    input_trained_model = None
    while True:
        input_trained_model = input("Name the folder of the reward function model: \n")
        if os.path.exists(f"../data/rewardFunctionData/{input_trained_model}"):
            print("\n")
            break
        else:
            print("Invalid file name. Try again\n")

    env, _, _ = settings.load_environment(f"../data/rewardFunctionData/{input_trained_model}", render_mode="human")
    model = settings.load_model(f"../data/rewardFunctionData/{input_trained_model}")

    theta = None

    with open(f"../data/rewardFunctionData/{input_trained_model}/reward_function.data", 'rb') as file:
        theta = pickle.load(file)

    env.reward_function = "max_entropy"
    env.max_entropy_configure(theta)

    for i in range(env.observation_space.n):
        for j in range(env.action_space.n):
            f = env.transform_state_action_to_feature(i, j)
            print(f"state: {i}, action: {j}: feature: {f} reward: {theta@f}")

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

    input_train_or_evaluate = helper.get_valid_input("Choose one of the following options?  \
                                                     \n1) Train an agent \
                                                     \n2) Evaluate an agent \
                                                     \n3) Create demonstration data \
                                                     \n4) Show demonstration data \
                                                     \n5) Perform inverse RL to learn reward function \
                                                     \n6) Show reward function \
                                                     \n7) Run model with learned reward function \n",\
                                                     ["1","2", "3", "4", "5", "6", "7"])

    if input_train_or_evaluate == "1":
        train() 
    elif input_train_or_evaluate == "2":
        evaluate() 
    elif input_train_or_evaluate == "3":
        create_demonstration_data()
    elif input_train_or_evaluate == "4":
        show_demonstration_data() 
    elif input_train_or_evaluate == "5":
        inverse_RL() 
    elif input_train_or_evaluate == "6":
        showRewardFunction() 
    else:
        show_inverse_RL() 

if __name__ == "__main__":
    main()