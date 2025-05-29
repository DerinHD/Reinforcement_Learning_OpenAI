import pickle
import settings
import numpy as np
import os
from pathvalidate import is_valid_filename
from collections import defaultdict

# OpenAi wrappers
from gymnasium.wrappers import RecordVideo # used to record training as video
from stable_baselines3.common.monitor import Monitor # used to monitor rewards during training
from inverseRL.approx_IRL import approx_IRL

import helper
import pygame

"""
-------------------------------------------------------------------
This file contains the main function of the project
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
- The trained model will be saved in the folder trainedModels/.
- The folder contains
    - <environment_name>.environment: parameters of the environment
    - <model_name>.model: saved RL model
    - rewards.data: rewards for each episode in the training process

-------------------------------------------------------------------
Option 2: User choose to evaluate a model:

Input:
- Enter folder name of trained mode

Output: 
- An episode will be shown with the trained models

-------------------------------------------------------------------

Option 3: User chooses to create demonstration data for inverse reinforcement learning:
Input:
- Enter folder name of demonstration data
- Choose environment and configure its parameters
Output:
- The demonstration data will be saved in the folder demonstrationData/.
- The folder contains
    - <environment_name>.environment: parameters of the environment
    - demonstration.data: demonstration data for inverse reinforcement learning
- The demonstration data is saved in a dictionary with the following structure:
    - {episode_number: {step_number: (state, action, reward)}}
    - The state is the observation of the environment
    - The action is the action taken by the model
    - The reward is the reward received by the model

-------------------------------------------------------------------

Option 4: User chooses to show demonstration data:
Input:
- Enter folder name of demonstration data
Output:
- The demonstration data will be shown in the terminal
- The demonstration data contains the state, action and reward for each step in the episode

-------------------------------------------------------------------

Option 5: User chooses to perform inverse reinforcement learning:
Input:
- Enter folder name of demonstration data
- Enter folder name of reward function data
Output:
- The reward function will be saved in the folder rewardFunctionData/.
- The folder contains
    - <environment_name>.environment: parameters of the environment
    - reward_function.data: reward function for the environment
    - <model_name>.model: saved RL model trained on the reconstructed reward function
- The reward function is learned from the demonstration data using inverse reinforcement learning

-------------------------------------------------------------------
Option 6: User chooses to show reward function:
Input:
- Enter folder name of reward function data
Output:
- The reward function will be shown in the terminal
- The reward function is the learned reward function from the inverse reinforcement learning
-------------------------------------------------------------------
Option 7: User chooses to run a model with the learned reward function:
Input:
- Enter folder name of reward function data
Output:
- An episode will be shown with the trained model
-------------------------------------------------------------------

Example how to use the module:
cd into the directory where the file is located and run the following command:
python main.py
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
            try:
                env = helper.TextRecordWrapper(env, f"../data/trainedModels/{input_foldername}/recordings", episode_trigger=lambda e: e in episodes_to_record)
            except AttributeError:
                print("Text recording not supported")
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
    folder_names = []
    while True: 
        folder_name = input(f"Name folder of trained model: \nAnswer: ")
        try:
            if os.path.exists(f"../data/trainedModels/{folder_name}") and folder_name not in folder_names:
                folder_names.append(folder_name)

                input_stop = helper.get_valid_input("\nStop (If not, you can enter more trained models for comparison)?\n1)Yes\n2)No", ["1", "2"])
                if input_stop == "1":
                    break
            else:
                raise Exception
        except Exception as e:
            print("Invalid file name. Try again\n")

    for input_trained_model in folder_names:
        # 2. Load environment and model
        env, _, _ = settings.load_environment(f"../data/trainedModels/{input_trained_model}")
        model = settings.load_model(f"../data/trainedModels/{input_trained_model}")

        n_eval_episodes = 1 # number of episodes to evaluate the model
        episode_rewards = []
        for i in range(n_eval_episodes):
            total_rewards = 0

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
                total_rewards+= reward

                episode_over = terminated or truncated
                env.render()
                obs = observation

            episode_rewards.append(total_rewards)
            
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

    input_video_or_text = helper.get_valid_input("Does the env render graphically or prints the output on terminal?\n1)GUI \n2)Terminal", ["1", "2"])
    is_gui = input_video_or_text == "1"

    # Run a specific number of episodes to generate demonstration data
    demonstration_data = []
    data_for_df = []
    done = False
    while not done:
        print("Actions available:")
        action_names = settings.get_action_names(environment_name)
        for key, value in action_names.items():
            print(f"Action index: {key}, value: {value}")
        print("\n")

        obs,_ = env.reset()
        transitions = {}
        i = 0
        episode_over = False
        while not episode_over:
            if is_gui:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        episode_over = True
                    if event.type == pygame.KEYDOWN:
                        if event.key in key_to_action and key_to_action[event.key] in action_names.keys():
                            action = key_to_action[event.key]  

                            observation, reward, terminated, truncated, info = env.step(action)

                            transitions[i] = [obs, action, reward]

                            episode_over = terminated or truncated

                            obs = observation
                            i +=1
            else:
                action = helper.get_valid_input("Enter the action index: ", [str(i) for i in action_names.keys()])
                action = int(action)

                observation, reward, terminated, truncated, info = env.step(action)

                transitions[i] = [obs, action, reward]

                episode_over = terminated or truncated

                obs = observation
                i +=1
            
            env.render()
        
        demonstration_data.append(transitions)
        input_done = helper.get_valid_input("\nDo you want to play another episode? \n1)Yes\n2)No", ["1", "2"])

        if input_done == "2":
            done = True


    with open(f"../data/demonstrationData/{input_foldername}/demonstration.data", 'wb') as file:
        pickle.dump(demonstration_data, file)

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

    num_episodes_train = helper.valid_parameter("Number of episodes to train the model", int, [1, np.inf])
    num_episodes_simulate = helper.valid_parameter("Number of episodes to simulate the model", int, [1, np.inf])

    theta, model = approx_IRL(model, demonstration_data, num_iterations = 300, alpha = 0.1, num_episodes_train=num_episodes_train,num_episodes_simulate=num_episodes_simulate)

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
  _____  _          _____          _      _                    
 |  __ \| |        / ____|        | |    | |                   
 | |__) | |       | |     ___   __| | ___| |__   __ _ ___  ___ 
 |  _  /| |       | |    / _ \ / _` |/ _ \ '_ \ / _` / __|/ _ \\
 | | \ \| |____   | |___| (_) | (_| |  __/ |_) | (_| \__ \  __/
 |_|  \_\______|   \_____\___/ \__,_|\___|_.__/ \__,_|___/\___|                                                                                                                                       
        """

    welcomeText = "\nWelcome to my reinforcement learning Open AI codebase. \
\nThis program allows the user to train or evaluate a model on a specific environment via terminal. \
\nThe model as well as the environment can be custom or from the OpenAI gymnasium/stablebaselines3 library.\
\nTo visualize results of the trained model(e.g. rewards during training), run the file visualizations.py.\n"
    
    print(logo+welcomeText)

    prompt = "What do you want to do?\n"

    main_options = [
        "Train an agent",
        "Evaluate an agent",
        "Create demonstration data",
        "Show demonstration data",
        "Perform inverse RL to learn reward function",
        "Show reward function",
        "Run model with learned reward function"    
    ]

    pairs, single_choice_prompt = helper.create_single_choice(main_options)

    input_single_choice_idx = helper.get_valid_input(prompt + single_choice_prompt,  pairs.keys()) # get index of environment 
    main_name = pairs[input_single_choice_idx]

    if main_name == main_options[0]:
        train() 
    elif main_name == main_options[1]:  
        evaluate() 
    elif main_name == main_options[2]:
        create_demonstration_data()
    elif main_name == main_options[3]:
        show_demonstration_data() 
    elif main_name == main_options[4]:
        inverse_RL() 
    elif main_name == main_options[5]:
        showRewardFunction() 
    else:
        show_inverse_RL() 

if __name__ == "__main__":
    main()