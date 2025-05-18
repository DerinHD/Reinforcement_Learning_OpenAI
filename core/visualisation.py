import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from environment.custom.trustgame_env import TrustGameEnv
import helper
import settings

"""
-------------------------------------------------------------
This file contains visualization functions for the project.
-------------------------------------------------------------

Option 1: Visualize rewards during training
Input:
    - folder names of trained models
    - window size for averaging the rewards
Output:
    - plot of average rewards over the specified window size for each trained model
    - x-axis: episodes
    - y-axis: reward (average over window size)
    - legend: trained models
    - title: "Average rewards over window size"

-------------------------------------------------------------
Option 2: Visualize reward function of frozenlake environment
Input:
    - folder name of the trained model
Output:
    - plot of the reward function for each action
    - legend: actions
    - title: "Reward function of frozenlake environment"
    - colorbar: scalar value

--------------------------------------------------------------
Option 3: Visualize trust game probabilities
Input:
    - None
Output:
    - plot of the payback probabilities for each state
    - x-axis: payback
    - y-axis: probability
    - title: "Payback Probabilities"
    - legend: states

-------------------------------------------------------------
Option 4: Visualize trust game actions
Input:
    - folder names of trained models
Output:
    - plot of the normalized action histogram for each state
    - x-axis: action taken
    - y-axis: proportion
    - title: "Normalized Action Histogram for State {state}"
    - legend: states
-------------------------------------------------------------

Example how to use this module:

cd into the directory where the file is located and run the following command:
python visualizations.py
-------------------------------------------------------------
"""


def getSuccessRateWindowed(window_size, rewards):
    """
    Calculate the success rate over a window size.
    Args:
        window_size (int): The size of the window.
        rewards (list): List of rewards.
    Returns:
        tuple: A tuple containing the windows and success rate.
    """
    num_windows = int(np.ceil(len(rewards) / window_size)) # number of windows
    
    # calculate success rate for each window
    success_counts = [ 
            sum(rewards[i * window_size:(i + 1) * window_size]) / len(rewards[i * window_size:(i + 1) * window_size]) for i in range(num_windows)
        ]
    windows = np.arange(window_size, len(rewards)+window_size, window_size) # window size
    return windows, success_counts

def visualize_rewards_training():
    """
    Visualize the rewards during training for different trained models.
    The user can input the folder names of the trained models and specify the window size for averaging the rewards.
    The function will plot the average rewards over the specified window size for each trained model.
    """
   
    rewards = {}
    folder_names = []
    while True: 
        folder_name = input(f"Name folder of trained model ({int.__name__}): \nAnswer: ")
        try:
            if os.path.exists(f"../data/trainedModels/{folder_name}") and folder_name not in folder_names:
                folder_names.append(folder_name)
                
                reward_path = f"../data/trainedModels/{folder_name}/rewards.data"
                with open(reward_path, 'rb') as file:
                    rewards[folder_name] = pickle.load(file)

                input_stop = helper.get_valid_input("\nStop (If not, you can enter more trained models for comparison)?\n1)Yes\n2)No", ["1", "2"])
                if input_stop == "1":
                    break
            else:
                raise Exception
        except Exception as e:
            print("Invalid file name. Try again\n")

    window_size = None

    while True:
        try:
            user_input = input(f"\nSpecify the value for window_size ({int.__name__}): \n ")
            window_size = int(user_input)
            if window_size <= 0:
                raise ValueError
            break
        except ValueError:
            print(f"Invalid input. Try again.")

    for key,value in rewards.items():
        windows, success_rate = getSuccessRateWindowed(window_size, value)

        plt.plot(windows, success_rate, label = key)
    
    plt.xlabel('Episodes')
    plt.ylabel(f'Reward (Average over window size={window_size})')
    plt.legend()
    plt.show()

def frozen_lake_reward_visualization():
    """
    Visualize the reward function of a trained model on the FrozenLake environment.
    The user can input the folder name of the trained model, and the function will load the model and visualize the reward function.
    """
    input_trained_model = None
    while True:
        input_trained_model = input("Name the folder of the reward function model: \n")
        if os.path.exists(f"../data/rewardFunctionData/{input_trained_model}"):
            print("\n")
            break
        else:
            print("Invalid file name. Try again\n")

    env, _, _ = settings.load_environment(f"../data/rewardFunctionData/{input_trained_model}", render_mode="human")

    theta = None

    with open(f"../data/rewardFunctionData/{input_trained_model}/reward_function.data", 'rb') as file:
        theta = pickle.load(file)

    map_size = int(np.sqrt(env.observation_space.n))
    grid = np.zeros((4, map_size, map_size ))

    for i in range(env.observation_space.n):
        position_2D__x = i % map_size
        position_2D__y = i // map_size
        rewards = np.zeros(4)

        for j in range(env.action_space.n):
            f = env.transform_state_action_to_feature(i, j)
            reward = theta@f
            rewards[j] = reward
            grid[j][position_2D__x][position_2D__y] = reward
        
    
    action_mapping = {
        0: "left",
        1:"down",
        2: "right",
        3: "up"
    }
    fig, ax = plt.subplots(1,4)
    for i in range(4):
        im = ax[i].imshow(grid[i], cmap='hot', interpolation='nearest')
        fig.colorbar(im, ax=ax[i], label='Scalar Value')
        ax[i].set_title(f'Action {action_mapping[i]}')
        ax[i].set_xlabel('X Position')
        ax[i].set_ylabel('Y Position')

    plt.show()

def visualize_trust_game_probabilities():
    """
    Visualize the payback probabilities for different states in the Trust Game environment.
    The function will plot the payback probabilities for each state.
    """

    env = TrustGameEnv(num_rounds=10)

    p_0 = env.payback_prob(state=0, investment=50)
    p_1 = env.payback_prob(state=1, investment=50)
    p_2 = env.payback_prob(state=2, investment=50)

    p = [p_0, p_1, p_2]

    for i in range(len(p)):
        x = list(p[i].keys())
        y = list(p[i].values())

        plt.plot(x, y, label=f"State {i}")
    plt.xlabel('Payback')
    plt.ylabel('Probability')
    plt.title('Payback Probabilities')
    plt.legend()
    plt.show()


def visualize_trust_game_actions():
    """
    Visualize the normalized action histogram for different states in the Trust Game environment.
    The user can input the folder names of the trained models, and the function will load the models and visualize the actions taken in each state.
    """
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
        # 1. Load the environment and model
        # 2. Load the trained model
        env, _, _ = settings.load_environment(f"../data/trainedModels/{input_trained_model}", render_mode="rgb_array")
        model = settings.load_model(f"../data/trainedModels/{input_trained_model}")

        n_eval_episodes = 100

        # 2. Evaluate the model
        state_0_actions = []
        state_1_actions = []
        state_2_actions = []

        for i in range(n_eval_episodes):
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

                # Store the action taken in the corresponding state
                if obs == 0:
                    state_0_actions.append(action)
                elif obs == 1:
                    state_1_actions.append(action)
                elif obs == 2:
                    state_2_actions.append(action)

                observation, reward, terminated, truncated, info = env.step(action)

                episode_over = terminated or truncated

                obs = observation

        # 3. Visualize actions
        state_actions = {
            0: state_0_actions,
            1: state_1_actions,
            2: state_2_actions
        }

        for state, actions in state_actions.items():
            values, counts = np.unique(actions, return_counts=True)
            proportions = counts / counts.sum()

            # bar‐chart of proportions
            plt.figure()
            plt.bar(values, proportions, width=0.6, edgecolor='k')
            plt.xticks(values)  # ensure each action value is shown
            plt.ylim(0, 1)
            plt.xlabel("Action Taken")
            plt.ylabel("Proportion")
            plt.title(f"Normalized Action Histogram for State {state}")
            plt.tight_layout()
            plt.show()


# add more visualization methods
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
\nThis program allows the user to visualize on a specific environment via terminal. \
\nThe model as well as the environment can be custom or from the OpenAI gymnasium/stablebaselines3 library.\
\nTo visualize results of the trained model(e.g. rewards during training), run the file visualizations.py.\n"
    
    print(logo+welcomeText)

    prompt = "Which of the following visualizations do you want to use? \n"


    # This file contains visualization functions for the project.
    visualization_options = [
        "visualize rewards during training",
        "visualize reward function of frozenlake environment",
        "visualize trust game probabilities",
        "visualize trust game actions",
    ]

    pairs, single_choice_prompt = helper.create_single_choice(visualization_options)

    input_single_choice_idx = helper.get_valid_input(prompt + single_choice_prompt,  pairs.keys()) # get index of environment 
    visualization_name = pairs[input_single_choice_idx]

    if visualization_name == visualization_options[0]:
        visualize_rewards_training()
    if visualization_name == visualization_options[1]:
        frozen_lake_reward_visualization()
    if visualization_name == visualization_options[2]:
        visualize_trust_game_probabilities()
    if visualization_name == visualization_options[3]:
        visualize_trust_game_actions()

if __name__ == "__main__":
    main()