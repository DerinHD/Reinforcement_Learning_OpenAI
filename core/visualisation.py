import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import helper
import settings

visualization_options = {
    "visualize rewards during training",
    "visualize reward function of frozenlake environment"
}

def getSuccessRateWindowed(window_size, rewards):
    num_windows = int(np.ceil(len(rewards) / window_size))
    success_counts = [
            sum(rewards[i * window_size:(i + 1) * window_size]) /window_size for i in range(num_windows)
        ]
    windows = np.arange(window_size, len(rewards)+window_size, window_size)
    return windows, success_counts

def visualize_rewards_training():
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
    
    plt.legend()
    plt.show()

def frozen_lake_reward_visualization():
    input_trained_model = None
    while True:
        input_trained_model = input("Name the folder of the reward function model: \n")
        if os.path.exists(f"../reward_function_data/{input_trained_model}"):
            print("\n")
            break
        else:
            print("Invalid file name. Try again\n")

    env, _, _ = settings.load_environment(f"../reward_function_data/{input_trained_model}", render_mode="human")
    model = settings.load_model(f"../reward_function_data/{input_trained_model}")

    theta = None

    with open(f"../reward_function_data/{input_trained_model}/reward_function.data", 'rb') as file:
        theta = pickle.load(file)

    map_size = int(np.sqrt(env.observation_space.n))
    grid = np.zeros((4, map_size, map_size ))
    grid_arrow = np.zeros((map_size, map_size ))

    arrow = {
        3: (0, 0.5),
        1: (0, -0.5),
        0: (-0.5, 0),
        2: (0.5, 0)
        }

    for i in range(env.observation_space.n):
        position_2D__x = i % map_size
        position_2D__y = i // map_size
        rewards = np.zeros(4)

        for j in range(env.action_space.n):
            f = env.transform_state_action_to_feature(i, j)
            reward = theta@f
            rewards[j] = reward
            grid[j][position_2D__x][position_2D__y] = reward

        max_reward = np.argmax(rewards)
        grid_arrow[position_2D__x][position_2D__y] = max_reward
        
    
    action_mapping = {
        0: "left",
        1:"down",
        2: "right",
        3: "up"
    }
    fig, ax = plt.subplots(1,5)
    for i in range(4):
        print(grid[i])
        im = ax[i].imshow(grid[i], cmap='hot', interpolation='nearest',  vmin=-100, vmax=100)
        fig.colorbar(im, ax=ax[i], label='Scalar Value')
        ax[i].set_title(f'Action {action_mapping[i]}')
    
    X, Y = np.meshgrid(np.arange(4), np.arange(4))
    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)

    for i in range(4):
        for j in range(4):
            dx, dy = arrow[grid_arrow[i, j]]
            U[i, j] = dx
            V[i, j] = dy

    ax[4].quiver(X, Y, U, V, pivot='middle', color='r', scale=1, scale_units='xy')

    plt.show()

# add more visualization methods
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
\nThis program allows the user to visualize on a specific environment via terminal. \
\nThe model as well as the environment can be custom or from the OpenAI gymnasium/stablebaselines3 library.\
\nTo visualize results of the trained model(e.g. rewards during training), run the file visualizations.py.\n"
    
    print(logo+welcomeText)

    prompt = "Which of the following visualizations do you want to use? \n"

    pairs, single_choice_prompt = helper.create_single_choice(visualization_options)

    input_single_choice_idx = helper.get_valid_input(prompt + single_choice_prompt,  pairs.keys()) # get index of environment 
    visualization_name = pairs[input_single_choice_idx]
    print(visualization_name)

    if visualization_name == "visualize rewards during training":
        visualize_rewards_training()
    if visualization_name == "visualize reward function of frozenlake environment":
        frozen_lake_reward_visualization()

if __name__ == "__main__":
    main()