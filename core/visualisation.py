import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import helper

visualization_options = {
    "visualize_rewards_training"
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
            if os.path.exists(f"../trained_models/{folder_name}") and folder_name not in folder_names:
                folder_names.append(folder_name)
                
                reward_path = f"../trained_models/{folder_name}/rewards.data"
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

    if visualization_name == "visualize_rewards_training":
        visualize_rewards_training()

if __name__ == "__main__":
    main()