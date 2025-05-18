import numpy as np
import os
import io
import sys
from gymnasium import Wrapper

"""
This file contains helper functions for the project.
"""


def valid_parameter(parameter_name: str, parameter_type,  boundaries = [-np.inf, np.inf]):
    """
    Function which ensures that the user gives a valid input for the parameter.
    Works for numerical data 

    parameters:

    parameter_name:
        name of the parameter

    parameter_type:
        type of the parameter

    boundaries:
        define boundary of parameter (min and max allowed value)

    returns:
        parameter
    """

    while True:
        try: 
            user_input = input(f"Specify the value for {parameter_name} ({parameter_type.__name__}): \n")
            parameter = parameter_type(user_input)
            if parameter < boundaries[0] or parameter>boundaries[1]:
                raise ValueError
            print("\n")
            return parameter

        except ValueError:
            print(f"Invalid input. Try again.\n")


def valid_parameter_list(parameter_name: str, parameter_type,  boundaries = [-np.inf, np.inf]):
    """
    Function which ensures that the user gives valid inputs for a list of parameters. 
    The inputs will be saved in an array.
    Works for numerical data

    parameters:

    parameter_name:
        name of the parameter

    parameter_type:
        type of the parameter

    boundaries:
        define boundary of parameter (min and max allowed value)
    
    returns:
        parameter_list
            list of parameters
    """
    parameter_list = []
    while True: 
        try:
            user_input = input(f"\n{parameter_name} ({parameter_type.__name__}): \nAnswer:")
            episode = parameter_type(user_input)
            if episode < boundaries[0] or episode > boundaries[1] or episode in parameter_list:
                raise ValueError
            
            parameter_list.append(episode)
            
            input_stop = get_valid_input("\nStop?\n1)Yes\n2)No", ["1", "2"])
            if input_stop == "1":
                return parameter_list
        except ValueError:
            print("Invalid input. Please try again\n")

def get_valid_input(prompt, valid_choices):
    """ 
    Function which ensures that the user gives a valid input by providing valid choices.
    If not, the user is asked again.

    parameters:
    
    prompt: 
        Description of input
    
    valid_choices: 
        list of valid inputs

    returns:
        input of user
    """
    while True:
        prompt = prompt + "\nAnswer: "
        user_input = input(prompt)
        if user_input in valid_choices:
            print("\n")
            return user_input
        else:
            print("\nInvalid input. Please choose one of the following options.\n")

def create_single_choice(list):
    """
    Creates a single choice from list
    
    parameters:

    list:
        list with size >= 1

    returns:
        pairs:
            (number, list element) pair
        prompt_list:
            single choice list as string
    """

    pairs = {}
    prompt_list = ""

    for idx, name in enumerate(list):
        pairs[str(idx+1)] = name
        prompt_list += f"{idx+1}) {name}\n"

    return pairs, prompt_list


class TextRecordWrapper(Wrapper):
    """
    A wrapper that records the text output of the environment to a file.
    This is useful for environments that print information to the console.
    The wrapper will create a new file for each episode, and the file will
    be named according to the episode number.
    The wrapper will also write the episode number to the file at the start
    and end of each episode.

    Parameters:
    env: The environment to wrap.
    logdir: The directory where the log files will be saved.
    episode_trigger: A function that takes the episode number as input and
        returns True if the episode should be logged, and False otherwise.
        This can be used to log only certain episodes, for example, every 10th episode.

    Example usage:
        env = gym.make('CartPole-v1')
        env = TextRecordWrapper(env, logdir='logs', episode_trigger=lambda ep: ep % 10 == 0)
        obs = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
    """
    def __init__(self, env, logdir, episode_trigger=lambda ep: True):
        super().__init__(env) # Initialize the wrapper
        
        self.logdir = logdir
        os.makedirs(self.logdir, exist_ok=True) # Create the log directory if it doesn't exist
        
        self.episode_trigger = episode_trigger # Function to determine if the episode should be logged
        self.episode = 0 # Current episode number
        self._logfile = None # File object for logging

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs) # Reset the environment
        self.episode += 1

        if self.episode_trigger(self.episode): # Check if the episode should be logged
            path = os.path.join(self.logdir, f"episode_{self.episode:03d}.txt")
            self._logfile = open(path, 'w')
            self._write(f"=== Episode {self.episode} START ===\n")
        else:
            self._logfile = None # If not logging, set logfile to None

        self._maybe_record_render()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action) # Take a step in the environment

        self._maybe_record_render() # Record the render output if logging

        if terminated or truncated:
            self._close_log()
        return obs, reward, terminated, truncated, info

    def _maybe_record_render(self):
        if not self._logfile:
            return
        
        buf = io.StringIO() # Create a buffer to capture the output
        old_stdout = sys.stdout # Save the current stdout
        try: 
            sys.stdout = buf # Redirect stdout to the buffer
            self.env.render() # Render the environment         
        finally: 
            sys.stdout = old_stdout # Restore stdout to its original state  

        text = buf.getvalue().rstrip() # Get the captured output
        self._write(text + "\n")

    def _write(self, txt):
        self._logfile.write(txt) # Write the text to the log file

    def _close_log(self):
        if self._logfile:
            self._logfile.write(f"\n=== Episode {self.episode} END ===\n") # Write the end of episode marker
            self._logfile.close()
            self._logfile = None
