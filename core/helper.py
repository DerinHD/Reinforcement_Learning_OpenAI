import numpy as np
import os

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
    Function which ensures that the user gives a valid inputs for the parameter. 
    The inputs will be saved in an array.
    Works for numerical data

    parameters:

    parameter_name:
        name of the parameter

    parameter_type:
        type of the parameter

    boundaries:
        define boundary of parameter (min and max allowed value)

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
    If not the user is asked again.

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