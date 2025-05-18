import os
import pickle
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv
import core.helper as helper
import numpy as np

from environment.baseEnvironment import BaseEnvironment


"""
This file contains the frozenlake environment.


Example how to use this environment:
Option 1: Create from terminal
env = FrozenLake.create_environment()
Option 2: Create manually
env = FrozenLake(is_slippery=True, map_size=4, seed=42, prob_frozen=0.8)
"""


class FrozenLake(BaseEnvironment, FrozenLakeEnv):
    """
    The frozenlake environment is a grid world where the agent has to reach the goal while avoiding holes.
    The environment is slippery, meaning that the agent can slip and fall into holes.
    The environment is a discrete action space with 4 actions: up, down, left, right.
    The environment is a grid world with a size of map_size x map_size.

    Parameters:
    is_slippery: bool
        If True, the environment is slippery, meaning that the agent can slip and fall into holes.
    map_size: int
        The size of the grid world. The environment is a grid world with a size of map_size x map_size.
    seed: int
        The seed for the random number generator. This is used to generate the random map.
    prob_frozen: float
        The probability that a tile is frozen. This is used to generate the random map.
    render_mode: str
        The render mode for the environment. This is used to render the environment.   
    """
    def __init__(self, 
                 is_slippery, 
                 map_size, 
                 seed,
                 prob_frozen,
                 render_mode = "human"):
        
        BaseEnvironment.__init__(self)
        FrozenLakeEnv.__init__(self, 
                        is_slippery=is_slippery, 
                        desc= generate_random_map(size=map_size, seed=seed, p=prob_frozen), render_mode= render_mode
        )

        feature = self.transform_state_action_to_feature(0,0)
        self.feature_size = len(feature)
        
        self.current_state, info = self.reset()

        self.parameters = {"is_slippery": is_slippery, 
                "map_size": map_size, 
                'seed': seed, 
                'prob_frozen':  prob_frozen,
        }

        self.seed = seed

        self.max_round = map_size * map_size * 10
    

    def transform_state_action_to_feature(self, state , action):
        """
        Transform the state and action to a feature vector.
        Args:
            state (int): The state of the environment.
            action (int): The action taken by the agent.
        Returns:
            feature (numpy.ndarray): The feature vector.    
        """
        num_states = self.observation_space.n
        num_actions = self.action_space.n
        
        states = np.zeros(num_states)
        actions = np.zeros(num_actions)

        states[state] = 1
        actions[action] = 1

        feature = np.concatenate([states, actions])
        
        """
        map_size = int(np.sqrt(num_states))
        
        # Option 2: 
        feature = np.zeros(2)
        state_2d_x = state % map_size
        state_2d_y = state // map_size

        is_wall = state_2d_x == 0 or state_2d_x == map_size - 1 or state_2d_y == 0 or state_2d_y == map_size - 1

        if is_wall:
            if (state_2d_x == 0 and action in [0, 1, 3]) or \
            (state_2d_x == map_size - 1 and action in [2, 1, 3]) or \
            (state_2d_y == 0 and action in [3, 0, 2]) or \
            (state_2d_y == map_size - 1 and action in [1, 0, 2]):
                feature[0] = 1

        if action == 0:  
            if state_2d_x > 0:
                state_2d_x -= 1

        elif action == 2:  
            if state_2d_x < map_size - 1:
                state_2d_x += 1

        elif action == 3:  
            if state_2d_y > 0:
                state_2d_y -= 1

        elif action == 1:  
            if state_2d_y < map_size - 1:
                state_2d_y += 1


        goal_2d_x = (num_states-1) % map_size
        goal_2d_y = (num_states-1) // map_size

        feature[1] = abs(state_2d_x - goal_2d_x) + abs(state_2d_y - goal_2d_y)
        """

        return feature
    
    def reset(self, seed=None,  options=None):
        """
        Reset the environment to its initial state.
        Args:
            seed (int): The seed for the random number generator.
            options (dict): The options for the environment.
        Returns:
            state (int): The initial state of the environment.
            obs (numpy.ndarray): The observation of the environment.
        """
        # Reset the environment
        state, obs = FrozenLakeEnv.reset(self, seed=seed, options=options)

        self.current_state = state
        self.round = 0

        return state, obs
    
    def env_step(self, action):
        """
        Perform a step in the environment.
        Args:
            action (int): The action taken by the agent.
        Returns:
            next_state (int): The next state of the environment.
            reward_env (float): The reward received from the environment.
            termination (bool): Whether the episode has terminated.
            truncation (bool): Whether the episode has been truncated.
            info (dict): Additional information about the environment.
        """
        next_state, reward_env, termination, truncation, info = FrozenLakeEnv.step(self, action)

        self.round += 1
        if self.round >= self.max_round:
            truncation = True
        
        return next_state, reward_env, termination, truncation, info

    def create_environment(render_mode = "human"):
        """
        Create the frozenlake environment.
        Args:
            render_mode (str): The render mode for the environment.
        Returns:"
            env (FrozenLake): The frozenlake environment.
        """
        print("Specify parameters for the frozenlake environment\n")

        is_slippery_idx = helper.get_valid_input(f"Environment is slippery \n1)True \n2)False", ["1", "2"])
        is_slippery = False
        
        if is_slippery_idx == "1":
            is_slippery = True

        map_size = helper.valid_parameter("Map size", int, [2, 20])
        
        seed = helper.valid_parameter("Environment seed", int)

        prob_frozen = helper.valid_parameter("Probability that tile is frozen", float, [0.0,1.0])

        return FrozenLake(is_slippery=is_slippery, map_size=map_size, seed=seed, prob_frozen=prob_frozen, render_mode=render_mode)
    
    def render(self):
        return FrozenLakeEnv.render(self)

    @staticmethod
    def get_action_names():
        """
        Get the list of names for the actions 
        """
        action_names = {
            0: "Move left",
            1: "Move down",
            2: "Move right",
            3: "Move up"
        }

        return action_names