import os
import pickle
import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control import MountainCarEnv
from core import helper
from environment.baseEnvironment import BaseEnvironment


"""
This file contains the mountaincar environment.
Example how to use this environment:
Option 1: Create from terminal
env = MountainCarContinuous.create_environment()
Option 2: Create manually
env = MountainCarContinuous(seed=42, render_mode="human")
"""
class MountainCarContinuousState(BaseEnvironment, MountainCarEnv):
    """
    The mountaincar environment is a continuous control task where the agent has to reach the goal while avoiding falling down.
    The environment is a continuous action space with 3 actions: accelerate to the left, don’t accelerate, accelerate to the right.
    
    
    Parameters:
    seed: int
        The seed for the random number generator. This is used to generate the random map.
    render_mode: str
        The render mode for the environment. This is used to render the environment.
    """
    def __init__(
        self,
        seed: int = None,
        render_mode: str = "human"
    ):
        BaseEnvironment.__init__(self)
        MountainCarEnv.__init__(self, render_mode=render_mode)

        self.seed = seed
        self.parameters = {"seed": seed}

        self.action_space.seed(self.seed)
        self.observation_space.seed(self.seed)

        sample_state = self.observation_space.sample()
        sample_action = self.action_space.sample()
        feature = self.transform_state_action_to_feature(sample_state, sample_action)
        self.feature_size = len(feature)

        self.current_state, _ = self.reset(seed=self.seed)
        self.round = 0
        
        self.max_round = 200

    def transform_state_action_to_feature(self, state: np.ndarray, action: int):
        """
        Transforms the state and action into a feature vector.
        Args:
            state (np.ndarray):
                State
            action (int):
                Action
        Returns:
            feature_vector (np.ndarray):
                Feature vector
        """
        state_feat = np.array(state, dtype=float)
        
        num_actions = self.action_space.n
        action_feat = np.zeros(num_actions, dtype=float)
        action_feat[action] = 1.0
        return np.concatenate([state_feat, action_feat])

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state. This function is used to reset the environment after each episode.
        Args:
            seed (int):
                Seed for the random number generator
            options (dict):
                Options for resetting the environment
        Returns:
            state (np.ndarray):
                Initial state of the environment
            info (dict):
                Additional information about the environment
        """
        state, info = MountainCarEnv.reset(self, seed=seed, options=options)
        self.current_state = state
        self.round = 0
        return state, info

    def env_step(self, action):
        """
        Performs a step in the environment using the given action.
        Args:
            action (int):
                Action to be performed
        Returns:
            next_state (np.ndarray):
                Next state after action was performed
            reward (float):
                Reward received after action was performed
            termination (bool):
                Whether the episode has terminated
            truncation (bool):
                Whether the episode has been truncated
            info (dict):
                Additional information about taking a step
        """
        raw_next, reward_env, termination, truncation, info = MountainCarEnv.step(self, action)

        self.round += 1
        
        if self.round >= self.max_round:
            truncation = True
        return raw_next, reward_env, termination, truncation, info

    @classmethod
    def create_environment(cls, render_mode: str = "human"):
        """
        Creates the Mountain Car environment with the specified parameters.
        Args:
            render_mode (str):
                Render mode for the environment
        Returns:
            env (MountainCarContinuous):
        """
        print("Specify parameters for the MountainCar continuous environment\n")
        seed = helper.valid_parameter("Environment seed", int, [-1, np.inf])
        if seed == -1:
            seed = None
        return cls(seed=seed, render_mode=render_mode)

    def render(self):
        return MountainCarEnv.render(self)
    
    @staticmethod
    def get_action_names():
        """
        Returns:
            action_names (dict):
                Dictionary mapping action indices to action names
        """
        return {
            0: "Accelerate to the left",
            1: "Don’t accelerate",
            2: "Accelerate to the right"
        }
