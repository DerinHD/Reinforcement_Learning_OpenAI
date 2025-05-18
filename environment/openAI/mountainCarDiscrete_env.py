import os
import pickle
import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control import MountainCarEnv
from gymnasium.spaces import Discrete

from core import helper
from environment.baseEnvironment import BaseEnvironment


"""
This file contains the mountaincar environment.
Example how to use this environment:
Option 1: Create from terminal
env = MountainCarDiscrete.create_environment()
Option 2: Create manually
env = MountainCarDiscrete(seed=42, n_bins=(20, 20), render_mode="human")
"""


class MountainCarDiscreteState(BaseEnvironment, MountainCarEnv):
    """
    The mountaincar environment is a discrete control task where the agent has to reach the goal while avoiding falling down.
    The environment is a discrete action space with 3 actions: accelerate to the left, don’t accelerate, accelerate to the right.

    Parameters:
    seed: int
        The seed for the random number generator. This is used to generate the random map.
    n_bins: tuple[int, int]
        The number of bins for the position and velocity. This is used to discretize the state space.
    render_mode: str
        The render mode for the environment. This is used to render the environment.        
    """ 
    def __init__(
        self,
        seed: int = None,
        n_bins: tuple[int, int] = (20, 20),
        render_mode: str = "human"
    ):

        BaseEnvironment.__init__(self)
        MountainCarEnv.__init__(self, render_mode=render_mode)

        self.seed = seed
        self.n_bins = n_bins  # (position_bins, velocity_bins)
        self.parameters = {"seed": seed, "n_bins": n_bins}

        self.action_space.seed(self.seed)
        print("Action space: ", self.action_space)

        raw_state, _ = MountainCarEnv.reset(self, seed=self.seed)

        self.obs_low, self.obs_high = self.observation_space.low, self.observation_space.high
        self.bin_edges = [
            np.linspace(self.obs_low[i], self.obs_high[i], self.n_bins[i] + 1) 
            for i in range(len(self.obs_low))
        ]
 
        self.max_round = 200

        self.observation_space = Discrete(self.n_bins[0] * self.n_bins[1]) 

        disc_state = self.discretize_state(raw_state)
        sample_action = self.action_space.sample()
        feature = self.transform_state_action_to_feature(disc_state, sample_action)
        self.feature_size = len(feature)

        self.current_state = disc_state
        self.round = 0

    def discretize_state(self, state: np.ndarray) -> int:
        """
        Discrete state
        Args:
            state (np.ndarray):
                continuous state
        Returns:
            bins (int):
                discrete state
        """
        bins = []
        for i, val in enumerate(state):
            idx = np.digitize(val, self.bin_edges[i]) - 1
            idx = min(max(idx, 0), self.n_bins[i] - 1)
            bins.append(idx)
        return bins[0] * self.n_bins[1] + bins[1]

    def transform_state_action_to_feature(self, state: int, action: int) -> np.ndarray:
        """
        Transforms the state and action into a feature vector.
        Args:
            state (int):
                State
            action (int):
                Action
        Returns:
            feature_vector (np.ndarray):
                Feature vector
        """
        state_feat = np.zeros(self.observation_space.n, dtype=float)
        state_feat[state] = 1.0
        action_feat = np.zeros(self.action_space.n, dtype=float)
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
            state (int):
                Initial state of the environment
            info (dict):
                Additional information about the environment
        """
        raw_state, info = MountainCarEnv.reset(self, seed=seed, options=options)
        disc_state = self.discretize_state(raw_state)
        self.current_state = disc_state
        self.round = 0
        return disc_state, info

    def env_step(self, action):
        """
        Performs a step in the environment using the given action.
        Args:
            action (int):
                Action to be performed
        Returns:"""
        raw_next, reward_env, termination, truncation, info = MountainCarEnv.step(self, action)
        next_state = self.discretize_state(raw_next)

        self.round += 1
        
        if self.round >= self.max_round:
            truncation = True
        return next_state, reward_env, termination, truncation, info

    @classmethod
    def create_environment(cls, render_mode: str = "human"):
        """
        Creates the environment from the command line. This function is used to create the environment from the command line.
        Args:
            render_mode (str):
                Render mode for the environment
        Returns:
            env (MountainCarDiscrete):
                Environment
        """
        # Check if the environment is available
        print("Specify parameters for the MountainCar environment\n")
        seed = helper.valid_parameter("Environment seed", int, [-1, np.inf])
        pos_bins = helper.valid_parameter("Number of position bins", int, [2, 100])
        vel_bins = helper.valid_parameter("Number of velocity bins", int, [2, 100])

        if seed == -1:
            seed = None
        return cls(seed=seed, n_bins=(pos_bins, vel_bins), render_mode=render_mode)

    def render(self):
        return MountainCarEnv.render(self)

    @staticmethod
    def get_action_names():
        """
        Returns:
            action_names (dict):
                Dictionary with action names
        """
        return {
            0: "Accelerate to the left",
            1: "Don’t accelerate",
            2: "Accelerate to the right"
        }
    
