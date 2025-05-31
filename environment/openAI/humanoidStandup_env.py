import gymnasium as gym
import numpy as np
from gymnasium.envs.mujoco.humanoidstandup_v4 import HumanoidStandupEnv
from environment.baseEnvironment import BaseEnvironment


class HumanoidStandup(BaseEnvironment, HumanoidStandupEnv):
    """
    Parameters:
    render_mode: str
        The render mode (e.g., "human", "rgb_array").
    """

    def __init__(self, render_mode="human"):
        BaseEnvironment.__init__(self)
        HumanoidStandupEnv.__init__(self, render_mode=render_mode)

        self.current_state, _ = self.reset()
        # not needed
        #self.feature_size = self.transform_state_action_to_feature(self.current_state, 0).shape[0]

        self.parameters = {
        }

        self.max_round = 1000  

    def transform_state_action_to_feature(self, state, action):
        """
        Transform the state and action to a feature vector.
        Args:
            state (int): The state of the environment.
            action (int): The action taken by the agent.
        Returns:
            feature (numpy.ndarray): The feature vector.    
        """
        
        return np.concatenate([state, action])

    def reset(self, seed=None, options=None):
        """
        Reset the environment.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Reset options.

        Returns:
            observation (np.ndarray), info (dict)
        """
        state, info = HumanoidStandupEnv.reset(self, seed=seed, options=options)
        self.current_state = state
        self.round = 0
        return state, info

    def env_step(self, action):
        """
        Args:
            action (np.ndarray): The action to take.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        next_state, reward, terminated, truncated, info = HumanoidStandupEnv.step(self, action)

        self.round += 1
        if self.round >= self.max_round:
            truncated = True

        return next_state, reward, terminated, truncated, info

    def render(self):
        return HumanoidStandupEnv.render(self)

    @staticmethod
    def create_environment(render_mode="human"):
        return HumanoidStandup(render_mode=render_mode)
