import os
import pickle
from stable_baselines3 import DQN
import gymnasium as gym

from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
import numpy as np

from core import helper
from models.baseModel import BaseModel

"""
Create Deep Q Network (DQN) via terminal. For more details, view https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html. 
"""


class DQNModel(BaseModel):
    def __init__(self,environment: gym.Env, model):
        super().__init__(environment)
        self.model = model

    @staticmethod
    def create_model(env: gym.Env):
        print("Specify parameters for the DQN model")

        # Choose policy
        list_of_policies = ["MlpPolicy", "CnnPolicy", "MultiInputPolicy"]
        prompt = "Which of the following policies?\n"
        pairs, single_choice_prompt = helper.create_single_choice(list_of_policies)
        idx = helper.get_valid_input(prompt + single_choice_prompt, pairs.keys())
        input_policy = pairs[idx]

        # Get hyperparameters
        learning_rate = helper.valid_parameter("Learning rate", float, [0, 1])
        gamma = helper.valid_parameter("Discount factor", float, [0, 1])
        epsilon_init = helper.valid_parameter("Initial Exploration rate", float, [0, 1])
        epsilon_fraction = helper.valid_parameter("Exploration fraction", float, [0, 1])
        epsilon_end = helper.valid_parameter("Final Exploration rate", float, [0, 1])
        buffer_size           = helper.valid_parameter("Replay buffer size", int, [1, np.inf])
        batch_size            = helper.valid_parameter("Batch size", int, [1, np.inf])
        learning_starts       = helper.valid_parameter("Learning starts after", int, [0, np.inf])
        train_freq            = helper.valid_parameter("Train every n steps", int, [1, np.inf])
        gradient_steps        = helper.valid_parameter("Gradient steps per update", int, [1, np.inf])
        target_update_interval= helper.valid_parameter("Target network update interval", int, [1, np.inf])

        config = {
            "policy": input_policy,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "exploration_initial_eps": epsilon_init,
            "exploration_fraction": epsilon_fraction,
            "exploration_final_eps": epsilon_end,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "learning_starts": learning_starts,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "target_update_interval": target_update_interval,
            "verbose": 1
        }

        model = DQN(env=env, policy_kwargs=dict(net_arch=[256, 256]), **config)

        return DQNModel(env, model)

    def reset(self):
        policy = getattr(self.model, 'policy', None)
        policy_kwargs = getattr(self.model, 'policy_kwargs', None)
        config = {
            'policy': policy,
            'env': self.env,
            'learning_rate': self.model.learning_rate,
            'gamma': self.model.gamma,
            'exploration_initial_eps': self.model.exploration_initial_eps,
            'exploration_fraction': self.model.exploration_fraction,
            'exploration_final_eps': self.model.exploration_final_eps,
            'buffer_size': self.model.buffer_size,
            'batch_size': self.model.batch_size,
            'learning_starts': self.model.learning_starts,
            'train_freq': self.model.train_freq,
            'gradient_steps': self.model.gradient_steps,
            'target_update_interval': self.model.target_update_interval,
            'verbose': self.model.verbose,
            'policy_kwargs': policy_kwargs,
        }
        self.model = DQN(env=self.env, **config)

    def predict(self, state, is_training: bool=False):
        action, _ = self.model.predict(state, deterministic=not is_training)
        return action

    def learn(self, num_episodes: int):
        self.model.learn(total_timesteps=num_episodes)


    def save(self, file_path: str):
        self.model.save(file_path)
        print(f"Model saved to {file_path}")

    @staticmethod
    def load(file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        model = DQN.load(file_path)

        return DQNModel(model.env, model)
