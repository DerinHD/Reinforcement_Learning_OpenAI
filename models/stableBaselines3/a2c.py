import os
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from core import helper
from models.baseModel import BaseModel


class A2CModel(BaseModel):
    def __init__(self, environment: gym.Env, model):
        super().__init__(environment)
        self.model = model

    @staticmethod
    def create_model(env: gym.Env):
        print("Specify parameters for the A2C model")

        list_of_norm = {"0": False, "1": True}
        norm_prompt = "Normalize observations and rewards? (0: False, 1: True): "
        norm_choice = helper.get_valid_input(norm_prompt, list_of_norm.keys())
        normalize = list_of_norm[norm_choice]


        list_of_policies = ["MlpPolicy", "CnnPolicy", "MultiInputPolicy"]
        policy_prompt = "Which of the following policies?\n"
        pairs, single_choice_prompt = helper.create_single_choice(list_of_policies)
        idx = helper.get_valid_input(policy_prompt + single_choice_prompt, pairs.keys())
        policy = pairs[idx]

        # Entropy coefficient
        ent_coef = helper.valid_parameter("Entropy coefficient", float, [0, 1])
        gamma = helper.valid_parameter("Discount factor (gamma)", float, [0, 1])
        n_steps = helper.valid_parameter("Number of steps per rollout (n_steps)", int, [1, float('inf')])
        learning_rate = helper.valid_parameter("Learning rate", float, [0, 1])
        vf_coef = helper.valid_parameter("Value function coefficient", float, [0, 1])
        max_grad_norm = helper.valid_parameter("Max gradient norm", float, [0, 1])
        gae_lambda = helper.valid_parameter("GAE lambda", float, [0, 1])

        policy_kwargs = dict(
            net_arch=[64, 64],
            activation_fn= "relu"
        )

        # Build model
        model = A2C(
            policy=policy,
            env=env,
            ent_coef=ent_coef,
            verbose=1,
            gamma=gamma,
            n_steps=n_steps,
            learning_rate=learning_rate,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            gae_lambda=gae_lambda,
        )

        return A2CModel(env, model)

    def predict(self, state, is_training: bool = False):
        action, _ = self.model.predict(state, deterministic=not is_training)
        return action

    def learn(self, num_episodes: int = 1000):
        self.model.learn(total_timesteps=num_episodes)

    def save(self, file_path: str):
        self.model.save(file_path)
        print(f"Model saved to {file_path}")

    def reset(self):
        pass

    @staticmethod
    def load(file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Load model
        model = A2C.load(file_path)

        return A2CModel(model.env, model)
