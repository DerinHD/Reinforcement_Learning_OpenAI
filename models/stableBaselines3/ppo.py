import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from core import helper
from models.baseModel import BaseModel


class PPOModel(BaseModel):
    def __init__(self, environment: gym.Env, model):
        super().__init__(environment)
        self.model = model

    @staticmethod
    def create_model(env: gym.Env):
        print("Specify parameters for the PPO model")

        list_of_norm = {"0": False, "1": True}
        norm_prompt = "Normalize observations and rewards? (0: False, 1: True): "
        norm_choice = helper.get_valid_input(norm_prompt, list_of_norm.keys())
        normalize = list_of_norm[norm_choice]


        list_of_policies = ["MlpPolicy", "CnnPolicy", "MultiInputPolicy"]
        policy_prompt = "Which of the following policies?\n"
        pairs, single_choice_prompt = helper.create_single_choice(list_of_policies)
        idx = helper.get_valid_input(policy_prompt + single_choice_prompt, pairs.keys())
        policy = pairs[idx]

        n_steps = helper.valid_parameter("Number of steps per rollout (n_steps)", int, [1, float('inf')])
        gae_lambda = helper.valid_parameter("GAE lambda", float, [0, 1])
        gamma = helper.valid_parameter("Discount factor (gamma)", float, [0, 1])
        n_epochs = helper.valid_parameter("Number of PPO epochs", int, [1, float('inf')])
        ent_coef = helper.valid_parameter("Entropy coefficient", float, [0, 1])
        learning_rate = helper.valid_parameter("Learning rate", float, [0, 1])
        batchsize = helper.valid_parameter("Batch size", int, [1, float('inf')])


        if normalize:
            env = DummyVecEnv([lambda: env])
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
        # Initialize the PPO model
        model = PPO(
            policy=policy,
            env=env,
            n_steps=n_steps,
            batch_size=batchsize,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            verbose=1,
            learning_rate= learning_rate
        )

        return PPOModel(env, model)

    def predict(self, state, is_training: bool = False):
        action, _ = self.model.predict(state, deterministic=not is_training)
        return action

    def learn(self, num_episodes):
        self.model.learn(total_timesteps=num_episodes)

    def reset():
        """
        Resets the environment
        """
        pass

    

    def save(self, file_path: str):
        self.model.save(file_path)
        print(f"Model saved to {file_path}")

    @staticmethod
    def load(file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        model = PPO.load(file_path)

        return PPOModel(model.env, model)
