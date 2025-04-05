from abc import ABC, abstractmethod
import pickle
import gymnasium as gym

class BaseModel(ABC):
    def __init__(self, environment: gym.Env):
        super().__init__()

        self.env = environment

    @abstractmethod
    def reset():
        pass

    @abstractmethod
    def predict(state, is_training: bool=False):
        pass

    @abstractmethod
    def learn(self, num_episodes: int):
        pass

    @abstractmethod
    def save(self, file_path: str):
        pass

    @staticmethod
    def load(file_path: str):
        pass
    