from abc import ABC, abstractmethod
import os
import pickle
import gymnasium as gym

from environment.baseEnvironment import BaseEnvironment

class BaseModel(ABC):
    def __init__(self, environment: BaseEnvironment):
        super().__init__()

        self.env = environment

    @abstractmethod
    def reset():
        """ 
        Resets the model to its initial state
        """
        pass 

    @abstractmethod
    def predict(state, is_training: bool=False):
        """
        Predicts the best action for a given state
        Args:
            state (any): Current state of the environment
            is_training (bool): Whether the model is in training mode or not
        Returns:
            action (any): Best action for the given state
        """ 
        pass

    @abstractmethod
    def learn(self, num_episodes: int):
        """
        Trains the model for a given number of episodes
        Args:
            num_episodes (int): Number of episodes to train the model
        """ 

        pass

    def save(self, file_path: str):
        """
        Saves the model's state to a file
        Args:
            file_path (str): Path to the file to save the model to
        """
        self.env = None

        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
        print(f"Model saved to {file_path}")
        pass

    @staticmethod
    def load(file_path: str):
        """
        Loads a saved model from a file
        Args:
            file_path (str): Path to the file to load the model from
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        with open(file_path, 'rb') as file:
            agent = pickle.load(file)
        print(f"Model loaded from {file_path}")
        return agent

    @staticmethod
    def create_model(env: gym.Env):
        """
        Creates a model for the given environment

        Args:
            env (gym.Env): Environment to create the model for

        Returns:
            model (BaseModel): Model for the given environment
        """
        raise NotImplementedError("create_model method not implemented in child class")
    