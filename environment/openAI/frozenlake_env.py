import os
import pickle
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv
import core.helper as helper
import numpy as np

from environment.baseEnvironment import BaseEnvironment

class FrozenLake(BaseEnvironment, FrozenLakeEnv):
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
    

    def transform_state_action_to_feature(self, state , action):
        # Option 1: One hot encoding
        
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
        state, obs = FrozenLakeEnv.reset(self, seed=seed, options=options)

        self.current_state = state
        self.round = 0

        return state, obs
    
    def create_environment(render_mode = "human"):
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