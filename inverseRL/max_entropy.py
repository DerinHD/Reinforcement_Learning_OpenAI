import numpy as np
import gymnasium as gym
from collections import defaultdict
from tqdm import tqdm
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

def max_entropy(model, demonstration_data, num_iterations = 1000, alpha = 0.1):
    mu_agent_feature = np.zeros(model.env.feature_size)

    for e in range(len(demonstration_data)):
        for key, value in demonstration_data[e].items():
            state = value[0]
            action = value[1]
            feature = model.env.transform_state_action_to_feature(state, action)
            mu_agent_feature += feature

    mu_agent_feature /=  len(demonstration_data)

    theta = np.random.uniform(-1, 1, model.env.feature_size)  
    model.env.max_entropy_configure(theta)
    model.env.reward_function = "max_entropy"

    for _ in tqdm(range(num_iterations)):
        model.reset()
        model.learn(num_episodes=1000)

        mu_policy_feature = np.zeros(model.env.feature_size)
        i=0
        # simulate policy
        for _ in range(100): 
            state, _ = model.env.reset()
            done = False
            while not done:
                action = model.predict(state=state)
                next_state, reward, terminated, truncated, _ = model.env.step(action)
                mu_policy_feature += model.env.transform_state_action_to_feature(state, action)
                done = terminated or truncated
                state = next_state
            i+=1

        mu_policy_feature /= i
        print(mu_policy_feature)
        
        # update theta
        theta += alpha * (mu_agent_feature - mu_policy_feature)
        print(f"loss: {np.linalg.norm((mu_agent_feature - mu_policy_feature))}")

        model.env.max_entropy_configure(theta)
    
    return theta, model