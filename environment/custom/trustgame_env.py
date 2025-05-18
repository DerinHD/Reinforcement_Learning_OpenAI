import os
import pickle
import numpy as np
import gymnasium as gym
import math

from core import helper
from environment.baseEnvironment import BaseEnvironment

"""
This is a custom environment for the trust game, compatible with OpenAI Gymnasium.

#Example how to use this environment:
Option 1: Create from terminal
env = TrustGameEnv.create_environment(render_mode="human")
Option 2: Create manually
env = TrustGameEnv(num_rounds=10, render_mode="human")
-------------------------------------------------------------
"""

# Probability of state transitions for the trustee.
# State transition probability is dependent on (state, action) with state= [1,2,3] an action = [0,10,20,30,40,50]
state_transition_probabilites = {
    # High cooperation (State 0)
    (0, 0):  {0: 0.60, 1: 0.30, 2: 0.10},
    (0, 10): {0: 0.65, 1: 0.25, 2: 0.10},
    (0, 20): {0: 0.70, 1: 0.20, 2: 0.10},
    (0, 30): {0: 0.75, 1: 0.18, 2: 0.07},
    (0, 40): {0: 0.78, 1: 0.17, 2: 0.05},
    (0, 50): {0: 0.80, 1: 0.15, 2: 0.05},

    # Neutral (State 1)
    (1, 0):  {0: 0.20, 1: 0.60, 2: 0.20},
    (1, 10): {0: 0.25, 1: 0.55, 2: 0.20},
    (1, 20): {0: 0.30, 1: 0.50, 2: 0.20},
    (1, 30): {0: 0.35, 1: 0.45, 2: 0.20},
    (1, 40): {0: 0.40, 1: 0.45, 2: 0.15},
    (1, 50): {0: 0.45, 1: 0.45, 2: 0.10},

    # Low cooperation (State 2)
    (2, 0):  {0: 0.10, 1: 0.30, 2: 0.60},
    (2, 10): {0: 0.15, 1: 0.35, 2: 0.50},
    (2, 20): {0: 0.20, 1: 0.40, 2: 0.40},
    (2, 30): {0: 0.25, 1: 0.45, 2: 0.30},
    (2, 40): {0: 0.30, 1: 0.50, 2: 0.20},
    (2, 50): {0: 0.35, 1: 0.50, 2: 0.15},
}


# Mapping of state to string
# State 0: High cooperation (Trustee is more likely to return a high payback)
# State 1: Neutral
# State 2: Low cooperation (Trustee is more likely to return a low payback))
def state_to_string(state):
    """
    Convert state to string

    parameters:
    
    state:
        trustee state (0,1 or 2)

    returns:
        string representation of the state
    """
    dict_state = {
     0: "High Cooperation",
     1: "Neutral",
     2: "Low_cooperation"
    }

    return dict_state[state]

# Probability of payback based on the state and investment
def get_payback_probabilities(state, investment):
    """
    Probability of the payback.
    Payback probability is only dependent on state. The action (:=investment) is only used to compute maximum possible payback

    parameters:
    
    state:
        state of trustee

    investment:
        action of the investor

    returns:
        probabilities of payback
    """

    max_payback = 3 * investment # maximum possible payback (0, 10, 20, 30, 40, 50) * 3
    possible_paybacks = list(range(0, max_payback + 1))  # All integers from 0 to 3 * investment

    probabilities = {} # probabilities of payback. Dictionary with keys = payback and values = probability

    if state == 0:  # High cooperation => polynomial curve with higher probability to get high payback
        power = 5 # change this parameter to modify curve
        
        probabilities = {p: (p ** power) for p in possible_paybacks} # polynomial curve
        total_weight = sum(probabilities.values()) # sum of all probabilities 
        # Normalize the probabilities to sum to 1
        probabilities = {p: prob / total_weight for p, prob in probabilities.items()}

    elif state == 1: # Neutral => normal distribution to 
        mean = (investment * 3) / 2  
        
        std_dev = investment / 4  # change this parameter to modify curve

        pdf_values = [np.exp(-(p - mean)**2 / (2 * std_dev**2)) for p in possible_paybacks]
        total_prob = sum(pdf_values)
        probabilities = {p: pdf / total_prob for p, pdf in zip(possible_paybacks, pdf_values)}

    elif state == 2:  # Low cooperation => polynomial curve with higher probability to get low payback
        power = 5  # change this parameter to modify curve
        
        probabilities = {p: ((max_payback - p) ** power) for p in possible_paybacks} # polynomial curve
        total_weight = sum(probabilities.values())  # sum of all probabilities
        probabilities = {p: prob / total_weight for p, prob in probabilities.items()} # Normalize the probabilities to sum to 1

    else:
        raise ValueError("Invalid state")

    return probabilities

class TrustGameEnv(BaseEnvironment):
    """
    The trust game is a two-player game where one player (the investor) decides how much money to invest in the other player (the trustee).
    The trustee then decides how much of the invested money to return to the investor.
    The environment simulates the trustee's behavior based on their cooperation state (high, neutral, low) and the investor's investment.

    -------------------------------------------------------------
    Attributes:

    num_rounds: int
        Number of rounds to invest for one episode

    budget: int
        Budget of investor at each round

    investment: array
        Investment options for the investor

    payback_prob: Function
        Custom probability function for payback which returns the probabilities for pair (state, action)
        
    state_transition_prob:
        Custom probability function for state transition which returns the probabilities for pair (state, action)

    qtable: np.ndarray
        The qtable for storing the expected rewards for state-action pairs
    -------------------------------------------------------------
    """
    def __init__(self, num_rounds, state_transition_prob = state_transition_probabilites, payback_prob = get_payback_probabilities, render_mode="human"):
        BaseEnvironment.__init__(self)

        self.render_mode = render_mode
        # Environment parameters 
        self.num_rounds = num_rounds # Number of rounds for the player to invest money
        
        self.budget = 50 
        self.investment = [0,10,20,30,40,50] # Investment

        # Cooperation state
        # State 0: High cooperation (Trustee is more likely to return a high payback)
        # State 1: Neutral
        # State 2: Low cooperation (Trustee is more likely to return a low payback))
        self.states = [0,1,2]

        # Open AI configuration: Set action and observation space  
        self.action_space = gym.spaces.Discrete(len(self.investment))
        self.observation_space = gym.spaces.Discrete(len(self.states))

        # Initialize payback and state transition probabilities 
        self.payback_prob = payback_prob # needs to be a function with parameters (state, action) (e.g. see default one on top of file)
        self.state_transition_prob = state_transition_prob # needs to be a dictionary in the format as the default one (see on top of file)
        
        self.parameters = {"num_rounds": num_rounds}

        # reset the environment
        self.reset()

    def reset(self, seed=None,  options=None):
        """
        Reset all numerical parameters to zero

        Note: Don't provide seed and options as parameters
        """
        self.current_state = np.random.choice(self.states)
        self.last_state = self.current_state
        self.invested = 0
        self.payback = 0
        self.current_round = 0
        self.total_reward = 0

        return self.current_state, {}
    
    def env_step(self, action):
        """   
        Perform an action on the environment

        parameters:

        action:
            investment

        returns:
            next_state:
                Next state received from the environment

            reward:
                Reward received from the environment

            done:
                Boolean indicating the termination of the episode

            truncated:
                Boolean indicating the truncation of the episode (e.g. maximum number of steps possible)

            info:
                Additional information
        """

        self.last_state = self.current_state # used for monitoring
        investment = self.investment[action] # investment is the action taken by the investor
        self.invested = investment # used for montitoring

        self.payback = 0 

        if investment != 0: # Get payback probabilities only if investment is higher than zero
            payback_prob = self.payback_prob(self.current_state, investment)

            paybacks = list(payback_prob.keys())
            probs = list(payback_prob.values())

            self.payback = np.random.choice(paybacks, p=probs)

        # Compute reward
        reward = self.budget - investment + self.payback

        self.total_reward += reward # for monitoring

        # Transition to next state
        transition_prob = self.state_transition_prob.get((self.current_state, investment))
        probs = list(transition_prob.values())
        
        self.current_state = np.random.choice(self.states,p=probs)

        # Increment round += 1
        self.current_round += 1
        done = self.current_round >= self.num_rounds

        return self.current_state, reward, done, False, {}
    

    def render(self, mode='human'):
        """
        Render the environment by printing the result on terminal.  

        Note: Don't provide mode as parameter
        """
        print(f"Round: {self.current_round:<{math.floor(math.log10(abs(self.num_rounds))) + 1}}/{self.num_rounds:<2}  "
          f"Last State: {state_to_string(self.last_state):<18}    "
          f"Invested: {self.invested:>3}    "
          f"Payback: {self.payback:>3}    "
          f"New State: {state_to_string(self.current_state):<18}    "
          f"Total Reward: {self.total_reward:>1}")

    @classmethod
    def create_environment(cls, render_mode: str = "human"):
        """
        Create environment by configuring its parameters via terminal
        """
        print("Specify parameters for the trustgame environment")

        num_rounds = helper.valid_parameter("Specify the value for num_rounds", int, [1, np.inf])

        return cls(num_rounds=num_rounds, render_mode=render_mode) 
    

    @staticmethod
    def get_action_names():
        """
        Get the list of names for the actions 

        Returns:
            action_names (dict):
                Dictionary with keys = action and values = action name
        """
        action_names = {
            0: "Don't invest",
            1: "Invest 10",
            2: "Invest 20",
            3: "Invest 30",
            4: "Invest 40",
            5: "Invest 50",
        }

        return action_names