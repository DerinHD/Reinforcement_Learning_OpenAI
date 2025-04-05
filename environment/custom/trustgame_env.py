import os
import pickle
import numpy as np
import gymnasium as gym
import math

"""
Create, save and load Custom Trust game environment.
More details about environment can be found here at trust_game.pdf 
"""

# Probability of state transitions for the trustee.
# State transition probability is dependent on (state, action) with state= [1,2,3] an action = [0,10,20,30,40,50]
state_transition_probabilites = {
    # High cooperation (State 0)
    (0, 0): {0: 0.2, 1: 0.4, 2: 0.4},
    (0, 10): {0: 0.3, 1: 0.4, 2: 0.3},
    (0, 20): {0: 0.5, 1: 0.3, 2: 0.2},
    (0, 30): {0: 0.6, 1: 0.3, 2: 0.1},
    (0, 40): {0: 0.7, 1: 0.2, 2: 0.1},
    (0, 50): {0: 0.7, 1: 0.2, 2: 0.1},

    # Neutral (State 1)
    (1, 0): {0: 0.1, 1: 0.3, 2: 0.6},
    (1, 10): {0: 0.2, 1: 0.3, 2: 0.5},
    (1, 20): {0: 0.3, 1: 0.4, 2: 0.3},
    (1, 30): {0: 0.3, 1: 0.4, 2: 0.3},
    (1, 40): {0: 0.6, 1: 0.3, 2: 0.1},
    (1, 50): {0: 0.7, 1: 0.2, 2: 0.1},

    # Low cooperation (State 2)
    (2, 0): {0: 0.5, 1: 0.5, 2: 0.0},
    (2, 10): {0: 0.4, 1: 0.4, 2: 0.2},
    (2, 20): {0: 0.1, 1: 0.3, 2: 0.6},
    (2, 30): {0: 0.1, 1: 0.3, 2: 0.6},
    (2, 40): {0: 0.05, 1: 0.25, 2: 0.7},
    (2, 50): {0: 0.0, 1: 0.2, 2: 0.8},
}

def state_to_string(state):
    """
    Convert state to string

    parameters:
    
    state:
        trustee state (0,1 or 2)
    """
    dict_state = {
     0: "High Cooperation",
     1: "Neutral",
     2: "Low_cooperation"
    }

    return dict_state[state]

def get_payback_probabilities(state, investment):
    """
    Probability of the payback.
    Payback probability is only dependent on state. The action (:=investment) is only used to compute maximum possible payback

    parameters:
    
    state:
        state of trustee

    investment:
        action of the investor
    """
    max_payback = 3 * investment
    possible_paybacks = list(range(0, max_payback + 1))  # All integers from 0 to 3 * investment

    probabilities = {}

    if state == 0:  # High cooperation => polynomial curve with higher probability to get high payback
        power = 5 # change this parameter to modify curve
        
        probabilities = {p: (p ** power) for p in possible_paybacks}
        total_weight = sum(probabilities.values())
        probabilities = {p: prob / total_weight for p, prob in probabilities.items()}

    elif state == 1: # Neutral => normal distribution to 
        mean = investment / 2  
        
        std_dev = investment / 4  # change this parameter to modify curve

        pdf_values = [np.exp(-(p - mean)**2 / (2 * std_dev**2)) for p in possible_paybacks]
        total_prob = sum(pdf_values)
        probabilities = {p: pdf / total_prob for p, pdf in zip(possible_paybacks, pdf_values)}

    elif state == 2:  # Low cooperation => polynomial curve with higher probability to get low payback
        power = 5  # change this parameter to modify curve
        
        probabilities = {p: ((max_payback - p) ** power) for p in possible_paybacks}
        total_weight = sum(probabilities.values())  
        probabilities = {p: prob / total_weight for p, prob in probabilities.items()}

    else:
        raise ValueError("Invalid state")

    return probabilities

class TrustGameEnv(gym.Env):
    """
    Trustgame compatible with Open AI gymnasium library
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

    action_space:
        Action space of environment

    observation_space:
        Observation space of environment
        

    -------------------------------------------------------------

    Methods:

    reset:
        Reset all numerical parameters to zero

    step:
        Perform an action on the environment

    render:
        Render the environment by printing the result on terminal

    create_trustgame_environment:
        Create environment by configuring its parameters via terminal

    load_trustgame_env:
        Load environment 

    save_trustgame_env:
        Save environment 

    """
    def __init__(self, num_rounds, state_transition_prob = state_transition_probabilites, payback_prob = get_payback_probabilities):
        super(TrustGameEnv, self).__init__()

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
    
    def step(self, action):
        """   
        Perform an action on the environment

        parameters:

        action:
            investment
        """
        self.last_state = self.current_state # used for montitoring
        investment = self.investment[action]
        self.invested = investment # used for montitoring

        self.payback = 0

        if investment != 0: # Get payback probabilities only of investment is higher than zero
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

        return self.current_state, reward, done, {}, {}
    

    def render(self, mode='human'):
        """
        Render the environment by printing the result on terminal

        Note: Don't provide mode as parameter
        """
        print(f"Round: {self.current_round:<{math.floor(math.log10(abs(self.num_rounds))) + 1}}/{self.num_rounds:<2}  "
          f"Last State: {state_to_string(self.last_state):<18}    "
          f"Invested: {self.invested:>3}    "
          f"Payback: {self.payback:>3}    "
          f"New State: {state_to_string(self.current_state):<18}    "
          f"Total Reward: {self.total_reward:>1}")

def create_trustgame_environment():
    """
    Create environment by configuring its parameters via terminal
    """
    print("Specify parameters for the trustgame environment")

    parameters = []

    print("\n")
    num_rounds = None
    while True:
        try:
            user_input = input(f"Specify the value for num_rounds ({int.__name__}): \n ")
            num_rounds = int(user_input)
            if num_rounds <= 0:
                raise ValueError
            break
        except ValueError:
            print(f"Invalid input. Try again.")

    parameters = {"num_rounds": num_rounds}

    return TrustGameEnv(num_rounds=num_rounds), parameters


def load_trustgame_env(file_path):
    """
    Load environment

    parameters:

    file_path:
        path of file to be load to

    returns:
        environment parameters
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'rb') as file:
        parameters = pickle.load(file)
            
    return TrustGameEnv(num_rounds=parameters["num_rounds"]), parameters

def save_trustgame_env(file_path, parameters):
    """
    Save environment parameters

    parameters:

    file_path:
        path of file to be saved at
        
    parameters:
        environment parameters
    """

    with open(file_path, 'wb') as file:
        pickle.dump(parameters, file)

def get_action_names_trust_game_environment():
    """
    Get the list of names for the actions 
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