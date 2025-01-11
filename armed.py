import os
import shutil
import numpy as np


# Create directories
os.makedirs('K-armed/multi_armed_bandit', exist_ok=True)
os.makedirs('K-armed/dynamic_programming', exist_ok=True)
os.makedirs('K-armed/model_free_learning', exist_ok=True)
os.makedirs('K-armed/data/results', exist_ok=True)

# Save code files

multi_armed_bandit_code = {
    'greedy_algorithm.py':

    class GreedyAlgorithm:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
    
    def select_arm(self):
        return np.argmax(self.values)
    
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value
,
    'epsilon_greedy_algorithm.py': 
import numpy as np
import random

class EpsilonGreedyAlgorithm:
    def __init__(self, n_arms, epsilon):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
    
    def select_arm(self):
        if random.random() > self.epsilon:
            return np.argmax(self.values)
        else:
            return random.randint(0, self.n_arms - 1)
    
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value
,
    'time_varying_epsilon_greedy_algorithm.py': 
import numpy as np
import random

class TimeVaryingEpsilonGreedyAlgorithm:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.time = 0
    
    def select_arm(self):
        self.time += 1
        epsilon = 1 / self.time
        if random.random() > epsilon:
            return np.argmax(self.values)
        else:
            return random.randint(0, self.n_arms - 1)
    
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value
,
    'successive_elimination_algorithm.py':
import numpy as np

class SuccessiveEliminationAlgorithm:
    def __init__(self, n_arms, horizon):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.eliminated_arms = set()
        self.horizon = horizon
    
    def select_arm(self):
        non_eliminated_arms = [arm for arm in range(self.n_arms) if arm not in self.eliminated_arms]
        return np.random.choice(non_eliminated_arms)
    
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value
        
        if self.counts[chosen_arm] >= self.horizon / self.n_arms:
            self.eliminate()
    
    def eliminate(self):
        mean_values = self.values
        threshold = np.mean(mean_values) - np.std(mean_values)
        for arm in range(self.n_arms):
            if self.values[arm] < threshold:
                self.eliminated_arms.add(arm)
,
    'ucb_algorithm.py': 
import numpy as np

class UCBAlgorithm:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_counts = 0
    
    def select_arm(self):
        self.total_counts += 1
        ucb_values = self.values + np.sqrt(2 * np.log(self.total_counts) / (self.counts + 1e-5))
        return np.argmax(ucb_values)
    
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value
,
    'thompson_sampling.py': 
import numpy as np

class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.successes = np.zeros(n_arms)
        self.failures = np.zeros(n_arms)
    
    def select_arm(self):
        samples = [np.random.beta(self.successes[i] + 1, self.failures[i] + 1) for i in range(self.n_arms)]
        return np.argmax(samples)
    
    def update(self, chosen_arm, reward):
        if reward == 1:
            self.successes[chosen_arm] += 1
        else:
            self.failures[chosen_arm] += 1
,
    'comparative_analysis.py': 
import numpy as np
import matplotlib.pyplot as plt
from greedy_algorithm import GreedyAlgorithm
from epsilon_greedy_algorithm import EpsilonGreedyAlgorithm
from time_varying_epsilon_greedy_algorithm import TimeVaryingEpsilonGreedyAlgorithm
from successive_elimination_algorithm import SuccessiveEliminationAlgorithm
from ucb_algorithm import UCBAlgorithm
from thompson_sampling import ThompsonSampling

def run_simulation(algo, n_arms, n_simulations, horizon):
    rewards = np.zeros(horizon)
    for _ in range(n_simulations):
        algo_instance = algo(n_arms)
        for t in range(horizon):
            chosen_arm = algo_instance.select_arm()
            reward = np.random.binomial(1, np.random.rand())  # Random reward for simulation
            algo_instance.update(chosen_arm, reward)
            rewards[t] += reward
    return rewards / n_simulations

def plot_results(results, algorithms, horizon):
    for algo, rewards in results.items():
        plt.plot(range(horizon), rewards, label=algo)
    plt.xlabel('Time')
    plt.ylabel('Average Reward')
    plt.title('Multi-Armed Bandit Algorithms Comparative Analysis')
    plt.legend()
    plt.show()

def main():
    n_arms = 10
    n_simulations = 500
    horizon = 1000
    algorithms = {
        'Greedy': GreedyAlgorithm,
        'Epsilon-Greedy': EpsilonGreedyAlgorithm,
        'Time-Varying Epsilon-Greedy': TimeVaryingEpsilonGreedyAlgorithm,
        'Successive Elimination': SuccessiveEliminationAlgorithm,
        'UCB': UCBAlgorithm,
        'Thompson Sampling': ThompsonSampling
    }
    results = {algo: run_simulation(algo_class, n_arms, n_simulations, horizon) for algo, algo_class in algorithms.items()}
    plot_results(results, algorithms, horizon)

if __name__ == "__main__":
    main()

}

dynamic_programming_code = {
    'policy_iteration.py': 
import numpy as np

class PolicyIteration:
    def __init__(self, states, actions, transition_prob, rewards, gamma, theta):
        self.states = states
        self.actions = actions
        self.transition_prob = transition_prob
        self.rewards = rewards
        self.gamma = gamma
        self.theta = theta
        self.policy = np.random.choice(actions, len(states))
        self.value_function = np.zeros(len(states))
    
    def policy_evaluation(self):
        while True:
            delta = 0
            for s in self.states:
                v = self.value_function[s]
                a = self.policy[s]
                self.value_function[s] = sum([self.transition_prob[s][a][s_prime] * (self.rewards[s][a][s_prime] + self.gamma * self.value_function[s_prime]) for s_prime in self.states])
                delta = max(delta, abs(v - self.value_function[s]))
            if delta < self.theta:
                break
    
    def policy_improvement(self):
        policy_stable = True
        for s in self.states:
            old_action = self.policy[s]
            action_values = [sum([self.transition_prob[s][a][s_prime] * (self.rewards[s][a][s_prime] + self.gamma * self.value_function[s_prime]) for s_prime in self.states]) for a in self.actions]
            self.policy[s] = self.actions[np.argmax(action_values)]
            if old_action != self.policy[s]:
                policy_stable = False
        return policy_stable
    
    def iterate(self):
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break
        return self.policy, self.value_function
,
    'value_iteration.py':
import numpy as np

class ValueIteration:
    def __init__(self, states, actions, transition_prob, rewards, gamma, theta):
        self.states = states
        self.actions = actions
        self.transition_prob = transition_prob
        self.rewards = rewards
        self.gamma = gamma
        self.theta = theta
        self.value_function = np.zeros(len(states))
        self.policy = np.zeros(len(states), dtype=int)
    
    def iterate(self):
        while True:
            delta = 0
            for s in self.states:
                v = self.value_function[s]
                action_values = [sum([self.transition_prob[s][a][s_prime] * (self.rewards[s][a][s_prime] + self.gamma * self.value_function[s_prime]) for s_prime in self.states]) for a in self.actions]
                self.value_function[s] = max(action_values)
                delta = max(delta, abs(v - self.value_function[s]))
            if delta < self.theta:
                break
        for s in self.states:
            action_values = [sum([self.transition_prob[s][a][s_prime] * (self.rewards[s][a][s_prime] + self.gamma * self.value_function[s_prime]) for s_prime in self.states]) for a in self.actions]
            self.policy[s] = np.argmax(action_values)
        return self.policy, self.value_function
,
    'comparative_analysis.py': 
import numpy as np
import matplotlib.pyplot as plt
from policy_iteration import PolicyIteration
from value_iteration import ValueIteration

def run_simulation(algo, states, actions, transition_prob, rewards, gamma, theta):
    algo_instance = algo(states, actions, transition_prob, rewards, gamma, theta)
    policy, value_function = algo_instance.iterate()
    return policy, value_function

def plot_results(value_functions, algorithms):
    for algo, values in value_functions.items():
        plt.plot(range(len(values)), values, label=algo)
    plt.xlabel('States')
    plt.ylabel('Value Function')
    plt.title('Dynamic Programming Algorithms Comparative Analysis')
    plt.legend()
    plt.show()

def main():
    states = list(range(16))
    actions = [0, 1, 2, 3]  # Up, Down, Left, Right
    transition_prob = np.zeros((16, 4, 16))
    rewards = np.zeros((16, 4, 16))
    gamma = 0.9
    theta = 0.0001
    
    # Define transitions and rewards here (simplified example)
    
    algorithms = {
        'Policy Iteration': PolicyIteration,
        'Value Iteration': ValueIteration
    }
    results = {algo: run_simulation(algo_class, states, actions, transition_prob, rewards, gamma, theta)[1] for algo, algo_class in algorithms.items()}
    plot_results(results, algorithms)

if __name__ == "__main__":
    main()

}

model_free_learning_code = {
    'sarsa.py': 
import numpy as np

class SARSA:
    def __init__(self, states, actions, alpha, gamma, epsilon, episodes):
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = np.zeros((len(states), len(actions)))
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(self.q_table[state])
    
    def learn(self, s, a, r, s_, a_):
        predict = self.q_table[s, a]
        target = r + self.gamma * self.q_table[s_, a_]
        self.q_table[s, a] += self.alpha * (target - predict)
    
    def run(self, environment):
        for _ in range(self.episodes):
            state = environment.reset()
            action = self.choose_action(state)
            while True:
                next_state, reward, done = environment.step(state, action)
                next_action = self.choose_action(next_state)
                self.learn(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
                if done:
                    break
        return self.q_table
,
    'q_learning.py': 
import numpy as np

class QLearning:
    def __init__(self, states, actions, alpha, gamma, epsilon, episodes):
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = np.zeros((len(states), len(actions)))
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(self.q_table[state])
    
    def learn(self, s, a, r, s_):
        predict = self.q_table[s, a]
        target = r + self.gamma * np.max(self.q_table[s_])
        self.q_table[s, a] += self.alpha * (target - predict)
    
    def run(self, environment):
        for _ in range(self.episodes):
            state = environment.reset()
            while True:
                action = self.choose_action(state)
                next_state, reward, done = environment.step(state, action)
                self.learn(state, action, reward, next_state)
                state = next_state
                if done:
                    break
        return self.q_table
,
    'comparative_analysis.py': 
import numpy as np
import matplotlib.pyplot as plt
from sarsa import SARSA
from q_learning import QLearning

class DummyEnvironment:
    def __init__(self):
        self.current_state = 0
    
    def reset(self):
        self.current_state = 0
        return self.current_state
    
    def step(self, state, action):
        next_state = state + action
        reward = np.random.rand()
        done = next_state >= 10
        return next_state, reward, done

def run_simulation(algo, states, actions, alpha, gamma, epsilon, episodes, environment):
    algo_instance = algo(states, actions, alpha, gamma, epsilon, episodes)
    q_table = algo_instance.run(environment)
    return q_table

def plot_results(q_tables, algorithms):
    for algo, q_table in q_tables.items():
        plt.plot(range(len(q_table)), q_table[:, 0], label=algo)
    plt.xlabel('States')
    plt.ylabel('Q-Values')
    plt.title('Model-Free Learning Algorithms Comparative Analysis')
    plt.legend()
    plt.show()

def main():
    states = list(range(10))
    actions = [0, 1]  # Simplified actions
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    episodes = 1000
    
    environment = DummyEnvironment()
    
    algorithms = {
        'SARSA': SARSA,
        'Q-Learning': QLearning
    }
    results = {algo: run_simulation(algo_class, states, actions, alpha, gamma, epsilon, episodes, environment) for algo, algo_class in algorithms.items()}
    plot_results(results, algorithms)

if __name__ == "__main__":
    main()

}
simulation_code = {
    'enviroment.py': 
import numpy as np

class BanditEnvironment:
    def __init__(self, n_arms, probabilities):
        self.n_arms = n_arms
        self.probabilities = probabilities
    
    def reset(self):
        return 0
    
    def step(self, action):
        reward = np.random.binomial(1, self.probabilities[action])
        return reward
,
    'run_simulation.py': 
import numpy as np
import matplotlib.pyplot as plt
from multi_armed_bandit.greedy_algorithm import GreedyAlgorithm
from multi_armed_bandit.epsilon_greedy_algorithm import EpsilonGreedyAlgorithm
from multi_armed_bandit.time_varying_epsilon_greedy_algorithm import TimeVaryingEpsilonGreedyAlgorithm
from multi_armed_bandit.successive_elimination_algorithm import SuccessiveEliminationAlgorithm
from multi_armed_bandit.ucb_algorithm import UCBAlgorithm
from multi_armed_bandit.thompson_sampling import ThompsonSampling
from environment import BanditEnvironment

def run_simulation(algo, env, n_simulations, horizon):
    rewards = np.zeros(horizon)
    for _ in range(n_simulations):
        algo_instance = algo(env.n_arms)
        for t in range(horizon):
            chosen_arm = algo_instance.select_arm()
            reward = env.step(chosen_arm)
            algo_instance.update(chosen_arm, reward)
            rewards[t] += reward
    return rewards / n_simulations

def plot_results(results, algorithms, horizon):
    for algo, rewards in results.items():
        plt.plot(range(horizon), rewards, label=algo)
    plt.xlabel('Time')
    plt.ylabel('Average Reward')
    plt.title('Multi-Armed Bandit Algorithms Comparative Analysis')
    plt.legend()
    plt.show()

def main():
    n_arms = 5
    probabilities = [0.1, 0.3, 0.5, 0.7, 0.9]  # True probabilities of each arm
    env = BanditEnvironment(n_arms, probabilities)
    n_simulations = 1000
    horizon = 500
    algorithms = {
        'Greedy': GreedyAlgorithm,
        'Epsilon-Greedy': lambda n_arms: EpsilonGreedyAlgorithm(n_arms, 0.1),
        'Time-Varying Epsilon-Greedy': TimeVaryingEpsilonGreedyAlgorithm,
        'Successive Elimination': lambda n_arms: SuccessiveEliminationAlgorithm(n_arms, horizon),
        'UCB': UCBAlgorithm,
        'Thompson Sampling': ThompsonSampling
    }
    results = {algo: run_simulation(algo_class, env, n_simulations, horizon) for algo, algo_class in algorithms.items()}
    plot_results(results, algorithms, horizon)

if __name__ == "__main__":
    main()
}


for file_name, code in multi_armed_bandit_code.items():
    with open(f'K-armed/multi_armed_bandit/{file_name}', 'w') as f:
        f.write(code)

for file_name, code in dynamic_programming_code.items():
    with open(f'K-armed/dynamic_programming/{file_name}', 'w') as f:
        f.write(code)

for file_name, code in model_free_learning_code.items():
    with open(f'K-armed/model_free_learning/{file_name}', 'w') as f:
        f.write(code)
        
for file_name, code in simulation_code.items():
    with open(f'K-armed/simulation/{file_name}', 'w') as f:
        f.write(code)

# Create a README file
with open('K-armed/README.md', 'w') as f:
    f.write('''
# K-armed Project

This project explores various algorithms such as Greedy Algorithm, Epsilon-Greedy Algorithm, Time-Varying Epsilon-Greedy Algorithm, Successive Elimination Algorithm, Upper-Confidence-Bound (UCB) Algorithm, and Thompson Sampling (TS) for optimal rewards in a multi-armed bandit situation.

Additionally, it includes the implementation and comparative analysis of Policy Iteration and Value Iteration algorithms from scratch, and Model-Free Learning Algorithms like SARSA and Q-Learning.

## Project Structure
K-armed/
├── README.md
├── multi_armed_bandit/
│ ├── init.py
│ ├── greedy_algorithm.py
│ ├── epsilon_greedy_algorithm.py
│ ├── time_varying_epsilon_greedy_algorithm.py
│ ├── successive_elimination_algorithm.py
│ ├── ucb_algorithm.py
│ ├── thompson_sampling.py
│ └── comparative_analysis.py
├── dynamic_programming/
│ ├── init.py
│ ├── policy_iteration.py
│ ├── value_iteration.py
│ └── comparative_analysis.py
├── model_free_learning/
│ ├── init.py
│ ├── sarsa.py
│ ├── q_learning.py
│ └── comparative_analysis.py
└── data/
└── results/
    ''')

# Create a ZIP file for the folder
shutil.make_archive('K-armed', 'zip', 'K-armed')
