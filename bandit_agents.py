import numpy as np

# --- Base Bandit Agent Class ---
class BanditAgent:
    """
    Base class for multi-armed bandit agents.
    Provides common initialization and reward tracking.
    """
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.q_estimates = np.zeros(num_arms) # Estimated Q-values (average reward) for each arm
        self.n_pulls = np.zeros(num_arms)     # Count of how many times each arm has been pulled
        self.total_reward = 0
        self.cumulative_rewards = []          # List to store cumulative reward at each step

    def choose_arm(self, step=0):
        """
        Abstract method to choose an arm.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("choose_arm method must be implemented by subclasses")

    def update(self, chosen_arm, reward):
        """
        Updates the agent's estimates and reward based on the chosen arm and received reward.
        """
        self.total_reward += reward
        self.cumulative_rewards.append(self.total_reward)
        self.n_pulls[chosen_arm] += 1
        # Update Q-estimate using incremental update formula
        # Q_new = Q_old + (1/n) * (Reward - Q_old)
        self.q_estimates[chosen_arm] = self.q_estimates[chosen_arm] + \
                                      (1 / self.n_pulls[chosen_arm]) * (reward - self.q_estimates[chosen_arm])

# --- Epsilon-Greedy Agent Implementation ---
class EpsilonGreedyAgent(BanditAgent):
    """
    Implements the Epsilon-Greedy algorithm.
    """
    def __init__(self, num_arms, epsilon):
        super().__init__(num_arms)
        self.epsilon = epsilon

    def choose_arm(self, step=0):
        """
        Chooses an arm using the epsilon-greedy strategy.
        """
        if np.random.rand() < self.epsilon:
            # Explore: Choose a random arm
            return np.random.randint(self.num_arms)
        else:
            # Exploit: Choose the arm with the highest estimated Q-value
            return np.argmax(self.q_estimates)

# --- UCB (Upper Confidence Bound) Agent Implementation ---
class UCBAgent(BanditAgent):
    """
    Implements the UCB (Upper Confidence Bound) algorithm.
    """
    def __init__(self, num_arms, c_param):
        super().__init__(num_arms)
        self.c_param = c_param
        self.step_count = 0 # To track total steps for UCB formula

    def choose_arm(self, step):
        """
        Chooses an arm using the UCB strategy.
        Initial phase: play each arm once.
        """
        self.step_count = step + 1 # UCB formula uses 1-indexed step count (t)

        # Initial phase: Play each arm once to get initial estimates and avoid division by zero
        if self.step_count <= self.num_arms:
            return self.step_count - 1 # Return 0, 1, 2... for first num_arms steps
        else:
            ucb_values = np.zeros(self.num_arms)
            for arm in range(self.num_arms):
                # UCB formula: Q(a) + c * sqrt(log(t) / N(a))
                # Note: log(self.step_count) because step is 0-indexed, and we need total pulls
                ucb_values[arm] = self.q_estimates[arm] + \
                                  self.c_param * np.sqrt(np.log(self.step_count) / self.n_pulls[arm])
            return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        """
        Updates the agent's estimates and reward for UCB.
        Initial phase needs specific handling for q_estimates.
        """
        self.total_reward += reward
        self.cumulative_rewards.append(self.total_reward)
        self.n_pulls[chosen_arm] += 1
        
        # For initial pulls, q_estimate is just the reward received
        if self.n_pulls[chosen_arm] == 1:
            self.q_estimates[chosen_arm] = reward
        else:
            # Standard incremental update after the first pull
            self.q_estimates[chosen_arm] = self.q_estimates[chosen_arm] + \
                                          (1 / self.n_pulls[chosen_arm]) * (reward - self.q_estimates[chosen_arm])

# --- Thompson Sampling Agent Implementation ---
class ThompsonSamplingAgent(BanditAgent):
    """
    Implements the Thompson Sampling algorithm for Bernoulli bandit problems.
    """
    def __init__(self, num_arms):
        super().__init__(num_arms)
        # Initialize parameters for the Beta distribution for each arm
        # alpha = number of successes + 1
        # beta = number of failures + 1
        self.alphas = np.ones(num_arms)
        self.betas = np.ones(num_arms)

    def choose_arm(self, step=0):
        """
        Chooses an arm by sampling from Beta distributions.
        """
        # Sample a theta (probability) from the Beta distribution for each arm
        sampled_thetas = [np.random.beta(self.alphas[arm], self.betas[arm]) for arm in range(self.num_arms)]
        # Choose the arm with the highest sampled theta
        return np.argmax(sampled_thetas)

    def update(self, chosen_arm, reward):
        """
        Updates the Beta distribution parameters based on the reward.
        Note: n_pulls and q_estimates are not strictly used by TS,
        but updated for consistency with BanditAgent's tracking.
        """
        self.total_reward += reward
        self.cumulative_rewards.append(self.total_reward)
        self.n_pulls[chosen_arm] += 1 # Track pulls for stats, not used by TS core logic

        if reward == 1:
            self.alphas[chosen_arm] += 1
        else:
            self.betas[chosen_arm] += 1
        # Q-estimates can still be calculated for monitoring, though not used for decision making in TS
        # This calculation needs to handle the initial state where alphas+betas might be small
        if (self.alphas[chosen_arm] + self.betas[chosen_arm] - 2) > 0:
            self.q_estimates[chosen_arm] = (self.alphas[chosen_arm] - 1) / (self.alphas[chosen_arm] + self.betas[chosen_arm] - 2)
        else: # Handle division by zero for initial pulls where a success/failure hasn't updated counts yet
             self.q_estimates[chosen_arm] = reward
