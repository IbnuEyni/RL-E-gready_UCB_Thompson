from bandit_environment import pull_arm
from bandit_agents import BanditAgent

def run_simulation(agent: BanditAgent, true_probabilities: list, num_steps: int):
    """
    Runs a simulation for a given bandit agent.

    Args:
        agent (BanditAgent): An instance of a bandit agent (e.g., EpsilonGreedyAgent, UCBAgent, ThompsonSamplingAgent).
        true_probabilities (list): True success probabilities for the arms.
        num_steps (int): Total steps to run.

    Returns:
        tuple: (total_reward, cumulative_rewards, arm_pull_counts) from the agent.
    """
    for step in range(num_steps):
        chosen_arm = agent.choose_arm(step)
        reward = pull_arm(true_probabilities[chosen_arm])
        agent.update(chosen_arm, reward)
    return agent.total_reward, agent.cumulative_rewards, agent.n_pulls
