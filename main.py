import numpy as np
import matplotlib.pyplot as plt

# Import agents and simulation function from their respective files
from bandit_agents import EpsilonGreedyAgent, UCBAgent, ThompsonSamplingAgent
from simulation import run_simulation

if __name__ == "__main__":
    NUM_ARMS = 5
    # True success probabilities for each arm. Arm 4 has the highest probability.
    TRUE_PROBABILITIES = [0.1, 0.3, 0.6, 0.2, 0.8]
    NUM_STEPS = 2000 # Increased steps for better visualization of convergence

    print(f"--- Multi-Armed Bandit Experiment ---")
    print(f"Number of Arms: {NUM_ARMS}")
    print(f"True Probabilities: {TRUE_PROBABILITIES}")
    print(f"Number of Steps: {NUM_STEPS}")
    print(f"Optimal Arm (index): {np.argmax(TRUE_PROBABILITIES)} (Probability: {TRUE_PROBABILITIES[np.argmax(TRUE_PROBABILITIES)]})")
    print("-" * 40)

    results = {} # To store results for plotting

    # --- Epsilon-Greedy Experiments ---
    print("\n--- Epsilon-Greedy Experiments ---")
    epsilon_values = [0.01, 0.1, 0.3]
    for epsilon in epsilon_values:
        agent = EpsilonGreedyAgent(NUM_ARMS, epsilon)
        total_reward, cumulative_rewards, arm_pull_counts = run_simulation(agent, TRUE_PROBABILITIES, NUM_STEPS)
        results[f'Epsilon-Greedy (ε={epsilon})'] = cumulative_rewards
        print(f"Epsilon-Greedy (epsilon={epsilon}): Total Reward = {total_reward:.2f}")
        print(f"  Arm Pull Counts: {arm_pull_counts}")
        print(f"  % Optimal Arm Pulled: {arm_pull_counts[np.argmax(TRUE_PROBABILITIES)] / NUM_STEPS * 100:.2f}%")
        print("-" * 30)

    # --- UCB Algorithm Experiments ---
    print("\n--- UCB Algorithm Experiments ---")
    c_values = [0.1, 1.0, 2.0]
    for c in c_values:
        agent = UCBAgent(NUM_ARMS, c)
        total_reward, cumulative_rewards, arm_pull_counts = run_simulation(agent, TRUE_PROBABILITIES, NUM_STEPS)
        results[f'UCB (c={c})'] = cumulative_rewards
        print(f"UCB (c={c}): Total Reward = {total_reward:.2f}")
        print(f"  Arm Pull Counts: {arm_pull_counts}")
        print(f"  % Optimal Arm Pulled: {arm_pull_counts[np.argmax(TRUE_PROBABILITIES)] / NUM_STEPS * 100:.2f}%")
        print("-" * 30)

    # --- Thompson Sampling Experiments ---
    print("\n--- Thompson Sampling Experiments ---")
    agent = ThompsonSamplingAgent(NUM_ARMS)
    total_reward, cumulative_rewards, arm_pull_counts = run_simulation(agent, TRUE_PROBABILITIES, NUM_STEPS)
    results['Thompson Sampling'] = cumulative_rewards
    print(f"Thompson Sampling: Total Reward = {total_reward:.2f}")
    print(f"  Arm Pull Counts: {arm_pull_counts}")
    print(f"  % Optimal Arm Pulled: {arm_pull_counts[np.argmax(TRUE_PROBABILITIES)] / NUM_STEPS * 100:.2f}%")
    print("-" * 30)

    # --- Visualization ---
    plt.figure(figsize=(12, 8))
    for label, rewards in results.items():
        plt.plot(rewards, label=label, alpha=0.8)

    plt.title('Cumulative Reward Over Time for Different Bandit Algorithms')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Additional Plot: Average Reward per Step ---
    plt.figure(figsize=(12, 8))
    for label, rewards in results.items():
        # Calculate average reward per step
        average_rewards = np.array(rewards) / (np.arange(1, len(rewards) + 1))
        plt.plot(average_rewards, label=label, alpha=0.8)

    plt.title('Average Reward Per Step Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward per Step')
    plt.ylim(0, 1) # Probabilities are between 0 and 1
    plt.legend()
    plt.grid(True)
    plt.show()
