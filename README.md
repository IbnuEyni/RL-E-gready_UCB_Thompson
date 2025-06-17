# Multi-Armed Bandit Algorithms: Exploration vs. Exploitation

This project implements and compares three classic algorithms for solving the Multi-Armed Bandit (MAB) problem: **Epsilon-Greedy**, **Upper Confidence Bound (UCB)**, and **Thompson Sampling**. The goal is to understand how these algorithms balance the trade-off between exploring new options (arms) and exploiting the best-known option to maximize cumulative reward over time.


## Table of Contents
- [Project Description](#project-description)
- [Algorithms Implemented](#algorithms-implemented)
- [File Structure](#file-structure)
- [How to Run](#how-to-run)
- [Experiment Analysis](#experiment-analysis)
- [Requirements](#requirements)


## Project Description

The Multi-Armed Bandit problem is a classic reinforcement learning scenario where an agent must repeatedly choose between several "arms" (actions), each providing a reward from an unknown probability distribution. The agent's objective is to maximize the total reward collected over a series of trials. This involves effectively balancing:

- **Exploration**: Trying different arms to learn their reward distributions.
- **Exploitation**: Choosing the arm currently believed to be the best.

This project provides Python implementations of three popular bandit algorithms and a framework to simulate their performance against a set of arms with predefined true reward probabilities.


## Algorithms Implemented

- **Epsilon-Greedy**  
  With a probability `ϵ`, chooses a random arm (*exploration*), and with probability `1−ϵ`, chooses the arm with the highest estimated average reward (*exploitation*).

- **UCB (Upper Confidence Bound)**  
  Selects arms based on an upper confidence bound of their true reward. Favors arms with high estimated rewards and those pulled less frequently, encouraging exploration of uncertain options.

- **Thompson Sampling**  
  A Bayesian approach. For each arm, samples from a posterior distribution (e.g., Beta for Bernoulli arms), and chooses the arm with the highest sample. Naturally balances exploration and exploitation using uncertainty.


## File Structure

The project is organized into the following Python files:

- **`bandit_environment.py`**  
  Contains the `pull_arm` function, simulating the reward from pulling an arm.

- **`bandit_agents.py`**  
  Defines the `BanditAgent` base class and its subclasses:
  - `EpsilonGreedyAgent`
  - `UCBAgent`
  - `ThompsonSamplingAgent`  
  Each encapsulates arm selection and reward update logic.

- **`simulation.py`**  
  Contains the `run_simulation` function, orchestrating interactions between agent and environment.

- **`main.py`**  
  The main script that initializes agents, runs simulations, prints metrics (total rewards, arm pulls), and generates plots.


## How to Run

To run the experiment and visualize the results:

1. **Save the files**  
   Ensure all four Python files are saved in the same directory.

2. **Install dependencies**  
   Use pip to install required packages:

   ```bash
   pip install numpy matplotlib

3. **Run the script**
   Navigate to the directory and execute:

   ```bash
   python main.py
   ```

4. **Output**
   The script prints experimental results and shows:

   * Cumulative reward over time
   * Average reward per step for each algorithm


## Experiment Analysis

The simulation uses a 5-armed bandit with true success probabilities:
`[0.1, 0.3, 0.6, 0.2, 0.8]` (optimal arm: 0.8).
Each algorithm runs for **2000 steps**.

### Epsilon-Greedy Analysis

* **High ϵ (e.g., 0.3)**
  More exploration → Faster optimal arm discovery, but lower long-term rewards due to random suboptimal pulls.
* **Low ϵ (e.g., 0.01)**
  More exploitation → High rewards if optimal arm found early, but risk of getting stuck on a suboptimal arm.
* **Conclusion**: Balanced ϵ (e.g., 0.1) typically performs best.

### UCB Algorithm Analysis

* **Exploration parameter (`c`)**

  * Higher `c`: More exploration
  * Lower `c`: Greedier behavior
* **Performance**:
  UCB typically outperforms Epsilon-Greedy due to more principled and efficient exploration.

### Thompson Sampling Analysis

* **Performance**:
  Performs consistently well across runs. Being Bayesian, it naturally balances exploration and exploitation.
* **Comparison**:
  Often rivals or exceeds UCB. Adaptively explores based on posterior uncertainty.


## Overall Comparison

| Metric                     | Epsilon-Greedy     | UCB        | Thompson Sampling |
| -------------------------- | ------------------ | ---------- | ----------------- |
| **Total Reward**           | Lower              | High       | Highest           |
| **Exploration Efficiency** | Random             | Systematic | Adaptive          |
| **Convergence Speed**      | Slow (ϵ-dependent) | Fast       | Fastest           |

* **Plots** confirm:
  UCB and Thompson Sampling show faster rises and maintain higher average rewards.


## Requirements

* Python 3.x
* numpy
* matplotlib
