import numpy as np

def pull_arm(true_probability):
    """
    Simulates pulling an arm with a given true success probability.
    Returns 1 for success (reward) or 0 for failure (no reward).
    """
    return 1 if np.random.rand() < true_probability else 0
