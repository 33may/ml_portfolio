import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(scores, window=5, xlabel="Game", ylabel="Score"):
    """
    Plots the running average of scores over episodes.

    Args:
        scores (list or np.ndarray): A sequence of episodic scores.
        window (int): The number of episodes to use for the moving average.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
    """
    scores = np.array(scores)
    moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(range(window - 1, len(scores)), moving_avg)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(False)
    plt.tight_layout()
    plt.show()
