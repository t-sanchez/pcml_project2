import numpy as np


def compute_cost(predictions, labels):
    return np.mean((predictions-labels)**2)
    