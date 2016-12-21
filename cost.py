import numpy as np


def compute_cost(predictions, labels):
    """
        Computes the RMSE between the predictions (what the algorithm outputs) and the labels (correct values from the training set)
        @param predictions : the array of predictions of an algorithm
        @param labels : the array of labels (correct values) corresponding to the same ids as the predictions
        @return Root Mean Square Error
    """
    return np.sqrt(np.mean((predictions-labels)**2))
    