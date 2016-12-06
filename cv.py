import numpy as np
from cost import *
from predictionAlgorithms import *

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_demo(data):
    seed = 1
    k_fold = 4
    numIterations = 20
    rank = 6
    lambdas = np.linspace(0.07, 0.08, 3)
    # split data in k fold
    k_indices = build_k_indices(data, k_fold, seed)
    # define lists to store the loss of training data and test data
    #rmse_tr = []
    mse_te = []
    best_lambda = lambdas[0]
    best_mse = 10
    # cross validation
    print("{k}-fold cross validation".format(k = k_fold))
    for lambda_ in lambdas:
        #rmse_tr_tmp = []
        mse_te_tmp = []
        print("lambda = {l}".format(l=lambda_))
        for k in range(k_fold):
            loss_te = cross_validationALS(data, k_indices, k, rank, lambda_, numIterations)
            print("k = {k} : loss = {l}".format(k=k, l = loss_te))
            #rmse_tr_tmp.append(loss_tr)
            mse_te_tmp.append(loss_te)
        #rmse_tr.append(np.mean(rmse_tr_tmp))
        mean_mse = np.mean(mse_te_tmp)
        print(mse_te_tmp)
        print("lambda = {l} : mse = {m}".format(l = lambda_, m = mean_mse))
        if mean_mse < best_mse:
            best_lambda = lambda_
        mse_te.append(mean_mse)
    return best_lambda, best_mse
    #cross_validation_visualization(lambdas, rmse_tr, rmse_te)

def cross_validationALS(data, k_indices, k, rank, lambda_, numIterations):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    train = data.loc[tr_indice]
    test = data.loc[te_indice]
    test.sort_values(['User', 'Movie'], ascending=[1,1], inplace=True)
    print("train : {s} elements".format(s = train.shape[0]))
    print("test : {s} elements".format(s = test.shape[0]))
    
    # ALS
    predictions = ALSPyspark(train, test, rank, lambda_, numIterations)
    print(predictions.head())
    print(test.head())
    # calculate the loss for train and test data
    #loss_tr = np.sqrt(2 * compute_mse(y_tr, tx_tr, w))
    loss_te = compute_cost(predictions._3.values, test.Prediction.values)
    return loss_te
    
def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    # form data with polynomial degree
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)
    # ridge regression
    w = ridge_regression(y_tr, tx_tr, lambda_)
    predictions = ALSPyspark(x_tr, x_te, rank, lambda_, numIterations)
    # calculate the loss for train and test data
    loss_tr = np.sqrt(2 * compute_mse(y_tr, tx_tr, w))
    loss_te = np.sqrt(2 * compute_mse(y_te, tx_te, w))
    return loss_tr, loss_te

def optimize_weights(predictions1, predictions2, predictions3, labels):
    resolution = 50
    cost_best = 10
    w1_best = 0
    w2_best = 0
    w3_best = 0
    for w1 in np.linspace(0,1,resolution):
        for w2 in np.linspace(0,1-w1,resolution*(1-w1)):
            w3 = 1-w1-w2
            predictions = w1*predictions1 + w2*predictions2 + w3*predictions3
            c = compute_cost(predictions, labels)
            #print("w1 = {w1}, w2 = {w2}, w3 = {w3} : loss = {l}".format(w1=w1, w2=w2, w3=w3, l=c))
            if c < cost_best:
                w1_best = w1
                w2_best = w2
                w3_best = w3
                cost_best = c
    return w1_best, w2_best, w3_best, cost_best
    