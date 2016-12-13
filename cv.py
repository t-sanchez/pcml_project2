import numpy as np
from cost import *
from predictionAlgorithms import *
from helpers import df_load, df_to_sparse
import pywFM

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
    """return the loss of Alternating Least Squares with Regularisation."""
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


def cross_validationALSBias_demo(data):
    """
        Full example of cross validation using the ALS with Bias method from pywFM. 
        It returns the best parameters after performing a grid search on the data we're giving.
        /!\ It takes a very long time to run, do not give too much parameters at a time /!\
    """

    # 1. Setting the lists of values on which we're doing the grid search as well as general parameters
    num_iter = 40; k_fold = 4
    std_init_vec=[0.43]; rank_vec=[7]; r0_reg_vec=[0.5]; r1_reg_vec=[15,20]; r2_reg_vec=[20,25]
    
    # 2. Loading the data and preparing the run.
    df = data
    k_indices = build_k_indices(df['Prediction'], k_fold, 12)

    rmse_te = np.zeros((len(std_init_vec) , len(rank_vec), len(r0_reg_vec),len(r1_reg_vec),len(r2_reg_vec)))    
    best_rmse = 1000000; best_std = 0; best_rank = 0;
    best_r0 = 0; best_r1 = 0; best_r2 = 0;
    
    print("\t\tStarting the ",k_fold,"-fold cross validation for ALS with bias\n")
    # 3. Cross-Validation (Hurray for the 6 for loops !)
    for i, std_init in enumerate(std_init_vec):
        for j, rank in enumerate(rank_vec):
            for k, r0_reg in enumerate(r0_reg_vec):
                for l, r1_reg in enumerate(r1_reg_vec):
                    for m, r2_reg in enumerate(r2_reg_vec):
                        rmse_te_tmp = []      
                    
                        for k_cv in range(0,k_fold):
                            rmse_te_tmp.append(cross_validationALSBias(df, k_indices, k_cv,
                                    num_iter , std_init, rank, r0_reg, r1_reg, r2_reg))
                        
                        mean_rmse = np.mean(rmse_te_tmp)    
                        print("RMSE = ", mean_rmse," (", num_iter, "iterations, std_init =",
                                std_init, ", k=", rank, ", r0_reg=", r0_reg,
                                ", r1_reg=", r1_reg,", r2_reg =", r2_reg,")")
    
                        if mean_rmse < best_rmse:
                            best_rmse = mean_rmse; best_std = std_init; 
                            best_rank = rank; best_r0 = r0_reg; best_r1 = r1_reg; best_r2 = r2_reg;   
                    
                        rmse_te[i][j][k][l][m] = mean_rmse

    return best_rmse, best_std, best_rank, best_r0, best_r1, best_r2  


def cross_validationALSBias(data, k_indices, k, num_iter, std_init, rank, r0_reg, r1_reg,r2_reg):
    """
        Runs the cross validation on the input data, using the ALS algorithm with the user bias included. 
        It splits the data into a training and testing fold, according to k_indices and k, and then runs 
        the ALS with bias on all the parameters (std_init, rank, r0_reg, r1_reg, r2_reg) for num_iter iterations.
        @param data : the DataFrame containing all our training data (on which we do the CV)
        @param k_indices : array of k-lists containing each of the splits of the data
        @param k : the number of folds of the cross-validation
        @param num_iter : the number of iterations of the algorithm
        @param std_init : the standard deviation for the initialisation of W and Z
        @param rank : the number of columns of W and Z
        @param r0_reg : the regularization parameter for the global bias term w0
        @param r1_reg : the regularization parameter of the user/item bias term w
        @param r2_reg : the regularization parameter for the ALS regularization (size of the entries of W and Z)
        @return loss_te : the RMSE loss for the run of the algorithm using libFM with these parameters.
        
    """
    # get k'th subgroup in test, others in train
    te_indices = k_indices[k]
    tr_indices = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indices = tr_indices.reshape(-1)
    
    train = data.loc[tr_indices]
    test = data.loc[te_indices]
    test.sort_values(['Movie', 'User'], ascending=[1,1], inplace=True)

    # format the DataFrames into the Sparse matrices we need to run with pywFM
    features_tr, target_tr =  df_to_sparse(train)
    features_te, target_te = df_to_sparse(test)

    # running the model
    fm = pywFM.FM(task = 'regression', learning_method='als', num_iter=num_iter, init_stdev = std_init,
                  k2 = rank, r0_regularization = r0_reg, r1_regularization = r1_reg, r2_regularization = r2_reg)
    
    model = fm.run(features_tr, target_tr, features_te, target_te)
    
    # getting the RMSE at the last run step.
    loss_te = model.rlog.rmse[num_iter-1]
                          
    return loss_te


def cross_validationMCMC_demo(data):
    
    # 1. Setting the lists of values on which we're doing the grid search as well as general parameters
    num_iter = 20; k_fold = 4
    std_init_vec=[0.375, 0.4, 0.45, 0.5];

    # 2. Loading the data and preparing the run.
    df = data
    k_indices = build_k_indices(df['Prediction'], k_fold, 12)

    best_rmse = 1000000;
    best_std = 0;
    
    rmse_te = []
    
    print("\t\tStarting the ",k_fold,"-fold cross validation for MCMC\n")
    for std_init in std_init_vec:
        rmse_te_tmp = []      
        for k_cv in range(0,k_fold):
            rmse_te_tmp.append(cross_validationMCMC(df, k_indices, k_cv, num_iter , std_init))
                        
        mean_rmse = np.mean(rmse_te_tmp)    
        print("RMSE = ", mean_rmse," (for MCMC with with ", num_iter, "iterations, std_init =", std_init,")")
    
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse; best_std = std_init;                     
        rmse_te.append(mean_rmse)
        
    return best_rmse, best_std

def cross_validationMCMC(data, k_indices, k, num_iter, std_init):
    """
        Runs the cross validation on the input data, using the Markov Chain Monte Carlo algorithm. 
        It splits the data into a training and testing fold, according to k_indices and k, and then runs 
        the MCMC on all the parameter std_init for num_iter iterations.
        @param data : the DataFrame containing all our training data (on which we do the CV)
        @param k_indices : array of k-lists containing each of the splits of the data
        @param k : the number of folds of the cross-validation
        @param num_iter : the number of iterations of the algorithm
        @param std_init : the standard deviation for the initialisation of the data
        @return loss_te : the RMSE loss for the run of the algorithm using libFM with these parameters.
        
    """
    # get k'th subgroup in test, others in train
    te_indices = k_indices[k]
    tr_indices = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indices = tr_indices.reshape(-1)
    
    train = data.loc[tr_indices]
    test = data.loc[te_indices]
    test.sort_values(['Movie', 'User'], ascending=[1,1], inplace=True)

    # format the DataFrames into the Sparse matrices we need to run with pywFM
    features_tr, target_tr =  df_to_sparse(train)
    features_te, target_te = df_to_sparse(test)

    # running the model
    fm = pywFM.FM(task='regression', num_iter= num_iter,init_stdev = std_init)
    
    model = fm.run(features_tr, target_tr, features_te, target_te)
    
    # getting the RMSE at the last run step.
    loss_te = model.rlog.rmse[num_iter-1]
                          
    return loss_te

def optimize_weights(predictions1, predictions2, predictions3, predictions4, predictions5, labels):
    resolution = 50
    cost_best = 10
    w1List = np.linspace(0,0.2,5)
    w2List = np.linspace(0,0.2,5)
    w3List = np.linspace(0,0.5,10)
    w4List = np.linspace(0,1,10)
    #w5List = np.linspace(0,1,10)
    w1_best = 0
    w2_best = 0
    w3_best = 0
    w4_best = 0
    w5_best = 0
    
    for w1 in w1List:
        for w2 in w2List:
            for w3 in w3List:
                for w4 in w4List:
                    if w1 + w2 + w3 + w4 <= 1:
                        w5 = 1 - (w1 + w2 + w3 + w4)
                        predictions = w1*predictions1 + w2*predictions2 + w3*predictions3 + w4*predictions4 + w5*predictions5
                        c = compute_cost(predictions, labels)
                        #print("w1 = {w1}, w2 = {w2}, w3 = {w3} : loss = {l}".format(w1=w1, w2=w2, w3=w3, l=c))
                        if c < cost_best:
                            w1_best = w1
                            w2_best = w2
                            w3_best = w3
                            w4_best = w4
                            w5_best = w5
                            cost_best = c
    return w1_best, w2_best, w3_best, w4_best, w5_best, cost_best
    
