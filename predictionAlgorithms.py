import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp

import findspark
findspark.init('/srv/spark')
import pyspark
sc = pyspark.SparkContext()
sql_sc = pyspark.sql.SQLContext(sc)
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

from helpers import df_to_sparse
import pywFM

# Returns the predicted labels of test set using user mean of train set
def baseline_user_mean(values_train, values_test):
    """baseline method: use the user means as the prediction."""    
    items = values_train['Movie']
    users = values_train['User']
    rates = values_train['Prediction']
    
    items_test = values_test['Movie']
    users_test = values_test['User']
    
    ratePerUser = np.zeros(len(np.unique(users))) # mean rate per user (over all movies)
    for i,user in enumerate(np.unique(users)):
        ratePerUser[i] = np.mean(rates[users == user])
        # ratePerUser[i] = mean rate given by user 'user' 

    return ratePerUser[users_test]


# Returns the predicted labels of test set using item mean of train set
def baseline_item_mean(values_train, values_test):
    """baseline method: use item means as the prediction."""
    items = values_train['Movie']
    users = values_train['User']
    rates = values_train['Prediction']
    
    items_test = values_test['Movie']
    users_test = values_test['User']
    
    ratePerMovie = np.zeros(len(np.unique(items))) # mean rate of each movie (over all users)
    for i,item in enumerate(np.unique(items)):
        ratePerMovie[i] = np.mean(rates[items == item])
    
    return ratePerMovie[items_test]

def ALSPysparkMe(train, test, rank, lambda_, numIterations):
    train_sdf = sql_sc.createDataFrame(train)
    
    model = ALS.train(train_sdf.rdd, rank, numIterations, lambda_)
    
    test_sdf = sql_sc.createDataFrame(test[['User', 'Movie']])
    predictionsTest = model.predictAll(test_sdf.rdd)
    predictions_df = predictionsTest.toDF()
    predictions_pd = predictions_df.toPandas()
    predictions_pd.columns = ['User', 'Movie', 'Prediction']
    predictions_pd.sort_values(['Movie','User'], ascending=[1,1], inplace=True)
    return predictions_pd


def ALSPyspark(train, test, rank, lambda_, numIterations):
    pDF = sql_sc.createDataFrame(train)
    spDF = pDF.rdd
    model = ALS.train(spDF, rank, numIterations,lambda_)
    
    pDF_test = sql_sc.createDataFrame(test[['User', 'Movie']])
    spDF_test = pDF_test.rdd
    testdata = spDF_test.map(lambda p: (p[0], p[1]))
    predictions = model.predictAll(testdata).map(lambda r: (r[0], r[1], r[2]))
    s  = predictions.toDF()
    lol = s.toPandas()
    lol.sort_values(['_1','_2'],ascending=[1,1],inplace=True)
    predic_end = lol.reset_index(drop=True)
    #test.Movie = test.Movie.values.astype(int)
    #test.User = test.User.values.astype(int)
    #new_test = test.sort_values(by=['User','Movie'],ascending=[True,True])
    #test_end = new_test.reset_index(drop=True)
    #test_end.head()
    
    return predic_end

def ALSBias_pywFM(train, test, num_iter=100, std_init = 0.43, rank = 7, r0_reg = 0.5, r1_reg = 15, r2_reg = 25):
    """
        Runs the ALS algorithm with the user bias included for num_iter iterations.
        @param train : the DataFrame containing all our training data.
        @param test : the DataFrame containing all our testing data.
        @param num_iter : the number of iterations of the algorithm
        @param std_init : the standard deviation for the initialisation of W and Z
        @param rank : the number of columns of W and Z
        @param r0_reg : the regularization parameter for the global bias term w0
        @param r1_reg : the regularization parameter of the user/item bias term w
        @param r2_reg : the regularization parameter for the ALS regularization (size of the entries of W and Z)
        @return loss_te : the RMSE loss for the run of the algorithm using libFM with these parameters.
        
    """
    # 1. Defining the model
    fm = pywFM.FM(task = 'regression', learning_method='als', num_iter=num_iter, init_stdev = std_init, k2 = rank,
             r0_regularization = r0_reg, r1_regularization = r1_reg, r2_regularization = r2_reg)
    
    # 2. Formatting the data
    features_tr, target_tr = df_to_sparse(train)
    features_te, target_te = df_to_sparse(test)
    
    # 3. Running the model
    model = fm.run(features_tr, target_tr, features_te, target_te)
    
    # 4. Outputs
    pred = model.predictions
    
    return np.array(pred)

def MCMC_pywFM(train, test, num_iter=100, std_init = 0.5):
    
    # 1. Defining the model
    fm = pywFM.FM(task='regression', num_iter= num_iter, init_stdev = std_init)
    
    # 2. Formatting the data
    features_tr, target_tr = df_to_sparse(train)
    features_te, target_te = df_to_sparse(test)
    
    # 3. Running the model
    model = fm.run(features_tr, target_tr, features_te, target_te)
    
    # 4. Outputs
    pred = model.predictions
    
    return np.array(pred)
