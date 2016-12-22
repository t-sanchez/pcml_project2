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

# Every algorithms are implemented in this file, and can all be used in a the same way. Each of them take as parameters the set on which the algorithm is trained, and the set to which the algorithm is applied for predictions. For the algorithm which have some parameters, the optimal ones are set by default, but can be modified by passing them as parameters of the function.

# Returns the predicted labels of test set using user mean of train set
def baseline_user_mean(values_train, values_test):
    """
        baseline method: use user means as the prediction.
        @param values_train : the DataFrame containing all our training data.
        @param values_test : the DataFrame containing all our testing data.
        
        @return ratePerUser[users_test] : the prediction values for all the data within the test set
    """ 
    items = values_train['Movie']
    users = values_train['User']
    rates = values_train['Prediction']
    
    items_test = values_test['Movie']
    users_test = values_test['User']
    
    ratePerUser = np.zeros(len(np.unique(users))) # mean rate per user (over all movies)
    for i,user in enumerate(np.unique(users)):
        # ratePerUser[i] : mean rate given by user 'user' 
        ratePerUser[i] = np.mean(rates[users == user])
        

    return ratePerUser[users_test]


# Returns the predicted labels of test set using item mean of train set
def baseline_item_mean(values_train, values_test):
    """
        baseline method: use item means as the prediction.
        @param values_train : the DataFrame containing all our training data.
        @param values_test : the DataFrame containing all our testing data.
        
        @return ratePerMovie[items_test] : the prediction values for all the data within the test set
    """
    items = values_train['Movie']
    users = values_train['User']
    rates = values_train['Prediction']
    
    items_test = values_test['Movie']
    users_test = values_test['User']
    
    ratePerMovie = np.zeros(len(np.unique(items))) # mean rate of each movie (over all users)
    for i,item in enumerate(np.unique(items)):
        # ratePerMovie[i] = mean rate given by to item 'item' 
        ratePerMovie[i] = np.mean(rates[items == item])
    
    return ratePerMovie[items_test]


def ALSPyspark(train, test, rank = 8, lambda_ = 0.081, numIterations = 20):
    """
        Runs the ALS algorithm without the user bias.
        N.B. The parameters passed by default are the best ones we found.

        @param train : the DataFrame containing all our training data.
        @param test : the DataFrame containing all our testing data.
        @param rank : the number of columns of W and Z
        @param lambda_ : the regularization parameter for the ALS regularization (size of the entries of W and Z)
        @param num_iter : the number of iterations of the algorithm
        
        @return predic_end._3.values : the prediction values for all the data within the test set
        
    """
    # 1. Format the data into RDD for Spark
    pDF = sql_sc.createDataFrame(train)
    spDF = pDF.rdd
    
    # 2. Train the model
    model = ALS.train(spDF, rank, numIterations,lambda_)
    
    # 3. Perform the prediction on the test data
    pDF_test = sql_sc.createDataFrame(test[['User', 'Movie']])
    spDF_test = pDF_test.rdd
    testdata = spDF_test.map(lambda p: (p[0], p[1]))
    predictions = model.predictAll(testdata).map(lambda r: (r[0], r[1], r[2]))
    
    # 4. Format the predictions before returning them. The predictions must sorted because Spark shuffled them during the predictions
    predictions_df = predictions.toDF()
    predictions_pd = predictions_df.toPandas()
    predictions_pd.sort_values(['_1','_2'],ascending=[1,1],inplace=True)
    predic_end = predictions_pd.reset_index(drop=True)
    
    return predic_end._3.values

def ALSBias_pywFM(train, test, num_iter=100, std_init = 0.43, rank = 7, r0_reg = 0.5, r1_reg = 15, r2_reg = 25):
    """
        Runs the ALS algorithm with the user bias included for num_iter iterations.
        N.B. The parameters passed by default are the best ones we found.
        
        @param train : the DataFrame containing all our training data.
        @param test : the DataFrame containing all our testing data.
        @param num_iter : the number of iterations of the algorithm
        @param std_init : the standard deviation for the initialisation of W and Z
        @param rank : the number of columns of W and Z
        @param r0_reg : the regularization parameter for the global bias term w0
        @param r1_reg : the regularization parameter of the user/item bias term w
        @param r2_reg : the regularization parameter for the ALS regularization (size of the entries of W and Z)
        @return np.array(pred) : the prediction values for all the data within the test set
        
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
    """
        Runs the ALS algorithm with MCMC for num_iter iterations.
        N.B. The parameters passed by default are the best ones we found.

        @param train : the DataFrame containing all our training data.
        @param test : the DataFrame containing all our testing data.
        @param num_iter : the number of iterations of the algorithm
        @param std_init : the standard deviation for the initialisation of W and Z

        @return np.array(pred) : the prediction values for all the data within the test set
        
    """
    
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

