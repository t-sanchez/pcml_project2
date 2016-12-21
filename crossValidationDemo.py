from cv import *
from helpers import *
import scipy.sparse as sp
import pandas as pd

# Performs the cross validation for all the three algorithms requiring parameter tuning, ie ALS unbiased, ALS biased and MCMC.
# THIS SCRIPT TAKES A VERY LONG TIME TO RUN PROPERLY, AS EVERY ALGORITHM IS CROSS VALIDATED.

# 1. Loading the Data
ratings = load_data('data_train.csv')

# 2. Formatting the Data
split = sp.find(ratings)
df = pd.DataFrame({'Movie':split[0], 'User': split[1], 'Prediction':split[2]})
df = df[['Movie', 'User', 'Prediction']]
      
# 3. Cross-Validating and printing the optimal parameters for each algorithm.
# N.B. The values on which the cross validation is done are set in the Demo functions.
print("Cross validation ALS biased")
best_rmse, best_std, best_rank, best_r0, best_r1, best_r2 = cross_validationALSBias_demo(df)
print("Best std : {std}, best rank : {r}, best r0 : {r0}, best r1 : {r1}, best r2 : {r2}"
      .format(std = best_std, r = best_rank, r0 = best_r0, r1 = best_r1, r2 = best_r2))
      
print("Cross validation MCMC")
best_rmse, best_std = cross_validationALS_demo(df)
print("Best std : {std}".format(std = best_std))
