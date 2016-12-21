import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt
from helpers import *
from cost import *
from predictionAlgorithms import *

trainDF, testDF = realDataSets() # Kaggle datasets (with no predictions in valuesTest)

# Training of each of the algorithms separately. The default parameters for each algorithm are the optimal ones, computed using cross validation, that is why we do not have any hardcoded parameter in the present script.

print("User mean")
predictionsUserMean = baseline_user_mean(trainDF, testDF)
print("Item mean")
predictionsItemMean = baseline_item_mean(trainDF, testDF)
print("ALS Spark")
predictionsALS = ALSPyspark(trainDF, testDF)
print("ALS with bias pywFM")
predictionsALSBias = ALSBias_pywFM(trainDF, testDF)
print("MCMC")
predictionsMCMC = MCMC_pywFM(trainDF, testDF)

# Blending of the algorithms
wUserMean = 0.0222222222222
wItemMean = 0.0
wALS = 0.157894736842
wALSBias = 0.315789473684
wMCMC = 0.504093567251

predictions = (wUserMean*predictionsUserMean + wItemMean*predictionsItemMean + wALS*predictionsALS +
               wALSBias * predictionsALSBias + wMCMC * predictionsMCMC) / (wUserMean + wItemMean + wALS + wALSBias + wMCMC)

# Submission
make_submission(predictions)