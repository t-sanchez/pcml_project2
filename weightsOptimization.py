import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt

from helpers import *
from cost import *
from predictionAlgorithms import *
from cv import *

# Optimize the weights for each of the algorithms. Takes about 10 minutes to run in total.


# Split the training set into training and testing set (70% for training, 30% for testing)
trainDF, testDF = crossValidationSets(0.7)
testDF.sort_values(['User', 'Movie'], ascending=[1,1], inplace=True)
trainDF.sort_values(['User', 'Movie'], ascending=[1,1], inplace=True)

# Train all the algorithms and performs prediction on testing set
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

wUser, wItem, wALS, wALSB, wMCMC, loss = optimize_weights(predictionsUserMean, predictionsItemMean, predictionsALS, predictionsALSBias, predictionsMCMC, testDF.Prediction.values)

print("Optimal weight for User Mean algorithm : {w}".format(w=wUser))
print("Optimal weight for Item Mean algorithm : {w}".format(w=wItem))
print("Optimal weight for ALS unbiased algorithm : {w}".format(w=wALS))
print("Optimal weight for ALS biased algorithm : {w}".format(w=wALSB))
print("Optimal weight for MCMC algorithm : {w}".format(w=wMCMC))
print(loss)