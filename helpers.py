# -*- coding: utf-8 -*-
"""some functions for help."""

from itertools import groupby

import numpy as np
import scipy.sparse as sp
import pandas as pd
from scipy.sparse import csr_matrix


def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()    
    
def load_data(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)


def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""
    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    #print("number of items: {}, number of users: {}".format(max_row, max_col))

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings


def group_by(data, index):
    """group list of list by a specific index."""
    sorted_data = sorted(data, key=lambda x: x[index])
    groupby_data = groupby(sorted_data, lambda x: x[index])
    return groupby_data


def build_index_groups(train):
    """build groups for nnz rows and cols."""
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices


def calculate_mse(real_label, prediction):
    """
    Computes MSE.
    @param real_label : the target labels we want to predict (typically from a split of training set or multiple-fold cross validation)
    @param prediction : the prediction of the algorithm on those labels.
    
    """
    t = real_label - prediction
    return 1.0 * t.dot(t.T) / len(t)

def make_submission(predictions):
    """
    Exports the submission to the csv file.
    @param predictions : the vector of predictions for the "sampleSubmission.csv" file. 
            N.B. It needs to be correctly reordered in order for the prediction fo actually work.
    """
    sampleSubmission = pd.read_csv('sampleSubmission.csv')
    sampleSubmission['Prediction'] = predictions
    sampleSubmission.to_csv('submission.csv', index = False)
    
def realDataSets():
    """
    Loads the training and testing data set, formats it, and returns two formatted dataframes
    @return train_df : returns the formatted DataFrame taken from "data_train.csv"
    @return test_df : returns the formatted DataFrame taken from "sampleSubmission.csv" (the ratings are not correct here,
    as these are the ones we need to predict).
    """
    
    #Loading the training Dataset
    path_dataset = "data_train.csv"
    train = load_data(path_dataset)
    values_train = sp.find(train)
    trainDF = pd.DataFrame(values_train[1], columns = ['User'])
    trainDF['Movie'] = values_train[0]
    trainDF['Prediction'] = values_train[2]
    
    #Loading the submission Dataset
    path_dataset = "sampleSubmission.csv"
    test = load_data(path_dataset)
    values_test = sp.find(test)
    testDF = pd.DataFrame(values_test[1], columns = ['User'])
    testDF['Movie'] = values_test[0]
    
    return trainDF, testDF


def df_to_sparse(df):
    """
        Rewrites our matrix of user movie association in the following format, starting from a 2 column csv file with :
        1st column : user id and movie id mixed, 2nd column : rating. The output matrix will take the form 
        
         Users  |     Movies    
        A  B  C | TI  NH  SW  ST
        [1, 0, 0,  1,  0,  0,  0],
        [1, 0, 0,  0,  1,  0,  0],
        [1, 0, 0,  0,  0,  1,  0],
        [0, 1, 0,  0,  0,  1,  0],
        [0, 1, 0,  0,  0,  0,  1],
        [0, 0, 1,  1,  0,  0,  0],
        [0, 0, 1,  0,  0,  1,  0] 
        ])
        
        target = [5, 3, 1, 4, 5, 1, 5]
        
        @param path : The path of the training/testing data
        @return features, target : the corresponding matrix and target values
    """
    
    #1. Extracting the info from the input DF
    user_index = np.squeeze(np.array(df['User']))
    movie_index = np.squeeze(np.array(df['Movie'] + max(user_index)))
    if 'Prediction' in df.columns:
        ratings = np.squeeze(np.array(df['Prediction']))
    else:
        ratings = np.zeros(df['Movie'].shape)
    #2.Formatting now the way we need to use libFM

    col_entries = np.r_[user_index,movie_index]
    indices = np.arange(0,len(user_index))
    row_entries = np.r_[indices,indices]
    entries = np.ones(len(row_entries))
    
    features = csr_matrix((entries,(row_entries, col_entries)),shape = (len(indices),len(col_entries)))
    
    return features, ratings

def write_submission(submission_data_path, prediction, out_path):
    """
        Given a submission_data_path (contains the data we loaded on the testing set), the corresponding predictions 
        from the algorithm and the out_path, formats and saves the result to the out_path.
        @param submission_data_path: the name of the data file which contains the data we want to do prediction on.
        @param prediction : the prediction for those data.
        @param out_path : the path to which we export the results.
    """
    df = df_load(submission_data_path)
    df['Prediction'] = prediction.astype(int)
    df[['Id','Prediction']].to_csv(out_path, index = False)

    
def cross_vad(y,ratio) :
    """
    Given a DataFrame y and a ratio ratio, returns the indices of the splitting ratio%-train,(1-ratio)%-test.
    @param y: a DataFrame containing the training data.
    @param ratio : the ratio of training elements we want after the split.
    @return k_train : the shuffled indices for our training set.
    @return k_test : the suffled indices for our tetsting set.
    """
    n = y.shape[0]
    interval = int(n *ratio)
    seed = 1
    np.random.seed(seed)
    index = np.random.permutation(n)
    k_train = index[0:interval]
    k_test  = index[interval:n]
    return k_train , k_test

def crossValidationSets(p):
    """
        Loads the training set and splits it into two train and test sets, given a ratio p. 
        It uses cross_vad to find the indices we need to get a partition p% of train, (1-p)% of test
        @param p : the percentage of elements that we want in the training split (has to be between 0 and 1)
        @return the train split and the test split.
    """
    path_dataset = "data_train.csv"
    train = load_data(path_dataset)
    values_train = sp.find(train)
    trainDF = pd.DataFrame(values_train[1], columns = ['User'])
    trainDF['Movie'] = values_train[0]
    trainDF['Prediction'] = values_train[2]
    
    index_train, index_test = cross_vad(trainDF, p)
    return trainDF.loc[index_train], trainDF.loc[index_test]
