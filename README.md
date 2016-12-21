# PCML - Project 2 : Recommender System

Here are our implementations of the Recommender Systems project, described more extensively on [kaggle](https://inclass.kaggle.com/c/epfml-rec-sys). 
## Files

### 1. Python scripts
- **run.py :** loads the training and testing datasets, trains all the 5 algorithms separately, and performs the predictions on the testing set. Then the blending is applied to the predictions, and the final prediction is saved in the file *submission.csv*. Note that `pySpark` produces a lot of warnings during a run bcause it is intended to be run on multiple cores which is not the case here, but still runs correctly here, so you do not need to worry about it.
- **crossValidationDemo.py :** loads the training datasets, runs the three cross validation demo functions that are in `cv.py` and prints the optimal parameters (i.e. the ones we want to use).
- **weightsOptimization.py :** loads the training dataset, trains the 5 algorithms on 70% of the daa, and optimizes the weights for each algorithm in order to minimize the error on the remaining part of the training set. The optimisation is performed using a grid search on possible values of the weights.

### 2. Notebook (.ipynb)
- **DataExploration.ipynb :** performs a simple first data exploration, in order to visualise how the rates are distributed per user, as well as per movie.

### 3. Python files (.py)
- **cost.py :** Standard RMSE cost function 
- **cv.py :** For each of the method, contains a *crossvalidation* function, which returns only the loss for a given input of parameters, as well as a *demo* function, which given data, return the best parameters chosen from the ones set in the function itself. There is another function, namely *optimize_weights*, which, given the prediction from the 5 methods as well as the reference labels, that returns the best combinaion of weights.
- **helpers.py :** helper functions, mostly for writing/reading from files, or formatting data the way we need.
- **plots.py :** various plots to visualise our data
- **predictionAlgorithms.py :** very important function, the one that actually does prediction. Given a set of parameters for a method as well as a training and a testing set, returns the vector of predictions for the testing set.

### 4. Files (.csv)
- **data_train.csv :** the training data, that we got from *Kaggle*.
- **sampleSubmission.csv :** the sample submission, containing the user-movie pairs for which we need a prediction.
- **submission.csv :** The best submission, actual file that we submitted on kaggle.

## Libraries 
**NOTE :** The installation is meant for a Linux operating system, and has been tested on another computer with Linux, in order to make sure that our code can be successfully run.


### 1. PySpark
We used the [PySpark](http://spark.apache.org/docs/0.9.0/python-programming-guide.html) library, which is a python binding programme which allows us to use of Spark written in Scala. Note that this software is under a license, that is explained [here](http://www.apache.org/licenses/). The installation is not trivial and done as follows. 
#### Prerequisites
We follow this [stackoverflow thread](http://askubuntu.com/questions/635265/how-do-i-get-pyspark-on-ubuntu) and summarise (or rewritten more simply) it here. Before starting the installation, make sure that you have **Java 7+** in addition to (obviously) **Python** installed. Then, you can download spark from the [download page](https://spark.apache.org/downloads.html) (Note that there might be some complications if you have scala installed and depending on its version). You have to be sure that both *java* and *python* programs are on your PATH or that the JAVA_HOME environment variable is set.

#### Installing pyspark
1. Go into the folder where you downloaded the archive of the pre-built binary distribution for spark.
2. Unzip and move the unzipped folder to a working directory (use *sudo*).

    ``` 
    tar -xzf spark-2.0.2-bin-hadoop2.7.tgz
    mv spark-2.0.2-bin-hadoop2.7 /srv/spark-2.0.2
    ```

3. Symlink the version of Spark to a spark directory 

    ``` ln -s /srv/spark-2.0.2 /srv/spark```
    

4. Edit your ~/.bashrc (or .bash_profile) and add Spark to your PATH, along with setting the SPARK_HOME environment vairable :
    
    ```
    export SPARK_HOME=/srv/spark
    export PATH=$SPARK_HOME/bin:$PATH
    ```
    
#### Importing pyspark in a python script
Now in order to import pyspark into a python script, you need to follow these last steps.

1. Install FindSpark, because PySpark isn't on sys.path by default.

  `pip install findspark`
  
2. Write and run a python script with the following lines to add PySpark to your sys.path

  ```
    import findspark
    findspark.init('/srv/spark')
  ```

You should now be able to do `import pyspark` :)

### 2. pywFM 

We used [pywFM](https://github.com/jfloff/pywFM), a python wrapper for the [libFM](http://libfm.org/) library, which was originally written in C++. This is a **Factorization Machine** library, which implements :
- Stochastic Gradient Descent (SGD),
- Adaptive Stochastic Gradient Descent (SGDA),
- Alternating Least Squares (ALS) with user and item bias,
- Markov Chain Monte Carlo (MCMC).

#### Installing libFM and pywFM

1. First you have to clone and compile `libFM` repository, and then set an environment variable to the `libFM` bin folder

    ```
    git clone https://github.com/srendle/libfm /home/libfm
    cd /home/libfm/ && make all
    ```

2. Edit your ~/.bashrc (or .bash_profile) and add libFM to your path
    
    ```
    export LIBFM_PATH=/home/libfm/bin/
    ```

3. Then, you simply install `pywFM` with `pip`
   
    ```
    pip install pywFM
    ```
    
And then `pywFM` should run from your computer !

### 3. pandas
We used this quite standard library, but mention it for completeness. This comes by default while installing Anaconda.


### 4. (Optional) 
After having installed everything, reload your ~/.bashrc to make sure that the changes are taken into account
    
    ```
    source ~/.bashrc
    ```
