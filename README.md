# PCML - Project 2 : Recommender System

Here are our implementations of the Recommender Systems project, described more extensively on [kaggle](https://inclass.kaggle.com/c/epfml-rec-sys). 

## Libraries 
### 1. PySpark
We used the [PySpark](http://spark.apache.org/docs/0.9.0/python-programming-guide.html) library, which is a python binding programme which allows us to use of Spark written in Scala. Note that this software is under a license, that is explained [here](http://www.apache.org/licenses/). The installation is not trivial and done as follows. 
#### Prerequisites
We follow this [stackoverflow thread](http://askubuntu.com/questions/635265/how-do-i-get-pyspark-on-ubuntu) and summarise (or rewritten more simply) it here. Before starting the installation, make sure that you have **Java 7+** in addition to (obviously) **Python** installed. Then, you can download spark from the (download page)[https://spark.apache.org/downloads.html] (Note that there might be some complications if you have scala installed and depending on its version). You have to be sure that both *java* and *python* programs are on your PATH or that the JAVA_HOME environment variable is set.
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
