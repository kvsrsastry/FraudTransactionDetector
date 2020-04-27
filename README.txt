==========================
Fraud Transaction Detector
==========================

The objective of this project is to come up with a classfication machine 
learning model which identifies anomaly data/records from genuine data/records 
given unclassified/unlabeled data as input. This generic objective has 
application in lot of domains like Healthcare, Stocks Trading, Banking, 
System Security etc. and few of the use cases are as below:

* Fradulent Medical Claim detection
* Fradulent Credit Card Transactions
* Early detection of insider trading
* Intrusion detection

Technologies used
=================

As the module needs to be scalable and handle Big Data involving 
Hundreds of Millions of records, I have chosen to use 

* Apache Spark
* H2o

My Approach
===========
Below is the approach taken and algorithms used to solve the problem 
at hand:

1. K-Means Clustering from Apache Spark MLlib
    * To identify clusters in the given unlabeled data
    * Handles Big Data and scales on a cluster of machines

2. Isolation Forest from H2o
    * To detect the Anamolies in each cluster identified in #1
    * Handles Big Data and works seamlessly with Apache Spark

3. Gradient Boosted Classification Trees from Spark MLlib 
    * To create Ensemble classification model
    * Handles Big Data and scales on a cluster of machines

4. Model optimization using Apache Spark MLlib CrossValidator

5. PCA
    * Dimensionality Reduction to visualize the data in 3D

How to import and use the package?
==================================
Below is the sample usage::

	from fraudtransactiondetector import FraudTransactionClassifier
	classifier = FraudTransactionClassifier(numClusters=num_clusters,
        	                                quantile=0.99)

	classifier.fit(df)
	print(classifier.modelValidationMetrics())

	# Apply it on entire Training data just to check
	results = classifier.transform(df)

	# Apply PCA and Visualize
	classifier.visualizeByApplyingPCA()

	# Select optimal number of clusters using Elbow Method
	classifier.selectOptimalClusters(df)

