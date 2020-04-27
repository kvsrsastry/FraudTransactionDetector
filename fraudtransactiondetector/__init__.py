#omgamganapathayenamaha
from pyspark.ml.clustering import KMeans
import h2o
import pandas as pd
import sys
import os

# This fixes Java gateway process exception
# This env is set by pysparkling
if "PYSPARK_SUBMIT_ARGS" in os.environ:
    del os.environ["PYSPARK_SUBMIT_ARGS"]
SPARK_PYTHON = os.environ.get("SPARK_HOME") + "/python"
sys.path.insert(0, SPARK_PYTHON)
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('Fraud-Transaction-Classifier').getOrCreate()

class FraudTransactionClassifier():
    '''
    '''
    def __init__(self, num_clusters=4, quantile=0.99):
        '''
        '''
        self.num_clusters = num_clusters
        self.quantile = quantile
        self.km = KMeans(k=num_clusters, initMode='k-means||', initSteps=10, maxIter=300)

    def fit(self, df):
        '''
        '''
        # Storing the Data Frame for future use if any ...
        self.df = df

        # Applying K-Means Clustering and Segmenting the data into Clusters
        print('Applying Scalable Clustering method and Segmenting the data into {} Clusters'.format(self.num_clusters))
        self.km = self.km.fit(df)
        newdf = self.km.transform(df)
        newdf = newdf.drop('features')
        newdf = newdf.withColumnRenamed('prediction', 'seg')
        self.kmeans_clustered_df = newdf
        print('Below is the Segmentation Summary: ')
        newdf.groupBy(newdf.seg).agg({'seg':'count'}).show()

        #return newdf

        # Applyig H2o Isolation Forest Outlier Detection on each identified cluster
        print('Applyig Outlier Detection on each of the {} Clusers'.format(self.num_clusters))
        anamoly_df = None
        considered_cols = newdf.columns
        considered_cols.remove('seg')
        for col,dtype in newdf.dtypes:
            if dtype == 'string' or dtype == 'vector':
                considered_cols.remove(col)

        for i in range(self.num_clusters):
            ad = h2o.estimators.H2OIsolationForestEstimator(ntrees=100, seed=12345)
            tmp_spark_df = newdf.filter(newdf['seg'] == i)
            tmp_pandas_df = tmp_spark_df.toPandas()
            #tmp_df = hc.asH2OFrame(tmp_spark_df)
            tmp_df = h2o.H2OFrame(tmp_pandas_df)
            ad.train(x=considered_cols, training_frame=tmp_df)
            predictions = ad.predict(tmp_df)
            quantile_frame = predictions.quantile([self.quantile])
            threshold = quantile_frame[0, "predictQuantiles"]
            liers = predictions["predict"] > threshold
            tmp_pandas_df['anomaly'] = h2o.as_list(liers)['predict'].values
            print('Anamolies identified in Cluster Number {} : {} : {}'.format(i, tmp_pandas_df['seg'].count(), tmp_pandas_df['anomaly'].count()))
            if anamoly_df is None:
                anamoly_df = tmp_pandas_df.copy()
            else:
                anamoly_df = pd.concat((anamoly_df, tmp_pandas_df.copy()), axis=0)


        newdf = spark.createDataFrame(anamoly_df)
        self.num_outliers = newdf.filter(newdf.anomaly == 1).count()
        self.num_inliers = newdf.filter(~(newdf.anomaly == 1)).count()
        print('Number of Outliers : {}'.format(self.num_outliers))
        print('Number of Genuine Samples : {}'.format(self.num_inliers))
        self.outliered_df = newdf

        return newdf

