#omgamganapathayenamaha
from pyspark.ml.clustering import KMeans
import h2o
import pandas as pd
import sys
import os
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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
    def __init__(self, numClusters=4, quantile=0.99):
        '''
        '''
        self.num_clusters = numClusters
        self.quantile = quantile
        self.km = KMeans(k=self.num_clusters, initMode='k-means||', initSteps=10, maxIter=300)

    def evaluate(self, labelCol, testResults):
        '''
        '''
        beval = BinaryClassificationEvaluator(labelCol=labelCol, rawPredictionCol='prediction')
        auroc = beval.evaluate(testResults)

        meval = MulticlassClassificationEvaluator(labelCol=labelCol,metricName='f1')
        f1 = meval.evaluate(testResults)

        meval = MulticlassClassificationEvaluator(labelCol=labelCol,metricName='accuracy')
        acc = meval.evaluate(testResults)

        return {'auroc': auroc, 'f1': f1, 'acc': acc}

    def modelValidationMetrics(self):
        '''
        '''
        return self.evaluation_results

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
            #print('Anamolies identified in Cluster Number {} : {} : {}'.format(i, tmp_pandas_df['seg'].count(), tmp_pandas_df['anomaly'].count()))
            print('Anamolies identified in Cluster Number {}'.format(i))
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

        # Applying Gradient Boosted Classifier and creating a Classification Model 
        # using the Anamoly Data we have generated
        col = 'anomaly'
        considered_cols = newdf.columns
        considered_cols.remove(col)
        considered_cols.remove('seg')
        for col,dtype in newdf.dtypes:
            if dtype == 'string' or dtype == 'vector':
                considered_cols.remove(col)

        vec = VectorAssembler(inputCols=considered_cols, outputCol='features')
        newdf = vec.transform(newdf)

        # Splitting the data for train and test
        train, test = newdf.randomSplit([0.8, 0.2], seed=12345)

        # Calling Gradient Boosted Classifier Constructor
        cla_model = GBTClassifier(labelCol=col)
        cla_model = cla_model.fit(train)

        # Testing the Model
        test_results = cla_model.transform(test)
        self.evaluation_results = self.evaluate(labelCol=col, testResults=test_results)

        # Save the Classification Model
        self.cla_model = cla_model

        return self

    def selectOptimalClusters(self, df):
        '''
        '''
        considered_cols = df.columns
        for col,dtype in newdf.dtypes:
            if dtype == 'string' or dtype == 'vector':
                considered_cols.remove(col)

        if 'features' in df.columns:
            df = df.drop('features')

        if 'feats' in df.columns:
            df = df.drop('feats')

        vec = VectorAssembler(inputCols=considered_cols, outputCol='features')
        df = vec.transform(df)

        wcss = []
        for i in range(2,11):
            km = KMeans(k=i, initMode='k-means||', initSteps=10, maxIter=300)
            km = km.fit(df)
            wcss.append(km.computeCost(df))

        plt.plot(list(range(2,11)), wcss[:9], color='blue')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.title('Elbow method')
        plt.show()

    def visualizeByApplyingPCA(self):
        vis_df = self.outliered_df
        col = 'anomaly'
        considered_cols = vis_df.columns
        considered_cols.remove(col)
        considered_cols.remove('seg')
        for col,dtype in newdf.dtypes:
            if dtype == 'string' or dtype == 'vector':
                considered_cols.remove(col)

        if 'features' in vis_df.columns:
            vis_df = vis_df.drop('features')

        if 'feats' in vis_df.columns:
            vis_df = vis_df.drop('feats')

        vec = VectorAssembler(inputCols=considered_cols, outputCol='feats')
        vis_df = vec.transform(vis_df)

        # Applying Standardization on the Data
        sc = StandardScaler(inputCol='feats', outputCol='features')
        vis_df = sc.fit(vis_df).transform(vis_df)

        # Reducing to 3 Dimensions so that we can visualize
        pca = PCA(k=3, inputCol='features', outputCol='pca_features')
        pca_model = pca.fit(vis_df)
        vis_df_transformed = pca_model.transform(vis_df)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel("PC3")
        inliers = vis_df_transformed.filter(~(vis_df_transformed.anomaly == 1)).select(vis_df_transformed.pca_features)
        inl = dict()
        for i in range(3):
            inl[i] = list()
        for row in inliers.collect():
            l = row.pca_features
            for i in range(3):
                inl[i].append(l[i])

        outliers = vis_df_transformed.filter(vis_df_transformed.anomaly == 1).select(vis_df_transformed.pca_features)
        outl = dict()
        for i in range(3):
            outl[i] = list()
        for row in outliers.collect():
            l = row.pca_features
            for i in range(3):
                outl[i].append(l[i])

        ax.scatter(inl[0], inl[1], zs=inl[2], s=4, lw=1, label="inliers",c="blue")
        ax.scatter(outl[0], outl[1], outl[2], lw=2, s=4, c="red", label="outliers")
        ax.legend()
        plt.show()

    def transform(self, testDf):
        '''
        '''
        if 'features' in testDf.columns:
            testDf = testDf.drop('features')

        col = 'anomaly'
        considered_cols = testDf.columns
        if considered_cols.count(col) > 0:
            considered_cols.remove(col)
        for col,dtype in testDf.dtypes:
            if dtype == 'string' or dtype == 'vector':
                considered_cols.remove(col)

        vec = VectorAssembler(inputCols=considered_cols, outputCol='features')
        testDf = vec.transform(testDf)

        testDfResults = self.cla_model.transform(testDf)
        return testDfResults

