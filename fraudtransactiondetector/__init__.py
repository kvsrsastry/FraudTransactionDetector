#omgamganapathayenamaha
from pyspark.ml.clustering import KMeans


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

        return newdf

        # Applyig H2o Isolation Forest Outlier Detection on each identified cluster
        print('Applyig Outlier Detection on each of the {} Clusers'.format(self.num_clusters))
        anamoly_df = None

