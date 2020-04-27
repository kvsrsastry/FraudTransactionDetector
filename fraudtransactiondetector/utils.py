#from xgboost import XGBRegressor
#from sklearn.model_selection import GridSearchCV
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

import numpy as np
import pandas as pd

# Finding Columns with NA Values
def get_cols_with_missing_values(df):
    '''
    '''
    cols = df.columns
    summ = dict()
    total = df.count()
    for col in cols:
        number_of_na = df.filter(df[col].isNull()).count()
        if number_of_na > 0:
            summ[col] = dict()
            summ[col]['missing'] = number_of_na
            summ[col]['percent_missing'] = number_of_na * 100 / total
    return summ

def _get_xgboost_regressor_model(col, train):
    '''
    '''
    print('Using Gradient Boosted Regressor Module to predict Missing Values ...')
    reg_model = GBTRegressor(labelCol=col)
    #params = ParamGridBuilder().addGrid(reg_model.maxDepth, [5, 10, 20]).\
    #                            addGrid(reg_model.minInfoGain, [0.0, 0.01, 1.0]).\
    #                            addGrid(reg_model.maxBins, [32, 20, 50, 100, 300]).build()
    #cv = CrossValidator(estimator=reg_model,
    #                   estimatorParamMaps=params,
    #                   evaluator=RegressionEvaluator(labelCol=col),
    #                   numFolds=10)
    reg_model = reg_model.fit(train)
    return reg_model

def fill_missing_vals_in_integer_feature(df, col, exempt_cols, considered_cols):
    '''
    '''
    vec = VectorAssembler(inputCols=considered_cols, outputCol='features')
    df = vec.transform(df)

    # Dividing the data into Train and Test data
    train = df.filter(~df[col].isNull())
    test = df.filter(df[col].isNull())

    # Getting the gradient boosted model
    reg_model = _get_xgboost_regressor_model(col, train)

    # Predicting the Missing Values and getting back the Data Frame into original shape
    test_results = reg_model.transform(test)
    test_results = test_results.drop(col)
    test_results = test_results.withColumnRenamed('prediction', col)
    test_results = test_results.drop('features')
    train = train.drop('features')
    df = train.union(test_results)
    return df

def _get_xgboost_classifier_model(col, train):
    '''
    '''
    print('Using Gradient Boosted Regressor Module to predict Missing Values ...')
    cla_model = GBTClassifier(labelCol=col)
    #params = ParamGridBuilder().addGrid(cla_model.maxDepth, [5, 10, 20]).\
    #                            addGrid(cla_model.minInfoGain, [0.0, 0.01, 1.0]).\
    #                            addGrid(cla_model.maxBins, [32, 20, 50, 100, 300]).build()
    #cv = CrossValidator(estimator=cla_model,
    #                   estimatorParamMaps=params,
    #                   evaluator=BinaryClassificationEvaluator(labelCol=col),
    #                   numFolds=10)
    cla_model = cla_model.fit(train)
    return cla_model

def fill_missing_vals_in_categorical_feature(df, col, exempt_cols, considered_cols):
    '''
    '''
    vec = VectorAssembler(inputCols=considered_cols, outputCol='features')
    df = vec.transform(df)

    # Dividing the data into Train and Test data
    train = df.filter(~df[col].isNull())
    test = df.filter(df[col].isNull())

    # Label Encoding
    col_indexed = col + '_indexed'
    si = StringIndexer(inputCol=col,outputCol=col_indexed)
    train = si.fit(train).transform(train)

    # Getting the gradient boosted model
    cla_model = _get_xgboost_classifier_model(col_indexed, train)

    # Predicting the Missing Values and getting back the Data Frame into original shape
    test_results = cla_model.transform(test)
    test_results = test_results.drop('features')
    train = train.drop('features')
    test_results = test_results.drop('gender')
    train = train.drop('gender')
    test_results = test_results.withColumnRenamed('prediction', 'gender')
    train = train.withColumnRenamed('gender_indexed', 'gender')
    test_results = test_results.drop('rawPrediction','probability')
    df = train.union(test_results)
    return df

