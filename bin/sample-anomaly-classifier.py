#omgamganapathayenamaha
import os
import sys
SPARK_PYTHON = os.environ.get("SPARK_HOME") + "/python"
sys.path.insert(0, SPARK_PYTHON)
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pysparkling import H2OContext
import h2o

sys.path.insert(0, r'C:\JUNIPER\JNPR_X1_Carbon\omgamganapathayenamaha\Sastry\AT_Technical_Assessment_Task\FraudTransactionDetector')
from fraudtransactiondetector.utils import get_cols_with_missing_values, fill_missing_vals_in_integer_feature, fill_missing_vals_in_categorical_feature

h2o.init()
del os.environ["PYSPARK_SUBMIT_ARGS"]

# Creating the Spark Session
spark = SparkSession.builder.appName('Fraud-Transaction-Classifier').master('local[*]').getOrCreate()

# H2O Context is needed for converting Pyspark DataFrames to H2O Frames
hc = H2OContext.getOrCreate(spark)

def get_pyspark_dataframe(filepath):
    print('Reading the file {} ...'.format(filepath))
    # Reading the Data Source and creating the PySpark Data Frame
    pandas_df = pd.read_excel(filepath, engine='odf')

    # Converting Nan to None before converting the Pandas Data Frame to Pyspark Data Frame
    pandas_df = pandas_df.where(pd.notnull(pandas_df), None)

    from pyspark.sql.types import IntegerType, FloatType, StringType, StructType, StructField
    schema = StructType(fields=[
        StructField('consumer_id', StringType(), True),
        StructField('gender', StringType(), True),
        StructField('has_gender', IntegerType(), True),
        StructField('has_first_name', IntegerType(), True),
        StructField('has_last_name', IntegerType(), True),
        StructField('has_email', IntegerType(), True),
        StructField('has_dob', IntegerType(), True),
        StructField('customer_age', FloatType(), True),
        StructField('account_age', IntegerType(), True),
        StructField('account_last_updated', IntegerType(), True),
        StructField('account_status', IntegerType(), True),
        StructField('app_downloads', IntegerType(), True),
        StructField('unique_offer_clicked', IntegerType(), True),
        StructField('total_offer_clicks', IntegerType(), True),
        StructField('unique_offer_rides', IntegerType(), True),
        StructField('total_offer_rides', IntegerType(), True),
        StructField('avg_claims', FloatType(), True),
        StructField('min_claims', IntegerType(), True),
        StructField('max_claims', IntegerType(), True),
        StructField('total_offers_claimed', IntegerType(), True)
    ])

    df = spark.createDataFrame(pandas_df, schema=schema)
    return df

# Get Pyspark Dataframe
df = get_pyspark_dataframe(r'../data/Technical test sample data.ods')
print('Pyspark Data Frame Size is : {} rows, {} columns'.format(df.count(), len(df.columns)))

# Finding Columns with NA Values
#summary = get_cols_with_missing_values(df)
#for key in summary.keys():
#    print('{} : {} NA_Values : {} % NA_Values'.format(key, summary[key]['missing'], summary[key]['percent_missing']))

# Handling Missing Values in the Columns one by one
# Handling 'customer_age' column
col = 'customer_age'
considered_cols = df.columns
considered_cols.remove(col)
considered_cols.remove('consumer_id')
considered_cols.remove('gender')
exempt_cols = ['consumer_id', 'gender']
df = fill_missing_vals_in_integer_feature(df, col, exempt_cols, considered_cols)
print('Pyspark Data Frame Size after handling NA in customer-age col is : {} rows, {} columns'.format(df.count(), len(df.columns)))

# Handling 'gender' column
col = 'gender'
considered_cols = ['has_gender','has_first_name', 'has_last_name', 'has_email', 'has_dob', 'account_age', 'account_last_updated', 'account_status', 'app_downloads', 'unique_offer_clicked', 'total_offer_clicks', 'unique_offer_rides', 'total_offer_rides', 'avg_claims', 'min_claims', 'max_claims', 'total_offers_claimed', 'customer_age']
exempt_cols = ['consumer_id']
df = fill_missing_vals_in_categorical_feature(df, col, exempt_cols, considered_cols)
print('Pyspark Data Frame Size after handling NA in gender col is : {} rows, {} columns'.format(df.count(), len(df.columns)))


