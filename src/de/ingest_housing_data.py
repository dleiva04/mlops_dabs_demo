# Databricks notebook source
import pandas as pd
import pyspark.sql.functions as F
import sklearn.datasets as sds
from pyspark.sql.types import *

# COMMAND ----------
catalog = dbutils.widgets.get("catalog")
schema = "de"

# COMMAND ----------

# Load the housing datasets
housing = sds.fetch_california_housing()

# Access the features and targets of the dataset
X = housing.data  # Features
y = housing.target  # Targets

# Access the feature names and target names of the dataset
feature_names = housing.feature_names
target_names = housing.target_names

# COMMAND ----------

df = pd.DataFrame(X, columns=feature_names)
df["MedHouseVal"] = y

# COMMAND ----------

df

# COMMAND ----------

sdf = spark.createDataFrame(df)
sdf = sdf.withColumn("HouseAge", F.col("HouseAge").cast("int"))
sdf = sdf.withColumn("Population", F.col("Population").cast("int"))
sdf.show()

# COMMAND ----------

sdf.write.saveAsTable(f"{catalog}.{schema}.california_housing", mode="overwrite")

# COMMAND ----------

housing_table = spark.table(f"{catalog}.{schema}.california_housing")
display(housing_table)
