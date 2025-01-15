# Databricks notebook source
# COMMAND ----------
%pip install databricks-feature-engineering

# COMMAND ----------
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.ml.feature import StandardScaler
from pyspark.sql.functions import pandas_udf

# COMMAND ----------
catalog = dbutils.widgets.get("catalog")
schema = "ml"


# COMMAND ----------

dataset = spark.table(f"{catalog}.de.california_housing")

# COMMAND ----------

dataset = dataset.withColumn("ID", F.monotonically_increasing_id())
dataset.show()


# COMMAND ----------

columns = [
    "ID",
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
    "MedHouseVal",
]

# COMMAND ----------


dataset.select(columns).write.saveAsTable(
    f"{catalog}.{schema}.california_housing", mode="overwrite"
)


# COMMAND ----------

housing_set = spark.table(f"{catalog}.{schema}.california_housing").select(columns)

# COMMAND ----------

housing_set.show()

# COMMAND ----------

feature_columns = [
    "ID",
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
features = housing_set.select(feature_columns)

# COMMAND ----------


# DBTITLE 1,Create Non-linear feature
@pandas_udf("float")
def nl_sigmoid(feat_a: pd.Series, feat_b: pd.Series) -> pd.Series:
    """
    Let's assume that feat_1 and feat_2 are scaled.
    """
    import numpy as np

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    result = (feat_a + feat_b) / (sigmoid(feat_a) + sigmoid(feat_b)) * 100
    return result


def _scale(column_name, pdf):
    max_val = pdf.select(F.max(column_name)).collect()[0][0]
    min_val = pdf.select(F.min(column_name)).collect()[0][0]
    result = pdf.withColumn(
        "scaled_" + column_name,
        (F.col(column_name) - min_val) / (max_val - min_val) - 0.5,
    )
    return result


def non_lin_sig(column_a, column_b, pdf, new_feature_name):
    scaled_values = _scale(column_a, pdf)
    scaled_values = _scale(column_b, scaled_values)
    column_a = "scaled_" + column_a
    column_b = "scaled_" + column_b

    result = scaled_values.withColumn(new_feature_name, nl_sigmoid(column_a, column_b))
    result = result.drop(column_a)
    result = result.drop(column_b)

    return result


# COMMAND ----------

features = non_lin_sig("Population", "HouseAge", features, "f1")
display(features)

# COMMAND ----------

# DBTITLE 1,Creating Feature Store
feature_table_name = f"{catalog}.ml.california_housing_features"

spark.sql("DROP TABLE IF EXISTS {}".format(feature_table_name))

fe = FeatureEngineeringClient()

housing_feature_table = fe.create_table(
    name=feature_table_name,
    primary_keys="ID",
    df=features,
    schema=features.schema,
    description="California housing features.",
)

# COMMAND ----------
