# Databricks notebook source
# COMMAND ----------
%pip install databricks-feature-engineering

# COMMAND ----------
import warnings

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelSignature, infer_signature
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking.client import MlflowClient
from mlflow.types.utils import _infer_schema
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

"""
This is a legacy feature store client and it is deprecated. 
from databricks.feature_store import FeatureStoreClient

Use the following API instead:
from databricks.feature_engineering import FeatureEngineeringClient
"""
# COMMAND ----------
catalog = dbutils.widgets.get("catalog")
schema = "ml"

# COMMAND ----------

"""
Setting this Notebook Experiment to register models under UC. 
References: 
https://docs.databricks.com/en/_extras/notebooks/source/mlflow/models-in-uc-example.html
https://docs.databricks.com/en/mlflow/models-in-uc-example.html
"""
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.xgboost.autolog(log_input_examples=True, log_models=False)
# If you are using a custom model, and even a non-ml model, you can still
# log different artificals if necessary using ml flow, you just have to make the call to the logging functions.
seed = 777

"""
https://docs.databricks.com/en/_extras/notebooks/source/machine-learning/feature-store-with-uc-basic-example.html
"""
# Start the Unity Catalog Feature Store client.
fe = FeatureEngineeringClient()

# COMMAND ----------

# DBTITLE 1,Load Data
# California_housing is not a feature table, but contains the ID (primary key) and MedHouseVal (prediction target) of all the data in the feature table
# {catalog}.ml.california_housing_features
full_data = spark.table(f"{catalog}.{schema}.california_housing").select(
    "ID", "MedHouseVal"
)
# We are not going to use all the data in the feature table for training, but a subset. We'll create two sets: a training set and another set for batch scoring (inference).
total_rows = full_data.count()
print("total data: {}".format(total_rows))
training_ids = full_data.sample(withReplacement=False, fraction=0.2, seed=seed)
inference_ids = full_data.subtract(training_ids)

display(training_ids)

# COMMAND ----------

display(inference_ids)

# COMMAND ----------

fe_table_name = f"{catalog}.{schema}.california_housing_features"
feature_names = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "AveOccup",
    "Latitude",
    "Longitude",
    "f1",
]

# In the FeatureLookup, if you do not provide the `feature_names` parameter, all features except primary keys are returned
model_feature_lookups = [
    FeatureLookup(
        table_name=fe_table_name, feature_names=feature_names, lookup_key="ID"
    )
]

# fe.create_training_set looks up features in model_feature_lookups that match the primary key from inference_data_df
training_set = fe.create_training_set(
    df=training_ids,
    feature_lookups=model_feature_lookups,
    label="MedHouseVal",
    exclude_columns="ID",
)
training_pd = training_set.load_df().toPandas()

# Create train and test datasets
X = training_pd.drop("MedHouseVal", axis=1)
y = training_pd["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# COMMAND ----------

display(X_train)

# COMMAND ----------

X_train[0:1]

# COMMAND ----------

with mlflow.start_run() as mlflow_run:
    params = {
        "colsample_bytree": 0.7206688950233426,
        "learning_rate": 0.00540449522464545,
        "max_depth": 12,
        "min_child_weight": 5,
        "n_estimators": 1813,
        "n_jobs": 100,
        "subsample": 0.6673098475456878,
        "verbosity": 0,
        "random_state": 779051502,
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    input_example = X_train[0:1]

    # Log the model using the Feature Engineering API
    fe.log_model(
        model=model,
        artifact_path="house_val_model",
        flavor=mlflow.xgboost,
        training_set=training_set,
        input_example=input_example,
    )

    training_eval_set = X_train.assign(**{"MedHouseVal": y_train})
    training_eval_set["predictions"] = model.predict(X_train)

    # Let's obtain some performance metrics using the training and validation sets.
    training_eval_result = mlflow.evaluate(
        data=training_eval_set,
        targets="MedHouseVal",
        predictions="predictions",
        model_type="regressor",
        evaluator_config={
            "log_model_explainability": False,
            "metric_prefix": "training_",
        },
    )

    validation_eval_set = X_test.assign(**{"MedHouseVal": y_test})
    validation_eval_set["predictions"] = model.predict(X_test)
    # Log metrics for the test set
    test_eval_result = mlflow.evaluate(
        data=validation_eval_set,
        targets="MedHouseVal",
        predictions="predictions",
        model_type="regressor",
        evaluator_config={"log_model_explainability": False, "metric_prefix": "test_"},
    )
    xgb_test_metrics = test_eval_result.metrics
    loss = xgb_test_metrics["test_root_mean_squared_error"]

    # Truncate metric key names so they can be displayed together
    xgb_test_metrics = {k.replace("test_", ""): v for k, v in xgb_test_metrics.items()}

    result = {
        "loss": loss,
        "test_metrics": xgb_test_metrics,
        "model": model,
        "run": mlflow_run,
    }

# COMMAND ----------

print(result)
print()

# COMMAND ----------

# DBTITLE 1,Register Model Under Unity Catalog
# Note that the UC model name follows the pattern
# <catalog_name>.<schema_name>.<model_name>, corresponding to
# the catalog, schema, and registered model name
# in Unity Catalog under which to create the version
# The registered model will be created if it doesn't already exist
autolog_run = mlflow.last_active_run()
model_uri = "runs:/{}/house_val_model".format(autolog_run.info.run_id)
model_name = f"{catalog}.{schema}.cali_housing_value"
mlflow.register_model(model_uri, model_name)

# COMMAND ----------

# Must be run on a cluster running Databricks Runtime for Machine Learning.
# This model was packaged by Feature Store.
# To retrieve features prior to scoring, call FeatureStoreClient.score_batch.
batch_predictions = fe.score_batch(model_uri=model_uri, df=inference_ids)
display(batch_predictions)
