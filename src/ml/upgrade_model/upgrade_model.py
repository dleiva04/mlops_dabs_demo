# Databricks notebook source
# COMMAND ----------
source_catalog = dbutils.widgets.get("source_catalog")
source_schema = "ml"
prod_catalog = dbutils.widgets.get("prod_catalog")
prod_schema = dbutils.widgets.get("prod_schema")
# COMMAND ----------
import mlflow

mlflow.set_registry_uri("databricks-uc")

client = mlflow.tracking.MlflowClient()
src_model_name = f"{source_catalog}.{source_schema}.cali_housing_value"
src_model_version = "1"
src_model_uri = f"models:/{src_model_name}/{src_model_version}"
dst_model_name = f"{prod_catalog}.{prod_schema}.cali_housing_value"
copied_model_version = client.copy_model_version(src_model_uri, dst_model_name)

# COMMAND ----------
client = mlflow.tracking.MlflowClient()
client.set_registered_model_alias(
    name=dst_model_name, alias="Champion", version=copied_model_version.version
)
