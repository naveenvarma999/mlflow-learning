import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()
client.set_registered_model_alias(
    name="iris-classifier",
    alias="champion",
    version=1)
print("champion = version 1")