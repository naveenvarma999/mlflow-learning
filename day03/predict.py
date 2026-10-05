import mlflow
from sklearn.datasets import load_iris

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model = mlflow.pyfunc.load_model("models:/iris-classifier@champion")

X, y = load_iris(return_X_y=True)
print("predictions:", model.predict(X[:5]))
print("actual:     ", y[:5])