import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

n_estimators = 150

mlflow.set_tracking_uri("sqlite:///mlflow.db")

with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, name="model", input_example=X_test[:2], skops_trusted_types=["sklearn.tree._tree.Tree"])

    print("accuracy:", acc)
    print("run_id:", run.info.run_id)