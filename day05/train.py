from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DB = Path(__file__).parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{DB}")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

n_estimators = 150

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", acc)

    info = mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=X_test[:2],
        registered_model_name="iris-classifier",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

print(f"accuracy={acc}  version={info.registered_model_version}")