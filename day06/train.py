from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DB = Path(__file__).parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{DB}")
mlflow.set_experiment("iris-baseline")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

configs = [
    ("random_forest", RandomForestClassifier(n_estimators=50, random_state=42)),
    ("random_forest", RandomForestClassifier(n_estimators=200, random_state=42)),
    ("random_forest", RandomForestClassifier(n_estimators=5, max_depth=1, random_state=42)),
    ("logistic_regression", LogisticRegression(max_iter=200)),
]

for model_type, model in configs:
    with mlflow.start_run():
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("dataset", "iris")
        mlflow.set_tag("author", "naveen")

        mlflow.log_params(model.get_params())

        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", acc)

        print(f"{model_type}: {acc:.4f}")