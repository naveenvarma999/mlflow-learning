import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")

for n_estimators in [50, 150, 300]:
    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)

        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(
            model,
            name="random-forest-classifier",
            input_example=X_test[:2],
            registered_model_name="iris-classifier",skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        print(f"n_estimators={n_estimators}  accuracy={acc}")