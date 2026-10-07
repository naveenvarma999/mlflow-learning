from pathlib import Path
import mlflow

DB = Path(__file__).parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{DB}")

runs = mlflow.search_runs(
    experiment_names=["iris-baseline"],
    filter_string="tags.model_type = 'random_forest' and metrics.accuracy > 0.8",
    order_by=["metrics.accuracy DESC"],
)

print(runs[["run_id", "metrics.accuracy", "params.n_estimators", "tags.model_type"]])
print()
print("best run:", runs.iloc[0]["run_id"])