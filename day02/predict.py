import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# find the most recent finished run
runs = mlflow.search_runs(
    filter_string="attributes.status = 'FINISHED'",
    order_by=["start_time DESC"],
    max_results=1,
)
RUN_ID = runs.iloc[0]["run_id"]
print("using run:", RUN_ID)

model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/model")
print(model.predict([[5.1, 3.5, 1.4, 0.2]]))