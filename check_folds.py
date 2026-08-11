import mlflow
mlflow.set_tracking_uri("file:///D:/ML Projects/EDA/gnn_eda/mlruns")
runs = mlflow.search_runs(experiment_names=["qor-rq2-leave-circuits-out"])
parents = runs[runs["params.n_train_circuits"].notna()]
cols = ["run_id", "params.n_train_circuits", "params.n_test_circuits", "params.seed", "start_time"]
print(parents[cols].sort_values("start_time"))