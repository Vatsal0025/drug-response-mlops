import mlflow

mlflow.set_experiment("test")

with mlflow.start_run():

    mlflow.log_param("learning_rate", 0.0003)

    mlflow.log_metric("accuracy", 0.82)

print("Done")