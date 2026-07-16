import mlflow
import mlflow.pytorch


def setup_experiment(experiment_name):

    mlflow.set_experiment(experiment_name)


def start_run(run_name=None):

    return mlflow.start_run(run_name=run_name)


def log_params(params):

    mlflow.log_params(params)


def log_metrics(metrics):

    mlflow.log_metrics(metrics)


def log_metric(name, value, step=None):

    mlflow.log_metric(name, value, step=step)


def save_model(model, artifact_path="model"):

    mlflow.pytorch.log_model(

        pytorch_model=model,

        name=artifact_path,

        serialization_format="pickle"

    )


def save_artifact(path):

    mlflow.log_artifact(path)