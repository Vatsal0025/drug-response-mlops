import os

import matplotlib.pyplot as plt

from src.components.mlflow_logger import save_artifact


def save_loss_curve(

    train_losses,

    val_losses,

    experiment_name

):

    os.makedirs(

        "experiments",

        exist_ok=True

    )

    path = (

        f"experiments/"

        f"{experiment_name.lower()}_loss.png"

    )

    plt.figure(

        figsize=(8, 5)

    )

    plt.plot(

        train_losses,

        label="Train Loss"

    )

    plt.plot(

        val_losses,

        label="Validation Loss"

    )

    plt.xlabel(

        "Epoch"

    )

    plt.ylabel(

        "Loss"

    )

    plt.title(

        experiment_name

    )

    plt.legend()

    plt.savefig(

        path

    )

    plt.close()

    save_artifact(

        path

    )