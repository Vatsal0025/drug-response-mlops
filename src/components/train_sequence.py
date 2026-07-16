import time
import torch
from src.components.visualize import save_loss_curve
from src.components.mlflow_logger import (
    setup_experiment,
    start_run,
    log_params,
    log_metric,
    save_model
)


def train_sequence_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    experiment_name="GRU",
    epochs=20
):

    train_losses = []
    val_losses = []

    setup_experiment(experiment_name)

    with start_run():

        log_params({

            "epochs": epochs,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "batch_size": train_loader.batch_size

        })

        for epoch in range(epochs):

            start = time.time()

            ####################
            # TRAIN
            ####################

            model.train()

            train_loss = 0

            for struct, smiles, target in train_loader:

                struct = struct.to(device)

                smiles = smiles.to(device)

                target = target.to(device)

                optimizer.zero_grad()

                output = model(struct, smiles)

                if isinstance(output, tuple):

                    output, attention_weights = output

                loss = criterion(output, target)

                loss.backward()

                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            ####################
            # VALIDATION
            ####################

            model.eval()

            val_loss = 0

            all_preds = []

            all_targets = []

            with torch.no_grad():

                for struct, smiles, target in val_loader:

                    struct = struct.to(device)

                    smiles = smiles.to(device)

                    target = target.to(device)

                    output = model(struct, smiles)

                    if isinstance(output, tuple):

                        output, attention_weights = output

                    loss = criterion(output, target)

                    val_loss += loss.item()

                    all_preds.append(output)

                    all_targets.append(target)

            val_loss /= len(val_loader)

            train_losses.append(train_loss)

            val_losses.append(val_loss)

            ####################
            # SIGN ACCURACY
            ####################

            y_pred = torch.cat(all_preds)

            y_true = torch.cat(all_targets)

            sign_accuracy = (

                (torch.sign(y_pred)
                 ==
                 torch.sign(y_true))

                .float()

                .mean()

                .item()

            )

            ####################
            # MLFLOW
            ####################

            log_metric(
                "train_loss",
                train_loss,
                step=epoch
            )

            log_metric(
                "val_loss",
                val_loss,
                step=epoch
            )

            log_metric(
                "sign_accuracy",
                sign_accuracy,
                step=epoch
            )

            end = time.time()

            print(

                f"Epoch {epoch+1}/{epochs} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | "
                f"Sign Acc: {sign_accuracy:.4f} | "
                f"Time: {end-start:.2f}s"

            )

        save_loss_curve(train_losses,val_losses,experiment_name)

        save_model(model)

    return train_losses, val_losses