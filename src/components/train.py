import time
import torch


def train_model(

    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs=20

):

    train_losses = []

    val_losses = []

    epoch_times = []

    for epoch in range(epochs):

        start = time.time()

        model.train()

        train_loss = 0

        for batch in train_loader:

            optimizer.zero_grad()

            loss = None

            # Placeholder

            optimizer.step()

        end = time.time()

        epoch_times.append(
            end - start
        )

    return (
        train_losses,
        val_losses,
        epoch_times
    )