import torch


def compute_sign_accuracy(
    y_pred,
    y_true
):

    return (

        torch.sign(y_pred)
        ==
        torch.sign(y_true)

    ).float().mean()