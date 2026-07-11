import torch
import random
import numpy as np


def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


def get_device():

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def compute_sign_accuracy(y_pred, y_true):

    return (
        (torch.sign(y_pred) == torch.sign(y_true))
        .float()
        .mean()
    )