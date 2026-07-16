import torch

from torch.utils.data import DataLoader

from torch_geometric.loader import (

    DataLoader as GraphDataLoader

)

from src.components.dataset import (

    DualInputDataset,

    GNNDataset

)


def build_sequence_loaders(

    X_struct_train,

    X_struct_val,

    X_smiles_train,

    X_smiles_val,

    y_train,

    y_val,

    batch_size=32

):

    train_dataset = DualInputDataset(

        X_struct_train,

        X_smiles_train,

        y_train

    )

    val_dataset = DualInputDataset(

        X_struct_val,

        X_smiles_val,

        y_val

    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False

    )

    return train_loader, val_loader


def build_graph_loaders(

    smiles_train_raw,

    smiles_val_raw,

    X_struct_train,

    X_struct_val,

    y_train,

    y_val,

    batch_size=32

):

    train_dataset = GNNDataset(

        smiles_train_raw,

        X_struct_train,

        y_train

    )

    val_dataset = GNNDataset(

        smiles_val_raw,

        X_struct_val,

        y_val

    )

    train_loader = GraphDataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True

    )

    val_loader = GraphDataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False

    )

    return train_loader, val_loader