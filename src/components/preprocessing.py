import pickle

import numpy as np
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def select_top_genes(df, top_k=3000):

    meta_columns = [
        "cell_type",
        "sm_name",
        "sm_lincs_id",
        "SMILES",
        "control"
    ]

    gene_columns = [
        c for c in df.columns
        if c not in meta_columns
    ]

    X = df[
        ["cell_type", "sm_name", "sm_lincs_id", "SMILES"]
    ]

    y = df[gene_columns]

    gene_variance = y.var()

    top_genes = (
        gene_variance
        .sort_values(ascending=False)
        .head(top_k)
        .index
    )

    y_reduced = y[top_genes]

    return X, y_reduced, top_genes




    
def encode_features(X):

    X_fe = X.copy()

    categorical_cols = [
        "cell_type",
        "sm_name",
        "sm_lincs_id"
    ]

    encoders = {}

    for col in categorical_cols:

        le = LabelEncoder()

        X_fe[col] = le.fit_transform(
            X_fe[col]
        )

        encoders[col] = le

    return X_fe, encoders



def split_data(
    X_struct,
    X_smiles,
    smiles_raw,
    y_array,
    cell_types
):

    return train_test_split(

        X_struct,
        X_smiles,
        smiles_raw,
        y_array,

        test_size=0.2,

        stratify=cell_types,

        random_state=42
    )
    


def scale_features(
    X_struct_train,
    X_struct_val
):

    scaler = StandardScaler()

    X_struct_train = scaler.fit_transform(
        X_struct_train
    )

    X_struct_val = scaler.transform(
        X_struct_val
    )

    return (
        X_struct_train,
        X_struct_val,
        scaler
    )