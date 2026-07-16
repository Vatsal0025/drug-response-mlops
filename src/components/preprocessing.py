import pickle
import os
import numpy as np
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def build_smiles_vocab(smiles_list):

    chars = set()

    for smiles in smiles_list:

        chars.update(smiles)

    char_to_idx = {

        char: idx + 1

        for idx, char in enumerate(

            sorted(chars)

        )

    }

    char_to_idx["<PAD>"] = 0

    return char_to_idx



def encode_smiles(

    smiles,

    char_to_idx,

    max_len=120

):

    encoded = [

        char_to_idx[c]

        for c in smiles

    ]

    encoded = encoded[:max_len]

    encoded += [

        0

    ] * (

        max_len

        - len(encoded)

    )

    return encoded

def prepare_data(

    X,
    y

):

    X_fe, encoders = encode_features(

        X

    )

    X_struct = X_fe[

        [

            "cell_type",

            "sm_name",

            "sm_lincs_id"

        ]

    ].values

    smiles_raw = X_fe[

        "SMILES"

    ].values

    char_to_idx = build_smiles_vocab(

        smiles_raw

    )

    X_smiles = np.array([

        encode_smiles(

            s,

            char_to_idx

        )

        for s in smiles_raw

    ])

    y_array = y.values

    return (

        X_struct,

        X_smiles,

        smiles_raw,

        y_array,

        char_to_idx,

        encoders

    )

def save_artifacts(

    scaler,

    encoders,

    char_to_idx,

    top_genes,

    artifact_dir="artifacts"

):

    os.makedirs(

        artifact_dir,

        exist_ok=True

    )

    with open(

        os.path.join(

            artifact_dir,

            "scaler.pkl"

        ),

        "wb"

    ) as f:

        pickle.dump(

            scaler,

            f

        )

    with open(

        os.path.join(

            artifact_dir,

            "encoders.pkl"

        ),

        "wb"

    ) as f:

        pickle.dump(

            encoders,

            f

        )

    with open(

        os.path.join(

            artifact_dir,

            "smiles_vocab.pkl"

        ),

        "wb"

    ) as f:

        pickle.dump(

            char_to_idx,

            f

        )

    with open(

        os.path.join(

            artifact_dir,

            "top_genes.pkl"

        ),

        "wb"

    ) as f:

        pickle.dump(

            top_genes,

            f
        )

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