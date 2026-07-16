import torch

from src.components.config_loader import load_config

from src.components.data_ingestion import load_data

from src.components.preprocessing import (

    select_top_genes,

    prepare_data,

    split_data,

    scale_features,

    save_artifacts

)

from src.components.data_loader import (

    build_sequence_loaders,

    build_graph_loaders

)

from src.components.train_graph import train_graph_model

from src.components.train_sequence import train_sequence_model


from src.models.gru import GRUModel
from src.models.gru_attention import GRUAttentionModel
from src.models.gnn import GNNModel
from src.models.gat import GATModel


def load_model(config):

    model_name = config["training"]["model_name"]

    if model_name == "GRU":

        params = config["models"]["gru"]

        return GRUModel(

            smiles_vocab_size=params["smiles_vocab_size"],
            smiles_embed_dim=params["smiles_embed_dim"],
            gru_hidden_dim=params["gru_hidden_dim"],
            structured_dim=params["structured_dim"],
            output_dim=params["output_dim"]

        )

    elif model_name == "GRU_Attention":

        params = config["models"]["gru_attention"]

        return GRUAttentionModel(

            smiles_vocab_size=params["smiles_vocab_size"],
            smiles_embed_dim=params["smiles_embed_dim"],
            gru_hidden_dim=params["gru_hidden_dim"],
            structured_dim=params["structured_dim"],
            output_dim=params["output_dim"]

        )

    elif model_name == "GNN":

        return GNNModel()

    elif model_name == "GAT":

        return GATModel()

    else:

        raise ValueError(

            f"Unknown model: {model_name}"

        )


def run_training():

    config = load_config()

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    print(f"\nUsing device: {device}")

    ##########################################
    # LOAD DATA
    ##########################################

    df, _ = load_data()

    ##########################################
    # SELECT GENES
    ##########################################

    X, y, top_genes = select_top_genes(

        df,

        top_k=config["data"]["top_genes"]

    )

    ##########################################
    # PREPARE DATA
    ##########################################

    (

        X_struct,

        X_smiles,

        smiles_raw,

        y_array,

        char_to_idx,

        encoders

    ) = prepare_data(

        X,

        y

    )

    ##########################################
    # TRAIN/VAL SPLIT
    ##########################################

    (

        X_struct_train,

        X_struct_val,

        X_smiles_train,

        X_smiles_val,

        smiles_train_raw,

        smiles_val_raw,

        y_train,

        y_val

    ) = split_data(

        X_struct,

        X_smiles,

        smiles_raw,

        y_array,

        X["cell_type"]

    )

    ##########################################
    # SCALE
    ##########################################

    (

        X_struct_train,

        X_struct_val,

        scaler

    ) = scale_features(

        X_struct_train,

        X_struct_val

    )

    ##########################################
    # SAVE ARTIFACTS
    ##########################################

    save_artifacts(

        scaler,

        encoders,

        char_to_idx,

        top_genes

    )

    ##########################################
    # TENSORS
    ##########################################

    X_struct_train = torch.tensor(

        X_struct_train,

        dtype=torch.float

    )

    X_struct_val = torch.tensor(

        X_struct_val,

        dtype=torch.float

    )

    X_smiles_train = torch.tensor(

        X_smiles_train,

        dtype=torch.long

    )

    X_smiles_val = torch.tensor(

        X_smiles_val,

        dtype=torch.long

    )

    y_train = torch.tensor(

        y_train,

        dtype=torch.float

    )

    y_val = torch.tensor(

        y_val,

        dtype=torch.float

    )

    ##########################################
    # MODEL
    ##########################################

    model_name = config["training"]["model_name"]

    model = load_model(

        config

    ).to(device)

    ##########################################
    # DATALOADERS
    ##########################################

    if model_name in [

        "GRU",

        "GRU_Attention"

    ]:

        train_loader, val_loader = (

            build_sequence_loaders(

                X_struct_train,

                X_struct_val,

                X_smiles_train,

                X_smiles_val,

                y_train,

                y_val,

                batch_size=config["training"]["batch_size"]

            )

        )

    else:

        train_loader, val_loader = (

            build_graph_loaders(

                smiles_train_raw,

                smiles_val_raw,

                X_struct_train,

                X_struct_val,

                y_train,

                y_val,

                batch_size=config["training"]["batch_size"]

            )

        )

    ##########################################
    # TRAINING
    ##########################################

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=config["training"]["learning_rate"]

    )

    criterion = torch.nn.L1Loss()

    if model_name in [

        "GRU",

        "GRU_Attention"

    ]:

        train_sequence_model(

            model,

            train_loader,

            val_loader,

            optimizer,

            criterion,

            device,

            experiment_name=model_name,

            epochs=config["training"]["epochs"]

        )

    else:

        train_graph_model(

            model,

            train_loader,

            val_loader,

            optimizer,

            criterion,

            device,

            experiment_name=model_name,

            epochs=config["training"]["epochs"]

        )


if __name__ == "__main__":

    run_training()