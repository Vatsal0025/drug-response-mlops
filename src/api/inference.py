import pickle
import torch
import numpy as np
from src.models.gru import GRUModel
from src.components.config_loader import load_config
from src.components.preprocessing import encode_smiles

config = load_config()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load artifacts

with open("artifacts/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)


print(encoders["cell_type"].classes_)
print(encoders["sm_name"].classes_[:10])
print(encoders["sm_lincs_id"].classes_[:10])

with open("artifacts/smiles_vocab.pkl", "rb") as f:
    char_to_idx = pickle.load(f)

with open("artifacts/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("artifacts/top_genes.pkl", "rb") as f:
    top_genes = pickle.load(f)






# Create model

params = config["models"]["gru"]

model = GRUModel(
    smiles_vocab_size=params["smiles_vocab_size"],
    smiles_embed_dim=params["smiles_embed_dim"],
    gru_hidden_dim=params["gru_hidden_dim"],
    structured_dim=params["structured_dim"],
    output_dim=params["output_dim"]
)

model.load_state_dict(
    torch.load(
        "artifacts/best_model.pth",
        map_location=device
    )
)





def predict_gene_expression(data):

    # Encode categorical variables

    cell_type = encoders["cell_type"].transform(
        [data.cell_type]
    )[0]

    sm_name = encoders["sm_name"].transform(
        [data.sm_name]
    )[0]

    sm_lincs_id = encoders["sm_lincs_id"].transform(
        [data.sm_lincs_id]
    )[0]

    # Encode smiles

    smiles_encoded = encode_smiles(

        data.smiles,

        char_to_idx

    )

    # Structured features

    structured = np.array([

        cell_type,

        sm_name,

        sm_lincs_id

    ]).reshape(1, -1)

    # Scale

    structured = scaler.transform(

        structured

    )

    # Convert to tensors

    structured_tensor = torch.tensor(

        structured,

        dtype=torch.float

    ).to(device)

    smiles_tensor = torch.tensor(

        [smiles_encoded],

        dtype=torch.long

    ).to(device)

    # Prediction

    with torch.no_grad():

        prediction = model(

            structured_tensor,

            smiles_tensor

        )

    prediction = prediction.squeeze().cpu().numpy()

    gene_predictions = [

        {

            "gene": gene,

            "value": float(value)

        }

        for gene, value in zip(

            top_genes,

            prediction

        )

    ]

    gene_predictions = sorted(

        gene_predictions,

        key=lambda x: abs(x["value"]),

        reverse=True

    )

    top_k = data.top_k

    selected_genes = gene_predictions[:top_k]

    return {

        "top_genes": selected_genes

    }  




model.to(device)
model.eval()


def predict(data):

    return {
        "status": "Model loaded successfully"
    }