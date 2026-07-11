import torch
import torch.nn as nn


class GRUModel(nn.Module):

    def __init__(
        self,
        smiles_vocab_size,
        smiles_embed_dim=128,
        gru_hidden_dim=384,
        structured_dim=3,
        output_dim=3000
    ):

        super(GRUModel, self).__init__()

        # SMILES embedding
        self.embedding = nn.Embedding(
            smiles_vocab_size,
            smiles_embed_dim,
            padding_idx=0
        )

        # GRU
        self.gru = nn.GRU(
            input_size=smiles_embed_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True
        )

        # Structured features
        self.struct_fc = nn.Sequential(
            nn.Linear(structured_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )

        # Final layers
        self.fc = nn.Sequential(

            nn.Linear(gru_hidden_dim + 64, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, output_dim)
        )

    def forward(self, x_struct, x_smiles):

        # SMILES branch
        x_smiles = self.embedding(x_smiles)

        _, h = self.gru(x_smiles)

        h = h[-1]

        # Structured branch
        x_struct = self.struct_fc(x_struct)

        # Combine
        x = torch.cat([h, x_struct], dim=1)

        return self.fc(x)