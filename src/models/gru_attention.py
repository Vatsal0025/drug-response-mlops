import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()

        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, gru_outputs):
        # gru_outputs: (batch, seq_len, hidden_dim)

        scores = self.attn(gru_outputs)  # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)

        # weighted sum
        context = torch.sum(weights * gru_outputs, dim=1)

        return context, weights


class GRUAttentionModel(nn.Module):
    def __init__(self,
                 smiles_vocab_size,
                 smiles_embed_dim=64,
                 gru_hidden_dim=128,
                 structured_dim=3,
                 output_dim=3000):

        super(GRUAttentionModel, self).__init__()

        # SMILES embedding
        self.embedding = nn.Embedding(smiles_vocab_size, smiles_embed_dim, padding_idx=0)

        # GRU (return all outputs)
        self.gru = nn.GRU(
            input_size=smiles_embed_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True
        )

        # Attention
        self.attention = Attention(gru_hidden_dim)

        # Structured branch
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
            nn.Dropout(0.3),

            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, output_dim)
        )

    def forward(self, x_struct, x_smiles):

        # SMILES branch
        x_smiles = self.embedding(x_smiles)  # (batch, seq_len, embed_dim)

        gru_out, _ = self.gru(x_smiles)      # (batch, seq_len, hidden_dim)

        context, attn_weights = self.attention(gru_out)

        # Structured branch
        x_struct = self.struct_fc(x_struct)

        # Combine
        x = torch.cat([context, x_struct], dim=1)

        # Output
        out = self.fc(x)

        return out, attn_weights