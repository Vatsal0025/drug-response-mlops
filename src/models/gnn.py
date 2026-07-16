import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np

from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool




class GNNModel(nn.Module):
    def __init__(self, struct_dim=3, output_dim=3000):
        super(GNNModel, self).__init__()
        print("Loading GNN model...")
        print("conv1 input features =", 7)
        # Graph layers
        self.conv1 = GCNConv(7, 64)
        self.conv2 = GCNConv(64, 128)

        # Structured branch
        self.struct_fc = nn.Sequential(
            nn.Linear(struct_dim, 64),
            nn.ReLU()
        )

        # Final layers
        self.fc = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GNN
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        # Pool graph → vector
        x = global_mean_pool(x, batch)

        # Get batch size
        batch_size = data.num_graphs

        # Reshape struct properly
        struct = data.struct.view(batch_size, -1)

        # Pass through FC
        struct = self.struct_fc(struct)

        # Combine
        x = torch.cat([x, struct], dim=1)

        return self.fc(x)