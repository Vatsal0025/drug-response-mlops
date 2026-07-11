import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

class GATModel(nn.Module):
    def __init__(self, struct_dim=3, output_dim=3000):
        super(GATModel, self).__init__()

        # GAT layers
        self.conv1 = GATConv(7, 64, heads=4)        # -> 64*4 = 256
        self.bn1 = nn.BatchNorm1d(256)

        self.conv2 = GATConv(256, 128, heads=2)     # -> 128*2 = 256
        self.bn2 = nn.BatchNorm1d(256)

        # Structured branch
        self.struct_fc = nn.Sequential(
            nn.Linear(struct_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )

        # Final layers (FIXED DIMENSIONS)
        self.fc = nn.Sequential(
            nn.Linear(256*2 + 64, 256),  # (mean+max)=512 + struct=64 → 576
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # ---- GAT Block 1 ----
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        # ---- GAT Block 2 (Residual) ----
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)

        x = x1 + x2   # 🔥 residual connection

        # ---- Pooling ----
        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)  # 512

        # ---- Structured branch ----
        struct = data.struct.view(data.num_graphs, -1)
        struct = self.struct_fc(struct)

        # ---- Combine ----
        x = torch.cat([x, struct], dim=1)  # 512 + 64 = 576

        return self.fc(x)