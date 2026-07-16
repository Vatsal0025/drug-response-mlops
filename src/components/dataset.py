import torch

from torch.utils.data import Dataset

from torch_geometric.data import Data

from rdkit import Chem


def smiles_to_graph(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:

        raise ValueError(
            f"Invalid SMILES: {smiles}"
        )

    node_features = []

    for atom in mol.GetAtoms():

        node_features.append([

            atom.GetAtomicNum(),

            atom.GetDegree(),

            atom.GetFormalCharge(),

            int(atom.GetHybridization()),

            int(atom.GetIsAromatic()),

            atom.GetTotalNumHs(),

            atom.GetValence(
                Chem.ValenceType.IMPLICIT
            )

        ])

    x = torch.tensor(
        node_features,
        dtype=torch.float
    )

    x = (

        x - x.mean(dim=0)

    ) / (

        x.std(dim=0) + 1e-5

    )

    edge_index = []

    for bond in mol.GetBonds():

        i = bond.GetBeginAtomIdx()

        j = bond.GetEndAtomIdx()

        edge_index.append([i, j])

        edge_index.append([j, i])

    edge_index = torch.tensor(

        edge_index,
        dtype=torch.long

    ).t().contiguous()

    return x, edge_index


class DualInputDataset(Dataset):

    def __init__(

        self,
        X_struct,
        X_smiles,
        y

    ):

        self.X_struct = X_struct

        self.X_smiles = X_smiles

        self.y = y

    def __len__(self):

        return len(
            self.X_struct
        )

    def __getitem__(self, idx):

        return (

            self.X_struct[idx],

            self.X_smiles[idx],

            self.y[idx]

        )


class GNNDataset(Dataset):

    def __init__(

        self,
        smiles_list,
        struct_data,
        targets

    ):

        self.smiles = smiles_list

        self.struct = struct_data

        self.targets = targets

    def __len__(self):

        return len(
            self.smiles
        )

    def __getitem__(self, idx):

        x, edge_index = smiles_to_graph(

            self.smiles[idx]

        )

        data = Data(

            x=x,
            edge_index=edge_index

        )

        data.struct = (

            self.struct[idx]

            .clone()

            .detach()

            .float()

        )
        
        data.y = (

            self.targets[idx]

            .clone()

            .detach()

            .float()

            .unsqueeze(0)

        )

        return data