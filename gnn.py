import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import global_mean_pool

import torch
from rdkit import Chem
from torch_geometric.data import Data

def atom_features(atom):
    """ Create an atom feature vector. """
    return torch.tensor([
        atom.GetAtomicNum(),  # Atomic number
        atom.GetDegree(),     # Degree
        atom.GetFormalCharge(),  # Formal charge
        atom.GetIsAromatic(),  # Aromaticity
        # Add more features if needed
    ], dtype=torch.float)


def smiles_to_graph(smiles, target):
    mol = Chem.MolFromSmiles(smiles)
    atoms = mol.GetAtoms()

    # Create node features
    node_features = torch.stack([atom_features(atom) for atom in atoms])

    # Create edges
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))  # Add both directions

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=node_features, edge_index=edge_index, target=target)


class SimpleGNN(torch.nn.Module):
    def __init__(self):
        super(SimpleGNN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(4, 16)
        self.conv2 = pyg_nn.GCNConv(16, 32)
        self.fc = torch.nn.Linear(32, 1)  # Output layer

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = global_mean_pool(x, batch)
        x = torch.sigmoid(self.fc(x))
        return x
