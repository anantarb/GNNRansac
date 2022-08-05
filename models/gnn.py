import torch
from torch_geometric.nn import GraphConv
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm

class GNNSample(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = GraphConv(517, 256)
        self.graphbn1 = BatchNorm(256)
        self.conv2 = GraphConv(256, 128)
        self.graphbn2 = BatchNorm(128)
        self.conv3 = GraphConv(128, 64)
        self.graphbn3 = BatchNorm(64)
        self.linear = torch.nn.Linear(64, 1)

    def forward(self, x, edge_index, edge_weight):

        x = self.graphbn1(self.conv1(x, edge_index, edge_weight))
        x = F.relu(x)
        x = self.graphbn2(self.conv2(x, edge_index, edge_weight))
        x = F.relu(x)
        x = self.graphbn3(self.conv3(x, edge_index, edge_weight))
        x = F.relu(x)
        
        x = self.linear(x)
        
        x = F.logsigmoid(x)
        
        norm = torch.logsumexp(x, dim=0)
        norm = norm.expand(x.size(0), 1)

        x = x - norm

        return x