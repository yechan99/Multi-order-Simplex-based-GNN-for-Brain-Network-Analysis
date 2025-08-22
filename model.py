import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))

        self.reset_parameters()

    def reset_parameters(self): # Initialize weights and bias
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj): # Graph convolution
        support = torch.matmul(adj, x)
        output = torch.matmul(support, self.weight) + self.bias

        return output

class GCN(nn.Module):
    def __init__(self, n_e_num, in_dim, out_dim, dropout):
        super(GCN, self).__init__()
        self.gc = GraphConvolution(in_dim, out_dim)
        self.norm = nn.LayerNorm([out_dim, n_e_num])
        self.dropout = dropout

    def forward(self, x, adj): # Graph convolution part
        x = self.gc.forward(x, adj)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        return x
    
class Ours(nn.Module):
    def __init__(self, node_num, edge_num, in_dim_n, in_dim_e, hid_dim, out_dim, dropout, num_gcn_layers):
        super(Ours, self).__init__()
        self.dropout = dropout
        self.node_num = node_num
        self.edge_num = edge_num
        self.num_gcn_layers = num_gcn_layers

        self.norm1 = nn.LayerNorm([in_dim_n, node_num])
        self.norm2 = nn.LayerNorm([in_dim_e, edge_num])

        self.gcn_n = GCN(node_num, in_dim_n, hid_dim, dropout)
        self.gcn_n_with_e = GCN(node_num, in_dim_e, hid_dim, dropout)
        self.gcn_e = GCN(edge_num, in_dim_e, hid_dim, dropout)
        self.gcn_e_with_n = GCN(edge_num, in_dim_n, hid_dim, dropout)
        
        self.gcn_layers_n = nn.ModuleList([GCN(node_num, hid_dim, hid_dim, dropout) for _ in range(num_gcn_layers - 1)])
        self.gcn_layers_e = nn.ModuleList([GCN(edge_num, hid_dim, hid_dim, dropout) for _ in range(num_gcn_layers - 1)])

        self.mlp = nn.Sequential(
            nn.Linear((node_num + edge_num) * hid_dim, hid_dim * 8),
            nn.BatchNorm1d(hid_dim * 8),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hid_dim * 8, out_dim),
        )

    def forward(self, x_n, x_e, adj, hodge, inc):
        batch = x_n.shape[0]
        x_n = x_n.squeeze(dim=1)
        x_e = x_e.squeeze(dim=1)
        adj = adj.squeeze()
        hodge = hodge.squeeze()
        inc = inc.squeeze()
        inc_T = torch.transpose(inc, 1, 2)

        x_n = self.norm1(x_n.permute(0, 2, 1)).permute(0, 2, 1)
        x_e = self.norm2(x_e.permute(0, 2, 1)).permute(0, 2, 1)

        x_n_2 = self.gcn_n(x_n, adj)
        x_n_with_e = self.gcn_n_with_e(x_e, inc)
        x_e_2 = self.gcn_e(x_e, hodge)
        x_e_with_n = self.gcn_e_with_n(x_n, inc_T)
        
        x_n = (x_n_2 + x_n_with_e)/2
        x_e = (x_e_2 + x_e_with_n)/2

        for gcn_layer_n, gcn_layer_e in zip(self.gcn_layers_n, self.gcn_layers_e):
            x_n = gcn_layer_n(x_n, adj)
            x_e = gcn_layer_e(x_e, hodge)

        x = torch.cat([x_n, x_e], dim=1)
        x = self.mlp(x.view(batch, -1))

        out = F.log_softmax(x, dim=1)

        return out