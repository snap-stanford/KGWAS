from torch_geometric.nn import Linear, SAGEConv, GCNConv, SGConv, Sequential, to_hetero, HeteroConv
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from .conv import GATConv


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.FC_hidden = nn.Linear(input_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.ReLU = nn.ReLU()       
                
    def forward(self, x):
        h     = self.ReLU(self.FC_hidden(x))
        h     = self.ReLU(self.FC_hidden2(h))
        x_hat = self.FC_output(h)
        return x_hat

class HeteroGNN(torch.nn.Module):
    def __init__(self, pyg_data, hidden_channels, out_channels, num_layers, gnn_backbone, gnn_aggr, snp_init_dim_size, gene_init_dim_size, go_init_dim_size, gat_num_head, no_relu = False):
        super().__init__()
        edge_types = pyg_data.edge_types
        self.convs = torch.nn.ModuleList()
        
        self.snp_feat_mlp = SimpleMLP(snp_init_dim_size, hidden_channels, hidden_channels)
        self.go_feat_mlp = SimpleMLP(go_init_dim_size, hidden_channels, hidden_channels)
        self.gene_feat_mlp = SimpleMLP(gene_init_dim_size, hidden_channels, hidden_channels)
        self.ReLU = nn.ReLU()   
        for _ in range(num_layers):
            conv_layer = {}
            for i in edge_types:
                if gnn_backbone == 'SAGE':
                    conv_layer[i] = SAGEConv((-1, -1), hidden_channels)
                elif gnn_backbone == 'GAT':
                    conv_layer[i] = GATConv((-1, -1), hidden_channels, 
                                            heads = gat_num_head, 
                                            add_self_loops = False)
                elif gnn_backbone == 'GCN':
                    conv_layer[i] = GCNConv(-1, hidden_channels, add_self_loops = False)
                elif gnn_backbone == 'SGC':
                    conv_layer[i] = SGConv(-1, hidden_channels, add_self_loops = False)
            conv = HeteroConv(conv_layer, aggr=gnn_aggr)
            self.convs.append(conv)
        
        self.lin = Linear(hidden_channels, out_channels)
        self.no_relu = no_relu
        
    def forward(self, x_dict, edge_index_dict, batch_size, genotype = None, return_h = False, 
                return_attention_weights = False):
        
        x_dict['SNP'] = self.snp_feat_mlp(x_dict['SNP'])
        x_dict['Gene'] = self.gene_feat_mlp(x_dict['Gene'])
        x_dict['CellularComponent'] = self.go_feat_mlp(x_dict['CellularComponent'])
        x_dict['BiologicalProcess'] = self.go_feat_mlp(x_dict['BiologicalProcess'])
        x_dict['MolecularFunction'] = self.go_feat_mlp(x_dict['MolecularFunction'])
        
        
        attention_all_layers = []
        for conv in self.convs:
            if return_attention_weights:
                out = conv(x_dict, edge_index_dict, 
                              return_attention_weights_dict = dict(zip(list(edge_index_dict.keys()), 
                                                            [True] * len(list(edge_index_dict.keys())))))
                #attention_layer = {i: [x[1] for x in j[1]] for i,j in out.items()}
                mean_attention = torch.mean(torch.vstack([torch.vstack([x[1] for x in j[1]]) for i,j in out.items()]))
                x_dict = {i: j[0] for i,j in out.items()}
                attention_all_layers.append(mean_attention)
            else:
                x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        
        if return_h:
            return self.ReLU(self.lin(x_dict['SNP']))[:batch_size], x_dict['SNP'][:batch_size] 
        if return_attention_weights:
            return self.ReLU(self.lin(x_dict['SNP']))[:batch_size], attention_all_layers
        else:
            if self.no_relu:
                return self.lin(x_dict['SNP'])[:batch_size]
            else:
                return self.ReLU(self.lin(x_dict['SNP']))[:batch_size]
