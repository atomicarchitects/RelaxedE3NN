import torch
from torch_cluster import radius_graph
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter
from typing import Union
from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct
from e3nn.io import SphericalTensor
from e3nn.math import soft_one_hot_linspace
from e3nn.util.test import assert_equivariant
from e3nn.util.test import set_random_seeds
from typing import Union
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set random seeds

# have a version for scalar and tensor as an argument!!! 
set_random_seeds()
class RelaxedConvolution(torch.nn.Module):
    def __init__(self, irreps_in, irreps_out, irreps_edge, num_neighbors,fc_neurons,fc_edge_dim=1, relaxed=False,
                 irreps_relaxed: Union[int,o3.Irreps]=2, mul_relaxed_weights = 1, irreps_inter = None) -> None:
        super().__init__()

        self.num_neighbors = num_neighbors

        self.fc_neurons = fc_neurons
        self.fc_edge_dim = fc_edge_dim
        self.irreps_out = irreps_out
        self.relaxed = relaxed

        if relaxed:
            if isinstance(irreps_relaxed, int):
                irreps_relaxed_iter = o3.Irrep.iterator(irreps_relaxed)
                self.irreps_relaxed = o3.Irreps("")
                for irrep in irreps_relaxed_iter:
                    self.irreps_relaxed += mul_relaxed_weights*irrep
            else:
                self.irreps_relaxed = irreps_relaxed
            if irreps_inter == None:
                self.irreps_inter = self.irreps_relaxed #+ irreps_edge
            else:
                self.irreps_inter = irreps_inter
            self.relaxed_weights = []
            self.relaxed_edge_tp = FullyConnectedTensorProduct(
                self.irreps_relaxed,
                irreps_edge,
                self.irreps_inter)
            self.in_edge_tp = FullyConnectedTensorProduct(
                irreps_in,
                self.irreps_inter,
                irreps_out,
                shared_weights=False,
            )
            for i, (mul, ir_in) in enumerate(self.irreps_relaxed):
                # group different copies into different relaxed weights
                if mul == 1:
                    if ir_in == o3.Irrep("0e"):
                        self.relaxed_weights.append(torch.nn.Parameter(torch.ones(ir_in.dim)))
                    else:
                        self.relaxed_weights.append(torch.nn.Parameter(torch.zeros(ir_in.dim)))
                else:
                    relaxed_weights_mul = []
                    if ir_in == o3.Irrep("0e"):
                        for j in range(mul):
                            relaxed_weights_mul.append(torch.nn.Parameter(torch.ones(ir_in.dim)))
                        self.relaxed_weights.append(torch.cat(relaxed_weights_mul))
                    else:
                        # if this is non-zero, then the symmetry is broken
                        for j in range(mul):
                            relaxed_weights_mul.append(torch.nn.Parameter(torch.zeros(ir_in.dim)))
                        self.relaxed_weights.append(torch.cat(relaxed_weights_mul))
            self.relaxed_weights = torch.nn.Parameter(torch.cat(self.relaxed_weights))
        
        # it is not relaxed
        else:
            self.in_edge_tp = FullyConnectedTensorProduct(
                irreps_in,
                irreps_edge,
                irreps_out,
                shared_weights=False,
            )

        self.fc = FullyConnectedNet([self.fc_edge_dim,self.fc_neurons,self.in_edge_tp.weight_numel], torch.relu)
    def forward(self, node_input, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
        tp_weight = self.fc(edge_scalars)
        if self.relaxed:
            relaxed_weights = self.relaxed_edge_tp(self.relaxed_weights,edge_attr)
            new_edge_features = self.in_edge_tp(node_input[edge_src], relaxed_weights, tp_weight)
            node_features = scatter(new_edge_features, edge_dst, dim=0, dim_size=node_input.shape[0]).div(self.num_neighbors**0.5)
        else:
            edge_features = self.in_edge_tp(node_input[edge_src], edge_attr,tp_weight)
            node_features = scatter(edge_features, edge_dst, dim=0, dim_size=node_input.shape[0]).div(self.num_neighbors**0.5)
        return node_features
    
class RelaxedE3NN(torch.nn.Module):
    def __init__(self,irreps_in, irreps_out, irreps_edge,fc_neurons, num_neighbors, relaxed, max_radius = 1.8,num_layers = 2,
                 irreps_relaxed: Union[int,o3.Irreps]=2, mul_relaxed_weights = 1, irreps_inter=None) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors  # typical number of neighbors
        self.irreps_edge = irreps_edge

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.fc_neurons = fc_neurons
        self.max_radius = max_radius
        self.num_layers = num_layers
        self.mul_relaxed_weights = mul_relaxed_weights
        self.irreps_relaxed = irreps_relaxed
        self.irreps_inter = irreps_inter
        self.model = torch.nn.Sequential()
        # changing the input and output irreps?
        fc_edge_dim = 1
        module = RelaxedConvolution(self.irreps_in, self.irreps_out, self.irreps_edge,self.num_neighbors, self.fc_neurons, fc_edge_dim, relaxed, irreps_relaxed, mul_relaxed_weights, irreps_inter)
        self.model.add_module(f"RelaxedConvolution_{0}", module)
        for i in range(self.num_layers-1):
            # need to have the irreps out of the previous layer
            module = RelaxedConvolution(self.irreps_out, self.irreps_out, self.irreps_edge,self.num_neighbors, self.fc_neurons, fc_edge_dim, relaxed, irreps_relaxed, mul_relaxed_weights,irreps_inter)
            self.model.add_module(f"RelaxedConvolution_{i+1}", module)

    def forward(self, data) -> torch.Tensor:
        num_nodes = len(data.x)
        edge_src, edge_dst = radius_graph(x=data.pos, r=self.max_radius, batch=data.batch,max_num_neighbors=num_nodes - 1)
        edge_vec = data.pos[edge_dst] - data.pos[edge_src]
        edge_attr = o3.spherical_harmonics(
            l=self.irreps_edge,
            x=edge_vec,
            normalize=True,
            normalization='component'
        )

        # could put edge length embedding here in the future 
        # Pass data through the model sequentially
        x = data.x
        edge_length_embedding = edge_vec.norm(dim = 1, keepdim=True)#.view(-1,1)
        for module in self.model:
            x = module(x, edge_src, edge_dst, edge_attr, edge_length_embedding)

        return x
