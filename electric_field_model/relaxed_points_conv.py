import torch
from torch_scatter import scatter
from typing import Union
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct
from e3nn.util.jit import compile_mode


@compile_mode("script")
class RelaxedConvolution(torch.nn.Module):
    r"""equivariant convolution

    Parameters
    ----------
    irreps_node_input : `e3nn.o3.Irreps`
        representation of the input node features

    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the node attributes

    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes

    irreps_node_output : `e3nn.o3.Irreps` or None
        representation of the output node features

    fc_neurons : list of int
        number of neurons per layers in the fully connected network
        first layer and hidden layers but not the output layer

    num_neighbors : float
        typical number of nodes convolved over
    """

    def __init__(
        self, irreps_node_input, irreps_node_attr, irreps_edge_attr, irreps_node_output, fc_neurons, num_neighbors,
        relaxed, irreps_inter_relaxed = None, irreps_relaxed: Union[int, o3.Irreps] = 1, mul_relaxed_weights = 1
        ) -> None:
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_inter_relaxed = o3.Irreps(irreps_inter_relaxed)
        self.num_neighbors = num_neighbors
        self.relaxed = relaxed
        
        self.sc = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_output)

        self.lin1 = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_input)
        if not self.relaxed:
            irreps_mid = []
            instructions = []
            for i, (mul, ir_in) in enumerate(self.irreps_node_input):
                for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                    for ir_out in ir_in * ir_edge:
                        if ir_out in self.irreps_node_output or ir_out == o3.Irrep(0, 1):
                            k = len(irreps_mid)
                            irreps_mid.append((mul, ir_out))
                            instructions.append((i, j, k, "uvu", True))
            irreps_mid = o3.Irreps(irreps_mid)
            irreps_mid, p, _ = irreps_mid.sort()

            instructions = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instructions]

            tp = TensorProduct(
                self.irreps_node_input,
                self.irreps_edge_attr,
                irreps_mid,
                instructions,
                internal_weights=False,
                shared_weights=False,
            )
            
            self.fc = FullyConnectedNet(fc_neurons + [tp.weight_numel], torch.nn.functional.silu)
            self.tp = tp

        if relaxed:
            # make relaxed weights
            if isinstance(irreps_relaxed, int):
                irreps_relaxed_iter = o3.Irrep.iterator(irreps_relaxed)
                self.irreps_relaxed = o3.Irreps("")
                for irrep in irreps_relaxed_iter:
                    self.irreps_relaxed += mul_relaxed_weights*irrep
            else:
                self.irreps_relaxed = irreps_relaxed
            self.relaxed_weights = []
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
            
            self.relaxed_edge_tp = FullyConnectedTensorProduct(
                self.irreps_relaxed,
                self.irreps_edge_attr,
                self.irreps_inter_relaxed
            )
            # make intermediate irreps

            irreps_mid = []
            instructions = []
            for i, (mul, ir_in) in enumerate(self.irreps_node_input):
                for j, (_, ir_edge) in enumerate(self.irreps_inter_relaxed):
                    for ir_out in ir_in * ir_edge:
                        if ir_out in self.irreps_node_output or ir_out == o3.Irrep(0, 1):
                            k = len(irreps_mid)
                            irreps_mid.append((mul, ir_out))
                            instructions.append((i, j, k, "uvu", True))
            irreps_mid = o3.Irreps(irreps_mid)
            irreps_mid, p, _ = irreps_mid.sort()

            instructions = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instructions] 


            self.in_relaxed_tp = TensorProduct(
                self.irreps_node_input,
                self.irreps_inter_relaxed,
                irreps_mid,
                instructions,
                internal_weights=False,
                shared_weights=False,
            )
            self.fc_relaxed = FullyConnectedNet(fc_neurons + [self.in_relaxed_tp.weight_numel], torch.nn.functional.silu)
        
        self.lin2 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_node_output)
        self.lin3 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, "0e")

            

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
        #print(node_input.shape)
        #print(node_attr.shape) 
        node_self_connection = self.sc(node_input, node_attr)
        node_features = self.lin1(node_input, node_attr)
        if self.relaxed:
            weight = self.fc_relaxed(edge_scalars)
            relaxed_weights = self.relaxed_edge_tp(self.relaxed_weights,edge_attr)
            # node size 1
            node_features = self.in_relaxed_tp(node_features, relaxed_weights, weight) 
            #new_edge_features = self.in_relaxed_tp(node_features, relaxed_weights, weight)
            #node_features = scatter(new_edge_features, edge_dst, dim=0, dim_size=node_input.shape[0]).div(self.num_neighbors**0.5)
        else:
            weight = self.fc(edge_scalars)
            edge_features = self.tp(node_features[edge_src],edge_attr,weight)
            node_features = scatter(edge_features, edge_dst, dim=0, dim_size=node_input.shape[0]).div(self.num_neighbors**0.5)

        node_conv_out = self.lin2(node_features, node_attr)
        node_angle = 0.1 * self.lin3(node_features, node_attr)
        #            ^^^------ start small, favor self-connection

        cos, sin = node_angle.cos(), node_angle.sin()
        m = self.sc.output_mask
        sin = (1 - m) + sin * m
        return cos * node_self_connection + sin * node_conv_out
