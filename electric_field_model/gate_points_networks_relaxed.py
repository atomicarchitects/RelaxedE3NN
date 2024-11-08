"""
>>> test_simple_network()
>>> test_network_for_a_graph_with_attributes()
"""
from typing import Dict, Union

import torch
from torch_cluster import radius_graph
from torch_geometric.data import Data
from torch_scatter import scatter

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from .gate_points_message_passing_relaxed import MessagePassing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SimpleNetwork(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
        irreps_out,
        max_radius,
        num_neighbors: int,
        num_nodes: int,
        mul: int = 50,
        layers: int = 1,
        lmax: int = 2,
        pool_nodes: bool = True,
        relaxed: bool = False,
    ) -> None:
        super().__init__()

        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = 1
        self.num_nodes = num_nodes
        self.pool_nodes = pool_nodes
        self.relaxed = relaxed

        irreps_node_hidden = o3.Irreps([(mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])

        self.mp = MessagePassing(
            irreps_node_input=irreps_in,
            irreps_node_hidden=irreps_node_hidden,
            irreps_node_output=irreps_out,
            irreps_node_attr="0e",
            irreps_edge_attr=o3.Irreps.spherical_harmonics(lmax),
            layers=layers,
            fc_neurons=[self.number_of_basis, 100],
            num_neighbors=num_neighbors,
            relaxed = relaxed
        )

        self.irreps_in = self.mp.irreps_node_input
        self.irreps_out = self.mp.irreps_node_output

    def preprocess(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if "batch" in data:
            batch = data["batch"]
        else:
            batch = data["pos"].new_zeros(data["pos"].shape[0], dtype=torch.long)

        # Create graph
        edge_index = radius_graph(data["pos"], self.max_radius, batch, max_num_neighbors=len(data["pos"]) - 1)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        # Edge attributes
        edge_vec = data["pos"][edge_src] - data["pos"][edge_dst]
        return batch, data["x"], edge_src, edge_dst, edge_vec

    def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        batch, node_inputs, edge_src, edge_dst, edge_vec = self.preprocess(data)
        del data

        edge_attr = o3.spherical_harmonics(range(self.lmax + 1), edge_vec, True, normalization="component")

        # Edge length embedding
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis="cosine",  # the cosine basis with cutoff = True goes to zero at max_radius
            cutoff=True,  # no need for an additional smooth cutoff
        ).mul(self.number_of_basis**0.5)

        # Node attributes are not used here
        node_attr = node_inputs.new_ones(node_inputs.shape[0], 1)
        node_outputs = self.mp(node_inputs, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)

        if self.pool_nodes:
            return scatter(node_outputs, batch, dim=0).div(self.num_nodes**0.5)
        else:
            return node_outputs


class NetworkForAGraphWithAttributes(torch.nn.Module):
    def __init__(
        self,
        irreps_node_input: o3.Irreps,
        irreps_node_attr: o3.Irreps,
        irreps_node_output: o3.Irreps,
        max_radius,
        num_neighbors: int,
        num_nodes: int,
        mul: int = 50,
        layers: int = 3,
        lmax: int = 2,
        pool_nodes: bool = True,
        relaxed: bool = False,
    ) -> None:
        super().__init__()

        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = 1
        self.num_nodes = num_nodes
        self.irreps_node_input = o3.Irreps("1x0e+1x0o+1x1o+1x1e+1x2e+1x2o")
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)
        self.pool_nodes = pool_nodes
        self.relaxed = relaxed
        irreps_node_hidden = o3.Irreps([(mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])

        self.mp = MessagePassing(
            irreps_node_input=self.irreps_node_input,
            irreps_node_hidden=irreps_node_hidden,
            irreps_node_output=irreps_node_output,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=o3.Irreps.spherical_harmonics(lmax),
            layers=layers,
            fc_neurons=[self.number_of_basis, 100],
            num_neighbors=num_neighbors,
            relaxed = relaxed,
        )

        self.irreps_node_input = self.mp.irreps_node_input
        self.irreps_node_attr = self.mp.irreps_node_attr
        self.irreps_node_output = self.mp.irreps_node_output

    def preprocess(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if "batch" in data:
            batch = data["batch"]
        else:
            batch = data["pos"].new_zeros(data["pos"].shape[0], dtype=torch.long)

        # Create graph
        edge_index = radius_graph(data["pos"], self.max_radius, batch, max_num_neighbors=len(data["pos"]) - 1)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        # Edge attributes
        edge_vec = data["pos"][edge_src] - data["pos"][edge_dst]

        return batch, edge_src, edge_dst, edge_vec, data["node_attr"]
    
    def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        batch, edge_src, edge_dst, edge_vec, node_attr = self.preprocess(data)
        del data
        '''
        edge_attr = o3.spherical_harmonics(range(self.lmax + 1), edge_vec, True, normalization="component")
        #edge_attr = torch.cat([data["edge_attr"], edge_sh], dim=1)
        print(edge_attr)
        # Edge length embedding
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis="cosine",  # the cosine basis with cutoff = True goes to zero at max_radius
            cutoff=True,  # no need for an additional smooth cutoff
        ).mul(self.number_of_basis**0.5)
        '''

        # no node inputs
        node_inputs = torch.zeros(self.irreps_node_input.dim)
        node_inputs[0] = 1
        node_inputs = node_inputs.repeat(node_attr.shape[0], 1).to(device)
        edge_src, edge_dst = torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
        #edge_attr = torch.tensor([[1.0]], dtype=torch.float64).to(device)
        edge_attr = torch.eye(1, self.irreps_edge_attr.dim, dtype=torch.float64).to(device)
        edge_length_embedding = torch.tensor([[1.0]], dtype=torch.float64).to(device)
        node_outputs = self.mp(node_inputs, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)

        if self.pool_nodes:
            return scatter(node_outputs, batch, dim=0).div(self.num_nodes**0.5)
        else:
            return node_outputs


def test_simple_network() -> None:
    net = SimpleNetwork("3x0e + 2x1o", "4x0e + 1x1o", max_radius=2.0, num_neighbors=3.0, num_nodes=5.0)

    net({"pos": torch.randn(5, 3), "x": net.irreps_in.randn(5, -1)})


def test_network_for_a_graph_with_attributes() -> None:
    net = NetworkForAGraphWithAttributes(
        "3x0e + 2x1o", "4x0e + 1x1o", "1e", "3x0o + 1e", max_radius=2.0, num_neighbors=3.0, num_nodes=5.0
    )

    net(
        {
            "pos": torch.randn(3, 3),
            "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
            "node_input": net.irreps_node_input.randn(3, -1),
            "node_attr": net.irreps_node_attr.randn(3, -1),
            "edge_attr": net.irreps_edge_attr.randn(3, -1),
        }
    )
