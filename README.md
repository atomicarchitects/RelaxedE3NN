# RelaxedE3NN
[GRaM at ICML'24] Relaxed Equivariant Graph Neural Networks.

This repo contains code for Relaxed Graph Equivariant Neural Networks (https://arxiv.org/abs/2407.20471). 

## Install

### Dependencies

#### PyTorch

e3nn requires PyTorch >=1.8.0 For installation instructions, please see the [PyTorch homepage](https://pytorch.org/).

#### torch_geometric

First you have to install [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric). For `torch` 1.11 and no CUDA support:

```bash
CUDA=cpu

pip install --upgrade --force-reinstall torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+${CUDA}.html
pip install --upgrade --force-reinstall torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+${CUDA}.html
pip install torch-geometric
```

See [here](https://github.com/rusty1s/pytorch_geometric#installation) to get cuda support or newer versions.

#### e3nn

#### Stable (PyPI)

```bash
$ pip install e3nn
```

## Organization

We provide code for two toy experiments. 
### Shape Deformations
shape_deformations_3d.ipynb contains symmetry breaking examples deforming a cube into a rectangular prism/an asymmetric shape. We demonstrate that the relaxed weights are interpretable through plotting their spherical harmonic projections (see the paper for more detail).
### Electric Field Simulation
electric_field_sim.ipynb contains an example learning the direction of the electric and magnetic force for a charged particle in a magnetic field.
### Models
relaxed_e3nn_conv.py contains a simple relaxed e3nn convolution model made of stacked RelaxedConvolutions. The electric_field_model folder contains an example of how to incorporate the relaxed e3nn layer into a more complicated model based on the sample networks contained in [e3nn](https://docs.e3nn.org/en/latest/). The files modified to contain the relaxed equivariant layer are specifically based SimpleNetwork and NetworkForAGraphWithAttributes in e3nn [here](https://github.com/e3nn/e3nn/blob/main/e3nn/nn/models/v2103/gate_points_networks.py). Within the electric_field_model, relaxed_points_conv.py modifies the [points convolution](https://github.com/e3nn/e3nn/blob/main/e3nn/nn/models/v2103/points_convolution.py) in e3nn. gate_points_message_passing_relaxed.py modifies the [message passing](https://github.com/e3nn/e3nn/blob/main/e3nn/nn/models/v2103/gate_points_message_passing.py) to use the relaxed convolution. gate_points_networks_relaxed.py modifies the sample message passing graph neural network models in e3nn to use the relaxed convolution layer.
