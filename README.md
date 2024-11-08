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
