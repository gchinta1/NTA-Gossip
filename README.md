# NTA-Gossip

Official implementation of **NTA-Gossip: Learning When Not to Learn in Decentralised Federated Learning**.

NTA-Gossip is a decentralized federated learning approach designed to mitigate negative transfer during peer-to-peer model exchange. Each client evaluates an incoming peer contribution using local validation data and can reject updates that are potentially harmful.

## Datasets

Experiments are provided for:

- Fashion-MNIST
- CIFAR-10

The datasets are downloaded automatically through `torchvision`.

## Network Topologies

The implementation supports three decentralized network topologies:

- Ring
- Erdős-Rényi
- Barabási-Albert Scale-Free

## Methods

The following methods are included:

- Local
- Random Gossip
- Static Peer
- Reliability
- PENS
- NTA-Gossip Hard
- NTA-Gossip Soft
- Adaptive NTA-Gossip Hard
- Adaptive NTA-Gossip Soft
- D-PSGD
- MATCHA

D-PSGD and MATCHA are included as reference decentralized learning methods.

## Experimental Configuration

The main experimental configuration is shared across the two datasets:

| Parameter | Value |
|---|---:|
| Clients | 50 |
| Communication rounds | 100 |
| Local epochs | 1 |
| Batch size | 64 |
| Learning rate | 0.01 |
| Dirichlet alpha | 0.3 |
| Base mixing weight | 0.2 |
| Random seed | 42 |

Topology-specific parameters are:

- Ring: 2 neighbors per client
- Erdős-Rényi: expected degree 4
- Scale-Free: `m = 2`

## Repository Structure

```text
NTA-Gossip/
├── nta_gossip_fmnist.py
├── nta_gossip_cifar10.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/YOUR_USERNAME/NTA-Gossip.git
cd NTA-Gossip
pip install -r requirements.txt
```

## Running the Experiments

### Fashion-MNIST

```bash
python nta_gossip_fmnist.py
```

### CIFAR-10

```bash
python nta_gossip_cifar10.py
```

Each script executes the experiments for the supported network topologies and methods and stores the resulting metrics in CSV format.

## Requirements

- Python 3
- PyTorch
- torchvision
- NumPy
- scikit-learn

See `requirements.txt` for the required Python packages.

## Reproducibility

A fixed random seed is used to improve reproducibility. The implementation includes the dataset partitioning, decentralized topology generation, local training, peer selection, negative-transfer gating, reliability updates, model aggregation, and evaluation procedures required to reproduce the experiments.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{chinta_nta_gossip,
  title   = {NTA-Gossip: Learning When Not to Learn in Decentralised Federated Learning},
  author  = {Chinta, Gkentian and Xiao, Ke and Wang, Qiyuan and Kolomvatsos, Kostas and Anagnostopoulos, Christos},
  year    = {2026}
}
```

The complete publication information can be added once the final bibliographic details are available.
