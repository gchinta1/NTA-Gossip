import copy
import csv
import math
import random
from collections import defaultdict, deque

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
NUM_CLIENTS = 50
ROUNDS = 100
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01

DATASET_NAME = "fashion_mnist"

DIRICHLET_ALPHA = 0.3

MIX_ALPHA = 0.20

DPSGD_MIX_ALPHA = 0.20
MATCHA_MIX_ALPHA = 0.20

GATE_TAU = 0.005
NT_TAU = 0.01

NTA_WARMUP_ROUNDS = 0

RELIABILITY_BETA = 0.20
RELIABILITY_TEMPERATURE = 1.0
RELIABILITY_EPSILON = 0.10

RHO_MIN = 0.30
RELIABILITY_LAMBDA = 5.0

SOFT_GATE_GAMMA = 5.0

VAL_RATIO = 0.20

ER_PROB = 4.0 / (NUM_CLIENTS - 1)
SCALE_FREE_M = 2

OUTPUT_CSV = "fmnist_all_topologies_all_methods.csv"
DATASETS = ["fashion_mnist"]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, image_size=28, num_classes=10):
        super().__init__()

        feature_size = image_size // 4

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * feature_size * feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)

def get_targets(dataset):
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if isinstance(targets, list):
            return np.asarray(targets, dtype=np.int64)
        if torch.is_tensor(targets):
            return targets.cpu().numpy().astype(np.int64)
        return np.asarray(targets, dtype=np.int64)

    raise ValueError("Dataset has no targets attribute.")

def dirichlet_split(dataset, num_clients, alpha):
    labels = get_targets(dataset)
    num_classes = int(labels.max() + 1)

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet([alpha] * num_clients)
        split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        class_splits = np.split(class_indices, split_points)

        for client_id, split in enumerate(class_splits):
            client_indices[client_id].extend(split.tolist())

    for k in range(num_clients):
        random.shuffle(client_indices[k])

    return client_indices

def make_loaders(dataset, client_indices):
    train_loaders = []
    val_loaders = []

    kept_clients = 0

    for indices in client_indices:
        if len(indices) < 2:
            continue

        subset = Subset(dataset, indices)

        val_size = max(1, int(VAL_RATIO * len(subset)))
        train_size = len(subset) - val_size

        if train_size < 1:
            continue

        train_set, val_set = random_split(
            subset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(SEED + kept_clients),
        )

        train_loaders.append(DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True))
        val_loaders.append(DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False))

        kept_clients += 1

    return train_loaders, val_loaders

def load_dataset_once():
    name = DATASET_NAME.lower()

    if name in ["fashion_mnist", "fmnist", "f-mnist"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])

        train_dataset = datasets.FashionMNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform,
        )

        test_dataset = datasets.FashionMNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform,
        )

        in_channels = 1
        image_size = 28

    elif name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        train_dataset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform,
        )

        test_dataset = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform,
        )

        in_channels = 1
        image_size = 28

    elif name in ["cifar10", "cifar-10"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ])

        train_dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transform,
        )

        test_dataset = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transform,
        )

        in_channels = 3
        image_size = 32

    else:
        raise ValueError(f"Unknown DATASET_NAME: {DATASET_NAME}")

    client_indices = dirichlet_split(
        train_dataset,
        NUM_CLIENTS,
        DIRICHLET_ALPHA,
    )

    train_loaders, val_loaders = make_loaders(train_dataset, client_indices)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    return train_loaders, val_loaders, test_loader, in_channels, image_size

def ring_neighbors(num_clients):
    """Undirected ring: exactly two neighbours per client."""
    return {k: [(k - 1) % num_clients, (k + 1) % num_clients] for k in range(num_clients)}

def connected_components(adjacency):
    visited = set()
    components = []

    for start in adjacency:
        if start in visited:
            continue

        queue = deque([start])
        visited.add(start)
        component = []

        while queue:
            node = queue.popleft()
            component.append(node)

            for neigh in adjacency[node]:
                if neigh not in visited:
                    visited.add(neigh)
                    queue.append(neigh)

        components.append(component)

    return components

def er_neighbors(num_clients, p=0.25):
    """
    Erdos-Renyi undirected topology.
    Ensures no isolated clients and connects components if needed.
    """
    adjacency = {k: set() for k in range(num_clients)}

    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            if random.random() < p:
                adjacency[i].add(j)
                adjacency[j].add(i)

    for k in range(num_clients):
        if len(adjacency[k]) == 0:
            candidates = [x for x in range(num_clients) if x != k]
            j = random.choice(candidates)
            adjacency[k].add(j)
            adjacency[j].add(k)

    components = connected_components(adjacency)
    while len(components) > 1:
        a = random.choice(components[0])
        b = random.choice(components[1])
        adjacency[a].add(b)
        adjacency[b].add(a)
        components = connected_components(adjacency)

    return {k: sorted(list(v)) for k, v in adjacency.items()}

def scale_free_neighbors(num_clients, m=2):
    """
    Barabasi-Albert style scale-free undirected topology without networkx.
    Starts from a clique of size m+1 and adds nodes preferentially by degree.
    """
    if m < 1:
        raise ValueError("SCALE_FREE_M must be at least 1.")
    if m >= num_clients:
        raise ValueError("SCALE_FREE_M must be smaller than NUM_CLIENTS.")

    adjacency = {k: set() for k in range(num_clients)}

    initial_size = m + 1
    for i in range(initial_size):
        for j in range(i + 1, initial_size):
            adjacency[i].add(j)
            adjacency[j].add(i)

    for new_node in range(initial_size, num_clients):
        existing_nodes = list(range(new_node))
        degrees = np.array([len(adjacency[v]) for v in existing_nodes], dtype=np.float64)

        if degrees.sum() <= 0:
            probs = np.ones(len(existing_nodes), dtype=np.float64) / len(existing_nodes)
        else:
            probs = degrees / degrees.sum()

        chosen = np.random.choice(existing_nodes, size=m, replace=False, p=probs)

        for old_node in chosen:
            old_node = int(old_node)
            adjacency[new_node].add(old_node)
            adjacency[old_node].add(new_node)

    return {k: sorted(list(v)) for k, v in adjacency.items()}

def make_static_peers(neighbors):
    return {k: neighbors[k][0] for k in neighbors}

def topology_degree_stats(neighbors):
    degrees = [len(v) for v in neighbors.values()]
    return min(degrees), float(np.mean(degrees)), max(degrees)

def maximal_random_matching(neighbors, num_clients):
    """
    MATCHA-style greedy random maximal matching over graph edges.
    This is a matching-based reference, not the full official MATCHA optimizer.
    """
    edges = []
    for i in range(num_clients):
        for j in neighbors[i]:
            if i < j:
                edges.append((i, j))

    random.shuffle(edges)

    used = set()
    pairs = []

    for i, j in edges:
        if i not in used and j not in used:
            pairs.append((i, j))
            used.add(i)
            used.add(j)

    return pairs

def train_local(model, loader):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

def validation_loss(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            total_count += x.size(0)

    return total_loss / max(total_count, 1)

def evaluate(model, loader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return acc, f1

def evaluate_all_clients(models, test_loader):
    accs = []
    f1s = []

    for model in models:
        acc, f1 = evaluate(model, test_loader)
        accs.append(acc)
        f1s.append(f1)

    return {
        "accuracy": float(np.mean(accs)),
        "macro_f1": float(np.mean(f1s)),
        "worst_accuracy": float(np.min(accs)),
    }

def mix_models(model_a, model_b, alpha):
    alpha = float(max(0.0, min(1.0, alpha)))

    mixed = copy.deepcopy(model_a)

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    state_mixed = mixed.state_dict()

    with torch.no_grad():
        for key in state_a:
            if "num_batches_tracked" in key:
                state_mixed[key] = state_a[key]
            else:
                state_mixed[key] = (1.0 - alpha) * state_a[key] + alpha * state_b[key]

    mixed.load_state_dict(state_mixed)
    return mixed.to(DEVICE)

def average_neighbor_models(local_model, neighbor_models, alpha):
    """
    D-PSGD-style full-neighbour averaging.
    theta_new = (1-alpha) theta_local + alpha average(theta_neighbors)
    """
    alpha = float(max(0.0, min(1.0, alpha)))
    mixed = copy.deepcopy(local_model)

    state_local = local_model.state_dict()
    state_mixed = mixed.state_dict()
    neighbor_states = [m.state_dict() for m in neighbor_models]

    with torch.no_grad():
        for key in state_local:
            if "num_batches_tracked" in key:
                state_mixed[key] = state_local[key]
            else:
                avg_neighbor = torch.stack(
                    [ns[key].float() for ns in neighbor_states],
                    dim=0
                ).mean(dim=0)
                state_mixed[key] = (1.0 - alpha) * state_local[key] + alpha * avg_neighbor

    mixed.load_state_dict(state_mixed)
    return mixed.to(DEVICE)

def sigmoid(x):
    x = max(min(float(x), 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-x))

def reliability_factor(score):
    return RHO_MIN + (1.0 - RHO_MIN) * sigmoid(RELIABILITY_LAMBDA * score)

def reliability_peer(k, neighbors, reliability):
    candidate_peers = neighbors[k]
    scores = np.array([reliability[k][j] for j in candidate_peers], dtype=np.float64)

    scaled = scores / max(RELIABILITY_TEMPERATURE, 1e-12)
    scaled = scaled - np.max(scaled)

    exp_scores = np.exp(scaled)
    softmax_probs = exp_scores / np.sum(exp_scores)

    uniform_probs = np.ones(len(candidate_peers), dtype=np.float64) / len(candidate_peers)
    probs = (1.0 - RELIABILITY_EPSILON) * softmax_probs + RELIABILITY_EPSILON * uniform_probs

    return int(np.random.choice(candidate_peers, p=probs))

def pens_one_peer_select(k, neighbors, local_model, local_models, val_loader):
    before_loss = validation_loss(local_model, val_loader)

    best_peer = None
    best_delta = -float("inf")

    for peer in neighbors[k]:
        candidate_model = mix_models(local_model, local_models[peer], MIX_ALPHA)
        after_loss = validation_loss(candidate_model, val_loader)
        delta = before_loss - after_loss

        if delta > best_delta:
            best_delta = delta
            best_peer = peer

    return best_peer, best_delta, len(neighbors[k])

def soft_gate_alpha(delta):
    return MIX_ALPHA * sigmoid(SOFT_GATE_GAMMA * (delta + GATE_TAU))

def adaptive_soft_gate_alpha(delta, reliability_score):
    rho = reliability_factor(reliability_score)
    return MIX_ALPHA * sigmoid(SOFT_GATE_GAMMA * (delta + GATE_TAU)) * rho

def adaptive_hard_alpha(reliability_score):
    rho = reliability_factor(reliability_score)
    return MIX_ALPHA * rho

def entropy_from_counts(counts):
    total = sum(counts)

    if total <= 0:
        return 0.0

    entropy = 0.0

    for count in counts:
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)

    return entropy

def mean_selection_entropy(selection_counts, neighbors):
    entropies = []

    for k in neighbors:
        counts = [selection_counts[k][j] for j in neighbors[k]]
        entropies.append(entropy_from_counts(counts))

    return float(np.mean(entropies)) if entropies else 0.0

def method_display_name(method):
    names = {
        "local": "Local only",
        "random_gossip": "Random gossip",
        "static_peer": "Static peer",
        "reliability_only": "Reliability only",
        "pens_one_peer": "PENS-style one-peer",
        "nta_hard": "NTA-Gossip hard",
        "nta_soft": "NTA-Gossip soft",
        "adaptive_nta_hard": "Adaptive NTA-Gossip hard",
        "adaptive_nta_soft": "Adaptive NTA-Gossip soft",
        "dpsgd_full": "D-PSGD full-neighbour",
        "matcha_matching": "MATCHA-style matching",
    }
    return names[method]

def communication_regime(method):
    if method == "local":
        return "No communication"
    if method == "dpsgd_full":
        return "Full-neighbour reference"
    if method == "matcha_matching":
        return "Matching reference"
    return "One-peer budget"

def choose_peer(method, k, neighbors, reliability, local_model, local_models, val_loader, static_peers):
    candidate_cost = 0

    if method == "random_gossip":
        peer = random.choice(neighbors[k])

    elif method == "static_peer":
        peer = static_peers[k]

    elif method in [
        "reliability_only",
        "nta_hard",
        "nta_soft",
        "adaptive_nta_hard",
        "adaptive_nta_soft",
    ]:
        peer = reliability_peer(k, neighbors, reliability)

    elif method == "pens_one_peer":
        peer, _, candidate_cost = pens_one_peer_select(
            k,
            neighbors,
            local_model,
            local_models,
            val_loader,
        )

    else:
        raise ValueError(f"Unknown one-peer method: {method}")

    return peer, candidate_cost

def run_one_peer_experiment(
    method,
    topology_name,
    train_loaders,
    val_loaders,
    test_loader,
    neighbors,
    static_peers,
    initial_models,
):
    actual_clients = len(train_loaders)
    models = [copy.deepcopy(m).to(DEVICE) for m in initial_models]

    reliability = {
        k: {j: 0.0 for j in neighbors[k]}
        for k in range(actual_clients)
    }

    selection_counts = {
        k: defaultdict(int)
        for k in range(actual_clients)
    }

    total_proposed_nt = 0
    total_accepted_nt = 0
    total_rejections = 0.0
    total_exchanges = 0
    total_candidate_evaluations = 0
    total_alpha_sum = 0.0
    total_alpha_count = 0

    min_deg, mean_deg, max_deg = topology_degree_stats(neighbors)

    print("\n==============================")
    print(f"Running method: {method_display_name(method)}")
    print(f"Topology: {topology_name}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Clients: {actual_clients}")
    print(f"Degree min/mean/max: {min_deg}/{mean_deg:.2f}/{max_deg}")
    print(f"Device: {DEVICE}")
    print("==============================")

    for round_id in range(ROUNDS):
        local_models = []

        for k in range(actual_clients):
            model_copy = copy.deepcopy(models[k])
            train_local(model_copy, train_loaders[k])
            local_models.append(model_copy)

        new_models = []

        round_proposed_nt = 0
        round_accepted_nt = 0
        round_rejections = 0.0
        round_exchanges = 0
        round_alpha_sum = 0.0
        round_alpha_count = 0

        for k in range(actual_clients):
            local_model = local_models[k]

            peer, candidate_cost = choose_peer(
                method=method,
                k=k,
                neighbors=neighbors,
                reliability=reliability,
                local_model=local_model,
                local_models=local_models,
                val_loader=val_loaders[k],
                static_peers=static_peers,
            )

            total_candidate_evaluations += candidate_cost
            selection_counts[k][peer] += 1

            peer_model = local_models[peer]
            baseline_mixed_model = mix_models(local_model, peer_model, MIX_ALPHA)

            before_loss = validation_loss(local_model, val_loaders[k])
            after_loss = validation_loss(baseline_mixed_model, val_loaders[k])
            delta = before_loss - after_loss

            proposed_negative = delta < -NT_TAU

            total_exchanges += 1
            round_exchanges += 1

            if proposed_negative:
                total_proposed_nt += 1
                round_proposed_nt += 1

            if method in [
                "reliability_only",
                "nta_hard",
                "nta_soft",
                "adaptive_nta_hard",
                "adaptive_nta_soft",
            ]:
                reliability[k][peer] = (
                    (1.0 - RELIABILITY_BETA) * reliability[k][peer]
                    + RELIABILITY_BETA * delta
                )

            if method in ["random_gossip", "static_peer", "reliability_only", "pens_one_peer"]:
                new_models.append(baseline_mixed_model)

                if proposed_negative:
                    total_accepted_nt += 1
                    round_accepted_nt += 1

            elif method == "nta_hard":
                accepted = True if round_id < NTA_WARMUP_ROUNDS else delta >= -GATE_TAU

                if accepted:
                    new_models.append(baseline_mixed_model)

                    if proposed_negative:
                        total_accepted_nt += 1
                        round_accepted_nt += 1
                else:
                    new_models.append(local_model)
                    total_rejections += 1.0
                    round_rejections += 1.0

            elif method == "nta_soft":
                if proposed_negative and GATE_TAU <= NT_TAU:
                    alpha_t = 0.0
                elif round_id < NTA_WARMUP_ROUNDS:
                    alpha_t = MIX_ALPHA
                else:
                    alpha_t = soft_gate_alpha(delta)

                new_models.append(mix_models(local_model, peer_model, alpha_t))

                total_alpha_sum += alpha_t
                total_alpha_count += 1
                round_alpha_sum += alpha_t
                round_alpha_count += 1

                fractional_reject = 1.0 - (alpha_t / MIX_ALPHA)
                total_rejections += fractional_reject
                round_rejections += fractional_reject

                if proposed_negative and alpha_t > 0.0:
                    total_accepted_nt += 1
                    round_accepted_nt += 1

            elif method == "adaptive_nta_hard":
                accepted = True if round_id < NTA_WARMUP_ROUNDS else delta >= -GATE_TAU

                if accepted:
                    alpha_t = adaptive_hard_alpha(reliability[k][peer])
                    new_models.append(mix_models(local_model, peer_model, alpha_t))

                    total_alpha_sum += alpha_t
                    total_alpha_count += 1
                    round_alpha_sum += alpha_t
                    round_alpha_count += 1

                    if proposed_negative:
                        total_accepted_nt += 1
                        round_accepted_nt += 1
                else:
                    new_models.append(local_model)
                    total_rejections += 1.0
                    round_rejections += 1.0

            elif method == "adaptive_nta_soft":
                if proposed_negative and GATE_TAU <= NT_TAU:
                    alpha_t = 0.0
                elif round_id < NTA_WARMUP_ROUNDS:
                    alpha_t = MIX_ALPHA
                else:
                    alpha_t = adaptive_soft_gate_alpha(delta, reliability[k][peer])

                new_models.append(mix_models(local_model, peer_model, alpha_t))

                total_alpha_sum += alpha_t
                total_alpha_count += 1
                round_alpha_sum += alpha_t
                round_alpha_count += 1

                fractional_reject = 1.0 - (alpha_t / MIX_ALPHA)
                total_rejections += fractional_reject
                round_rejections += fractional_reject

                if proposed_negative and alpha_t > 0.0:
                    total_accepted_nt += 1
                    round_accepted_nt += 1

            else:
                raise ValueError(f"Unknown method: {method}")

        models = new_models

        if round_accepted_nt > round_proposed_nt:
            raise RuntimeError(
                f"Invalid metric in round {round_id+1}: "
                f"NTRacc count {round_accepted_nt} > NTRprop count {round_proposed_nt}"
            )

        if total_accepted_nt > total_proposed_nt:
            raise RuntimeError(
                f"Invalid cumulative metric: "
                f"NTRacc count {total_accepted_nt} > NTRprop count {total_proposed_nt}"
            )

        if GATE_TAU <= NT_TAU and NTA_WARMUP_ROUNDS == 0 and method in ["nta_hard", "adaptive_nta_hard"]:
            if total_accepted_nt != 0:
                raise RuntimeError(
                    f"Theorem violation for {method}: "
                    f"GATE_TAU <= NT_TAU but accepted NT count is {total_accepted_nt}"
                )

        metrics = evaluate_all_clients(models, test_loader)

        if round_exchanges > 0:
            round_ntr_prop = round_proposed_nt / round_exchanges
            round_ntr_acc = round_accepted_nt / round_exchanges
            round_reject = round_rejections / round_exchanges
        else:
            round_ntr_prop = 0.0
            round_ntr_acc = 0.0
            round_reject = 0.0

        msg = (
            f"Round {round_id + 1:03d} | "
            f"Mean Acc: {metrics['accuracy']:.4f} | "
            f"Worst Acc: {metrics['worst_accuracy']:.4f} | "
            f"Macro-F1: {metrics['macro_f1']:.4f} | "
            f"NTRprop: {round_ntr_prop:.4f} | "
            f"NTRacc: {round_ntr_acc:.4f} | "
            f"Reject: {round_reject:.4f}"
        )

        if round_alpha_count > 0:
            msg += f" | AvgAlpha: {round_alpha_sum / round_alpha_count:.4f}"

        print(msg)

    final_metrics = evaluate_all_clients(models, test_loader)

    if total_exchanges > 0:
        ntr_prop = total_proposed_nt / total_exchanges
        ntr_acc = total_accepted_nt / total_exchanges
        rejection_rate = total_rejections / total_exchanges
        c_model = 1.0
        c_cand = total_candidate_evaluations / total_exchanges
        h_sel = mean_selection_entropy(selection_counts, neighbors)
    else:
        ntr_prop = None
        ntr_acc = None
        rejection_rate = None
        c_model = 0.0
        c_cand = 0.0
        h_sel = None

    avg_alpha = None
    if total_alpha_count > 0:
        avg_alpha = total_alpha_sum / total_alpha_count

    if ntr_prop is not None and ntr_acc is not None and ntr_acc > ntr_prop + 1e-12:
        raise RuntimeError(f"Final invalid metric: NTRacc={ntr_acc} > NTRprop={ntr_prop}")

    if GATE_TAU <= NT_TAU and NTA_WARMUP_ROUNDS == 0 and method in ["nta_hard", "adaptive_nta_hard"]:
        if total_accepted_nt != 0:
            raise RuntimeError(
                f"Final theorem violation for {method}: accepted NT count {total_accepted_nt}"
            )

    return {
        "topology": topology_name,
        "method": method,
        "communication_regime": communication_regime(method),
        "accuracy": final_metrics["accuracy"],
        "macro_f1": final_metrics["macro_f1"],
        "worst_accuracy": final_metrics["worst_accuracy"],
        "ntr_prop": ntr_prop,
        "ntr_acc": ntr_acc,
        "rejection_rate": rejection_rate,
        "c_model": c_model,
        "total_model_exchanges": total_exchanges,
        "c_cand": c_cand,
        "selection_entropy": h_sel,
        "avg_alpha": avg_alpha,
    }

def run_local_experiment(topology_name, train_loaders, test_loader, initial_models):
    actual_clients = len(train_loaders)
    models = [copy.deepcopy(m).to(DEVICE) for m in initial_models]

    print("\n==============================")
    print("Running method: Local only")
    print(f"Topology: {topology_name}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Clients: {actual_clients}")
    print(f"Device: {DEVICE}")
    print("==============================")

    for round_id in range(ROUNDS):
        new_models = []

        for k in range(actual_clients):
            model_copy = copy.deepcopy(models[k])
            train_local(model_copy, train_loaders[k])
            new_models.append(model_copy)

        models = new_models
        metrics = evaluate_all_clients(models, test_loader)

        print(
            f"Round {round_id + 1:03d} | "
            f"Mean Acc: {metrics['accuracy']:.4f} | "
            f"Worst Acc: {metrics['worst_accuracy']:.4f} | "
            f"Macro-F1: {metrics['macro_f1']:.4f}"
        )

    final_metrics = evaluate_all_clients(models, test_loader)

    return {
        "topology": topology_name,
        "method": "local",
        "communication_regime": communication_regime("local"),
        "accuracy": final_metrics["accuracy"],
        "macro_f1": final_metrics["macro_f1"],
        "worst_accuracy": final_metrics["worst_accuracy"],
        "ntr_prop": None,
        "ntr_acc": None,
        "rejection_rate": None,
        "c_model": 0.0,
        "total_model_exchanges": 0,
        "c_cand": 0.0,
        "selection_entropy": None,
        "avg_alpha": None,
    }

def run_dpsgd_full_experiment(topology_name, train_loaders, test_loader, neighbors, initial_models):
    actual_clients = len(train_loaders)
    models = [copy.deepcopy(m).to(DEVICE) for m in initial_models]

    total_model_exchanges = 0
    min_deg, mean_deg, max_deg = topology_degree_stats(neighbors)

    print("\n==============================")
    print("Running method: D-PSGD full-neighbour")
    print(f"Topology: {topology_name}")
    print(f"Dataset: {DATASET_NAME}")
    print("Reference baseline: not one-peer budget-matched")
    print(f"Degree min/mean/max: {min_deg}/{mean_deg:.2f}/{max_deg}")
    print(f"Clients: {actual_clients}")
    print(f"Device: {DEVICE}")
    print("==============================")

    for round_id in range(ROUNDS):
        local_models = []

        for k in range(actual_clients):
            model_copy = copy.deepcopy(models[k])
            train_local(model_copy, train_loaders[k])
            local_models.append(model_copy)

        new_models = []

        for k in range(actual_clients):
            neighbor_models = [local_models[j] for j in neighbors[k]]
            new_model = average_neighbor_models(
                local_models[k],
                neighbor_models,
                DPSGD_MIX_ALPHA,
            )
            new_models.append(new_model)
            total_model_exchanges += len(neighbors[k])

        models = new_models
        metrics = evaluate_all_clients(models, test_loader)

        print(
            f"Round {round_id + 1:03d} | "
            f"Mean Acc: {metrics['accuracy']:.4f} | "
            f"Worst Acc: {metrics['worst_accuracy']:.4f} | "
            f"Macro-F1: {metrics['macro_f1']:.4f}"
        )

    final_metrics = evaluate_all_clients(models, test_loader)

    c_model = total_model_exchanges / max(actual_clients * ROUNDS, 1)

    return {
        "topology": topology_name,
        "method": "dpsgd_full",
        "communication_regime": communication_regime("dpsgd_full"),
        "accuracy": final_metrics["accuracy"],
        "macro_f1": final_metrics["macro_f1"],
        "worst_accuracy": final_metrics["worst_accuracy"],
        "ntr_prop": None,
        "ntr_acc": None,
        "rejection_rate": None,
        "c_model": float(c_model),
        "total_model_exchanges": total_model_exchanges,
        "c_cand": 0.0,
        "selection_entropy": None,
        "avg_alpha": None,
    }

def run_matcha_matching_experiment(topology_name, train_loaders, test_loader, neighbors, initial_models):
    actual_clients = len(train_loaders)
    models = [copy.deepcopy(m).to(DEVICE) for m in initial_models]

    total_model_exchanges = 0
    min_deg, mean_deg, max_deg = topology_degree_stats(neighbors)

    print("\n==============================")
    print("Running method: MATCHA-style matching")
    print(f"Topology: {topology_name}")
    print(f"Dataset: {DATASET_NAME}")
    print("Reference baseline: greedy matching, not identical to official MATCHA")
    print(f"Degree min/mean/max: {min_deg}/{mean_deg:.2f}/{max_deg}")
    print(f"Clients: {actual_clients}")
    print(f"Device: {DEVICE}")
    print("==============================")

    for round_id in range(ROUNDS):
        local_models = []

        for k in range(actual_clients):
            model_copy = copy.deepcopy(models[k])
            train_local(model_copy, train_loaders[k])
            local_models.append(model_copy)

        new_models = [copy.deepcopy(m).to(DEVICE) for m in local_models]

        pairs = maximal_random_matching(neighbors, actual_clients)

        for a, b in pairs:
            new_models[a] = mix_models(local_models[a], local_models[b], MATCHA_MIX_ALPHA)
            new_models[b] = mix_models(local_models[b], local_models[a], MATCHA_MIX_ALPHA)
            total_model_exchanges += 2

        models = new_models
        metrics = evaluate_all_clients(models, test_loader)

        print(
            f"Round {round_id + 1:03d} | "
            f"Mean Acc: {metrics['accuracy']:.4f} | "
            f"Worst Acc: {metrics['worst_accuracy']:.4f} | "
            f"Macro-F1: {metrics['macro_f1']:.4f}"
        )

    final_metrics = evaluate_all_clients(models, test_loader)

    c_model = total_model_exchanges / max(actual_clients * ROUNDS, 1)

    return {
        "topology": topology_name,
        "method": "matcha_matching",
        "communication_regime": communication_regime("matcha_matching"),
        "accuracy": final_metrics["accuracy"],
        "macro_f1": final_metrics["macro_f1"],
        "worst_accuracy": final_metrics["worst_accuracy"],
        "ntr_prop": None,
        "ntr_acc": None,
        "rejection_rate": None,
        "c_model": float(c_model),
        "total_model_exchanges": total_model_exchanges,
        "c_cand": 0.0,
        "selection_entropy": None,
        "avg_alpha": None,
    }

def format_metric(value, decimals=4):
    if value is None:
        return "--"
    return f"{float(value):.{decimals}f}"

def print_overleaf_table(results):
    print("\n\nMAIN TABLE VALUES FOR OVERLEAF")
    print(
        "Topology | Method | Accuracy | Macro-F1 | Worst Acc. | "
        "NTRprop | NTRacc | Rejection Rate | C_model | C_cand | H_sel | AvgAlpha"
    )

    for r in results:
        print(
            f"{r['topology']} | "
            f"{method_display_name(r['method'])} | "
            f"{r['accuracy']:.4f} | "
            f"{r['macro_f1']:.4f} | "
            f"{r['worst_accuracy']:.4f} | "
            f"{format_metric(r['ntr_prop'])} | "
            f"{format_metric(r['ntr_acc'])} | "
            f"{format_metric(r['rejection_rate'])} | "
            f"{format_metric(r['c_model'])} | "
            f"{format_metric(r['c_cand'])} | "
            f"{format_metric(r['selection_entropy'])} | "
            f"{format_metric(r['avg_alpha'])}"
        )

    print("\nLaTeX rows")
    for r in results:
        print(
            f"{r['topology']} & "
            f"{method_display_name(r['method'])} & "
            f"{r['accuracy']:.3f} & "
            f"{r['macro_f1']:.3f} & "
            f"{r['worst_accuracy']:.3f} & "
            f"{format_metric(r['ntr_prop'], 3)} & "
            f"{format_metric(r['ntr_acc'], 3)} & "
            f"{format_metric(r['rejection_rate'], 3)} & "
            f"{format_metric(r['c_model'], 2)} & "
            f"{format_metric(r['c_cand'], 2)} & "
            f"{format_metric(r['selection_entropy'], 3)} & "
            f"{format_metric(r['avg_alpha'], 3)} \\\\"
        )

def print_reference_table(results):
    ref_methods = {"dpsgd_full", "matcha_matching"}
    refs = [r for r in results if r["method"] in ref_methods]

    if not refs:
        return

    print("\n\nREFERENCE BASELINE TABLE FOR OVERLEAF")
    print("Topology | Method | Communication regime | C_model | Accuracy | Macro-F1 | Worst Acc.")

    for r in refs:
        print(
            f"{r['topology']} | "
            f"{method_display_name(r['method'])} | "
            f"{r['communication_regime']} | "
            f"{format_metric(r['c_model'], 2)} | "
            f"{r['accuracy']:.4f} | "
            f"{r['macro_f1']:.4f} | "
            f"{r['worst_accuracy']:.4f}"
        )

    print("\nLaTeX reference rows")
    for r in refs:
        print(
            f"{r['topology']} & "
            f"{method_display_name(r['method'])} & "
            f"{r['communication_regime']} & "
            f"{format_metric(r['c_model'], 2)} & "
            f"{r['accuracy']:.3f} & "
            f"{r['macro_f1']:.3f} & "
            f"{r['worst_accuracy']:.3f} \\\\"
        )

def save_results_csv(results, filename):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Topology",
            "Method",
            "Communication Regime",
            "Accuracy",
            "Macro-F1",
            "Worst Acc.",
            "NTRprop",
            "NTRacc",
            "Rejection Rate",
            "C_model",
            "Total Model Exchanges",
            "C_cand",
            "H_sel",
            "Avg Alpha",
            "Dataset",
            "Seed",
            "Clients",
            "Rounds",
            "Local Epochs",
            "Batch Size",
            "LR",
            "Dirichlet Alpha",
            "Mix Alpha",
            "DPSGD Mix Alpha",
            "MATCHA Mix Alpha",
            "Gate Tau",
            "NT Tau",
            "Warmup Rounds",
            "Reliability Beta",
            "Reliability Temperature",
            "Reliability Epsilon",
            "Rho Min",
            "Reliability Lambda",
            "Soft Gamma",
            "ER Prob",
            "Scale-free m",
        ])

        for r in results:
            writer.writerow([
                r["topology"],
                method_display_name(r["method"]),
                r["communication_regime"],
                r["accuracy"],
                r["macro_f1"],
                r["worst_accuracy"],
                "" if r["ntr_prop"] is None else r["ntr_prop"],
                "" if r["ntr_acc"] is None else r["ntr_acc"],
                "" if r["rejection_rate"] is None else r["rejection_rate"],
                r["c_model"],
                r["total_model_exchanges"],
                r["c_cand"],
                "" if r["selection_entropy"] is None else r["selection_entropy"],
                "" if r["avg_alpha"] is None else r["avg_alpha"],
                r.get("dataset", DATASET_NAME),
                SEED,
                NUM_CLIENTS,
                ROUNDS,
                LOCAL_EPOCHS,
                BATCH_SIZE,
                LR,
                DIRICHLET_ALPHA,
                MIX_ALPHA,
                DPSGD_MIX_ALPHA,
                MATCHA_MIX_ALPHA,
                GATE_TAU,
                NT_TAU,
                NTA_WARMUP_ROUNDS,
                RELIABILITY_BETA,
                RELIABILITY_TEMPERATURE,
                RELIABILITY_EPSILON,
                RHO_MIN,
                RELIABILITY_LAMBDA,
                SOFT_GATE_GAMMA,
                ER_PROB,
                SCALE_FREE_M,
            ])

if __name__ == "__main__":
    if GATE_TAU > NT_TAU:
        raise ValueError("For theorem-compatible hard NTA, GATE_TAU must be <= NT_TAU.")
    if NTA_WARMUP_ROUNDS != 0:
        raise ValueError("For theorem-compatible hard NTA, NTA_WARMUP_ROUNDS must be 0.")

    print("NTA-Gossip final reproducibility runner")
    print(f"Device={DEVICE} | Clients={NUM_CLIENTS} | Rounds={ROUNDS} | Local epochs={LOCAL_EPOCHS}")
    print(f"Batch={BATCH_SIZE} | LR={LR} | Dirichlet alpha={DIRICHLET_ALPHA} | Mixing alpha={MIX_ALPHA}")

    methods = [
        "local", "random_gossip", "static_peer", "reliability_only",
        "pens_one_peer", "nta_hard", "nta_soft", "adaptive_nta_hard",
        "adaptive_nta_soft", "dpsgd_full", "matcha_matching",
    ]
    all_results = []

    for dataset_name in DATASETS:
        DATASET_NAME = dataset_name
        print("\n" + "=" * 72)
        print(f"DATASET: {DATASET_NAME}")
        print("=" * 72)
        set_seed(SEED)
        train_loaders, val_loaders, test_loader, in_channels, image_size = load_dataset_once()
        actual_clients = len(train_loaders)
        if actual_clients != NUM_CLIENTS:
            raise RuntimeError(f"Expected {NUM_CLIENTS} clients, got {actual_clients}. Check partitioning.")
        initial_models = [SimpleCNN(in_channels=in_channels, image_size=image_size).to(DEVICE) for _ in range(actual_clients)]

        topology_builders = {
            "Ring": lambda n: ring_neighbors(n),
            "ER": lambda n: er_neighbors(n, p=ER_PROB),
            "Scale-free": lambda n: scale_free_neighbors(n, m=SCALE_FREE_M),
        }

        for topology_name, builder in topology_builders.items():
            set_seed(SEED)
            neighbors = builder(actual_clients)
            static_peers = make_static_peers(neighbors)
            min_deg, mean_deg, max_deg = topology_degree_stats(neighbors)
            print(f"\n{topology_name}: degree min/mean/max = {min_deg}/{mean_deg:.2f}/{max_deg}")

            for method in methods:
                set_seed(SEED)
                if method == "local":
                    result = run_local_experiment(topology_name, train_loaders, test_loader, initial_models)
                elif method == "dpsgd_full":
                    result = run_dpsgd_full_experiment(topology_name, train_loaders, test_loader, neighbors, initial_models)
                elif method == "matcha_matching":
                    result = run_matcha_matching_experiment(topology_name, train_loaders, test_loader, neighbors, initial_models)
                else:
                    result = run_one_peer_experiment(method, topology_name, train_loaders, val_loaders, test_loader, neighbors, static_peers, initial_models)
                result["dataset"] = DATASET_NAME
                all_results.append(result)

    print_overleaf_table(all_results)
    print_reference_table(all_results)
    save_results_csv(all_results, OUTPUT_CSV)
    print(f"\nSaved CSV: {OUTPUT_CSV}")
