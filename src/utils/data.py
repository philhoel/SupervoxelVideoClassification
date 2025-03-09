

import os

import urllib
import zipfile

import numpy as np
import torch
import tarfile


def create_dense_adjacency(data):
    adjacency = ...  # TODO: Transform data to dense NxN representations
    adjacency += torch.eye(len(adjacency))
    return adjacency


def train_val_test_split(
    labels,
    num_classes=7,
    train_per_class=20,
    val_size=500,
    test_size=1000
):
    indices = np.arange(len(labels))
    train_indices = []
    val_indices = []
    test_indices = []

    for c in range(num_classes):
        c_indices = indices[labels.numpy() == c]
        np.random.shuffle(c_indices)
        train_indices.extend(c_indices[:train_per_class])

    remaining = list(set(indices) - set(train_indices))
    np.random.shuffle(remaining)
    val_indices = remaining[:val_size]
    test_indices = remaining[val_size:val_size + test_size]

    return torch.tensor(train_indices, dtype=torch.long), \
        torch.tensor(val_indices, dtype=torch.long), \
        torch.tensor(test_indices, dtype=torch.long)


def download_and_extract_cora(data_dir='cora'):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
    filepath = os.path.join(data_dir, "cora.tgz")

    # Download the file if it doesn't exist
    if not os.path.exists(filepath):
        print("Downloading Cora dataset...")
        urllib.request.urlretrieve(url, filepath)
        print("Download complete.")
    else:
        print("Cora dataset already downloaded.")

    # Extract the tar.gz file
    if not os.path.exists(os.path.join(data_dir, "cora.cites")):
        print(f"Extracting Cora dataset from {filepath}...")
        with tarfile.open(filepath, 'r:gz') as tar_ref:
            tar_ref.extractall(data_dir)
        print("Extraction complete.")
    else:
        print("Cora dataset already extracted.")


def get_cora_data(data_dir='cora'):
    download_and_extract_cora(data_dir)
    content_file = os.path.join(data_dir, "cora.content")
    cites_file = os.path.join(data_dir, "cora.cites")

    # Load content
    idx_features_labels = []
    with open(content_file, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split()
            idx = parts[0]
            features = [int(x) for x in parts[1:-1]]
            label = parts[-1]
            idx_features_labels.append((idx, features, label))

    # Build indices
    idx = [x[0] for x in idx_features_labels]
    idx_map = {j: i for i, j in enumerate(idx)}
    features = np.array([x[1] for x in idx_features_labels], dtype=np.float32)
    labels_raw = [x[2] for x in idx_features_labels]
    # Encode labels
    classes = list(sorted(set(labels_raw)))
    class_map = {c: i for i, c in enumerate(classes)}
    labels = np.array([class_map[c] for c in labels_raw], dtype=np.int64)

    # Load cites and build adjacency matrix
    edges = []
    with open(cites_file, 'r') as f:
        for line in f:
            src, dst = line.strip().split()
            if src in idx_map and dst in idx_map:
                src_idx = idx_map[src]
                dst_idx = idx_map[dst]
                edges.append((src_idx, dst_idx))
    num_nodes = len(idx)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for src, dst in edges:
        adj[src, dst] = 1
        adj[dst, src] = 1  # Assuming undirected graph

    # Convert to torch tensors
    features = torch.tensor(features)
    labels = torch.tensor(labels)
    adj = torch.tensor(adj)

    return features, labels, adj
