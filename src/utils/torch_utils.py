import torch


def get_current_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def add_cls_tokens(features, edges, seg_indexes, cls_token, device):
    """
    Adds cls tokens to features and adjusts edges accordingly.

    Parameters:
    - features: Tensor of shape (num_nodes, num_features)
    - edges: Tensor of shape (2, num_edges)
    - seg_indexes: Tensor of shape (num_nodes,), indicating segment indices
    - cls_token: Tensor of shape (num_features,)
    - device: torch.device

    Returns:
    - new_features: Tensor with cls tokens added
    - final_edges: Adjusted edges tensor
    - cls_indexes: Tensor of shape (num_segments,), indicating the indices of the cls tokens
    """

    # Get counts and cumulative indices for segments
    # print("seg_indexes", seg_indexes)
    unique_segments, node_counts = torch.unique(
        seg_indexes, return_counts=True)
    node_indexes = torch.cumsum(node_counts, dim=0)
    num_segments = len(unique_segments)
    N = features.size(0)
    N_new = N + num_segments  # New total number of nodes after adding cls tokens

    # Initialize new features and mapping from old to new indices
    new_features = torch.zeros((N_new, features.size(1)), device=device)
    old_to_new_indices = torch.zeros(N, dtype=torch.long, device=device)

    # Prepare to adjust edges
    edges = edges.to(device)
    new_edges_list = []
    current_new_idx = 0

    for s in range(num_segments):
        # Segment indices
        start_idx = node_indexes[s - 1] if s > 0 else 0
        end_idx = node_indexes[s]
        segment_length = end_idx - start_idx

        # Copy segment features to new features tensor
        new_start_idx = current_new_idx
        new_end_idx = new_start_idx + segment_length
        new_features[new_start_idx:new_end_idx] = features[start_idx:end_idx]

        # Update mapping from old to new indices
        old_indices = torch.arange(start_idx, end_idx, device=device)
        new_indices = torch.arange(new_start_idx, new_end_idx, device=device)
        old_to_new_indices[old_indices] = new_indices

        # Insert cls token
        cls_idx = new_end_idx
        new_features[cls_idx] = cls_token.to(device)

        # Create edges between cls token and segment nodes
        node_indices = new_indices
        cls_indices = torch.full(
            (segment_length,), cls_idx, dtype=torch.long, device=device)
        edges_src = torch.cat([cls_indices, node_indices])
        edges_dst = torch.cat([node_indices, cls_indices])
        new_edges = torch.stack([edges_src, edges_dst], dim=0)
        new_edges_list.append(new_edges)

        # Update current index for next segment
        current_new_idx = cls_idx + 1

    # Adjust original edges to new indices
    edges_mapped = old_to_new_indices[edges.view(-1)].view(2, -1)

    # Concatenate all edges
    final_edges = torch.cat([edges_mapped] + new_edges_list, dim=1)

    # print(f"nodes: {N}, segments: {num_segments}, new nodes: {N_new}, node_indexes: {node_indexes}")
    i_range = torch.arange(num_segments, device=device)
    cls_indexes = node_indexes + i_range

    return new_features, final_edges, cls_indexes
