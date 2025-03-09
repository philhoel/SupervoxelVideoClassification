import torch


def accuracy(output, labels):
    # Ensure output and labels are tensors, handling batch processing directly
    if isinstance(output, list):
        output = torch.stack([o.cpu() for o in output]).flatten(
            start_dim=1, end_dim=-1)
    else:
        output = output.cpu()

    if isinstance(labels, list):
        labels = torch.stack([l.cpu() for l in labels]).flatten()
    else:
        labels = labels.cpu()

    preds = output.argmax(dim=1)
    correct = (preds == labels).float()
    acc = correct.mean()

    return acc


def top_5_accuracy(output, labels):
    # Ensure output and labels are tensors, handling batch processing directly
    if isinstance(output, list):
        output = torch.stack([o.cpu() for o in output]).flatten(
            start_dim=1, end_dim=-1)
    else:
        output = output.cpu()

    if isinstance(labels, list):
        labels = torch.stack([l.cpu() for l in labels]).flatten()
    else:
        labels = labels.cpu()

    # Get top 5 predictions
    # Indices of top-5 predictions (batch_size x 5)
    top5 = torch.topk(output, 5, dim=1)[1]

    # Compare each of the top-5 predictions with the true labels
    correct = top5.eq(labels.unsqueeze(1))  # (batch_size x 5)

    # Compute top-5 accuracy
    # True if the label is in the top-5 for each sample
    correct_5 = correct.any(dim=1).float()
    acc_5 = correct_5.mean()  # Average over the batch

    return acc_5


def confusion_matrix(output, labels):
    # Ensure output and labels are tensors, handling batch processing directly
    if isinstance(output, list):
        output = torch.stack([o.cpu() for o in output]).flatten(
            start_dim=1, end_dim=-1)
    else:
        output = output.cpu()

    if isinstance(labels, list):
        labels = torch.stack([l.cpu() for l in labels]).flatten()
    else:
        labels = labels.cpu()

    # Get predictions
    preds = output.argmax(dim=1)

    # Compute confusion matrix
    num_classes = output.size(1)
    labels = labels.long().view(-1)
    preds = preds.long().view(-1)

    # Create indices tensor
    indices = torch.stack([labels, preds], dim=0)
    values = torch.ones(labels.size(0), dtype=torch.int64)

    # Create the confusion matrix
    conf_matrix = torch.sparse_coo_tensor(indices, values, size=(
        num_classes, num_classes), dtype=torch.int64).to_dense()

    return conf_matrix
