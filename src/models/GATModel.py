import random
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from tqdm import tqdm

from modules import GATLayerSparse
from supervoxel import Supervoxel
from utils import add_cls_tokens, accuracy, format_time, top_5_accuracy, confusion_matrix
import os
from .scheduler import CosineDecay


# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)


class GATModel(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        num_heads,
        num_classes,
        dropout,
        time_patch,
        space_patch,
        weight_init='kaiming',
    ):
        super().__init__()
        self.gat1 = GATLayerSparse(
            in_feat=in_features,
            out_feat=hidden_features * num_heads,
            concat_hidden=True,
            num_heads=num_heads,
            dropout=dropout,
            activation='lrelu',
            v2=True,
            weight_init=weight_init
        )

        """
        self.gat2 = GATLayerSparse(
            in_feat=hidden_features * num_heads,
            out_feat=hidden_features * num_heads,
            concat_hidden=True,
            num_heads = num_heads,
            dropout=dropout,
            activation='lrelu',
            v2=True,
            weight_init=weight_init,
        )
        """
        
        self.gat3 = GATLayerSparse(
            in_feat=hidden_features * num_heads,
            out_feat=out_features,
            concat_hidden=False,
            num_heads=1,
            dropout=dropout,
            activation='none',  # No activation in the output layer
            v2=True,
            weight_init=weight_init
        )

        # Get supervoxels
        self.supervoxel = Supervoxel(
            device="cuda", time_patch=time_patch, space_patch=space_patch)
        node_feature_size = in_features
        self.cls_token = nn.Parameter(torch.zeros(1, node_feature_size))
        if weight_init == 'kaiming':
            nn.init.kaiming_uniform_(
                self.cls_token, a=0.2, nonlinearity='leaky_relu')
        elif weight_init == 'xavier':
            nn.init.xavier_uniform_(self.cls_token)
        else:
            raise ValueError(f"Invalid weight initialization: {weight_init}")

        self.batch_norm_1 = nn.LayerNorm(hidden_features * num_heads)
        self.batch_norm_2 = nn.LayerNorm(out_features)

        self.dropout = nn.Dropout(dropout)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(
            in_features=out_features, out_features=num_classes)

        self.train_losses = []
        self.train_top_5_accuracies = []

        self.epoch_losses = []
        self.epoch_accuracies = []

        self.validation_accuracies = []
        self.validation_losses = []
        self.val_top_5_accuracies = []

        self.model_details = (
            f"init_{weight_init}_in_feat_{in_features}_hid_feat_{hidden_features}_"
            f"out_feat_{out_features}_num_head_{num_heads}_"
            f"num_cls_{num_classes}_dropout_{dropout}"
        )

        self.epochs_trained = 0
        self.best_weights = None
        self.best_val_accuracy = 0
        self.best_epoch = 0

    def forward(self, x, adj, seg_index):
        x = self.gat1(x, adj)
        x = self.batch_norm_1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)

        """
        x = self.gat2(x,adj)
        x = self.batch_norm_1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        """
        

        x = self.gat3(x, adj)  # [batch_size, num_nodes, out_features]
        x = self.batch_norm_2(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        # Get CLS tokens
        x = x.squeeze(0)[seg_index]

        x = self.classifier(x)

        return x
