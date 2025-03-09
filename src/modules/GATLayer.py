import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class GATLayer(nn.Module):

    activation_functions = {
        'relu': nn.ReLU,
        'lrelu': lambda: nn.LeakyReLU(negative_slope=0.2),
        'tanh': nn.Tanh,
        'elu': nn.ELU,
        'gelu': nn.GELU,
        'none': lambda: lambda x: x
    }

    def __init__(
        self,
        in_feat: int,
        out_feat: int,
        concat_hidden: bool,  # Typically: Concatenate hidden layers and average last layer
        num_heads: int = 4,
        dropout: float = 0.4,
        activation: str = 'lrelu',
        v2: bool = True,
    ):
        """
        Activation functions: lrelu (default) none, relu, tanh, elu, gelu
        """

        super().__init__()

        # Set chosen activation function
        activation = self.activation_functions[activation]
        self.activation = activation()
        # self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.soft = nn.Softmax(dim=-1)
        self.dropout = dropout
        self.num_heads = num_heads
        self.concat_hidden = concat_hidden
        self.v2 = v2

        # Calculate sizes
        if concat_hidden:
            assert out_feat % num_heads == 0, "Output features must be divisible by number of heads"
            self.num_hidden = out_feat // num_heads
            self.out_feat = out_feat
        else:
            self.out_feat = out_feat
            self.num_hidden = out_feat

        # Initialize weights
        self.W = nn.Parameter(torch.empty(
            in_feat, self.num_hidden * num_heads))

        if v2:
            self.Wa = nn.Parameter(torch.empty(num_heads, self.num_hidden))
        else:
            self.Wa = nn.Parameter(torch.empty(num_heads, 2 * self.num_hidden))

        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.Wa)

    def get_gat_v1_attention(self, x):
        # [batch, num_heads, num_nodes, 1]
        x_i = torch.matmul(x, self.Wa[:, :self.num_hidden].unsqueeze(-1))
        # [batch, num_heads, num_nodes, 1]
        x_j = torch.matmul(x, self.Wa[:, self.num_hidden:].unsqueeze(-1))
        # [batch, num_heads, num_nodes, num_nodes]
        x_ij = x_i + x_j.transpose(2, 3)

        # broadcast add (Multihead attention)
        e_ij = self.activation(x_ij)
        return e_ij

    def get_gat_v2_attention(self, x):
        x_i = x.unsqueeze(3)  # [batch, num_heads, num_nodes, 1, num_hidden]
        x_j = x.unsqueeze(2)  # [batch, num_heads, 1, num_nodes, num_hidden]
        # [batch, num_heads, num_nodes, num_nodes, num_hidden]
        x_ij = x_i + x_j

        e_ij = self.activation(x_ij)
        e_ij = (e_ij * self.Wa.view(1, self.num_heads, 1, 1, self.num_hidden)
                ).sum(dim=-1)  # [batch, num_heads, num_nodes, num_nodes]
        return e_ij

    def forward(self, node_feat, adj_mtx):

        batch, num_nodes, _ = node_feat.shape

        x = torch.matmul(node_feat, self.W)
        x = F.dropout(x, self.dropout, training=self.training)

        # splitting heads by reshaping and putting heads dim first (num_heads, num_nodes, num_hidden)
        x = x.view(batch, num_nodes, self.num_heads, self.num_hidden).permute(
            0, 2, 1, 3)  # [batch, num_heads, num_nodes, num_hidden]

        # Get attention scores
        if self.v2:
            e_ij = self.get_gat_v2_attention(x)
        else:
            e_ij = self.get_gat_v1_attention(x)

        # Mask non-existent edges
        masked_attention_scores = e_ij.masked_fill(
            adj_mtx.unsqueeze(1) <= 0, float('-inf'))

        # Normalize attention scores
        attention = self.soft(masked_attention_scores)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Apply attention scores
        x_prime = torch.matmul(attention, x)

        if self.concat_hidden:
            out = x_prime.permute(0, 2, 1, 3).contiguous().view(
                batch, num_nodes, self.out_feat)  # [batch, num_nodes, out_feat]self.Wa.view
        else:
            out = x_prime.mean(dim=1)  # [bacth, num_nodes, out_feat]

        return out
