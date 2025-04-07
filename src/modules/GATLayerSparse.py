import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_softmax
#from .scatter_functions import scatter_softmax_2d, scatter_sum_2d


class GATLayerSparse(nn.Module):

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
        weight_init: str = 'kaiming'
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
        self.loop_fill = 1.0

        # Calculate sizes
        if concat_hidden:
            assert out_feat % num_heads == 0, "Output features must be divisible by number of heads"
            self.num_hidden = out_feat // num_heads
            self.out_feat = out_feat
        else:
            self.out_feat = out_feat
            self.num_hidden = out_feat

        # Initialize weights
        self.W_l = nn.Parameter(torch.empty(
            in_feat, self.num_hidden * num_heads))
        self.W_r = nn.Parameter(torch.empty(
            in_feat, self.num_hidden * num_heads))

        if v2:
            self.Wa = nn.Parameter(torch.empty(num_heads, self.num_hidden))
        else:
            self.Wa = nn.Parameter(torch.empty(num_heads, 2 * self.num_hidden))

        if weight_init == 'kaiming':
            nn.init.kaiming_uniform_(
                self.W_l, a=0.2, nonlinearity='leaky_relu')
            nn.init.kaiming_uniform_(
                self.W_r, a=0.2, nonlinearity='leaky_relu')
            nn.init.kaiming_uniform_(self.Wa, a=0.2, nonlinearity='leaky_relu')
        elif weight_init == 'xavier':
            nn.init.xavier_uniform_(self.W_l)
            nn.init.xavier_uniform_(self.W_r)
            nn.init.xavier_normal_(self.Wa)
        else:
            raise ValueError(
                f"Invalid weight initialization method: {weight_init}")

    def forward(self, node_feat, edge_index):
        # [batch_size, num_nodes, in_feat]
        batch_size, num_nodes, _ = node_feat.shape

        # [batch_size, num_nodes, num_heads * num_hidden]
        x_l = torch.matmul(node_feat, self.W_l)
        # [batch_size, num_nodes, num_heads * num_hidden]
        x_r = torch.matmul(node_feat, self.W_r)
        x_l = F.dropout(x_l, self.dropout, training=self.training)
        x_r = F.dropout(x_r, self.dropout, training=self.training)

        # Reshape and permute x_l
        x_l = x_l.view(batch_size, num_nodes, self.num_heads, self.num_hidden)
        # [batch_size, num_heads, num_nodes, num_hidden]
        x_l = x_l.permute(0, 2, 1, 3)

        # Reshape and permute x_r
        x_r = x_r.view(batch_size, num_nodes, self.num_heads, self.num_hidden)
        # [batch_size, num_heads, num_nodes, num_hidden]
        x_r = x_r.permute(0, 2, 1, 3)

        # Prepare edge indices
        _, _, num_edges = edge_index.shape  # [batch_size, 2, num_edges]

        # Expand edge_index to match num_heads: [batch_size, num_heads, 2, num_edges]
        edge_index = edge_index.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # Get source and target node indices: [batch_size, num_heads, num_edges]
        edge_src = edge_index[:, :, 0, :]  # Source nodes
        edge_dst = edge_index[:, :, 1, :]  # Target nodes

        # Gather node features for source and target nodes
        x_src = torch.gather(
            x_l, 2, edge_src.unsqueeze(-1).expand(-1, -1, -1, self.num_hidden))
        x_dst = torch.gather(
            x_r, 2, edge_dst.unsqueeze(-1).expand(-1, -1, -1, self.num_hidden))

        # Compute attention scores
        if self.v2:
            # [batch_size, num_heads, num_edges, num_hidden]
            e = self.activation(x_src + x_dst)
            Wa = self.Wa.unsqueeze(0).unsqueeze(
                2)  # [1, num_heads, 1, num_hidden]
            e = (e * Wa).sum(dim=-1)  # [batch_size, num_heads, num_edges]
        else:
            # Concatenate along the feature dimension
            a_input = torch.cat([x_src, x_dst], dim=-1)
            # [1, num_heads, 1, 2 * num_hidden]
            Wa = self.Wa.unsqueeze(0).unsqueeze(2)
            # [batch_size, num_heads, num_edges]
            e = self.activation((a_input * Wa).sum(dim=-1))

        # Normalize attention coefficients per target node using scatter_softmax
        # Reshape e and edge_dst to combine batch and heads for scatter_softmax
        # [batch_size * num_heads, num_edges]
        e = e.view(batch_size * self.num_heads, num_edges)
        # [batch_size * num_heads, num_edges]
        edge_dst = edge_dst.view(batch_size * self.num_heads, num_edges)

        # [batch_size * num_heads, num_edges]
        alpha = scatter_softmax(e, edge_dst, dim=1)
        #alpha = scatter_softmax_2d(e, edge_dst)
        # [batch_size * num_heads, num_edges]
        alpha = alpha.view(batch_size, self.num_heads, num_edges)
        alpha = F.dropout(alpha, self.dropout, training=self.training)

        # Message passing
        # [batch_size, num_heads, num_edges, num_hidden]
        m = x_src * alpha.unsqueeze(-1)

        # Flatten batch_size and num_heads into a single dimension
        batch_size, num_heads, num_edges, num_hidden = m.shape
        batch_num_heads = batch_size * num_heads

        # Reshape tensors
        # [batch_num_heads, num_edges, num_hidden]
        m_flat = m.view(batch_num_heads, num_edges, num_hidden)
        # [batch_num_heads, num_edges]
        index_flat = edge_dst.reshape(batch_num_heads, num_edges)

        # Aggregate messages to target nodes using torch_scatter.scatter_add
        out_flat = scatter_add(
            src=m_flat,
            index=index_flat,
            dim=1,
            dim_size=num_nodes
        )  # [batch_num_heads, num_nodes, num_hidden]

        #out_flat = scatter_sum_2d(m_flat, index_flat)

        
        # Reshape back to original dimensions
        out = out_flat.view(batch_size, num_heads, num_nodes, num_hidden)

        # Reshape and combine heads
        if self.concat_hidden:
            # Concatenate along the feature dimension
            out = out.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, -1)
        else:
            # Average over heads
            out = out.mean(dim=1)

        return out
