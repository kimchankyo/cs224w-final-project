from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool
from torch.nn import Linear, ReLU
import torch.nn as nn
import torch
# import torch.nn.functional as F


class BasicGCN(nn.Module):
  def __init__(self, num_features: int, num_classes: int, embedding_size: int = 128) -> None:
    super(BasicGCN, self).__init__()

    # GCN Embedding Layer
    self._embed_conv = GCNConv(num_features, embedding_size)

    # Message Layers
    self._msg1 = GCNConv(embedding_size, embedding_size)
    self._msg2 = GCNConv(embedding_size, embedding_size)

    # Aggregation Layer
    self._aggr = Linear(embedding_size*2, num_classes)

    # Non-Linear Layers
    self._relu = ReLU()

  def forward(self, x, edge_index, batch_index):
    # Embed Input
    embed = self._embed_conv(x, edge_index)
    embed = self._relu(embed)
    
    # Pass Message Layers
    msg = self._msg1(embed, edge_index)
    msg = self._relu(msg)
    msg = self._msg2(msg, edge_index)
    msg = self._relu(msg)

    # Aggregation Pooling
    aggr = torch.cat([global_mean_pool(msg, batch_index), global_max_pool(msg, batch_index)], dim=1)
    out = self._aggr(aggr)
    
    return out, aggr
