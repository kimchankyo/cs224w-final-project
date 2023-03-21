import torch
import torch_scatter
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, Size

from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool
from torch.nn import Linear, ReLU

# Poor excuse attempt
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
    print(x.shape)
    embed = self._embed_conv(x, edge_index)
    print(embed.shape)
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


# GNN Layers -> MessagePassing modules
# class GCN(MessagePassing):
#   def __init__(self, in_channels: int, out_channels: int,
#                  improved: bool = False, cached: bool = False,
#                  add_self_loops: bool = True, normalize: bool = True,
#                  bias: bool = True, **kwargs):

#         kwargs.setdefault('aggr', 'add')
#         super().__init__(**kwargs)

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.improved = improved
#         self.cached = cached
#         self.add_self_loops = add_self_loops
#         self.normalize = normalize

#         self._cached_edge_index = None
#         self._cached_adj_t = None

#         self.lin = Linear(in_channels, out_channels, bias=False,
#                           weight_initializer='glorot')

#         if bias:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()

# [docs]
#     def reset_parameters(self):
#         super().reset_parameters()
#         self.lin.reset_parameters()
#         zeros(self.bias)
#         self._cached_edge_index = None
#         self._cached_adj_t = None


# [docs]
#     def forward(self, x: Tensor, edge_index: Adj,
#                 edge_weight: OptTensor = None) -> Tensor:

#         if self.normalize:
#             if isinstance(edge_index, Tensor):
#                 cache = self._cached_edge_index
#                 if cache is None:
#                     edge_index, edge_weight = gcn_norm(  # yapf: disable
#                         edge_index, edge_weight, x.size(self.node_dim),
#                         self.improved, self.add_self_loops, self.flow, x.dtype)
#                     if self.cached:
#                         self._cached_edge_index = (edge_index, edge_weight)
#                 else:
#                     edge_index, edge_weight = cache[0], cache[1]

#             elif isinstance(edge_index, SparseTensor):
#                 cache = self._cached_adj_t
#                 if cache is None:
#                     edge_index = gcn_norm(  # yapf: disable
#                         edge_index, edge_weight, x.size(self.node_dim),
#                         self.improved, self.add_self_loops, self.flow, x.dtype)
#                     if self.cached:
#                         self._cached_adj_t = edge_index
#                 else:
#                     edge_index = cache

#         x = self.lin(x)

#         # propagate_type: (x: Tensor, edge_weight: OptTensor)
#         out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
#                              size=None)

#         if self.bias is not None:
#             out = out + self.bias

#         return out


#     def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
#         return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

#     def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
#         return spmm(adj_t, x, reduce=self.aggr)


# GraphSage Layer
class GraphSage(MessagePassing):
  def __init__(self, inChannels: int, outChannels: int, 
               normalize = True, bias: bool = False, **kwargs) -> None:  
    super(GraphSage, self).__init__(**kwargs)

    self.in_channels = inChannels
    self.out_channels = outChannels
    self.normalize = normalize

    self.lin_l = nn.Linear(self.in_channels, self.out_channels, bias=True)
    self.lin_r = nn.Linear(self.in_channels, self.out_channels, bias=True)

    self.reset_parameters()

  def reset_parameters(self) -> None:
    self.lin_l.reset_parameters()
    self.lin_r.reset_parameters()

  def forward(self, x: Tensor, edge_index: Adj, size: Size = None) -> Tensor:
    p = self.propagate(edge_index, size=size, x=(x, x))
    out = p + self.lin_l(x)
    out = F.normalize(out) if self.normalize else out
    return out

  def message(self, x_j: Tensor) -> Tensor:
    out = self.lin_r(x_j)
    return out

  def aggregate(self, inputs: Tensor, index: Tensor, 
                dim_size: int = None) -> Tensor:
    out = torch_scatter.scatter(
      inputs, index, self.node_dim, dim_size=dim_size, reduce='mean'
    )
    return out


# Generic GNN (Can interchange different layers)
class GNNConvLayer:
  SAGE = GraphSage


class GNN(nn.Module):
  def __init__(self, inputSize: int, hiddenSize: int, outputSize: int, 
               numLayers: int = 3, convLayer: nn.Module = GraphSage,
               heads: int = 1, dropout: float = 0.5) -> None:
    super(GNN, self).__init__()
    self.convs = nn.ModuleList()    # Convolution Layer List
    self.postMP = nn.Sequential(
      nn.Linear(heads*hiddenSize, hiddenSize), 
      nn.Dropout(dropout), 
      nn.Linear(hiddenSize, outputSize)
    )
    self.dropout = dropout
    self.numLayers = numLayers
    self.logSoftmax = nn.LogSoftmax(dim=1)

    self._initConvLayers(inputSize, hiddenSize, outputSize, 
                         numLayers, convLayer, heads)

  def _initConvLayers(self, inputSize: int, hiddenSize: int, outputSize: int, 
                      numLayers: int, convLayer: nn.Module, heads: int) -> None:
    if numLayers == 1:
      self.convs.append(convLayer(inputSize, outputSize))
    else:
      self.convs.append(convLayer(inputSize, hiddenSize))
      for _ in range(numLayers-1): 
        self.convs.append(convLayer(heads*hiddenSize, hiddenSize))

  def forward(self, data) -> Tensor:
    x, edgeIndex, batch = data.x, data.edge_index, data.batch
    for i in range(self.numLayers):
      x = self.convs[i](x, edgeIndex)
      x = F.relu(x)
      x = F.dropout(x, p=self.dropout, training=self.training)
    x = self.postMP(x)
    return self.logSoftmax(global_mean_pool(x, batch))
