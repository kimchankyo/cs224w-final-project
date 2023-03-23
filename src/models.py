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
from torch_geometric.nn.inits import zeros

from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool, Linear
from torch_geometric.utils import softmax

from typing import Dict


# Poor excuse attempt
# class BasicGCN(nn.Module):
#   def __init__(self, num_features: int, num_classes: int, embedding_size: int = 128) -> None:
#     super(BasicGCN, self).__init__()

#     # GCN Embedding Layer
#     self._embed_conv = GCNConv(num_features, embedding_size)

#     # Message Layers
#     self._msg1 = GCNConv(embedding_size, embedding_size)
#     self._msg2 = GCNConv(embedding_size, embedding_size)

#     # Aggregation Layer
#     self._aggr = Linear(embedding_size*2, num_classes)

#     # Non-Linear Layers
#     self._relu = ReLU()

#   def forward(self, x, edge_index, batch_index):
#     # Embed Input
#     print(x.shape)
#     embed = self._embed_conv(x, edge_index)
#     print(embed.shape)
#     embed = self._relu(embed)
    
#     # Pass Message Layers
#     msg = self._msg1(embed, edge_index)
#     # Pass Message Layers
#     msg = self._msg1(embed, edge_index)
#     msg = self._relu(msg)
#     msg = self._msg2(msg, edge_index)
#     msg = self._relu(msg)

#     # Aggregation Pooling
#     aggr = torch.cat([global_mean_pool(msg, batch_index), global_max_pool(msg, batch_index)], dim=1)
#     out = self._aggr(aggr)
    
#     return out, aggr


# GNN Layers -> MessagePassing modules
class GCN(MessagePassing):
  def __init__(self, in_channels: int, out_channels: int,
               improved: bool = False, cached: bool = False,
               add_self_loops: bool = True, normalize: bool = True,
               bias: bool = True, **kwargs):

    kwargs.setdefault('aggr', 'add')
    super().__init__(**kwargs)

    self.in_channels = in_channels
    self.out_channels = out_channels
    # self.improved = improved
    # self.cached = cached
    # self.add_self_loops = add_self_loops
    # self.normalize = normalize

    self.lin = Linear(in_channels, out_channels, bias=False,
                      weight_initializer='glorot')

    if bias:
        self.bias = nn.Parameter(torch.Tensor(out_channels))
    else:
        self.register_parameter('bias', None)

    self.reset_parameters()

  def reset_parameters(self):
    self.lin.reset_parameters()
    zeros(self.bias)


  def forward(self, x: Tensor, edge_index: Adj, 
              edge_weight: Tensor = None) -> Tensor:

    # if self.normalize:
    #     if isinstance(edge_index, Tensor):
    #         cache = self._cached_edge_index
    #         if cache is None:
    #             edge_index, edge_weight = gcn_norm(  # yapf: disable
    #                 edge_index, edge_weight, x.size(self.node_dim),
    #                 self.improved, self.add_self_loops, self.flow, x.dtype)
    #             if self.cached:
    #                 self._cached_edge_index = (edge_index, edge_weight)
    #         else:
    #             edge_index, edge_weight = cache[0], cache[1]

    # Linear Layer
    x = self.lin(x)

    # Propagate
    out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                         size=None)
    
    # Apply Bias
    if self.bias is not None:
      out = out + self.bias

    return out


  def message(self, x_j: Tensor, edge_weight: Tensor = None) -> Tensor:
    return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

  # def message_and_aggregate(self, adj_t: Tensor, x: Tensor) -> Tensor:
  #   return spmm(adj_t, x, reduce=self.aggr)

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

# GAT Layer
class GAT(MessagePassing):
  def __init__(self, in_channels, out_channels, heads = 2,
               negative_slope = 0.2, dropout = 0., **kwargs):
    super(GAT, self).__init__(node_dim=0, **kwargs)

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.heads = heads
    self.negative_slope = negative_slope
    self.dropout = dropout

    self.lin_l = nn.Linear(in_channels, heads * out_channels)
    self.lin_r = self.lin_l
    self.att_l = nn.Parameter(torch.randn(heads, out_channels))
    self.att_r = nn.Parameter(torch.randn(heads, out_channels))

    self.reset_parameters()

  def reset_parameters(self):
    nn.init.xavier_uniform_(self.lin_l.weight)
    nn.init.xavier_uniform_(self.lin_r.weight)
    nn.init.xavier_uniform_(self.att_l)
    nn.init.xavier_uniform_(self.att_r)

  def forward(self, x, edge_index, size = None):
    H, C = self.heads, self.out_channels
    x_l, x_r = self.lin_l(x).view(-1, H, C), self.lin_r(x).view(-1, H, C)
    a_l, a_r = x_l * self.att_l.unsqueeze(0), x_r * self.att_r.unsqueeze(0)
    out = self.propagate(edge_index, x=(x_l, x_r), alpha=(a_l, a_r))
    out = out.view(-1, H * C)
    return out

  def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
    a_f = alpha_j + alpha_i
    a_f = F.leaky_relu(a_f, negative_slope = self.negative_slope)
    a_f = softmax(a_f, index) if ptr is None else softmax()
    a_f = F.dropout(a_f, p=self.dropout)
    out = x_j * a_f
    return out

  def aggregate(self, inputs, index, dim_size = None):
    out = torch_scatter.scatter(inputs, index, dim=self.node_dim, reduce='sum')
    return out


# Generic GNN (Can interchange different layers)
class GNNConvLayer:
  SAGE = GraphSage
  GAT = GAT
  CONV = GCN

  STR_TO_TYPE = {
    'gat': GAT,
    'sage': SAGE,
    'gcn': CONV
  }


class GNN(nn.Module):
  def __init__(self, inputSize: int, hiddenSize: int, outputSize: int, 
               numLayers: int = 3, convLayerType: str = 'sage',
               heads: int = 1, dropout: float = 0.5, skip: bool = False) -> None:
    super(GNN, self).__init__()
    convLayer = GNNConvLayer.STR_TO_TYPE[convLayerType]
    
    self.inputSize = inputSize
    self.hiddenSize = hiddenSize
    self.outputSize = outputSize
    self.numLayers = numLayers
    self.convLayerType = convLayerType
    self.heads = heads
    self.dropout = dropout
    self.skip = skip
    
    self.convs = nn.ModuleList()    # Convolution Layer List
    self.bns = nn.ModuleList([nn.BatchNorm1d(heads * hiddenSize) for i in range(numLayers)])
    self.postMP = nn.Sequential(
      nn.Linear(heads*hiddenSize, hiddenSize), 
      nn.Dropout(dropout),
      nn.Linear(hiddenSize, outputSize)
    )
    
    self.logSoftmax = nn.LogSoftmax(dim=1)
    
    self._initConvLayers(inputSize, hiddenSize, outputSize, 
                         numLayers, convLayer, heads)

  def _initConvLayers(self, inputSize: int, hiddenSize: int, outputSize: int, 
                      numLayers: int, convLayer: nn.Module, heads: int) -> None:
    assert numLayers > 1
    self.convs.append(convLayer(inputSize, hiddenSize))
    for _ in range(numLayers-1): 
      self.convs.append(convLayer(heads*hiddenSize, hiddenSize))

  def getConfig(self) -> Dict:
    return {
      'inputSize': self.inputSize,
      'hiddenSize': self.hiddenSize,
      'outputSize': self.outputSize,
      'numLayers': self.numLayers,
      'convLayerType': self.convLayerType,
      'heads': self.heads,
      'dropout': self.dropout,
      'skip': self.skip
    }

  def forward(self, data) -> Tensor:
    x, edgeIndex, batch = data.x, data.edge_index, data.batch
    residual = x
    for i in range(self.numLayers):
      x = self.convs[i](x, edgeIndex)
      x = self.bns[i](x)
      x = x + residual if self.skip else x
      x = F.relu(x)
      x = F.dropout(x, p=self.dropout, training=self.training)
    x = self.postMP(x)
    return self.logSoftmax(global_add_pool(x, batch))
  
