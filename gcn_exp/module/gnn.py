import pdb
from typing import List, Optional, Tuple, Union
import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM,Parameter
import torch.nn as nn
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm

class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, spar_wei=False):
        super(SAGE, self).__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.spar_wei = spar_wei

        self.convs.append(SAGEConv(in_channels, hidden_channels, spar_wei=spar_wei))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, spar_wei=spar_wei))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, out_channels, spar_wei=spar_wei))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_mask: Optional[torch.Tensor] = None, wei_masks: Optional[List[torch.Tensor]] = None):
        device = next(self.parameters()).device
        x = x.to(device)
        adj_t = adj_t.to(device)
        if wei_masks is None:
            wei_masks = [None] * len(self.convs)
        for i, conv in enumerate(self.convs[:-1]):
            wei_mask = wei_masks[i] if i < len(wei_masks) else None
            x = conv(x, adj_t, edge_mask=edge_mask, wei_mask=wei_mask)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        wei_mask = wei_masks[-1] if len(wei_masks) == len(self.convs) else None
        x = self.convs[-1](x, adj_t, edge_mask=edge_mask, wei_mask=wei_mask)
        return x.log_softmax(dim=-1)

class BinaryStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class MaskedLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, spar_wei=False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.threshold = nn.Parameter(torch.empty(out_features))
        self.spar_wei = spar_wei
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.mask = None
        self.step = BinaryStep.apply
        self.reset_parameters()
        self.fixed_mask = False
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            self.threshold.fill_(0.)

    def forward(self, input: torch.Tensor, mask=None) -> torch.Tensor:
        if self.fixed_mask and self.mask is not None:
            return F.linear(input, self.weight * self.mask, self.bias)
        if mask is not None:
            return F.linear(input, self.weight * mask, self.bias)
        if not self.spar_wei:
            return F.linear(input, self.weight, self.bias)
        abs_weight = torch.abs(self.weight)
        threshold = self.threshold.view(-1, 1)
        mask = self.step(abs_weight - threshold)
        self.mask = mask
        masked_weight = self.weight * mask
        return F.linear(input, masked_weight, self.bias)

    def generate_wei_mask(self):
        if not self.spar_wei or self.mask is None:
            return []
        with torch.no_grad():
            wei_masks = self.mask.detach().cpu()
        return wei_masks

class SAGEConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, list, Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        spar_wei: bool = False,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project
        self.spar_wei = spar_wei

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', in_channels[0])
            kwargs['aggr_kwargs'].setdefault('out_channels', in_channels[0])

        super().__init__(aggr, **kwargs)

        if self.project:
            if self.spar_wei:
                self.lin = MaskedLinear(in_channels[0], in_channels[0], bias=True, spar_wei=True)
            else:

                self.lin = nn.Linear(in_channels[0], in_channels[0], bias=True)

        if self.aggr is None:
            self.fuse = False  # No "fused" message_and_aggregate.
            self.lstm = nn.LSTM(in_channels[0], in_channels[0], batch_first=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        if self.spar_wei:
            self.lin_l = MaskedLinear(aggr_out_channels, out_channels, bias=bias, spar_wei=True)
            if self.root_weight:
                self.lin_r = MaskedLinear(in_channels[1], out_channels, bias=False, spar_wei=True)
        else:
            self.lin_l = nn.Linear(aggr_out_channels, out_channels, bias=bias)
            #pdb.set_trace()
            if self.root_weight:
                #pdb.set_trace()
                self.lin_r = nn.Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.project and hasattr(self, 'lin'):
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight and hasattr(self, 'lin_r'):
            self.lin_r.reset_parameters()

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], edge_index: SparseTensor, edge_mask=None, wei_mask=None, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        if isinstance(edge_mask, torch.Tensor):
            self.edge_mask = edge_mask
        else:
            if not hasattr(self, "edge_mask"):
                self.register_parameter("edge_mask", None)

        if isinstance(x, torch.Tensor):
            x = (x, x)

        if self.project and hasattr(self, 'lin'):
            if self.spar_wei:
                x = (self.lin(x[0], wei_mask), x[1])
            else:
                x = (self.lin(x[0]), x[1])

        if self.spar_wei == 0:
            out = self.propagate(edge_index, x=x, size=size)
            out = self.lin_l(out)
        else:
            out = self.propagate(edge_index, x=x, size=size)
            out = self.lin_l(out, wei_mask)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            if self.spar_wei:
                out = out + self.lin_r(x_r, wei_mask)
            else:
                out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        if self.edge_mask is not None:
            x_j = x_j * self.edge_mask.unsqueeze(1)
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')