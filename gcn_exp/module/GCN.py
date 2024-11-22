import torch
import torch.nn as nn
import torch.nn.functional as F
from ogbn_arxiv.module.gnn import MaskedLinear

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2, device=None,
                 layer_norm_first=True, use_ln=True, spar_wei=False):
        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"

        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.spar_wei = spar_wei

        self.convs.append(GCNConvCustom(nfeat, nhid, spar_wei=spar_wei))
        self.lns.append(nn.LayerNorm(nfeat) if use_ln else nn.Identity())

        for _ in range(layer - 2):
            self.convs.append(GCNConvCustom(nhid, nhid, spar_wei=spar_wei))
            self.lns.append(nn.LayerNorm(nhid) if use_ln else nn.Identity())

        self.convs.append(GCNConvCustom(nhid, nclass, spar_wei=spar_wei))
        self.lns.append(nn.LayerNorm(nhid) if use_ln else nn.Identity())

        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, GCNConvCustom):
                m.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, mask=None, wei_masks=None):
        device = next(self.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)
        if mask is not None:
            mask = mask.to(device)
        if wei_masks is not None:
            wei_masks = [w.to(device) for w in wei_masks]
        x.requires_grad_(True)
        if self.layer_norm_first and self.use_ln:
            x = self.lns[0](x)

        for i, conv in enumerate(self.convs[:-1]):
            if self.spar_wei and wei_masks is not None and i < len(wei_masks):
                x = conv(x, edge_index, edge_weight, mask, wei_masks[i])
            else:
                x = conv(x, edge_index, edge_weight, mask, wei_mask=None)
            if self.use_ln:
                x = self.lns[i + 1](x)

            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        if self.spar_wei and wei_masks is not None and len(self.convs) - 1 < len(wei_masks):
            x = self.convs[-1](x, edge_index, edge_weight, mask, wei_masks[-1])
        else:
            x = self.convs[-1](x, edge_index, edge_weight, mask, wei_mask=None)

        log_softmax_output = F.log_softmax(x, dim=1)
        return log_softmax_output

from torch_geometric.nn import MessagePassing
from typing import Optional, Tuple, Union
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.typing import OptPairTensor


class GCNConvCustom(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, list, Aggregation]] = "add",
        normalize: bool = True,
        bias: bool = True,
        spar_wei: bool = False,
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.spar_wei = spar_wei

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = MaskedLinear(in_channels[0], out_channels, bias=bias, spar_wei=spar_wei) if spar_wei else nn.Linear(in_channels[0], out_channels, bias=bias)
        if spar_wei:
            self.lin_r = MaskedLinear(in_channels[1], out_channels, bias=False, spar_wei=True)
            print(f"Initialized MaskedLinear layer: {self.lin_l}")
            print(f"Initialized MaskedLinear root layer: {self.lin_r}")
        else:
            self.lin_r = nn.Linear(in_channels[1], out_channels, bias=False)
            print(f"Initialized Linear layer: {self.lin_l}")
            print(f"Initialized Linear root layer: {self.lin_r}")

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.lin_r is not None:
            self.lin_r.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, mask=None, wei_mask=None):
        if isinstance(x, torch.Tensor):
            x = (x, x)

        if self.spar_wei:
            x = (self.lin_l(x[0], wei_mask), x[1])
        else:
            x = (self.lin_l(x[0]), x[1])

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, mask=mask, wei_mask=wei_mask)

        if self.lin_r is not None:
            out = out + self.lin_r(x[1], wei_mask) if self.spar_wei else out + self.lin_r(x[1])

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: torch.Tensor, edge_weight: OptPairTensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if edge_weight is not None:
            x_j = x_j * edge_weight.unsqueeze(1)
        if mask is not None:
            x_j = x_j * mask.unsqueeze(1)
        return x_j

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels})'

