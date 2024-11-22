import pdb
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter, scatter_softmax
from torch_geometric.utils import degree
from torch import Tensor
import math

class BinaryStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, spar_wei=False, device=None, dtype=None):
        super(MaskedLinear, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.threshold = nn.Parameter(torch.empty(out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.mask = None
        self.step = BinaryStep.apply
        self.reset_parameters()
        self.fixed_mask = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            self.threshold.fill_(0.)

    def forward(self, input, weight_mask=None):
        if self.fixed_mask and self.mask is not None:
            return F.linear(input, self.weight * self.mask, self.bias)
        if weight_mask is not None:
            masked_weight = self.weight * weight_mask
            return F.linear(input, masked_weight, self.bias)
        if not self.spar_wei:
            return F.linear(input, self.weight, self.bias)
        abs_weight = torch.abs(self.weight)
        threshold = self.threshold.view(-1, 1)
        mask = self.step(abs_weight - threshold)
        assert isinstance(mask, torch.Tensor), "mask should be a Tensor"
        self.mask = mask
        #pdb.set_trace()
        masked_weight = self.weight * mask
        return F.linear(input, masked_weight, self.bias)

    def get_thresholds(self):
        return self.threshold.detach().cpu().numpy()

class GenMessagePassing(MessagePassing):
    def __init__(self, aggr='softmax',
                 t=1.0, learn_t=False,
                 p=1.0, learn_p=False,
                 y=0.0, learn_y=False):

        if aggr in ['softmax_sg', 'softmax', 'softmax_sum']:

            super(GenMessagePassing, self).__init__(aggr=None)
            self.aggr = aggr

            if learn_t and (aggr == 'softmax' or aggr == 'softmax_sum'):
                self.learn_t = True
                self.t = torch.nn.Parameter(torch.Tensor([t]), requires_grad=True)
            else:
                self.learn_t = False
                self.t = t

            if aggr == 'softmax_sum':
                self.y = torch.nn.Parameter(torch.Tensor([y]), requires_grad=learn_y)

        elif aggr in ['power', 'power_sum']:

            super(GenMessagePassing, self).__init__(aggr=None)
            self.aggr = aggr

            if learn_p:
                self.p = torch.nn.Parameter(torch.Tensor([p]), requires_grad=True)
            else:
                self.p = p

            if aggr == 'power_sum':
                self.y = torch.nn.Parameter(torch.Tensor([y]), requires_grad=learn_y)
        else:
            super(GenMessagePassing, self).__init__(aggr=aggr)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):

        if self.aggr in ['add', 'mean', 'max', None]:
            return super(GenMessagePassing, self).aggregate(inputs, index, ptr, dim_size)

        elif self.aggr in ['softmax_sg', 'softmax', 'softmax_sum']:

            if self.learn_t:
                out = scatter_softmax(inputs*self.t, index, dim=self.node_dim)
            else:
                with torch.no_grad():
                    out = scatter_softmax(inputs*self.t, index, dim=self.node_dim)

            out = scatter(inputs*out, index, dim=self.node_dim,
                          dim_size=dim_size, reduce='sum')

            if self.aggr == 'softmax_sum':
                self.sigmoid_y = torch.sigmoid(self.y)
                degrees = degree(index, num_nodes=dim_size).unsqueeze(1)
                out = torch.pow(degrees, self.sigmoid_y) * out

            return out


        elif self.aggr in ['power', 'power_sum']:
            min_value, max_value = 1e-7, 1e1
            torch.clamp_(inputs, min_value, max_value)
            out = scatter(torch.pow(inputs, self.p), index, dim=self.node_dim,
                          dim_size=dim_size, reduce='mean')
            torch.clamp_(out, min_value, max_value)
            out = torch.pow(out, 1/self.p)
            # torch.clamp(out, min_value, max_value)

            if self.aggr == 'power_sum':
                self.sigmoid_y = torch.sigmoid(self.y)
                degrees = degree(index, num_nodes=dim_size).unsqueeze(1)
                out = torch.pow(degrees, self.sigmoid_y) * out

            return out

        else:
            raise NotImplementedError('To be implemented')

class MsgNorm(torch.nn.Module):
    def __init__(self, learn_msg_scale=False):
        super(MsgNorm, self).__init__()

        self.msg_scale = torch.nn.Parameter(torch.Tensor([1.0]),
                                            requires_grad=learn_msg_scale)

    def forward(self, x, msg, p=2):
        msg = F.normalize(msg, p=p, dim=1)
        x_norm = x.norm(p=p, dim=1, keepdim=True)
        msg = msg * x_norm * self.msg_scale
        return msg

class GENConv(GenMessagePassing):
    def __init__(self, in_dim, emb_dim,
                 aggr='softmax',
                 t=1.0, learn_t=False,
                 p=1.0, learn_p=False,
                 y=0.0, learn_y=False,
                 msg_norm=False, learn_msg_scale=True,
                 encode_edge=False, bond_encoder=False,
                 edge_feat_dim=None,
                 norm='batch', mlp_layers=2,
                 eps=1e-7,
                 spar_wei=False):
        super(GENConv, self).__init__(aggr=aggr,
                                      t=t, learn_t=learn_t,
                                      p=p, learn_p=learn_p,
                                      y=y, learn_y=learn_y)
        self.spar_wei = spar_wei
        channels_list = [in_dim]
        for i in range(mlp_layers-1):
            channels_list.append(in_dim*2)
        channels_list.append(emb_dim)

        self.mlp = MLP(channels=channels_list,
                       norm=norm,
                       last_lin=True,
                       spar_wei=spar_wei)

        self.msg_encoder = torch.nn.ReLU()
        self.eps = eps

        self.msg_norm = msg_norm
        self.encode_edge = encode_edge
        self.bond_encoder = bond_encoder

        if msg_norm:
            self.msg_norm = MsgNorm(learn_msg_scale=learn_msg_scale)
        else:
            self.msg_norm = None

        if self.encode_edge:
            if self.bond_encoder:
                self.edge_encoder = BondEncoder(emb_dim=in_dim)
            else:
                self.edge_encoder = torch.nn.Linear(edge_feat_dim, in_dim)

    def forward(self, x, edge_index, edge_attr=None, edge_mask=None, wei_mask=None):
        if self.encode_edge and edge_attr is not None:
            edge_emb = self.edge_encoder(edge_attr)
        else:
            edge_emb = edge_attr

        if isinstance(edge_mask, Tensor):
            self.edge_mask = edge_mask.unsqueeze(1)
        elif not hasattr(self, 'edge_mask'):
            self.register_parameter("edge_mask", None)
        m = self.propagate(edge_index, x=x, edge_attr=edge_emb, wei_mask=wei_mask)

        if self.msg_norm is not None:
            m = self.msg_norm(x, m)

        h = x + m
        if self.spar_wei == 1 and wei_mask is not None:
            out = self.mlp(h, masks=wei_mask)
        else:
            out = self.mlp(h)
        return out

    def message(self, x_j, edge_attr=None, wei_mask=None):
        if edge_attr is not None:
            msg = x_j + edge_attr
        else:
            msg = x_j

        if self.edge_mask is not None:
            mask_msg = self.edge_mask * (self.msg_encoder(msg) + self.eps)
        else:
            mask_msg = self.msg_encoder(msg) + self.eps
        return mask_msg

    def update(self, aggr_out):
        return aggr_out


def act_layer(act_type, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act_type.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def norm_layer(norm_type, nc):
    # normalization layer 1d
    norm = norm_type.lower()
    if norm == 'batch':
        layer = nn.BatchNorm1d(nc, affine=True)
    elif norm == 'layer':
        layer = nn.LayerNorm(nc, elementwise_affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm1d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer

allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'possible_is_conjugated_list': [False, True],
}

def get_bond_feature_dims():
    return list(map(len, [
        allowable_features['possible_bond_type_list'],
        allowable_features['possible_bond_stereo_list'],
        allowable_features['possible_is_conjugated_list']
        ]))

class MLP(Seq):
    def __init__(self, channels, act='relu',
                 norm=None, bias=True,
                 drop=0., last_lin=False,
                 spar_wei=False):
        m = []
        for i in range(1, len(channels)):

            if spar_wei:
                linear_layer = MaskedLinear(channels[i - 1], channels[i], bias=bias, spar_wei=True)
            else:
                linear_layer = Lin(channels[i - 1], channels[i], bias)
            m.append(linear_layer)

            if (i == len(channels) - 1) and last_lin:
                pass
            else:
                if norm is not None and norm.lower() != 'none':
                    m.append(norm_layer(norm, channels[i]))
                if act is not None and act.lower() != 'none':
                    m.append(act_layer(act))
                if drop > 0:
                    m.append(nn.Dropout2d(drop))

        self.m = m
        super(MLP, self).__init__(*self.m)

    def forward(self, x, masks=None):
        for layer in self:
            if isinstance(layer, MaskedLinear):
                if masks and len(masks) > 0:
                    weight_mask = masks.pop(0)
                    x = layer(x, weight_mask=weight_mask)
                else:
                    x = layer(x)
            else:
                x = layer(x)
        return x

class BondEncoder(nn.Module):

    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = nn.ModuleList()
        full_bond_feature_dims = get_bond_feature_dims()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding

