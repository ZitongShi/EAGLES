import torch
from gnn_conv import GENConv, MaskedLinear, MLP
from gnn_conv import norm_layer
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import logging


class DeeperGCN(torch.nn.Module):
    def __init__(self, args):
        super(DeeperGCN, self).__init__()

        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.block = args.block

        self.checkpoint_grad = False

        hidden_channels = args.hidden_channels
        num_tasks = args.num_tasks
        conv = args.conv
        aggr = args.gcn_aggr

        t = args.t
        self.learn_t = args.learn_t
        p = args.p
        self.learn_p = args.learn_p
        y = args.y
        self.learn_y = args.learn_y

        self.msg_norm = args.msg_norm
        learn_msg_scale = args.learn_msg_scale

        conv_encode_edge = args.conv_encode_edge
        norm = args.norm
        mlp_layers = args.mlp_layers
        node_features_file_path = args.nf_path

        self.use_one_hot_encoding = args.use_one_hot_encoding

        # 使用梯度检查点减少显存消耗
        if aggr not in ['add', 'max', 'mean'] and self.num_layers > 15:
            self.checkpoint_grad = True
            self.ckp_k = 9

        print('The number of layers {}'.format(self.num_layers),
              'Aggregation method {}'.format(aggr),
              'block: {}'.format(self.block))

        if self.block == 'res+':
            print('LN/BN->ReLU->GraphConv->Res')
        elif self.block == 'res':
            print('GraphConv->LN/BN->ReLU->Res')
        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')
        elif self.block == "plain":
            print('GraphConv->LN/BN->ReLU')
        else:
            raise Exception('Unknown block Type')

        self.gcns = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        self.spar_wei = args.spar_wei
        for layer in range(self.num_layers):
            if conv == 'gen':
                gcn = GENConv(hidden_channels, hidden_channels,
                             aggr=aggr,
                             t=t, learn_t=self.learn_t,
                             p=p, learn_p=self.learn_p,
                             y=y, learn_y=self.learn_y,
                             msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale,
                             encode_edge=conv_encode_edge, edge_feat_dim=hidden_channels,
                             norm=norm, mlp_layers=mlp_layers,
                             spar_wei=args.spar_wei)  # 传递 spar_wei
            else:
                raise Exception('Unknown Conv Type')

            self.gcns.append(gcn)
            self.layer_norms.append(norm_layer(norm, hidden_channels))

        self.node_features = torch.load(node_features_file_path).to(args.device)

        if self.use_one_hot_encoding:
            self.node_one_hot_encoder = torch.nn.Linear(8, 8)
            self.node_features_encoder = torch.nn.Linear(8 * 2, hidden_channels)
        else:
            self.node_features_encoder = torch.nn.Linear(8, hidden_channels)

        self.edge_encoder = torch.nn.Linear(8, hidden_channels)

        self.node_pred_linear = torch.nn.Linear(hidden_channels, num_tasks)

    def print_parameter_stats(self):
        """
        打印GNN中所有层的参数总量及被掩码的参数总量
        """
        print("GNN Model Parameter Statistics:")

        # 检查gcns是否存在且不为空
        if not hasattr(self, 'gcns') or len(self.gcns) == 0:
            print("  No convolution layers found in gcns")
            return

        print(f"Total convolution layers in gcns: {len(self.gcns)}")

        for idx, layer in enumerate(self.gcns):
            print(f"Layer {idx + 1}: {layer.__class__.__name__}")

            # 检查是否有 MLP
            if hasattr(layer, 'mlp') and isinstance(layer.mlp, MLP):
                for sub_idx, sub_layer in enumerate(layer.mlp):
                    if isinstance(sub_layer, MaskedLinear):
                        total_params = sub_layer.weight.numel()
                        masked_params = (sub_layer.mask == 0).sum().item() if sub_layer.mask is not None else 0
                        print(
                            f"  MLP Layer {sub_idx + 1} - MaskedLinear: Total Params = {total_params}, Masked Params = {masked_params}")
                    else:
                        # 对于非 MaskedLinear 的层
                        if hasattr(sub_layer, 'weight') and sub_layer.weight is not None:
                            total_params = sub_layer.weight.numel()
                            masked_params = (sub_layer.mask == 0).sum().item() if hasattr(sub_layer,
                                                                                          'mask') and sub_layer.mask is not None else 0
                            print(
                                f"  MLP Layer {sub_idx + 1} - {sub_layer.__class__.__name__}: Total Params = {total_params}, Masked Params = {masked_params}")
            else:
                # 如果层中没有 MLP 或 MaskedLinear，则打印权重和掩码信息
                if hasattr(layer, 'weight') and layer.weight is not None:
                    total_params = layer.weight.numel()
                    masked_params = (layer.mask == 0).sum().item() if hasattr(layer,
                                                                              'mask') and layer.mask is not None else 0
                    print(f"  Weight Params: Total Params = {total_params}, Masked Params = {masked_params}")

            # 打印 bias 参数信息
            if hasattr(layer, 'bias') and layer.bias is not None:
                total_bias_params = layer.bias.numel()
                print(f"  Bias Params = {total_bias_params}")
        print("Finished printing parameter stats.")

    def forward(self, x, node_index, edge_index, edge_attr, edge_mask=None, wei_masks = None):

        node_features_1st = self.node_features[node_index]

        if self.use_one_hot_encoding:
            node_features_2nd = self.node_one_hot_encoder(x)
            # concatenate
            node_features = torch.cat((node_features_1st, node_features_2nd), dim=1)
        else:
            node_features = node_features_1st

        h = self.node_features_encoder(node_features)

        edge_emb = self.edge_encoder(edge_attr)

        if self.block == 'res+':
            # 第一层
            if self.spar_wei == 1:
                if wei_masks is not None and len(wei_masks) >=1:
                    h = self.gcns[0](h, edge_index, edge_emb, edge_mask=edge_mask, wei_mask=wei_masks[0])
                else:
                    h = self.gcns[0](h, edge_index, edge_emb, edge_mask=edge_mask, wei_mask=None)
            else:
                h = self.gcns[0](h, edge_index, edge_emb, edge_mask)

            if self.checkpoint_grad:
                for layer in range(1, self.num_layers):
                    h1 = self.layer_norms[layer-1](h)
                    h2 = F.relu(h1)
                    h2 = F.dropout(h2, p=self.dropout, training=self.training)
                    if layer % self.ckp_k != 0:
                        if self.spar_wei == 1:
                            if wei_masks is not None and len(wei_masks) > layer:
                                res = checkpoint(self.gcns[layer], h2, edge_index, edge_emb, wei_mask=wei_masks[layer])
                            else:
                                res = checkpoint(self.gcns[layer], h2, edge_index, edge_emb, wei_mask=None)
                        else:
                            res = checkpoint(self.gcns[layer], h2, edge_index, edge_emb)
                        h = res + h
                    else:
                        if self.spar_wei == 1:
                            if wei_masks is not None and len(wei_masks) > layer:
                                h = self.gcns[layer](h2, edge_index, edge_emb, edge_mask=edge_mask, wei_mask=wei_masks[layer]) + h
                            else:
                                h = self.gcns[layer](h2, edge_index, edge_emb, edge_mask=edge_mask, wei_mask=None) + h
                        else:
                            h = self.gcns[layer](h2, edge_index, edge_emb, edge_mask=edge_mask) + h
            else:
                for layer in range(1, self.num_layers):
                    h1 = self.layer_norms[layer-1](h)
                    h2 = F.relu(h1)
                    h2 = F.dropout(h2, p=self.dropout, training=self.training)
                    if self.spar_wei == 1:
                        if wei_masks is not None and len(wei_masks) > layer:
                            h = self.gcns[layer](h2, edge_index, edge_emb, edge_mask=edge_mask, wei_mask=wei_masks[layer]) + h
                        else:
                            h = self.gcns[layer](h2, edge_index, edge_emb, edge_mask=edge_mask, wei_mask=None) + h
                    else:
                        h = self.gcns[layer](h2, edge_index, edge_emb, edge_mask=edge_mask) + h

            h = F.relu(self.layer_norms[self.num_layers-1](h))
            h = F.dropout(h, p=self.dropout, training=self.training)

            return self.node_pred_linear(h)

        elif self.block == 'res':
            if self.spar_wei == 1:
                if wei_masks is not None and len(wei_masks) >=1:
                    h = F.relu(self.layer_norms[0](self.gcns[0](h, edge_index, edge_emb, edge_mask=edge_mask, wei_mask=wei_masks[0])))
                else:
                    h = F.relu(self.layer_norms[0](self.gcns[0](h, edge_index, edge_emb, edge_mask=edge_mask, wei_mask=None)))
            else:
                h = F.relu(self.layer_norms[0](self.gcns[0](h, edge_index, edge_emb, edge_mask=edge_mask)))

            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                if self.spar_wei == 1:
                    if wei_masks is not None and len(wei_masks) > layer:
                        h1 = self.gcns[layer](h, edge_index, edge_emb, edge_mask=edge_mask, wei_mask=wei_masks[layer])
                    else:
                        h1 = self.gcns[layer](h, edge_index, edge_emb, edge_mask=edge_mask, wei_mask=None)
                else:
                    h1 = self.gcns[layer](h, edge_index, edge_emb, edge_mask=edge_mask)
                h2 = self.layer_norms[layer](h1)
                h = F.relu(h2) + h
                h = F.dropout(h, p=self.dropout, training=self.training)

            return self.node_pred_linear(h)

        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')

        elif self.block == 'plain':
            if self.spar_wei == 1:
                if wei_masks is not None and len(wei_masks) >=1:
                    h = F.relu(self.layer_norms[0](self.gcns[0](h, edge_index, edge_emb, edge_mask=edge_mask, wei_mask=wei_masks[0])))
                else:
                    h = F.relu(self.layer_norms[0](self.gcns[0](h, edge_index, edge_emb, edge_mask=edge_mask, wei_mask=None)))
            else:
                h = F.relu(self.layer_norms[0](self.gcns[0](h, edge_index, edge_emb, edge_mask=edge_mask)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                if self.spar_wei == 1:
                    if wei_masks is not None and len(wei_masks) > layer:
                        h1 = self.gcns[layer](h, edge_index, edge_emb, edge_mask=edge_mask, wei_mask=wei_masks[layer])
                    else:
                        h1 = self.gcns[layer](h, edge_index, edge_emb, edge_mask=edge_mask, wei_mask=None)
                else:
                    h1 = self.gcns[layer](h, edge_index, edge_emb, edge_mask=edge_mask)
                h2 = self.layer_norms[layer](h1)
                h = F.relu(h2)
                h = F.dropout(h, p=self.dropout, training=self.training)

            return self.node_pred_linear(h)

        else:
            raise Exception('Unknown block Type')

    def print_params(self, epoch=None, final=False):

        if self.learn_t:
            ts = []
            for gcn in self.gcns:
                ts.append(gcn.t.item())
            if final:
                print('Final t {}'.format(ts))
            else:
                logging.info('Epoch {}, t {}'.format(epoch, ts))

        if self.learn_p:
            ps = []
            for gcn in self.gcns:
                ps.append(gcn.p.item())
            if final:
                print('Final p {}'.format(ps))
            else:
                logging.info('Epoch {}, p {}'.format(epoch, ps))

        if self.learn_y:
            ys = []
            for gcn in self.gcns:
                ys.append(gcn.sigmoid_y.item())
            if final:
                print('Final sigmoid(y) {}'.format(ys))
            else:
                logging.info('Epoch {}, sigmoid(y) {}'.format(epoch, ys))

        if self.msg_norm:
            ss = []
            for gcn in self.gcns:
                ss.append(gcn.msg_norm.msg_scale.item())
            if final:
                print('Final s {}'.format(ss))
            else:
                logging.info('Epoch {}, s {}'.format(epoch, ss))
