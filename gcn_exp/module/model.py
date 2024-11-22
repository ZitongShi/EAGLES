from ogbn_arxiv.module.moe import moe
from gnn import SAGE, MaskedLinear
import torch
import torch.nn as nn
from GCN import GCN


class model(nn.Module):
    def __init__(self, num_features, num_classes, args, device, params=None):
        super(model, self).__init__()
        self.args = args
        self.device = device
        self.k_list = torch.tensor(args["k_list"], device=device)

        self.expert = moe(input_size=num_features,
                          hidden_size=args["hidden_spl"],
                          num_experts=self.k_list.size(0),
                          nlayers=args["num_layers_spl"],
                          activation=nn.ReLU(),
                          k_list=self.k_list,
                          expert_select=args['expert_select'],
                          lam=args['lam'],
                          )
        if args['dataset'] == 'ogbn-arxiv' or args['dataset'] == 'ogbn-products':
            self.gnn = SAGE(
                in_channels=num_features,
                hidden_channels=args["hidden_channels"],
                out_channels=num_classes,
                num_layers=args["num_layers"],
                dropout=args["dropout"],
                spar_wei = args['spar_wei']
            )
        else:
            self.gnn = GCN(
                nfeat=num_features,
                nhid=args["hidden_channels"],
                nclass=num_classes,
                dropout=args["dropout"],
                lr=args["lr"],
                weight_decay=args["weight_decay"],
                device=device,
                use_ln=args.get('use_ln', False),
                spar_wei=args.get('spar_wei', False),
                layer_norm_first=False,
                layer = args["num_layers"]
            )

    def calculate_accuracy(self, y_true, y_pred):
        correct = (y_true == y_pred).sum().item()
        total = y_true.size(0)
        return correct / total

    def test(self, features, edge_index, edge_weight,shape, labels, split_idx, temp, args):

        self.eval()
        with torch.no_grad():
            mask, add_loss = self.expert(
                x=features,
                edge_index=edge_index,
                temp=temp,
                shape=shape,
                edge_attr=edge_weight,
                training=False
            )
            output = self.gnn(features, edge_index, mask, wei_masks=None)
            sparsity = torch.nonzero(mask).size(0) / mask.numel()
            y_pred = output.argmax(dim=-1)
            train_idx_tensor = torch.tensor(split_idx['train'], device=self.device)
            train_labels = labels[train_idx_tensor]
            train_preds = y_pred[train_idx_tensor]
            train_acc = self.calculate_accuracy(train_labels, train_preds)
            test_idx_tensor = torch.tensor(split_idx['test'], device=self.device)
            test_labels = labels[test_idx_tensor]
            test_preds = y_pred[test_idx_tensor]
            test_acc = self.calculate_accuracy(test_labels, test_preds)

        return train_acc, test_acc, sparsity


