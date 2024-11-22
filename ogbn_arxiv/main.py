import os
import pickle
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon
from ogbn_arxiv.config.args import parser_loader
from ogbn_arxiv.module.model import model
from ogbn_arxiv.module.gnn import MaskedLinear
from ogbn_arxiv.module.moe import moe
from ogbn_arxiv.helper.split import split_Random, split_Louvain
from datasets import  ogba_data,Amazon_data
import ot
def ogba_data(dataset):
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph = dataset[0]
    num_nodes = graph.num_nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    graph.train_mask = train_mask
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[val_idx] = True
    graph.val_mask = val_mask
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_idx] = True
    graph.test_mask = test_mask

    return graph

def train(model: model, features, indices, labels, values, shape, train_idx, temp, optimizer, args, mask=None):

    model.train()
    optimizer.zero_grad()

    mask, add_loss = model.expert(x=features, edge_index=indices,
                                  temp=temp, shape=shape,
                                  edge_attr=values, training=True)

    num_edges = mask.numel()
    num_masks = mask.sum().item()
    output = model.gnn(features, indices, mask,wei_masks=None)
    total_mask_ratio = 0
    if args['spar_wei'] == 1:
        linear_layers = [layer for layer in model.gnn.modules() if isinstance(layer, MaskedLinear)]
        total_non_zero_weights = 0
        total_num_weights = 0

        for layer in linear_layers:
            non_zero_weights = (layer.mask == 1).sum().item()
            total_weights = layer.mask.numel()
            total_non_zero_weights += non_zero_weights
            total_num_weights += total_weights
        if total_num_weights > 0:
            total_mask_ratio = total_non_zero_weights / total_num_weights
        else:
            total_mask_ratio = 0.0

    if labels.dim() == 2 and labels.size(1) == 1:
        labels = labels.squeeze(1)
    elif labels.dim() > 2:
        labels = labels.squeeze()
    labels = labels.long()
    if args['spar_wei'] == 1 and args['save_spar_wei'] == 1:
        wei_loss = 0
        for layer in model.gnn.modules():
            if isinstance(layer, MaskedLinear):
                wei_loss += args['w2loss'] * torch.sum(torch.exp(-layer.threshold))
        loss = F.nll_loss(output[train_idx], labels[train_idx]) + add_loss * args['lambda2'] + wei_loss
    else:
        loss = F.nll_loss(output[train_idx], labels[train_idx]) + add_loss * args['lambda2']
    loss.backward()
    optimizer.step()
    pruning_ratio = num_masks / num_edges if num_edges > 0 else 0
    print(f"Loss: {loss.item()}, pruning ratio: {100 * pruning_ratio:.2f}%")
    if args['spar_wei'] == 0:
        return loss.item(), pruning_ratio, 1
    return loss.item(), pruning_ratio,total_mask_ratio

@torch.no_grad()
def test(model: model, features, indices, labels, values, shape, split_idx, temp, evaluator, args, mask=None):
    model.eval()
    mask, add_loss = model.expert(
        x=features,
        edge_index=indices,
        temp=temp,
        shape=shape,
        edge_attr=values,
        training=False
    )
    output = model.gnn(features, indices, mask, wei_masks=None)

    sparsity = torch.nonzero(mask).size(0) / mask.numel()
    y_pred = output.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({
        'y_true': labels[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': labels[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    return train_acc, test_acc, sparsity


def fed_avg(global_model, client_models):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key], device=global_dict[key].device)
    for client_model in client_models:
        client_dict = client_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] += client_dict[key].to(global_dict[key].device)  # 确保在相同设备
    for key in global_dict.keys():
        global_dict[key] = global_dict[key] / len(client_models)
    global_model.load_state_dict(global_dict)

def get_versioned_file_path(file_path):
    if not os.path.exists(file_path):
        return file_path

    base, ext = os.path.splitext(file_path)
    version = 1
    new_file_path = f"{base}_version_{version}{ext}"

    while os.path.exists(new_file_path):
        version += 1
        new_file_path = f"{base}_version_{version}{ext}"

    return new_file_path



from collections import defaultdict

def load_masks_into_model(model, masks, fixed=True):

    masked_layers = [layer for layer in model.gnn.modules() if isinstance(layer, MaskedLinear)]
    if len(masks) != len(masked_layers):
        raise ValueError("加载的 mask 数量与模型中的 MaskedLinear 层数不匹配。")

    for layer, mask in zip(masked_layers, masks):
        if layer.mask is None:
            layer.mask = mask.to(layer.weight.device)
        else:
            layer.mask = mask.to(layer.mask.device)
        if fixed:
            layer.fixed_mask = True

def main():
    args = parser_loader()
    print(args)
    num_available_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_available_gpus}")
    device = torch.device("cuda:" + str(args['device'])) if torch.cuda.is_available() else torch.device("cpu")

    if args['dataset'] == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name=f"{args['dataset']}", root='./data/')
        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')

        data = ogba_data(dataset)
        print("class", int(data.y.max() + 1))
        print('==============================================================')
        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of edges: {data.num_edges}')

        topo_val = None
        print(data.keys())
    elif args['dataset'] in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root='./data/', \
                            name=args['dataset'], \
                            transform=T.LargestConnectedComponents())

        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
        data = dataset[0]  # Get the graph object.
        print("class", int(data.y.max() + 1))
        print('==============================================================')
        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of edges: {data.num_edges}')
    elif args['dataset'] in ['photo']:
        dataset = Amazon(root='./data/', name=args['dataset'], \
                        transform=T.NormalizeFeatures())
        data = Amazon_data(dataset)
        data.y = data.y.to(dtype=torch.long)
    print('=' * 20 + 'Start Splitting the Data' + '=' * 20)

    if args['is_iid'] == "iid":
        client_data = split_Random(args, data)
    elif args['is_iid'] == "non-iid-louvain":
        client_data = split_Louvain(args, data)

    for i in range(args['num_workers']):
        print(f"Client {i} data length: {len(client_data[i])}")

    print('=' * 20 + 'Display Client-side Data' + '=' * 20)
    for i in range(args['num_workers']):
        print(f"Client:{i}")
        print(client_data[i])
        print(f'Number of nodes: {client_data[i].num_nodes}')
        print(f'Number of edges: {client_data[i].num_edges}')
        print(f'Number of train: {client_data[i].train_mask.sum()}')
        print(f'Number of val: {client_data[i].val_mask.sum()}')
        print(f'Number of test: {client_data[i].test_mask.sum()}')

    print('=' * 20 + 'Start Preparing the Models' + '=' * 20)
    model_list = []
    for i in range(args['num_workers']):
        client_device = device
        client_model = model(data.num_features, dataset.num_classes, args, client_device).to(client_device)
        model_list.append(client_model)
    print(f"{len(model_list)} client models are initialized")
    global_model = model(data.num_features, dataset.num_classes, args, device).to(device)
    k_list_str = '_'.join(map(str, args['k_list']))
    w2loss_history = []
    pruning_rate_history = []
    best_acc_within_range = defaultdict(float)
    best_mask_within_range = {}
    convergence_threshold = 5
    current_epoch = 0
    pruning_ranges = [90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
    pruning_tolerance = 3
    loaded_masks = defaultdict(list)

    if args['load_spar_wei'] == 1:
        for pruning_point in pruning_ranges:
            if args['dataset'] == 'ogbn-arxiv':
                mask_file_path = f"saved_wei/num_layers_{args['num_layers']}_best_masks_pruning_{pruning_point}+-{pruning_tolerance}%.pth"
            else:
                mask_file_path = f"saved_wei/{args['dataset']}_num_layers_{args['num_layers']}_best_masks_pruning_{pruning_point}+-{pruning_tolerance}%.pth"
            if os.path.exists(mask_file_path):
                masks = torch.load(mask_file_path)
                loaded_masks[pruning_point] = masks
                print(f"Loaded mask for pruning rate {pruning_point}±{pruning_tolerance}%: {mask_file_path}")
            else:
                print(f"Mask file for pruning rate {pruning_point}±{pruning_tolerance}% not found.")

        selected_pruning_rate = args['selected_pruning_rate']
        if selected_pruning_rate in loaded_masks:
            print(f"Assigning mask for pruning rate {selected_pruning_rate}±{pruning_tolerance}% to global_model.")
            load_masks_into_model(global_model, loaded_masks[selected_pruning_rate])
        else:
            print(f"Mask for pruning rate {selected_pruning_rate}±{pruning_tolerance}% not loaded, continuing training.")

    if args['spar_wei'] == 0 and args['save_spar_wei'] == 0 and args['load_spar_wei'] == 0:
        os.makedirs('results/normal', exist_ok=True)
        acc_file_path = f"results/normal/dataset_{args['dataset']}_workers_{args['num_workers']}_num_layers_{args['num_layers']}_acc.txt"
    elif args['spar_wei'] == 1 and args['save_spar_wei'] == 1 and args['load_spar_wei'] == 0:
        os.makedirs('results/pretrain', exist_ok=True)
        acc_file_path = f"results/pretrain/dataset_{args['dataset']}_is_spar_wei_{args['spar_wei']}_w2loss_{args['w2loss']}_workers_{args['num_workers']}_num_layers_{args['num_layers']}_acc.txt"
    elif args['spar_wei'] == 1 and args['save_spar_wei'] == 0 and args['load_spar_wei'] == 1:
        os.makedirs('results/retrain', exist_ok=True)
        acc_file_path = f"results/retrain/dataset_{args['dataset']}_num_workers_{args['num_workers']}_{k_list_str}_num_layers_{args['num_layers']}_pruning_rate_{selected_pruning_rate}_acc.txt"

    acc_file_path = get_versioned_file_path(acc_file_path)
    optimizers = []
    for client_model in model_list:
        optimizer = torch.optim.Adam(client_model.parameters(), lr=args['lr'])
        optimizers.append(optimizer)

    if args['spar_wei'] == 1 and args['load_spar_wei'] == 0 and args['save_spar_wei'] == 1:
        current_pruning_index = 0
        total_pruning_points = len(pruning_ranges)
    else:
        current_pruning_index = None
        total_pruning_points = None
    if args['spar_wei'] == 1 and args['load_spar_wei'] == 0 and args['save_spar_wei'] == 1:
        loop_condition = lambda epoch, index: epoch < args['epochs'] and index < total_pruning_points
    else:
        loop_condition = lambda epoch, index: epoch < args['epochs']

    while loop_condition(current_epoch, current_pruning_index):
        if args['spar_wei'] == 1 and args['load_spar_wei'] == 0 and args['save_spar_wei'] == 1:
            current_pruning_point = pruning_ranges[current_pruning_index]
            lower_bound = current_pruning_point - pruning_tolerance
            upper_bound = current_pruning_point + pruning_tolerance
            print('=' * 20 + f"Epoch {current_epoch + 1}/{args['epochs']} Start (Pruning Point: {current_pruning_point}±{pruning_tolerance}%)" + '=' * 20)
        else:
            print('=' * 20 + f"Epoch {current_epoch + 1}/{args['epochs']} Start" + '=' * 20)

        for client_model in model_list:
            if any(k != 1 for k in args['k_list']) or args['spar_wei'] == 1:
                pass
            else:
                client_model.load_state_dict(global_model.state_dict())
            client_model.to(client_model.device)
            global_masked_layers = [layer for layer in global_model.gnn.modules() if isinstance(layer, MaskedLinear)]
            client_masked_layers = [layer for layer in client_model.gnn.modules() if isinstance(layer, MaskedLinear)]
            for global_layer, client_layer in zip(global_masked_layers, client_masked_layers):
                if global_layer.mask is not None:
                    client_layer.mask = global_layer.mask.clone().to(client_model.device)
                    if args['spar_wei'] == 1 and args['load_spar_wei'] == 1:
                        client_layer.fixed_mask = True
                    else:
                        if args['fix_mask'] == 1:
                            client_layer.fixed_mask = True
                        else:
                            client_layer.fixed_mask = False

        print('=' * 20 + "Global model has been distributed to all clients" + '=' * 20)
        client_eval_results = []
        client_sparsity_results = []
        all_wei_mask_ratio = []
        all_sparsity = []
        if args['dataset'] == 'ogbn-arxiv':
            evaluator = Evaluator(name=args['dataset'])
            if args['use_topo']:
                for client_model in model_list:
                    client_model.expert.topo_val = topo_val
            else:
                for client_model in model_list:
                    client_model.expert.topo_val = None

        for client_idx, client_model in enumerate(model_list):
            client_device = client_model.device
            client_features = client_data[client_idx]['x'].to(client_device)
            client_labels = client_data[client_idx]['y'].to(client_device)
            client_indices = client_data[client_idx]['edge_index'].to(client_device)
            client_values = torch.ones((client_data[client_idx].num_edges,), device=client_device)
            client_shape = (client_data[client_idx].num_nodes, client_data[client_idx].num_nodes)
            client_train_mask = client_data[client_idx].train_mask.to(client_device)
            client_test_mask = client_data[client_idx].test_mask.to(client_device)
            client_train_idx = client_train_mask.nonzero(as_tuple=False).view(-1).to(client_device)
            client_test_idx = client_test_mask.nonzero(as_tuple=False).view(-1).to(client_device)
            optimizer = optimizers[client_idx]
            all_sparsity_innner_epoch = []
            all_wei_mask_inner_ratio = []
            for inner_epoch in range(1, args['inner_epochs'] + 1):
                if (current_epoch - 1) % args["temp_N"] == 0:
                    decay_temp = np.exp(-1 * args["temp_r"] * inner_epoch)
                    temp = max(0.05, decay_temp)
                else:
                    temp = 1.0
                loss, pruning_ratio, wei_mask_ratio = train(
                    client_model,
                    client_features,
                    client_indices,
                    client_labels,
                    client_values,
                    client_shape,
                    client_train_idx,
                    temp,
                    optimizer,
                    args,
                    mask=None
                )
                all_sparsity_innner_epoch.append(pruning_ratio)
                all_wei_mask_inner_ratio.append(wei_mask_ratio)
                if inner_epoch == 1:
                    print(f"Client {client_idx}:")
                print(f"Inner Epoch {inner_epoch}, Loss: {loss:.4f}, sparse ratio: {100 * pruning_ratio:.2f}%")
            all_sparsity.append(np.mean(all_sparsity_innner_epoch))
            all_wei_mask_ratio.append(np.mean(all_wei_mask_inner_ratio))
            with torch.no_grad():
                client_model.eval()
                if args['dataset'] == 'ogbn-arxiv':
                    client_result = test(
                        client_model,
                        client_features,
                        client_indices,
                        client_labels,
                        client_values,
                        client_shape,
                        {
                            'train': client_train_idx,
                            'test': client_test_idx
                        },
                        temp,
                        evaluator,
                        args
                    )
                else:
                    client_result = client_model.test(
                        features=client_features,
                        edge_index=client_indices,
                        edge_weight=client_values,
                        shape=client_shape,
                        labels=client_labels,
                        split_idx={
                            'train': client_train_idx,
                            'test': client_test_idx
                        },
                        temp=temp,
                        args=args
                    )
                client_train_acc, client_test_acc, client_sparsity = client_result
                client_eval_results.append(client_test_acc)
                client_sparsity_results.append(client_sparsity)
                print(f'Client {client_idx}, Test Accuracy: {100 * client_test_acc:.2f}%, Sparsity: {100 * client_sparsity:.2f}%')
                print(f"Client {client_idx} Weight Mask Ratio: {100 * np.mean(all_wei_mask_inner_ratio):.2f}%")
            torch.cuda.empty_cache()

        if any(k != 1 for k in args['k_list']) or args['spar_wei'] == 1:
            global_model, model_list, average_wei_mask_ratio = EAGLE_AGG(global_model, model_list, args)
        else:
            average_wei_mask_ratio = 1
            fed_avg(global_model, model_list)

        if len(client_eval_results) > 0:
            average_test_acc = sum(client_eval_results) / len(client_eval_results)
            average_sparsity = sum(client_sparsity_results) / len(client_sparsity_results)
        else:
            average_test_acc = 0.0
            average_sparsity = 0.0
            print("Warning: No client evaluation results to average.")
        print(f'Average Test Accuracy across all clients: {100 * average_test_acc:.2f}%')
        print(f'Average Sparsity across all clients: {100 * average_sparsity:.2f}%')
        print(f'Average Weight Mask Ratio across all clients: {100 * average_wei_mask_ratio:.2f}%')
        print("FedAvg aggregation has been completed")
        with open(acc_file_path, 'a') as acc_file:
            acc_file.write(f"Epoch {current_epoch + 1}, acc:{average_test_acc:.5f}, spar:{average_sparsity:.5f}, wei_mask_ratio:{average_wei_mask_ratio} \n")
        if args['spar_wei'] == 1 and args['load_spar_wei'] == 0 and args['save_spar_wei'] == 1:
            current_w2loss = args['w2loss']
            w2loss_history.append(current_w2loss)
            pruning_rate_history.append(average_wei_mask_ratio)
            current_pruning_ratio = average_wei_mask_ratio
            current_pruning_percentage = current_pruning_ratio * 100
            if lower_bound <= current_pruning_percentage <= upper_bound:
                if average_test_acc > best_acc_within_range[current_pruning_point]:
                    best_acc_within_range[current_pruning_point] = average_test_acc
                    masks = [layer.mask.detach().cpu().clone() for layer in global_model.gnn.modules() if isinstance(layer, MaskedLinear)]
                    best_mask_within_range[current_pruning_point] = masks
                    print(f"Updated best accuracy for pruning rate {current_pruning_point}±{pruning_tolerance}%: {average_test_acc:.4f}%")
            elif current_pruning_percentage < lower_bound:
                if current_pruning_index is not None and current_pruning_point is not None:
                    if best_mask_within_range.get(current_pruning_point) is not None:
                        masks = best_mask_within_range[current_pruning_point]
                        if args['dataset'] == 'ogbn-arxiv':
                            mask_file_path = f"saved_wei/num_layers_{args['num_layers']}_best_masks_pruning_{current_pruning_point}+-{pruning_tolerance}%.pth"
                        else:
                            mask_file_path = f"saved_wei/{args['dataset']}_num_layers_{args['num_layers']}_best_masks_pruning_{current_pruning_point}+-{pruning_tolerance}%.pth"
                        torch.save(masks, mask_file_path)
                        print(f"Best mask for pruning rate {current_pruning_point}±{pruning_tolerance}% has been saved to {mask_file_path}")
                        load_masks_into_model(global_model, masks, fixed=True)
                        print(f"Mask for pruning rate {current_pruning_point}±{pruning_tolerance}% has been loaded into global_model.")
                    else:
                        print(f"No best mask recorded for pruning rate {current_pruning_point}±{pruning_tolerance}%.")

                    current_pruning_index += 1
                    if current_pruning_index < total_pruning_points:
                        next_pruning_point = pruning_ranges[current_pruning_index]
                        print(f"Switching to next pruning rate point: {next_pruning_point}±{pruning_tolerance}%")
                    else:
                        print("All pruning rate points have been processed.")

            if len(pruning_rate_history) >= convergence_threshold:
                recent_pruning = pruning_rate_history[-convergence_threshold:]
                pruning_diffs = [abs(recent_pruning[i] - recent_pruning[i - 1]) for i in
                                 range(1, convergence_threshold)]
                if all(diff < 1 for diff in pruning_diffs):
                    args['w2loss'] = args['w2loss'] * 2.5
                    print(f"Convergence detected based on pruning rate. Increasing w2loss to {args['w2loss']:.6f}")
                    pruning_rate_history = []

            if args['spar_wei'] == 1 and args['load_spar_wei'] == 0 and args['save_spar_wei'] == 1:
                epochs_remaining = args['epochs'] - (current_epoch + 1)
                if epochs_remaining <= 10 and (current_pruning_ratio * 100) > 4:
                    args['epochs'] += 10
                    print(f"Extending training epochs by 10. New total epochs: {args['epochs']}")
            if args['spar_wei'] == 1 and args['load_spar_wei'] == 0 and args['save_spar_wei'] == 1:
                if average_wei_mask_ratio <= 0.04:
                    print("Parameter sparsity rate has reached 4% or below, ending training early.")
                    masks = [layer.mask.detach().cpu().clone() for layer in global_model.gnn.modules() if
                             isinstance(layer, MaskedLinear)]
                    if args['dataset'] == 'ogbn-arxiv':
                        final_mask_file_path = f"saved_wei/num_layers_{args['num_layers']}_best_masks_pruning_0+-4%.pth"
                    else:
                        final_mask_file_path = f"saved_wei/{args['dataset']}_num_layers_{args['num_layers']}_best_masks_pruning_0+-4%.pth"
                    torch.save(masks, final_mask_file_path)
                    print(f"Final mask for pruning rate 0±4% has been saved to {final_mask_file_path}")
                    break
        current_epoch += 1
        print("Training completed.")


def EAGLE_AGG(global_model, client_models, args):

    num_clients = len(client_models)

    client_wei_masks = []
    for client_idx, client_model in enumerate(client_models):
        wei_masks = []
        for layer in client_model.gnn.modules():
            if isinstance(layer, MaskedLinear):
                wei_masks.append(layer.mask.clone().detach())
        client_wei_masks.append(wei_masks)

    num_layers = len(client_wei_masks[0])
    union_wei_masks = []
    for layer_idx in range(num_layers):
        union_mask = client_wei_masks[0][layer_idx].clone()
        for client_idx in range(1, num_clients):
            union_mask = torch.max(union_mask, client_wei_masks[client_idx][layer_idx])
        union_wei_masks.append(union_mask)

    masked_linear_layers = [layer for layer in global_model.gnn.modules() if isinstance(layer, MaskedLinear)]
    for layer, union_mask in zip(masked_linear_layers, union_wei_masks):
        layer.mask = union_mask.clone()

    client_mask_changes = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(num_clients):
            total_weights = 0
            changed_weights = 0
            for layer_idx in range(num_layers):
                mask_i = client_wei_masks[i][layer_idx]
                mask_j = client_wei_masks[j][layer_idx]
                total_weights += mask_i.numel()
                changed_weights += (mask_i != mask_j).sum().item()
            mask_change_ratio = changed_weights / total_weights
            client_mask_changes[i, j] = mask_change_ratio

    client_w_gates = []
    for client_idx, client_model in enumerate(client_models):
        if hasattr(client_model, 'learner'):
            moe = getattr(client_model, 'learner')
        else:
            print(f"Client {client_idx} does not have a 'learner' module.")
            raise AttributeError(f"Client {client_idx} does not have a 'learner' module.")
        if moe and isinstance(moe, moe):
            if hasattr(moe, 'w_gate'):
                w_gate = moe.w_gate.clone().detach()
                client_w_gates.append(w_gate)
                print(f"Client {client_idx}'s w_gate parameters have been extracted.")
            else:
                print(f"Client {client_idx}'s MoE module does not have a 'w_gate' parameter.")
                raise AttributeError(f"Client {client_idx}'s MoE module does not have a 'w_gate' parameter.")
        else:
            print(f"Client {client_idx}'s 'learner' module is not of MoE type.")
            raise AttributeError(f"Client {client_idx}'s 'learner' module is not of MoE type.")

    ot_distances = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(num_clients):
            w_gate_i = client_w_gates[i]
            w_gate_j = client_w_gates[j]
            flattened_i = w_gate_i.view(-1).cpu().numpy()
            flattened_j = w_gate_j.view(-1).cpu().numpy()
            a = np.ones(len(flattened_i)) / len(flattened_i)
            b = np.ones(len(flattened_j)) / len(flattened_j)
            M = ot.dist(flattened_i.reshape(-1, 1), flattened_j.reshape(-1, 1))
            ot_distance = ot.emd2(a, b, M)
            ot_distances[i, j] = ot_distance

    mask_changes_normalized = (client_mask_changes - client_mask_changes.min()) / (
            client_mask_changes.max() - client_mask_changes.min() + 1e-8)
    ot_distances_normalized = (ot_distances - ot_distances.min()) / (ot_distances.max() - ot_distances.min() + 1e-8)

    similarity_matrix = 1.0 - (mask_changes_normalized + ot_distances_normalized) / 2.0
    similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)
    aggregation_weights = similarity_matrix / (similarity_matrix.sum(axis=1, keepdims=True) + 1e-8)

    for client_idx in range(num_clients):
        weights = aggregation_weights[client_idx]
        print(f"Client {client_idx}'s aggregation weights: {weights}")

    personalized_state_dicts = {client_idx: {} for client_idx in range(num_clients)}
    model_keys = global_model.state_dict().keys()
    client_state_dicts = [client_model.state_dict() for client_model in client_models]

    for key in model_keys:
        params = [client_state_dict[key].float() for client_state_dict in client_state_dicts]
        for client_idx in range(num_clients):
            weights = aggregation_weights[client_idx]
            personalized_param = sum(w * p for w, p in zip(weights, params))
            personalized_state_dicts[client_idx][key] = personalized_param

    for client_idx, client_model in enumerate(client_models):
        client_model.load_state_dict(personalized_state_dicts[client_idx])
        print(f"Client {client_idx}'s model parameters have been personalized and updated.")
    total_mask_ratio = 0
    if args['spar_wei'] == 1:
        linear_layers = [layer for layer in global_model.gnn.modules() if isinstance(layer, MaskedLinear)]
        total_non_zero_weights = 0
        total_num_weights = 0
        for layer in linear_layers:
            non_zero_weights = (layer.mask == 1).sum().item()
            total_weights = layer.mask.numel()
            total_non_zero_weights += non_zero_weights
            total_num_weights += total_weights
        if total_num_weights > 0:
            total_mask_ratio = total_non_zero_weights / total_num_weights
        else:
            total_mask_ratio = 0.0

    return global_model, client_models, total_mask_ratio


if __name__ == "__main__":
    main()
