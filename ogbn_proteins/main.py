from collections import defaultdict

import os
import pickle
from ogbn_proteins.config.args import ArgsInit
from ogb.nodeproppred import Evaluator
from torch_geometric.utils import scatter
from ogbn_proteins.helper.helperfunc import fed_avg, EAGLE_AGG,load_masks_into_model,get_versioned_file_path
from ogbn_proteins.module.gnn_conv import MaskedLinear
from ogbn_proteins.helper.split import split_Louvain
from ogbn_proteins.module.model import model
import torch
import torch.nn as nn
import statistics
import numpy as np
import math
import logging

def train(client_graph, train_idx, valid_idx, test_idx, model, optimizer, criterion, temp, device, args, use_topo=True):
    loss_list = []
    model.train()
    device = next(model.parameters()).device

    client_graph.y = client_graph.y.to(device)
    client_graph.x = client_graph.x.to(device)
    client_graph.edge_index = client_graph.edge_index.to(device)
    train_y = client_graph.y[train_idx].to(device)
    train_x = client_graph.x[train_idx].float().to(device)
    edges_index = client_graph.edge_index.to(device)
    edges_attr = client_graph.edge_attr.to(device) if client_graph.edge_attr is not None else None

    num_global_nodes = client_graph.num_nodes
    mapper_tensor = torch.full((num_global_nodes,), -1, dtype=torch.long, device=device)
    mapper_tensor[train_idx] = torch.arange(train_idx.size(0), device=device)

    edges_index_mapped = mapper_tensor[edges_index]

    valid_edges_mask = (edges_index_mapped[0, :] != -1) & (edges_index_mapped[1, :] != -1)
    edges_index_mapped = edges_index_mapped[:, valid_edges_mask]

    if edges_attr is not None:
        edges_attr_mapped = edges_attr[valid_edges_mask]
    else:
        edges_attr_mapped = None

    optimizer.zero_grad()

    if use_topo:
        try:
            model.learner.get_topo_val(edges_index_mapped)
            print("Successfully called get_topo_val on learner.")
        except Exception as e:
            print(f"Error calling get_topo_val: {e}")

    try:
        mask, add_loss = model.learner(train_x, edges_index_mapped, temp, edges_attr_mapped, True)
    except Exception as e:
        print(f"Error during model.learner forward pass: {e}")
        raise

    if mask is None:
        raise ValueError("mask returned from model.learner is None.")
    if add_loss is None:
        raise ValueError("add_loss returned from model.learner is None.")

    total_flops = 0

    num_edges = mask.numel()
    num_masks = mask.sum().item()
    feature_dim = train_x.shape[1]
    # 假设每个特征维度涉及一次乘法和一次加法
    message_passing_flops = 2 * num_masks * feature_dim
    total_flops += message_passing_flops
    print(f"Message Passing FLOPs: {message_passing_flops}")

    try:
        if args.spar_wei == 1:
            pred = model.gnn(train_x, train_idx, edges_index_mapped, edges_attr_mapped, edge_mask=mask, wei_masks=None)
        else:
            pred = model.gnn(train_x, train_idx, edges_index_mapped, edges_attr_mapped, edge_mask=mask)
    except Exception as e:
        print(f"Error during model.gnn forward pass: {e}")
        raise

    num_nodes = train_x.shape[0]
    total_mask_ratio = 0
    if args.spar_wei == 1:
        # 稀疏线性层 FLOPs
        linear_layers = [layer for layer in model.gnn.modules() if isinstance(layer, MaskedLinear)]
        total_non_zero_weights = 0
        total_num_weights = 0
        for layer in linear_layers:
            non_zero_weights = (layer.mask == 1).sum().item()
            total_weights = layer.mask.numel()
            total_non_zero_weights += non_zero_weights
            total_num_weights += total_weights
            flops = 2 * num_nodes * non_zero_weights
            total_flops += flops
            #print(f"MaskedLinear Layer FLOPs: {flops}")
        if total_num_weights > 0:
            total_mask_ratio = total_non_zero_weights / total_num_weights
        else:
            total_mask_ratio = 0.0
    else:
        linear_layers = [layer for layer in model.gnn.modules() if isinstance(layer, nn.Linear)]
        for layer in linear_layers:
            in_features = layer.in_features
            out_features = layer.out_features
            flops = 2 * num_nodes * in_features * out_features
            total_flops += flops
            print(f"Linear Layer FLOPs: {flops}")

    moe_flops = 0

    input_size = model.learner.w_gate.shape[0]
    num_experts = model.learner.num_experts

    gate_flops = 2 * num_nodes * input_size * num_experts  # Multiply and Add
    moe_flops += gate_flops
    print(f"MoE Gate Linear Transformation FLOPs: {gate_flops}")

    softmax_flops = 2 * num_nodes * num_experts
    topk_flops = num_nodes * num_experts * math.log(num_experts, 2)
    moe_flops += softmax_flops + topk_flops
    print(f"MoE Softmax FLOPs: {softmax_flops}, Top-K FLOPs: {topk_flops}")

    for expert in model.learner.experts:
        for layer in expert.layers:
            in_features = layer.in_features
            out_features = layer.out_features
            flops = 2 * num_nodes * in_features * out_features
            moe_flops += flops
            print(f"MoE Expert Layer FLOPs: {flops}")

    total_flops += moe_flops
    print(f"MoE Total FLOPs: {moe_flops}")

    relu_flops = train_x.numel()
    total_flops += relu_flops
    print(f"ReLU FLOPs: {relu_flops}")

    bn_layers = [layer for layer in model.gnn.modules() if isinstance(layer, nn.BatchNorm1d)]
    for bn in bn_layers:
        bn_flops = 2 * bn.num_features * num_nodes
        total_flops += bn_flops
        print(f"BatchNorm FLOPs: {bn_flops}")

    binary_step_flops = num_edges
    total_flops += binary_step_flops
    print(f"BinaryStep FLOPs: {binary_step_flops}")

    print(f"Total FLOPs in this iteration: {total_flops}")

    target = train_y.to(torch.float32)

    try:
        if model.gnn.spar_wei:
            wei_loss = 0
            for layer in model.gnn.modules():
                if isinstance(layer, MaskedLinear):
                    wei_loss += args.w2loss * torch.sum(torch.exp(-layer.threshold))
            loss = criterion(pred.to(torch.float32), target) + add_loss * args.lambda2 + wei_loss
        else:
            loss = criterion(pred.to(torch.float32), target) + add_loss * args.lambda2
    except Exception as e:
        print(f"Error during loss computation: {e}")
        raise

    loss.backward()
    optimizer.step()
    loss_list.append(loss.item())

    pruning_ratio = num_masks / num_edges if num_edges > 0 else 0
    print(f"Loss: {loss.item()}, pruning ratio: {100 * pruning_ratio:.2f}%")

    if args.spar_wei == 0:
        return statistics.mean(loss_list), pruning_ratio, total_flops, 1
    return statistics.mean(loss_list), pruning_ratio, total_flops, total_mask_ratio

@torch.no_grad()
def multi_evaluate(client_graph, model, evaluator, temp, device,use_topo=True, train_idx=None, valid_idx=None,
                   test_idx=None, args=None):
    model.eval()
    device = next(model.parameters()).device

    client_graph.y = client_graph.y.to(device)
    client_graph.x = client_graph.x.to(device)
    client_graph.edge_index = client_graph.edge_index.to(device)
    if client_graph.edge_attr is not None:
        client_graph.edge_attr = client_graph.edge_attr.to(device)

    train_idx = train_idx.to(device)
    valid_idx = valid_idx.to(device)
    test_idx = test_idx.to(device)

    if use_topo:
        model.learner.get_topo_val(client_graph.edge_index)

    mask, add_loss = model.learner(client_graph.x, client_graph.edge_index, temp, client_graph.edge_attr, False)

    if mask is None:
        raise ValueError("mask returned from model.learner is None.")
    if add_loss is None:
        raise ValueError("add_loss returned from model.learner is None.")

    try:
        if args.spar_wei == 1:
            pred = model.gnn(client_graph.x, torch.arange(client_graph.num_nodes).to(device), client_graph.edge_index,
                             client_graph.edge_attr, edge_mask=mask, wei_masks=None)
        else:
            pred = model.gnn(client_graph.x, torch.arange(client_graph.num_nodes).to(device), client_graph.edge_index,
                             client_graph.edge_attr, edge_mask=mask)
    except Exception as e:
        print(f"Error during model.gnn forward pass: {e}")
        raise

    num_edges = mask.numel()
    num_masks = mask.sum().item()
    feature_dim = client_graph.x.shape[1]
    message_passing_flops = num_masks * feature_dim
    num_nodes = client_graph.x.shape[0]

    if args.spar_wei == 1:
        linear_layers = [layer for layer in model.gnn.modules() if isinstance(layer, MaskedLinear)]
        feature_transform_flops = sum(2 * num_nodes * (layer.mask == 1).sum().item() for layer in linear_layers)
    else:
        linear_layers = [layer for layer in model.gnn.modules() if isinstance(layer, torch.nn.Linear)]
        feature_transform_flops = sum(2 * num_nodes * layer.in_features * layer.out_features for layer in linear_layers)

    total_flops = message_passing_flops + feature_transform_flops
    target = client_graph.y.cpu()

    train_pred = pred[train_idx.cpu()].cpu().numpy()
    valid_pred = pred[valid_idx.cpu()].cpu().numpy()
    test_pred = pred[test_idx.cpu()].cpu().numpy()

    train_y = target[train_idx.cpu()].cpu().numpy()
    valid_y = target[valid_idx.cpu()].cpu().numpy()
    test_y = target[test_idx.cpu()].cpu().numpy()

    eval_result = {}

    input_dict = {"y_true": train_y, "y_pred": train_pred}
    eval_result["train"] = evaluator.eval(input_dict)

    input_dict = {"y_true": valid_y, "y_pred": valid_pred}
    eval_result["valid"] = evaluator.eval(input_dict)

    input_dict = {"y_true": test_y, "y_pred": test_pred}
    eval_result["test"] = evaluator.eval(input_dict)

    return eval_result, total_flops


def extract_node_features(client_data, num_workers,aggr='add',idx=None):
    file_path = 'node_features/Client_workers_{}_{}_init_node_features_{}.pt'.format(num_workers,idx, aggr)
    if os.path.isfile(file_path):
        print('{} exists'.format(file_path))
    else:
        if aggr in ['add', 'mean', 'max']:
            node_features = scatter(client_data.edge_attr,
                                    client_data.edge_index[0],
                                    dim=0,
                                    dim_size=client_data.y.shape[0],
                                    reduce=aggr)
        else:
            raise Exception('Unknown Aggr Method')
        torch.save(node_features, file_path)
        print('Node features extracted are saved into file {}'.format(file_path))
    return file_path




def main():
    args = ArgsInit().save_exp()
    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")
    logging.info(f'Device: {device}')

    from ogb.nodeproppred import PygNodePropPredDataset
    dataset = PygNodePropPredDataset(name='ogbn-proteins', root='./data/')
    data = dataset[0]

    if data.edge_attr is not None:
        row, col = data.edge_index
        data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce='sum')
    _, f_dim = data.x.size()
    logging.info(f'ogbn-proteins Number of features: {f_dim}')
    logging.info(f"data.y shape: {data.y.shape}")
    print(data)
    args.num_tasks = dataset.num_tasks

    logging.info(f'Args: {args}')

    evaluator = Evaluator(args.dataset)
    criterion = torch.nn.BCEWithLogitsLoss()
    print('=' * 20 + 'Start Splitting the Data' + '=' * 20)

    client_data = split_Louvain(args, data)

    for i in range(args.num_workers):
        print(f"Client {i} data length: {len(client_data[i])}")
    print('=' * 20 + 'Display Client-side Data' + '=' * 20)

    for client_idx, client_graph in enumerate(client_data):
        print(f"Client {client_idx} graph: {client_graph}")
        print(f'Number of nodes: {client_graph.num_nodes}')
        print(f'Number of edges: {client_graph.num_edges}')
        if client_graph.x is not None:
            print(f"client_graph.x shape: {client_graph.x.shape}")
        else:
            raise ValueError(f"client_graph {client_idx}.x is None. Please check split_Louvain function.")

    print('=' * 20 + 'Start Preparing the Models' + '=' * 20)
    nf_path = extract_node_features(client_data[0], args.num_workers, 'add', 0)
    args.nf_path = nf_path
    global_model = model(data.x.size(1), dataset.num_tasks, args, device).to(device)
    k_list_str = '_'.join(map(str, args.k_list))
    w2loss_history = []
    pruning_rate_history = []
    best_acc_within_range = defaultdict(float)
    best_mask_within_range = {}
    convergence_threshold = 5
    current_epoch = 0

    pruning_ranges = [90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
    pruning_tolerance = 3
    loaded_masks = defaultdict(list)

    if args.load_spar_wei == 1:
        for pruning_point in pruning_ranges:
            mask_file_path = f"saved_wei/num_layers_{args.num_layers}_best_masks_pruning_{pruning_point}+-{pruning_tolerance}%.pth"
            if os.path.exists(mask_file_path):
                masks = torch.load(mask_file_path)
                loaded_masks[pruning_point] = masks
                print(f"Load the pruning rate num_layers_{args.num_layers}_best_masks_pruning_{pruning_point}+-{pruning_tolerance}% mask: {mask_file_path}")
            else:
                print(f"A mask file with pruning rate num_layers_{args.num_layers}_best_masks_pruning_{pruning_point}+-{pruning_tolerance}% was not found")

        selected_pruning_rate = args.selected_pruning_rate
        if selected_pruning_rate in loaded_masks:
            print(f"The mask with pruning rate {selected_pruning_rate}±{pruning_tolerance}% is assigned to global_model.")
            load_masks_into_model(global_model, loaded_masks[selected_pruning_rate])
        else:
            print(f"Training continues with the mask with the pruning rate {selected_pruning_rate}±{pruning_tolerance}% not loaded.")

    os.makedirs('Aba_results', exist_ok=True)
    acc_file_path = f"Aba_results/dataset_{args.dataset}_{k_list_str}_lambda2_{args.lambda2}_acc.txt"
    acc_file_path = get_versioned_file_path(acc_file_path)

    client_models = []
    for idx in range(args.num_workers):
        nf_path = extract_node_features(client_data[idx], args.num_workers,'add', idx)
        args.nf_path = nf_path
        client_model = model(data.x.size(1), dataset.num_tasks, args, device).to(device)
        client_models.append(client_model)

    for idx, client_model in enumerate(client_models):
        assert global_model is not None, "global_model is None before loading state_dict"
        assert client_model is not None, f"client_model {idx} is None before loading state_dict"
        client_model.load_state_dict(global_model.state_dict())


    train_idx_list = []
    valid_idx_list = []
    test_idx_list = []

    for client_idx, client_graph in enumerate(client_data):
        num_nodes = client_graph.num_nodes
        all_indices = torch.randperm(num_nodes)

        train_size = int(0.6 * num_nodes)
        valid_size = int(0.2 * num_nodes)

        train_idx = all_indices[:train_size]
        valid_idx = all_indices[train_size:train_size + valid_size]
        test_idx = all_indices[train_size + valid_size:]

        train_idx_list.append(train_idx)
        valid_idx_list.append(valid_idx)
        test_idx_list.append(test_idx)

    optimizers = []
    for client_model in client_models:
        optimizer = torch.optim.Adam(client_model.parameters(), lr=args.lr)
        optimizers.append(optimizer)

    if args.spar_wei == 1 and args.load_spar_wei == 0 and args.save_spar_wei == 1:
        current_pruning_index = 0
        total_pruning_points = len(pruning_ranges)
    else:
        current_pruning_index = None
        total_pruning_points = None

    total_upload_bytes = 0
    total_download_bytes = 0
    total_bytes =0

    for epoch in range(1, args.epochs + 1):
        if args.spar_wei == 1 and args.load_spar_wei == 0 and args.save_spar_wei == 1:
            current_pruning_point = pruning_ranges[current_pruning_index]
            lower_bound = current_pruning_point - pruning_tolerance
            upper_bound = current_pruning_point + pruning_tolerance
            print('=' * 20 + f"Epoch {epoch }/{args.epochs} Start (Pruning Point: {current_pruning_point}±{pruning_tolerance}%)" + '=' * 20)
        else:
            print('=' * 20 + f"Epoch {epoch }/{args.epochs} Start" + '=' * 20)

        for client_model in client_models:
            if args.spar_wei != 1:
                client_model.load_state_dict(global_model.state_dict())
            else:
                pass
            client_model.to(client_model.device)
            global_masked_layers = [layer for layer in global_model.gnn.modules() if isinstance(layer, MaskedLinear)]
            client_masked_layers = [layer for layer in client_model.gnn.modules() if isinstance(layer, MaskedLinear)]
            for global_layer, client_layer in zip(global_masked_layers, client_masked_layers):
                if global_layer.mask is not None:
                    client_layer.mask = global_layer.mask.clone().to(client_model.device)
                    if args.spar_wei == 1 and args.load_spar_wei == 1:
                        client_layer.fixed_mask = True
                    else:
                        client_layer.fixed_mask = False
        print('=' * 20 + "Global model has been distributed to all clients" + '=' * 20)
        all_train_results = []
        all_valid_results = []
        all_test_results = []
        all_sparsity = []
        all_wei_mask_ratio = []
        total_flops_per_epoch = 0
        total_flops_per_epoch_eval = 0
        all_client_train_flops = []
        all_client_eval_flops = []
        for client_idx in range(args.num_workers):

            optimizer = optimizers[client_idx]
            device = next(client_models[client_idx].parameters()).device
            train_idx = train_idx_list[client_idx].to(device)
            valid_idx = valid_idx_list[client_idx].to(device)
            test_idx = test_idx_list[client_idx].to(device)
            all_sparsity_innner_epoch = []
            all_wei_mask_inner_ratio = []
            for inner_epoch in range(1, args.inner_epochs + 1):
                if (epoch - 1) % args.temp_N == 0:
                    decay_temp = np.exp(-args.temp_r * inner_epoch)
                    temp = max(0.05, decay_temp)
                    logging.debug(f"Updated temperature: {temp}")

                epoch_loss, sparsity,flops,wei_mask_ratio = train(
                    client_data[client_idx],
                    train_idx,
                    valid_idx,
                    test_idx,
                    client_models[client_idx],
                    optimizer,criterion,
                    temp,
                    device,
                    args,
                    args.use_topo)
                total_flops_per_epoch += flops

                if inner_epoch == 1:
                    print(f"Client {client_idx}:")
                print(f"Inner Epoch {inner_epoch}, Loss: {epoch_loss:.4f}, sparse ratio: {100 * sparsity:.2f}%")
                all_sparsity_innner_epoch.append(sparsity)
                all_wei_mask_inner_ratio.append(wei_mask_ratio)
            all_sparsity.append(np.mean(all_sparsity_innner_epoch) if all_sparsity_innner_epoch else 0.0)
            all_wei_mask_ratio.append(np.mean(all_wei_mask_inner_ratio) if all_wei_mask_inner_ratio else 0.0)
            result,eval_flops = multi_evaluate(
                client_data[client_idx],
                client_models[client_idx],
                evaluator,
                temp,
                device,
                args.use_topo,
                train_idx, valid_idx, test_idx,
                args=args
            )
            total_flops_per_epoch_eval += eval_flops
            total_flops_per_epoch = total_flops_per_epoch / args.inner_epochs
            all_client_train_flops.append(total_flops_per_epoch)
            all_client_eval_flops.append(total_flops_per_epoch_eval)
            print(f"Client {client_idx} Evaluation Results: {result}, sparse ratio: {100 * np.mean(all_sparsity_innner_epoch):.2f}%")
            print(f"Client {client_idx} train flops: {total_flops_per_epoch}, eval flops: {total_flops_per_epoch_eval}")
            print(f"Client {client_idx} Weight Mask Ratio: {100 * np.mean(all_wei_mask_inner_ratio):.2f}%")

        if args.spar_wei == 1:
            global_model, client_models, communication_bytes,average_wei_mask_ratio = EAGLE_AGG(global_model, client_models, args)
            total_upload_bytes += communication_bytes['upload_bytes']
            total_download_bytes += communication_bytes['download_bytes']
            total_bytes += communication_bytes['upload_bytes'] + communication_bytes['download_bytes']
        else:
            average_wei_mask_ratio = 1
            global_model, upload_bytes, download_bytes = fed_avg(global_model, client_models)
            communication_bytes = {
                'upload_bytes': upload_bytes,
                'download_bytes': download_bytes
            }
            total_upload_bytes += communication_bytes['upload_bytes']
            total_download_bytes += communication_bytes['download_bytes']
            total_bytes += communication_bytes['upload_bytes'] + communication_bytes['download_bytes']

        if 'train' in result:
            all_train_results.append(result['train']['rocauc'])
        if 'valid' in result:
            all_valid_results.append(result['valid']['rocauc'])
        if 'test' in result:
            all_test_results.append(result['test']['rocauc'])
        average_train_acc = np.mean(all_train_results) if all_train_results else 0.0
        average_valid_acc = np.mean(all_valid_results) if all_valid_results else 0.0
        average_test_acc = np.mean(all_test_results) if all_test_results else 0.0
        average_sparsity = np.mean(all_sparsity) if all_sparsity else 0.0
        average_train_flops = np.mean(all_client_train_flops) if all_client_train_flops else 0.0
        average_eval_flops = np.mean(all_client_eval_flops) if all_client_eval_flops else 0.0
        print(f'Average Accuracy across all clients: train ROC AUC:{100 * average_train_acc:.2f}, '
              f'valid ROC AUC:{100 * average_valid_acc:.2f}, test ROC AUC:{100 * average_test_acc:.2f}%')
        print(f'Average Sparsity across all clients: {100 * average_sparsity:.2f}%')
        print(f'Average FLOPs across all clients: train FLOPs:{average_train_flops}, eval FLOPs:{average_eval_flops}')
        print(f'Average Weight Mask Ratio across all clients: {100 * average_wei_mask_ratio:.2f}%')

        with open(acc_file_path, 'a') as acc_file:
            acc_file.write(f"Epoch:{epoch }, acc:{average_test_acc}, spar:{average_sparsity}, train_flops:{average_train_flops:.4f}, eval_flops:{average_eval_flops}, wei_mask_ratio:{average_wei_mask_ratio}, total_bytes:{total_bytes} \n")

        if args.spar_wei == 1 and args.load_spar_wei == 0 and args.save_spar_wei == 1:
            current_w2loss = args.w2loss
            w2loss_history.append(current_w2loss)
            pruning_rate_history.append(average_wei_mask_ratio)
            current_pruning_ratio = average_wei_mask_ratio
            current_pruning_percentage = current_pruning_ratio * 100
            if lower_bound <= current_pruning_percentage <= upper_bound:
                if average_test_acc > best_acc_within_range[current_pruning_point]:
                    best_acc_within_range[current_pruning_point] = average_test_acc
                    masks = [layer.mask.detach().cpu().clone() for layer in global_model.gnn.modules() if isinstance(layer, MaskedLinear)]
                    best_mask_within_range[current_pruning_point] = masks
                    print(f"The best accuracy for pruning rate {current_pruning_point}±{pruning_tolerance}% is updated: {average_test_acc:.4f}%")
            elif current_pruning_percentage < lower_bound:
                if current_pruning_index is not None and current_pruning_point is not None:
                    if best_mask_within_range.get(current_pruning_point) is not None:
                        masks = best_mask_within_range[current_pruning_point]
                        mask_file_path = f"saved_wei/num_layers_{args.num_layers}_best_masks_pruning_{current_pruning_point}+-{pruning_tolerance}%.pth"
                        torch.save(masks, mask_file_path)
                        print(f"The mask for the optimal pruning rate {current_pruning_point}±{pruning_tolerance}% is saved to {mask_file_path}")
                        load_masks_into_model(global_model, masks, fixed=True)
                        print(f"A mask of pruning rate {current_pruning_point}±{pruning_tolerance}% has been loaded into global_model")
                    else:
                        print(f"The best mask for the pruning rate {current_pruning_point}±{pruning_tolerance}% was not recorded")
                    current_pruning_index += 1
                    if current_pruning_index < total_pruning_points:
                        next_pruning_point = pruning_ranges[current_pruning_index]
                        print(f"Switch to the next pruning rate point: {next_pruning_point}±{pruning_tolerance}%")
                    else:
                        print("All pruning rate points have been processed.")

            if args.spar_wei == 1 and args.load_spar_wei == 0 and args.save_spar_wei == 1:
                epochs_remaining = args.epochs - (current_epoch + 1)
                if epochs_remaining <= 10 and (current_pruning_ratio * 100) > 4:
                    args.epochs += 10
                    print(f"The number of training rounds was extended by 10 rounds. New total rounds: {args.epochs}")
            if args.spar_wei == 1 and args.load_spar_wei == 0 and args.save_spar_wei == 1:
                if average_wei_mask_ratio <= 0.04:
                    print("End training early if the parameter sparsification rate has reached 4% or less.")
                    masks = [layer.mask.detach().cpu().clone() for layer in global_model.gnn.modules() if
                             isinstance(layer, MaskedLinear)]
                    final_mask_file_path = f"saved_wei/num_layers_{args.num_layers}_final_masks_pruning_0+-4%.pth"
                    torch.save(masks, final_mask_file_path)
                    print(f"A mask with a final pruning rate of 0±4% is saved to {final_mask_file_path}")
                    break
        print("Training completed.")


if __name__ == "__main__":
    main()