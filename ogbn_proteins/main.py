import os
import pdb
import pickle
import torch_geometric.utils as tg_utils
import torch
import torch.optim as optim
import statistics
from args import ArgsInit
import time
import numpy as np
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.utils import scatter
from gnn_conv import MaskedLinear
from split import split_Louvain
import logging
from MoG import MoG
from MoE import MoE
import ot

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def random_partition_graph(num_nodes, cluster_number=10):
    """
    Randomly partition the nodes into different clusters.
    :param num_nodes: The total number of nodes in the graph
    :param cluster_number: The number of clusters
    :return an array of length num_nodes where each element represents the cluster number to which the node belongs
    """
    parts = np.random.randint(0, cluster_number, size=num_nodes)
    return parts

def create_edge_index_dict(edge_index):
    """
    A dictionary is created based on edge_index for quick edge lookup.
    :param edge_index: Edge index of the graph, of shape [2, num_edges]
    :return: A dictionary that maps (node 1, node 2) to the index of the edge
    """
    edge_index_dict = {}
    edge_index_np = edge_index.numpy()

    for idx in range(edge_index_np.shape[1]):
        node_pair = (edge_index_np[0, idx], edge_index_np[1, idx])
        edge_index_dict[node_pair] = idx

    return edge_index_dict

def generate_sub_graphs(data, parts, cluster_number=10, batch_size=1):
    """
    The subgraph is generated according to the partition result.
    :param data: Raw graph data
    :param parts: An array of partitioned cluster labels
    :param cluster_number: The number of clusters
    :param batch_size: Number of clusters per batch
    :return: node, edge, edge index, and original edge list of the subgraph
    """
    no_of_batches = cluster_number // batch_size
    print('The number of clusters: {}'.format(cluster_number))
    sg_nodes = [[] for _ in range(no_of_batches)]
    sg_edges = [[] for _ in range(no_of_batches)]
    sg_edges_orig = [[] for _ in range(no_of_batches)]
    sg_edges_index = [[] for _ in range(no_of_batches)]
    edges_no = 0
    edge_index = data.edge_index
    edge_index_dict = create_edge_index_dict(edge_index)

    for cluster in range(no_of_batches):
        sg_nodes[cluster] = torch.tensor(np.where(parts == cluster)[0], dtype=torch.long)
        sg_edges[cluster] = tg_utils.subgraph(sg_nodes[cluster], edge_index, relabel_nodes=True)[0]  # 提取子图的边
        edges_no += sg_edges[cluster].shape[1]
        # mapper
        mapper = {nd_idx: nd_orig_idx for nd_idx, nd_orig_idx in enumerate(sg_nodes[cluster])}
        # map edges to original edges
        sg_edges_orig[cluster] = edge_list_mapper(mapper, sg_edges[cluster])
        # edge index
        sg_edges_index[cluster] = [edge_index_dict.get((edge[0], edge[1]), -1) for edge in sg_edges_orig[cluster].t().numpy()]
        sg_edges_index[cluster] = [idx for idx in sg_edges_index[cluster] if idx != -1]

    print('Total number edges of sub graphs: {}, of whole graph: {}, {:.2f} % edges are lost'.
          format(edges_no, data.num_edges, (1 - edges_no / data.num_edges) * 100))

    return sg_nodes, sg_edges, sg_edges_index, sg_edges_orig

def edge_list_mapper(mapper, sg_edges_list):
    """
    Map the edge index in sg_edges_list to the corresponding index value of the mapper.
    :param mapper: A node mapping dictionary that maps the original node ID to the new node ID
    :param sg_edges_list: The original edge list
    :return: The mapped edge list
    """
    idx_1st = list(map(lambda x: mapper[x], sg_edges_list[0].tolist()))
    idx_2nd = list(map(lambda x: mapper[x], sg_edges_list[1].tolist()))
    sg_edges_orig = torch.LongTensor([idx_1st, idx_2nd])
    return sg_edges_orig
def calculate_model_size(model):
    """计算模型参数的总字节数"""
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    return total_bytes

def fed_avg(global_model, client_models):
    """Aggregating client model parameters with FedAvg，并计算通信字节量

    :param global_model: 全局模型
    :param client_models: 客户端模型列表
    :return: 总上传字节量, 总下载字节量
    """
    num_clients = len(client_models)

    # 计算单个模型的字节大小
    model_size = calculate_model_size(global_model)

    # 计算总上传字节量（每个客户端上传一个模型）
    total_upload_bytes = num_clients * model_size

    # 计算总下载字节量（服务器将全局模型发送给每个客户端）
    total_download_bytes = num_clients * model_size

    # 执行FedAvg聚合
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
    #pdb.set_trace()
    return global_model, total_upload_bytes, total_download_bytes

def my_fedaggregation(global_model, client_models, args):
    """执行个性化联邦聚合，每个客户端根据与其他客户端的相似度计算自己的聚合权重，
    并从所有客户端的模型参数中聚合得到个性化的模型参数。同时计算通信字节量。

    :param global_model: 全局模型，用于掩码对齐
    :param client_models: 客户端模型列表
    :param args: 参数
    :return: 更新后的全局模型和个性化的客户端模型列表，以及通信字节量信息
    """
    num_clients = len(client_models)
    communication_bytes = {
        'upload_bytes': 0,
        'download_bytes': 0
    }

    # 步骤 1: 获取每个客户端的 wei_mask
    client_wei_masks = []
    for client_idx, client_model in enumerate(client_models):
        wei_masks = []
        for layer in client_model.gnn.modules():
            if isinstance(layer, MaskedLinear):
                # 确保 layer.mask 与计算图分离
                wei_masks.append(layer.mask.clone().detach())
        client_wei_masks.append(wei_masks)

    # 步骤 2: 获取所有客户端的 wei_mask 的并集
    num_layers = len(client_wei_masks[0])  # MaskedLinear 层的数量
    union_wei_masks = []
    for layer_idx in range(num_layers):
        # 从第一个客户端的掩码开始
        union_mask = client_wei_masks[0][layer_idx].clone()
        for client_idx in range(1, num_clients):
            union_mask = torch.max(union_mask, client_wei_masks[client_idx][layer_idx])
        union_wei_masks.append(union_mask)

    # 步骤 3: 使用并集 wei_mask 更新 global_model 的 wei_mask
    masked_linear_layers = [layer for layer in global_model.gnn.modules() if isinstance(layer, MaskedLinear)]
    for layer, union_mask in zip(masked_linear_layers, union_wei_masks):
        layer.mask = union_mask.clone()

    # 步骤 4: 计算客户端之间的掩码变化比例（两两计算）
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

    # 步骤 5: 提取每个客户端的融合专家参数 (w_gate)
    client_w_gates = []
    for client_idx, client_model in enumerate(client_models):
        moe = None
        if hasattr(client_model, 'learner'):
            moe = getattr(client_model, 'learner')
        else:
            print(f"客户端 {client_idx} 没有 'learner' 模块。")
            raise AttributeError(f"客户端 {client_idx} 没有 'learner' 模块。")
        if moe and isinstance(moe, MoE):
            if hasattr(moe, 'w_gate'):
                w_gate = moe.w_gate.clone().detach()
                client_w_gates.append(w_gate)
                print(f"客户端 {client_idx} 的 w_gate 参数已提取。")
            else:
                print(f"客户端 {client_idx} 的 MoE 模块没有 'w_gate' 参数。")
                raise AttributeError(f"客户端 {client_idx} 的 MoE 模块没有 'w_gate' 参数。")
        else:
            print(f"客户端 {client_idx} 的 'learner' 模块不是 MoE 类型。")
            raise AttributeError(f"客户端 {client_idx} 的 'learner' 模块不是 MoE 类型。")

    # 步骤 6: 计算客户端之间的 OT 距离（两两计算）
    ot_distances = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(num_clients):
            w_gate_i = client_w_gates[i]
            w_gate_j = client_w_gates[j]
            # 扁平化 w_gate 张量以进行 OT 距离计算
            flattened_i = w_gate_i.view(-1).cpu().numpy()
            flattened_j = w_gate_j.view(-1).cpu().numpy()
            # 生成两个均匀分布
            a = np.ones(len(flattened_i)) / len(flattened_i)
            b = np.ones(len(flattened_j)) / len(flattened_j)
            # 计算 OT 距离（Earth Mover's Distance）
            M = ot.dist(flattened_i.reshape(-1, 1), flattened_j.reshape(-1, 1))  # 成对距离矩阵
            ot_distance = ot.emd2(a, b, M)  # OT 距离
            ot_distances[i, j] = ot_distance

    # 步骤 7: 将掩码变化比例和 OT 距离归一化到 [0,1]
    mask_changes_normalized = (client_mask_changes - client_mask_changes.min()) / (
            client_mask_changes.max() - client_mask_changes.min() + 1e-8)
    ot_distances_normalized = (ot_distances - ot_distances.min()) / (ot_distances.max() - ot_distances.min() + 1e-8)

    # 步骤 8: 计算客户端之间的相似度矩阵
    # 相似度 = 1 - (归一化的掩码变化比例 + 归一化的 OT 距离) / 2
    similarity_matrix = 1.0 - (mask_changes_normalized + ot_distances_normalized) / 2.0
    # 将相似度矩阵中的负值截断为零
    similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)

    # 步骤 9: 对相似度矩阵的每一行进行归一化，得到每个客户端的聚合权重
    aggregation_weights = similarity_matrix / (similarity_matrix.sum(axis=1, keepdims=True) + 1e-8)

    for client_idx in range(num_clients):
        weights = aggregation_weights[client_idx]
        print(f"客户端 {client_idx} 的聚合权重: {weights}")

    # 步骤 10: 使用聚合权重对所有客户端进行个性化聚合
    personalized_state_dicts = {client_idx: {} for client_idx in range(num_clients)}
    model_keys = global_model.state_dict().keys()
    client_state_dicts = [client_model.state_dict() for client_model in client_models]
    if args.spar_wei == 1 and args.save_spar_wei == 1 and args.load_spar_wei == 0:
        # 计算上传通信字节量（客户端上传模型参数和mask）
        for client_idx, client_model in enumerate(client_models):
            state_dict = client_state_dicts[client_idx]
            # 计算GNN参数大小
            gnn_params = {k: v for k, v in state_dict.items() if 'gnn' in k}
            gnn_bytes = sum(v.numel() * v.element_size() for v in gnn_params.values())
            # 计算MoE参数大小（w_gate和experts）
            learner = getattr(client_model, 'learner', None)
            moe_bytes = 0
            if learner:
                w_gate = getattr(learner, 'w_gate', None)
                if w_gate is not None:
                    moe_bytes += w_gate.numel() * w_gate.element_size()
                experts = getattr(learner, 'experts', None)
                if experts:
                    for expert in experts:
                        for param in expert.parameters():
                            moe_bytes += param.numel() * param.element_size()
            #pdb.set_trace()
            # 计算mask参数大小（每个客户端上传mask）
            masks = [layer.mask for layer in client_model.gnn.modules() if isinstance(layer, MaskedLinear)]
            mask_bytes = sum(mask.numel() * mask.element_size() for mask in masks) / 8
            # 总上传字节量
            upload_bytes = gnn_bytes + moe_bytes + mask_bytes
            communication_bytes['upload_bytes'] += upload_bytes
            print(f"客户端 {client_idx} 上传字节量: {upload_bytes} Bytes")
        download_bytes_per_client = 0
        for layer, union_mask in zip(masked_linear_layers, union_wei_masks):
            non_zero_mask = (union_mask != 0)
            # 假设 layer.weight 是一个参数
            non_zero_params = layer.weight[non_zero_mask].numel()
            download_bytes_per_client += non_zero_params * layer.weight.element_size()
        total_download_bytes = download_bytes_per_client * num_clients
        communication_bytes['download_bytes'] += total_download_bytes
        #pdb.set_trace()
        print(f"服务器下载参数mask共识字节量: {total_download_bytes} Bytes")

    elif args.spar_wei == 1 and args.save_spar_wei == 0 and args.load_spar_wei == 1:
        # 这是读取的文件中保存的参数稀疏化，所有客户端的mask已然是一样的
        # 模拟客户端仅上传未被mask的参数，不需要上传mask_bytes

        # 步骤 1: 获取全局掩码（假设所有客户端的掩码一致）
        global_mask = [layer.mask.clone().detach() for layer in global_model.gnn.modules() if isinstance(layer, MaskedLinear)]

        # 步骤 2: 计算全局未被mask的GNN参数总字节量
        total_unmasked_gnn_bytes = 0
        masked_linear_layers = [layer for layer in global_model.gnn.modules() if isinstance(layer, MaskedLinear)]
        for layer, mask in zip(masked_linear_layers, global_mask):
            # 仅保留未被mask的参数
            unmasked_params_num = mask.sum().item()  # 计算未被mask的参数数量
            total_unmasked_gnn_bytes += unmasked_params_num * layer.weight.element_size()

        # 步骤 3: 计算每个客户端的上传字节量（仅上传未被mask的GNN参数和MoE参数）
        for client_idx, client_model in enumerate(client_models):
            gnn_bytes = 0
            client_masked_layers = [layer for layer in client_model.gnn.modules() if isinstance(layer, MaskedLinear)]
            for layer, mask in zip(client_masked_layers, global_mask):
                # 计算未被mask的参数数量
                unmasked_params_num = mask.sum().item()
                gnn_bytes += unmasked_params_num * layer.weight.element_size()
            # 计算MoE参数大小（w_gate和experts）
            learner = getattr(client_model, 'learner', None)
            moe_bytes = 0
            if learner:
                w_gate = getattr(learner, 'w_gate', None)
                if w_gate is not None:
                    moe_bytes += w_gate.numel() * w_gate.element_size()
                experts = getattr(learner, 'experts', None)
                if experts:
                    for expert in experts:
                        for param in expert.parameters():
                            moe_bytes += param.numel() * param.element_size()
                        upload_bytes = gnn_bytes + moe_bytes
            communication_bytes['upload_bytes'] += upload_bytes
            print(f"客户端 {client_idx} 上传字节量: {upload_bytes} Bytes")
            #pdb.set_trace()  # 调试时可用，生产环境中请移除
        # 步骤 4: 计算下载通信字节量（服务器发送未被mask的GNN参数给每个客户端）
        # 由于所有客户端的mask相同，下载字节量等于未被mask的GNN参数大小乘以客户端数量
        total_download_bytes = total_unmasked_gnn_bytes * num_clients
        communication_bytes['download_bytes'] += total_download_bytes
        print(f"服务器下载参数mask共识字节量: {total_download_bytes} Bytes")
        #pdb.set_trace()  # 调试时可用，生产环境中请移除

    # 生成个性化参数
    for key in model_keys:
        # 收集所有客户端的对应参数
        params = [client_state_dict[key].float() for client_state_dict in client_state_dicts]
        # 对于每个客户端，计算其个性化的参数
        for client_idx in range(num_clients):
            weights = aggregation_weights[client_idx]
            personalized_param = sum(w * p for w, p in zip(weights, params))
            personalized_state_dicts[client_idx][key] = personalized_param

    # 更新每个客户端的模型参数
    for client_idx, client_model in enumerate(client_models):
        client_model.load_state_dict(personalized_state_dicts[client_idx])
        print(f"客户端 {client_idx} 的模型参数已个性化更新。")
    total_mask_ratio = 0
    if args.spar_wei == 1:
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

    return global_model, client_models, communication_bytes, total_mask_ratio


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

def cache_split_data(args, client_data, cache_type="louvain"):
    """
    Different types of data caches are kept.
    :param args: Parameter object that contains relevant configuration information
    :param client_data: The data to cache.
    :param cache_type: Cache type, "louvain" to hold the Louvain segmentation results, "processed" to hold the client_data after final processing.
    """
    if cache_type == "louvain":
        file_name = f"{args.is_iid}_{args.num_workers}_louvain_split.pkl"
    elif cache_type == "processed":
        file_name = f"{args.is_iid}_{args.num_workers}_processed_client_data.pkl"

    file_path = os.path.join("cache", file_name)

    if not os.path.exists("cache"):
        os.makedirs("cache")

    with open(file_path, 'wb') as f:
        pickle.dump(client_data, f)

    print(f"Data split saved to {file_path}")

def load_cached_data(args, cache_type="louvain"):
    """
    Load different types of data caches.
    :param args: Parameter object that contains relevant configuration information
    :param cache_type: Cache type, "louvain" to load the Louvain segmentation result, "processed" to load the final processed client_data.
    :return: client-side data to load, or None if the cache doesn't exist
    """
    if cache_type == "louvain":
        file_name = f"{args.is_iid}_{args.num_workers}_louvain_split.pkl"
    elif cache_type == "processed":
        file_name = f"{args.is_iid}_{args.num_workers}_processed_client_data.pkl"

    file_path = os.path.join("cache", file_name)

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            client_data = pickle.load(f)
        print(f"Loaded cached data from {file_path}")
        return client_data
    return None

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

def load_masks_into_model(model, masks, fixed=True):
    """
    将加载的 masks 分配给模型中的 MaskedLinear 层，并根据需要固定 mask。

    Args:
        model (torch.nn.Module): 模型。
        masks (list of torch.Tensor): 与每个 MaskedLinear 层对应的 mask 列表。
        fixed (bool): 是否固定 mask。
    """
    masked_layers = [layer for layer in model.gnn.modules() if isinstance(layer, MaskedLinear)]
    if len(masks) != len(masked_layers):
        raise ValueError("加载的 mask 数量与模型中的 MaskedLinear 层数不匹配。")

    for layer, mask in zip(masked_layers, masks):
        if layer.mask is None:
            # 如果 layer.mask 是 None，直接赋值并设置 fixed_mask
            layer.mask = mask.to(layer.weight.device)
        else:
            # 如果已有 mask，覆盖并设置 fixed_mask
            layer.mask = mask.to(layer.mask.device)
        if fixed:
            layer.fixed_mask = True  # 固定 mask


def main():
    args = ArgsInit().save_exp()
    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")
    logging.info(f'Device: {device}')

    # 加载 ogbn-proteins 数据集
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

    client_data = load_cached_data(args, cache_type="louvain")
    if client_data is None:
        client_data = split_Louvain(args, data)
        cache_split_data(args, client_data, cache_type="louvain")

    for i in range(args.num_workers):
        print(f"Client {i} data length: {len(client_data[i])}")
    print('=' * 20 + 'Display Client-side Data' + '=' * 20)

    for client_idx, client_graph in enumerate(client_data):
        print(f"Client {client_idx} graph: {client_graph}")
        print(f'Number of nodes: {client_graph.num_nodes}')
        print(f'Number of edges: {client_graph.num_edges}')
        #print(f"client_graph.x is None: {client_graph.x is None}")
        if client_graph.x is not None:
            print(f"client_graph.x shape: {client_graph.x.shape}")
        else:
            raise ValueError(f"client_graph {client_idx}.x is None. Please check split_Louvain function.")
    #pdb.set_trace()

    print('=' * 20 + 'Start Preparing the Models' + '=' * 20)
    nf_path = extract_node_features(client_data[0], args.num_workers, 'add', 0)
    args.nf_path = nf_path
    global_model = MoG(data.x.size(1), dataset.num_tasks, args, device).to(device)
    #os.makedirs('results/acc', exist_ok=True)
    #os.makedirs('results/sparsity', exist_ok=True)
    from collections import defaultdict
    k_list_str = '_'.join(map(str, args.k_list))
    w2loss_history = []
    pruning_rate_history = []
    best_acc_within_range = defaultdict(float)  # 记录剪枝率范围内的最优准确率
    best_mask_within_range = {}  # 记录剪枝率范围内的最优 mask
    convergence_threshold = 5  # 收敛阈值，连续多少次未显著变化则认为收敛
    current_epoch = 0

    # 定义剪枝率范围，例如 90±2, 80±2, 70±2 等
    pruning_ranges = [90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
    pruning_tolerance = 4  # ±2%
    loaded_masks = defaultdict(list)

    if args.load_spar_wei == 1:
        # ==================== 加载 wei_mask ====================
        for pruning_point in pruning_ranges:
            mask_file_path = f"saved_wei/num_layers_{args.num_layers}_best_masks_pruning_{pruning_point}+-{pruning_tolerance}%.pth"
            if os.path.exists(mask_file_path):
                masks = torch.load(mask_file_path)
                loaded_masks[pruning_point] = masks
                print(f"加载剪枝率 num_layers_{args.num_layers}_best_masks_pruning_{pruning_point}+-{pruning_tolerance}% 的 mask：{mask_file_path}")
            else:
                print(f"未找到剪枝率 num_layers_{args.num_layers}_best_masks_pruning_{pruning_point}+-{pruning_tolerance}% 的 mask 文件。")

        selected_pruning_rate = args.selected_pruning_rate  # 可根据需要修改
        if selected_pruning_rate in loaded_masks:
            print(f"将剪枝率 {selected_pruning_rate}±{pruning_tolerance}% 的 mask 分配给 global_model。")
            load_masks_into_model(global_model, loaded_masks[selected_pruning_rate])
        else:
            print(f"未加载剪枝率 {selected_pruning_rate}±{pruning_tolerance}% 的 mask，继续训练。")
        # ==================== 完成加载 wei_mask ====================
    #global_model.print_parameter_stats()
    #pdb.set_trace()

    # if args.spar_wei == 0 and args.save_spar_wei == 0 and args.load_spar_wei == 0:
    #     os.makedirs('results/normal', exist_ok=True)
    #     acc_file_path = f"results/normal/workers_{args.num_workers}_num_layers_{args.num_layers}_acc.txt"
    # elif args.spar_wei == 1 and args.save_spar_wei == 1 and args.load_spar_wei == 0:
    #     os.makedirs('results/pretrain', exist_ok=True)
    #     acc_file_path = f"results/pretrain/is_spar_wei_{args.spar_wei}_w2loss_{args.w2loss}_workers_{args.num_workers}_num_layers_{args.num_layers}_acc.txt"
    # elif args.spar_wei == 1 and args.save_spar_wei == 0 and args.load_spar_wei == 1:
    #     os.makedirs('results/retrain', exist_ok=True)
    #     acc_file_path = f"results/retrain/num_workers_{args.num_workers}_{k_list_str}_num_layers_{args.num_layers}_pruning_rate_{selected_pruning_rate}_acc.txt"
    #
    os.makedirs('Aba_results', exist_ok=True)
    acc_file_path = f"Aba_results/dataset_{args.dataset}_{k_list_str}_lambda2_{args.lambda2}_acc.txt"
    acc_file_path = get_versioned_file_path(acc_file_path)


    client_models = []
    for idx in range(args.num_workers):
        nf_path = extract_node_features(client_data[idx], args.num_workers,'add', idx)
        args.nf_path = nf_path
        client_model = MoG(data.x.size(1), dataset.num_tasks, args, device).to(device)
        client_models.append(client_model)

    for idx, client_model in enumerate(client_models):
        assert global_model is not None, "global_model is None before loading state_dict"
        assert client_model is not None, f"client_model {idx} is None before loading state_dict"
        client_model.load_state_dict(global_model.state_dict())

    results = {
        'highest_valid': 0,
        'final_train': 0,
        'final_test': 0,
        'highest_train': 0
    }

    start_time = time.time()
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

        # 定义当前处理的 pruning_point 索引，仅在特定条件下初始化
    if args.spar_wei == 1 and args.load_spar_wei == 0 and args.save_spar_wei == 1:
        current_pruning_index = 0
        total_pruning_points = len(pruning_ranges)
    else:
        current_pruning_index = None  # 不使用 pruning_index
        total_pruning_points = None

    total_upload_bytes = 0
    total_download_bytes = 0
    total_bytes =0

    # main
    for epoch in range(1, args.epochs + 1):
        if args.spar_wei == 1 and args.load_spar_wei == 0 and args.save_spar_wei == 1:
            current_pruning_point = pruning_ranges[current_pruning_index]
            lower_bound = current_pruning_point - pruning_tolerance
            upper_bound = current_pruning_point + pruning_tolerance
            print('=' * 20 + f"Epoch {epoch }/{args.epochs} Start (Pruning Point: {current_pruning_point}±{pruning_tolerance}%)" + '=' * 20)
        else:
            print('=' * 20 + f"Epoch {epoch }/{args.epochs} Start" + '=' * 20)


        #if epoch == 300:
            #pdb.set_trace()
        for client_model in client_models:
            if args.spar_wei != 1:
                client_model.load_state_dict(global_model.state_dict())
            else:
                pass
            client_model.to(client_model.device)
            # 复制 global_model 的 mask 到每个 client_model
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
        #client_models[0].print_parameter_stats()
        #pdb.set_trace()
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

            #optimizer = optim.Adam(client_models[client_idx].parameters(), lr=args.lr)
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

                #client_models[client_idx].train()

                # client_models[client_idx].print_parameter_stats()
                # pdb.set_trace()
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
                #logging.info(f"Client {client_idx}, Inner Epoch {inner_epoch}, Loss: {epoch_loss:.4f}")
                all_sparsity_innner_epoch.append(sparsity)
                all_wei_mask_inner_ratio.append(wei_mask_ratio)
                #client_models[client_idx].print_parameter_stats()
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
            client_models[client_idx].print_parameter_stats()
            #pdb.set_trace()
            #logging.info(f"Client {client_idx} Evaluation Results: {result}")

        #pdb.set_trace()
        if args.spar_wei == 1:
            global_model, client_models, communication_bytes,average_wei_mask_ratio = my_fedaggregation(global_model, client_models, args)
            total_upload_bytes += communication_bytes['upload_bytes']
            total_download_bytes += communication_bytes['download_bytes']
            total_bytes += communication_bytes['upload_bytes'] + communication_bytes['download_bytes']
            #pdb.set_trace()
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
            #sparsity_file.write(f"Epoch {epoch }, {average_sparsity}\n")

        if args.spar_wei == 1 and args.load_spar_wei == 0 and args.save_spar_wei == 1:
            # 记录当前w2loss和剪枝率
            current_w2loss = args.w2loss
            w2loss_history.append(current_w2loss)
            pruning_rate_history.append(average_wei_mask_ratio)
            current_pruning_ratio = average_wei_mask_ratio  # 使用平均权重掩码比例作为剪枝率
            # 确定当前剪枝率是否超出当前 pruning_point 的范围
            current_pruning_percentage = current_pruning_ratio * 100  # 转换为百分比
            if lower_bound <= current_pruning_percentage <= upper_bound:
                # 当前剪枝率在范围内，检查是否需要更新最佳准确率和 mask
                if average_test_acc > best_acc_within_range[current_pruning_point]:
                    best_acc_within_range[current_pruning_point] = average_test_acc
                    # 获取所有 MaskedLinear 层的 mask
                    masks = [layer.mask.detach().cpu().clone() for layer in global_model.gnn.modules() if isinstance(layer, MaskedLinear)]
                    best_mask_within_range[current_pruning_point] = masks
                    print(f"更新了剪枝率 {current_pruning_point}±{pruning_tolerance}% 的最佳准确率: {average_test_acc:.4f}%")
            elif current_pruning_percentage < lower_bound:
                # 当前剪枝率已经超出当前 pruning_point 的范围，保存最佳 mask 并加载到 global_model
                if current_pruning_index is not None and current_pruning_point is not None:
                    if best_mask_within_range.get(current_pruning_point) is not None:
                        # 保存之前记录的最佳 mask
                        masks = best_mask_within_range[current_pruning_point]
                        mask_file_path = f"saved_wei/num_layers_{args.num_layers}_best_masks_pruning_{current_pruning_point}+-{pruning_tolerance}%.pth"
                        torch.save(masks, mask_file_path)
                        print(f"最佳剪枝率 {current_pruning_point}±{pruning_tolerance}% 的 mask 已保存到 {mask_file_path}")
                        #pdb.set_trace()
                        # 将 mask 加载到 global_model 中
                        load_masks_into_model(global_model, masks, fixed=True)
                        print(f"已将剪枝率 {current_pruning_point}±{pruning_tolerance}% 的 mask 加载到 global_model。")
                    else:
                        #pdb.set_trace()
                        print(f"没有记录到剪枝率 {current_pruning_point}±{pruning_tolerance}% 的最佳 mask。")

                    # 移动到下一个 pruning_point
                    current_pruning_index += 1
                    if current_pruning_index < total_pruning_points:
                        next_pruning_point = pruning_ranges[current_pruning_index]
                        print(f"切换到下一个剪枝率点: {next_pruning_point}±{pruning_tolerance}%")
                    else:
                        print("所有剪枝率点已处理完毕。")

            # 检查是否收敛（基于剪枝率）
            if len(pruning_rate_history) >= convergence_threshold:
                recent_pruning = pruning_rate_history[-convergence_threshold:]
                pruning_diffs = [abs(recent_pruning[i] - recent_pruning[i - 1]) for i in
                                 range(1, convergence_threshold)]
                if all(diff < 1 for diff in pruning_diffs):
                    # 收敛，增加w2loss
                    args.w2loss = args.w2loss * 2
                    print(f"基于剪枝率检测到收敛。将 w2loss 增加到 {args.w2loss:.6f}")
                    # 重置剪枝率历史记录
                    pruning_rate_history = []

            # 检查是否接近最大 epoch，若是且剪枝率未达到 5%，则扩展训练
            if args.spar_wei == 1 and args.load_spar_wei == 0 and args.save_spar_wei == 1:
                epochs_remaining = args.epochs - (current_epoch + 1)
                if epochs_remaining <= 10 and (current_pruning_ratio * 100) > 4:
                    args.epochs += 10
                    print(f"将训练轮数扩展 10 轮。新的总轮数: {args.epochs}")
            # **新增逻辑：当参数稀疏化率达到4%或以下时，提前退出循环并保存当前 mask**
            if args.spar_wei == 1 and args.load_spar_wei == 0 and args.save_spar_wei == 1:
                if average_wei_mask_ratio <= 0.04:
                    print("参数稀疏化率已达到4%或以下，提前结束训练。")

                    # 获取所有 MaskedLinear 层的 mask
                    masks = [layer.mask.detach().cpu().clone() for layer in global_model.gnn.modules() if
                             isinstance(layer, MaskedLinear)]

                    # 定义保存路径（您可以根据需要修改路径和文件名）
                    final_mask_file_path = f"saved_wei/num_layers_{args.num_layers}_final_masks_pruning_0+-4%.pth"

                    # 保存 mask 到磁盘
                    torch.save(masks, final_mask_file_path)
                    print(f"最终剪枝率 0±4% 的 mask 已保存到 {final_mask_file_path}")

                    # **可选**：如果需要，将 mask 加载到 global_model 中（虽然此时已经达到目标稀疏率）
                    # load_masks_into_model(global_model, masks, fixed=True)
                    # print(f"已将最终剪枝率 0±4% 的 mask 加载到 global_model。")

                    # 退出训练循环
                    break

    # if args.save_spar_wei == 1:
    #     for pruning_point, acc in best_acc_within_range.items():
    #         masks = best_mask_within_range.get(pruning_point, None)
    #         if masks is not None:
    #             mask_file_path = f"saved_wei/num_workers_{args.num_workers}_num_layers_{args.num_layers}_best_masks_pruning_{pruning_point}+-{pruning_tolerance}%.pth"
    #             torch.save(masks, mask_file_path)
    #             print(f"Best masks for pruning rate {pruning_point}±{pruning_tolerance}% saved to {mask_file_path}")
    #         else:
    #             print(f"No mask recorded for pruning rate {pruning_point}±{pruning_tolerance}%")
        print("Training completed.")


    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f'Total time: {time.strftime("%H:%M:%S", time.gmtime(total_time))}')

if __name__ == "__main__":
    main()