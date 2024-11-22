import numpy as np
import os
import pickle
import torch_geometric.utils as tg_utils
from ogbn_proteins.config.args import ArgsInit
from ogb.nodeproppred import Evaluator
from torch_geometric.utils import scatter
import torch
import torch.nn as nn
import statistics
import numpy as np
import math
import logging
import ot
from ogbn_proteins.module.moe import MoE
from ogbn_proteins.module.gnn_conv import MaskedLinear

def load_masks_into_model(model, masks, fixed=True):

    masked_layers = [layer for layer in model.gnn.modules() if isinstance(layer, MaskedLinear)]
    if len(masks) != len(masked_layers):
        raise ValueError("The number of masks loaded does not match the number of MaskedLinear layers in the model.")
    for layer, mask in zip(masked_layers, masks):
        if layer.mask is None:
            layer.mask = mask.to(layer.weight.device)
        else:
            layer.mask = mask.to(layer.mask.device)
        if fixed:
            layer.fixed_mask = True

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

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def random_partition_graph(num_nodes, cluster_number=10):

    parts = np.random.randint(0, cluster_number, size=num_nodes)
    return parts

def create_edge_index_dict(edge_index):

    edge_index_dict = {}
    edge_index_np = edge_index.numpy()
    for idx in range(edge_index_np.shape[1]):
        node_pair = (edge_index_np[0, idx], edge_index_np[1, idx])
        edge_index_dict[node_pair] = idx
    return edge_index_dict

def generate_sub_graphs(data, parts, cluster_number=10, batch_size=1):

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
        sg_edges[cluster] = tg_utils.subgraph(sg_nodes[cluster], edge_index, relabel_nodes=True)[0]
        edges_no += sg_edges[cluster].shape[1]
        mapper = {nd_idx: nd_orig_idx for nd_idx, nd_orig_idx in enumerate(sg_nodes[cluster])}
        sg_edges_orig[cluster] = edge_list_mapper(mapper, sg_edges[cluster])
        sg_edges_index[cluster] = [edge_index_dict.get((edge[0], edge[1]), -1) for edge in sg_edges_orig[cluster].t().numpy()]
        sg_edges_index[cluster] = [idx for idx in sg_edges_index[cluster] if idx != -1]

    print('Total number edges of sub graphs: {}, of whole graph: {}, {:.2f} % edges are lost'.
          format(edges_no, data.num_edges, (1 - edges_no / data.num_edges) * 100))

    return sg_nodes, sg_edges, sg_edges_index, sg_edges_orig

def edge_list_mapper(mapper, sg_edges_list):

    idx_1st = list(map(lambda x: mapper[x], sg_edges_list[0].tolist()))
    idx_2nd = list(map(lambda x: mapper[x], sg_edges_list[1].tolist()))
    sg_edges_orig = torch.LongTensor([idx_1st, idx_2nd])
    return sg_edges_orig

def fed_avg(global_model, client_models):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key], device=global_dict[key].device)
    for client_model in client_models:
        client_dict = client_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] += client_dict[key].to(global_dict[key].device)
    for key in global_dict.keys():
        global_dict[key] = global_dict[key] / len(client_models)
    global_model.load_state_dict(global_dict)
    return global_model

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
        if moe and isinstance(moe, MoE):
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
    if args.spar_wei == 1 and args.save_spar_wei == 1 and args.load_spar_wei == 0:
        for client_idx, client_model in enumerate(client_models):
            state_dict = client_state_dicts[client_idx]
            gnn_params = {k: v for k, v in state_dict.items() if 'gnn' in k}
            gnn_bytes = sum(v.numel() * v.element_size() for v in gnn_params.values())
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
            masks = [layer.mask for layer in client_model.gnn.modules() if isinstance(layer, MaskedLinear)]
            mask_bytes = sum(mask.numel() * mask.element_size() for mask in masks) / 8
            upload_bytes = gnn_bytes + moe_bytes + mask_bytes
            print(f"Client {client_idx} upload bytes: {upload_bytes} Bytes")
        download_bytes_per_client = 0
        for layer, union_mask in zip(masked_linear_layers, union_wei_masks):
            non_zero_mask = (union_mask != 0)
            non_zero_params = layer.weight[non_zero_mask].numel()
            download_bytes_per_client += non_zero_params * layer.weight.element_size()
        total_download_bytes = download_bytes_per_client * num_clients
        print(f"Server download parameter mask consensus bytes: {total_download_bytes} Bytes")

    elif args.spar_wei == 1 and args.save_spar_wei == 0 and args.load_spar_wei == 1:
        global_mask = [layer.mask.clone().detach() for layer in global_model.gnn.modules() if isinstance(layer, MaskedLinear)]

        total_unmasked_gnn_bytes = 0
        masked_linear_layers = [layer for layer in global_model.gnn.modules() if isinstance(layer, MaskedLinear)]
        for layer, mask in zip(masked_linear_layers, global_mask):
            unmasked_params_num = mask.sum().item()
            total_unmasked_gnn_bytes += unmasked_params_num * layer.weight.element_size()

        for client_idx, client_model in enumerate(client_models):
            gnn_bytes = 0
            client_masked_layers = [layer for layer in client_model.gnn.modules() if isinstance(layer, MaskedLinear)]
            for layer, mask in zip(client_masked_layers, global_mask):
                unmasked_params_num = mask.sum().item()
                gnn_bytes += unmasked_params_num * layer.weight.element_size()
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
            print(f"Client {client_idx} upload bytes: {upload_bytes} Bytes")
        total_download_bytes = total_unmasked_gnn_bytes * num_clients
        print(f"Server download parameter mask consensus bytes: {total_download_bytes} Bytes")

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

    return global_model, client_models, total_mask_ratio
