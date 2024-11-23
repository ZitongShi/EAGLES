import os

import numpy as np
import ot
import torch

from gcn_exp.module.gnn import MaskedLinear


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
