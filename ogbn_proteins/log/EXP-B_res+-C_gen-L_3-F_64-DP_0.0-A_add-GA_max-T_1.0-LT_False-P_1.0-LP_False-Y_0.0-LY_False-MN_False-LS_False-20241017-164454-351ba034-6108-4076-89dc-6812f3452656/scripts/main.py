import os
import pdb
import pickle
import torch_geometric.utils as tg_utils

import torch
import torch.optim as optim
import statistics
from dataset import OGBNDataset
from args import ArgsInit
import time
import numpy as np
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.utils import scatter

from gnn_conv import MaskedLinear
from split import split_Louvain
from utils import save_ckpt,intersection, process_indexes
import logging
from MoG import MoG

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

def fed_avg(global_model, client_models):
    """The client model parameters are aggregated using FedAvg"""
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

def my_fedaggregation(global_model, client_models, args):
    """
    Perform federated aggregation using the proposed method.
    :param global_model: The global model
    :param client_models: The list of client models
    :param args: The arguments
    :return: The updated global model
    """

    # Step 1: Get wei_mask from each client
    client_wei_masks = []
    for client_model in client_models:
        wei_masks = []
        for layer in client_model.gnn.modules():
            if isinstance(layer, MaskedLinear):
                # Ensure that layer.mask is detached from computation graph
                wei_masks.append(layer.mask.clone().detach())
        client_wei_masks.append(wei_masks)

    # Step 2: Get the union of wei_mask from each client
    # Assuming masks are binary (0 or 1), union is the element-wise maximum
    num_clients = len(client_models)
    num_layers = len(client_wei_masks[0])  # Number of MaskedLinear layers
    union_wei_masks = []
    for layer_idx in range(num_layers):
        # Start with the mask from the first client
        union_mask = client_wei_masks[0][layer_idx].clone()
        for client_idx in range(1, num_clients):
            union_mask = torch.max(union_mask, client_wei_masks[client_idx][layer_idx])
        union_wei_masks.append(union_mask)

    # Step 3: Update global_model's wei_mask using the union wei_mask
    masked_linear_layers = [layer for layer in global_model.gnn.modules() if isinstance(layer, MaskedLinear)]
    for layer, union_mask in zip(masked_linear_layers, union_wei_masks):
        layer.mask = union_mask.clone()

    # Step 4: Compute the change in wei_mask for each client
    client_mask_changes = []
    for client_idx in range(num_clients):
        total_weights = 0
        changed_weights = 0
        for layer_idx in range(num_layers):
            client_mask = client_wei_masks[client_idx][layer_idx]
            union_mask = union_wei_masks[layer_idx]
            total_weights += client_mask.numel()
            changed_weights += (client_mask != union_mask).sum().item()
        mask_change_ratio = changed_weights / total_weights
        client_mask_changes.append(mask_change_ratio)

    # Step 5: Compute the aggregation weight for each client
    # Clients with less mask change get higher weight
    client_weights = [1.0 - change for change in client_mask_changes]
    total_weight = sum(client_weights)
    if total_weight == 0:
        # If all clients have completely different masks, use equal weights
        client_weights = [1.0 / num_clients] * num_clients
    else:
        client_weights = [w / total_weight for w in client_weights]

    # Step 6: Use the aggregation weight to update the parameters of global_model
    global_state_dict = global_model.state_dict()
    for key in global_state_dict.keys():
        global_state_dict[key] = torch.zeros_like(global_state_dict[key])

    for client_weight, client_model in zip(client_weights, client_models):
        client_state_dict = client_model.state_dict()
        for key in global_state_dict.keys():
            global_state_dict[key] += client_weight * client_state_dict[key]

    global_model.load_state_dict(global_state_dict)

    return global_model

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

    num_edges = mask.numel()
    num_masks = mask.sum().item()
    feature_dim = train_x.shape[1]
    message_passing_flops = num_masks * feature_dim
    total_flops = message_passing_flops

    print(f"Total FLOPs in this iteration: {total_flops}")

    try:
        if args.spar_wei == 1:
            pred = model.gnn(train_x, train_idx, edges_index_mapped, edges_attr_mapped, edge_mask=mask, wei_masks=None)
        else:
            pred = model.gnn(train_x, train_idx, edges_index_mapped, edges_attr_mapped, edge_mask=mask)
    except Exception as e:
        print(f"Error during model.gnn forward pass: {e}")
        raise

    num_nodes = train_x.shape[0]
    total_num_weights = 0
    total_mask_ratio = []
    if args.spar_wei == 1:
        linear_layers = [layer for layer in model.gnn.modules() if isinstance(layer, MaskedLinear)]
        for layer in linear_layers:
            non_zero_weights = (layer.mask == 1).sum().item()
            total_num_weights = layer.mask.numel()
            total_mask_ratio.append(non_zero_weights / total_num_weights)
            #pdb.set_trace()
            flops = 2 * num_nodes * non_zero_weights
            total_flops += flops
    else:
        linear_layers = [layer for layer in model.gnn.modules() if isinstance(layer, torch.nn.Linear)]
        for layer in linear_layers:
            in_features = layer.in_features
            out_features = layer.out_features
            flops = 2 * num_nodes * in_features * out_features  # 标准 Linear 层的 FLOPs
            total_flops += flops

    target = train_y.to(torch.float32)

    try:
        if model.gnn.spar_wei:
            wei_loss = 0
            for layer in model.gnn.modules():
                if isinstance(layer, MaskedLinear):
                    wei_loss += args.w2loss * torch.sum(torch.exp(-layer.threshold))
            loss = criterion(pred.to(torch.float32), target) + add_loss * 0.1 + wei_loss
        else:
            loss = criterion(pred.to(torch.float32), target) + add_loss * 0.1
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
    return statistics.mean(loss_list), pruning_ratio, total_flops, np.mean(total_mask_ratio)

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
    #print(f"Total FLOPs in evaluation: {total_flops}")

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

def extract_node_features(client_data, aggr='add',idx=None):
    file_path = 'Client_{}_init_node_features_{}.pt'.format(idx, aggr)
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

def main():
    args = ArgsInit().save_exp()

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")

    logging.info(f'Device: {device}')

    # Load the ogbn-proteins dataset
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
        if client_graph.x is not None:
            print(f"client_graph.x shape: {client_graph.x.shape}")
        else:
            raise ValueError(f"client_graph {client_idx}.x is None. Please check split_Louvain function.")

    print('=' * 20 + 'Start Preparing the Models' + '=' * 20)
    global_model = MoG(data.x.size(1), dataset.num_tasks, args, device).to(device)
    os.makedirs('results/acc', exist_ok=True)
    os.makedirs('results/sparsity', exist_ok=True)
    k_list_str = '_'.join(map(str, args.k_list))
    if args.spar_wei == 1:
        acc_file_path = f"results/acc/is_spar_wei_{args.spar_wei}_w2loss_{args.w2loss}_{k_list_str}_acc.txt"
    else:
        acc_file_path = f"results/acc/is_spar_wei_{args.spar_wei}_{k_list_str}_acc.txt"
    acc_file_path = get_versioned_file_path(acc_file_path)

    client_models = []
    for idx in range(args.num_workers):
        nf_path = extract_node_features(client_data[idx], 'add', idx)
        args.nf_path = nf_path
        client_model = MoG(data.x.size(1), dataset.num_tasks, args, device).to(device)
        client_models.append(client_model)

    for idx, client_model in enumerate(client_models):
        assert global_model is not None, "global_model is None before loading state_dict"
        assert client_model is not None, f"client_model {idx} is None before loading state_dict"
        client_model.load_state_dict(global_model.state_dict())

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

    for epoch in range(1, args.epochs + 1):
        print('=' * 20 + f"Epoch {epoch }/{args.epochs} Start" + '=' * 20)

        if epoch == 200:
            pdb.set_trace()
        for client_model in client_models:
            client_model.load_state_dict(global_model.state_dict())
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
            optimizer = optim.Adam(client_models[client_idx].parameters(), lr=args.lr)
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

                client_models[client_idx].train()
                epoch_loss, sparsity, flops, wei_mask_ratio = train(
                    client_data[client_idx],
                    train_idx,
                    valid_idx,
                    test_idx,
                    client_models[client_idx],
                    optimizer, criterion,
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
            result, eval_flops = multi_evaluate(
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
        average_wei_mask_ratio = np.mean(all_wei_mask_ratio) if all_wei_mask_ratio else 0.0
        print(f'Average Accuracy across all clients: train ROC AUC:{100 * average_train_acc:.2f}, '
              f'valid ROC AUC:{100 * average_valid_acc:.2f}, test ROC AUC:{100 * average_test_acc:.2f}%')
        print(f'Average Sparsity across all clients: {100 * average_sparsity:.2f}%')
        print(f'Average FLOPs across all clients: train FLOPs:{average_train_flops:.4f}, eval FLOPs:{average_eval_flops}')
        print(f'Average Weight Mask Ratio across all clients: {100 * average_wei_mask_ratio:.2f}%')

        with open(acc_file_path, 'a') as acc_file:
            acc_file.write(f"Epoch:{epoch }, acc:{average_test_acc}, spar:{average_sparsity}, train_flops:{average_train_flops:.4f}, eval_flops:{average_eval_flops}, wei_mask_ratio:{average_wei_mask_ratio}\n")

        # Use the new aggregation function
        if args.spar_wei == 1:
            global_model = my_fedaggregation(global_model, client_models, args)
        else:
            global_model = fed_avg(global_model, client_models)
        for client_model in client_models:
            client_model.load_state_dict(global_model.state_dict())

        logging.debug("Performed federated aggregation.")
        logging.info(f"Round {epoch} Results: {result}")

    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f'Total time: {time.strftime("%H:%M:%S", time.gmtime(total_time))}')

if __name__ == "__main__":
    main()
