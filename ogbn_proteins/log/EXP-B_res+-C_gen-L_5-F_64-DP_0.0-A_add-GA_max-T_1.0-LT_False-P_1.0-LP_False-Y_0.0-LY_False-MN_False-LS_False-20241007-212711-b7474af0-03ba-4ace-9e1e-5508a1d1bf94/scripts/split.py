from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import community as community_louvain
import random
from tqdm import tqdm

import torch_geometric

def split_Random(args,data):
    """
    original code link： https://github.com/alibaba/FederatedScope/blob/fe1806b36b4629bb0057e84912d5f42a79f4461d/federatedscope/core/splitters/graph/random_splitter.py#L14
    :param args: args.overlapping_rate(float):Additional samples of overlapping data, \
            eg. ``'0.4'``;
                    args.drop_edge(float): Drop edges (drop_edge / client_num) for each \
            client within overlapping part.
    :param data:
    :param clients:
    :return:
    """
    args.dropout = 0
    ovlap = args.overlapping_rate
    drop_edge = args.dropout
    client_num = args.num_workers

    sampling_rate = (np.ones(client_num) -
                          ovlap) / client_num

    data.index_orig = torch.arange(data.num_nodes)


    print("Graph to Networkx")
    G = to_networkx(
        data,
        node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
        to_undirected=True)
    print("Setting node attributes")
    nx.set_node_attributes(G,
                           dict([(nid, nid)
                                 for nid in range(nx.number_of_nodes(G))]),
                           name="index_orig")
    print("Calculating  partition")
    client_node_idx = {idx: [] for idx in range(client_num)}
    indices = np.random.permutation(data.num_nodes)

    sum_rate = 0
    for idx, rate in enumerate(sampling_rate):
        client_node_idx[idx] = indices[round(sum_rate *
                                             data.num_nodes):round(
            (sum_rate + rate) *
            data.num_nodes)]
        sum_rate += rate

    if ovlap:
        ovlap_nodes = indices[round(sum_rate * data.num_nodes):]
        for idx in client_node_idx:
            client_node_idx[idx] = np.concatenate(
                (client_node_idx[idx], ovlap_nodes))

    # Drop_edge index for each client
    if drop_edge:
        ovlap_graph = nx.Graph(nx.subgraph(G, ovlap_nodes))
        ovlap_edge_ind = np.random.permutation(
            ovlap_graph.number_of_edges())
        drop_all = ovlap_edge_ind[:round(ovlap_graph.number_of_edges() *
                                         drop_edge)]
        drop_client = [
            drop_all[s:s + round(len(drop_all) / client_num)]
            for s in range(0, len(drop_all),
                           round(len(drop_all) / client_num))
        ]

    graphs = []
    for owner in client_node_idx:
        nodes = client_node_idx[owner]
        sub_g = nx.Graph(nx.subgraph(G, nodes))
        if drop_edge:
            sub_g.remove_edges_from(
                np.array(ovlap_graph.edges)[drop_client[owner]])
        graphs.append(from_networkx(sub_g))

    return graphs


def split_Louvain(args, data):
    """
    original code link： https://github.com/alibaba/FederatedScope/blob/fe1806b36b4629bb0057e84912d5f42a79f4461d/federatedscope/core/splitters/graph/random_splitter.py#L14
    :param args: args.overlapping_rate(float):Additional samples of overlapping data, \
            eg. ``'0.4'``;
                    args.drop_edge(float): Drop edges (drop_edge / client_num) for each \
            client within overlapping part.
    :param data:
    :param clients:
    :return:
    """
    args.delta = 40

    delta = args.delta
    client_num = args.num_workers
    data.index_orig = torch.arange(data.num_nodes)

    print("Graph to Networkx")

    # 更新 node_attrs 以仅包含存在的属性
    node_attrs = ['edge_index', 'node_species', 'y', 'edge_attr']
    G = to_networkx(data, node_attrs=node_attrs, to_undirected=True)
    Large_data_list = ['Reddit']
    print("Setting node attributes")
    nx.set_node_attributes(G,
                           dict([(nid, nid)
                                 for nid in tqdm(range(nx.number_of_nodes(G)))]),
                           name="index_orig")

    print("Calculating community partition")
    args.dataset = 'ogbn-proteins'
    if args.dataset in Large_data_list:
        partition = community_louvain.best_partition(G, resolution=0.1)
    else:
        partition = community_louvain.best_partition(G)

    cluster2node = {}
    for node in partition:
        cluster = partition[node]
        if cluster not in cluster2node:
            cluster2node[cluster] = [node]
        else:
            cluster2node[cluster].append(node)

    max_len = len(G) // client_num - delta
    max_len_client = len(G) // client_num

    tmp_cluster2node = {}
    for cluster in cluster2node:
        while len(cluster2node[cluster]) > max_len:
            tmp_cluster = cluster2node[cluster][:max_len]
            tmp_cluster2node[len(cluster2node) + len(tmp_cluster2node) +
                             1] = tmp_cluster
            cluster2node[cluster] = cluster2node[cluster][max_len:]
    cluster2node.update(tmp_cluster2node)

    orderedc2n = (zip(cluster2node.keys(), cluster2node.values()))
    orderedc2n = sorted(orderedc2n, key=lambda x: len(x[1]), reverse=True)

    client_node_idx = {idx: [] for idx in range(client_num)}
    idx = 0
    for (cluster, node_list) in orderedc2n:
        while len(node_list) + len(
                client_node_idx[idx]) > max_len_client + delta:
            idx = (idx + 1) % client_num
        client_node_idx[idx] += node_list
        idx = (idx + 1) % client_num

    graphs = []
    for owner in client_node_idx:
        nodes = client_node_idx[owner]
        graphs.append(from_networkx(nx.subgraph(G, nodes)))

    return graphs
