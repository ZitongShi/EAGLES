import os
import pdb
import pickle

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx, subgraph
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


import os
import pickle
import torch
from tqdm import tqdm
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx

def split_Louvain(args, data):
    client_num = args.num_workers
    data.index_orig = torch.arange(data.num_nodes)

    graph_file = "saved_graph_louvain.pkl"

    if os.path.exists(graph_file):
        print(f"Loading graph from {graph_file}")
        with open(graph_file, 'rb') as f:
            G = pickle.load(f)
    else:
        print("Converting graph to NetworkX")
        node_attrs = ['x', 'y', 'node_species']
        edge_attrs = ['edge_attr']
        # 将图直接转换为无向图
        G = to_networkx(data, node_attrs=node_attrs, edge_attrs=edge_attrs, to_undirected=True)
        with open(graph_file, 'wb') as f:
            pickle.dump(G, f)
        print(f"Graph saved to {graph_file}")

    # 检查图是否为无向图，如果不是，转换为无向图
    if G.is_directed():
        print("Converting directed graph to undirected graph")
        G = G.to_undirected()

    # 继续您的代码
    print("Checking edge attributes in NetworkX graph...")
    missing_edge_attrs = 0
    for u, v, attrs in G.edges(data=True):
        if 'edge_attr' not in attrs:
            missing_edge_attrs += 1

    print(f"Total edges missing 'edge_attr': {missing_edge_attrs}")

    nx.set_node_attributes(G, {nid: nid for nid in range(nx.number_of_nodes(G))}, name="index_orig")

    print("Calculating community partition")
    args.dataset = 'ogbn-proteins'
    if args.dataset in ['Reddit']:
        partition = community_louvain.best_partition(G, resolution=0.1)
    else:
        partition = community_louvain.best_partition(G)
    print("Calculating community partition done!")

    print("Splitting nodes to clients")
    cluster2node = {}
    for node, cluster in partition.items():
        cluster2node.setdefault(cluster, []).append(node)
    print(f"Number of clusters formed: {len(cluster2node)}")

    max_len_client = len(G) // client_num
    delta = int(0.1 * max_len_client)
    print(f"Maximum nodes per client: {max_len_client}, with initial delta: {delta}")

    # 分割过大的簇
    tmp_cluster2node = {}
    for cluster in cluster2node:
        while len(cluster2node[cluster]) > max_len_client - delta:
            tmp_cluster = cluster2node[cluster][:max_len_client - delta]
            tmp_cluster2node[len(cluster2node) + len(tmp_cluster2node) + 1] = tmp_cluster
            cluster2node[cluster] = cluster2node[cluster][max_len_client - delta:]
            print(f"Splitting large cluster {cluster} into smaller clusters.")

    cluster2node.update(tmp_cluster2node)
    print(f"Total number of clusters after splitting: {len(cluster2node)}")

    orderedc2n = sorted(cluster2node.items(), key=lambda x: len(x[1]), reverse=True)
    print("Clusters sorted by size (largest to smallest).")

    client_node_idx = {idx: [] for idx in range(client_num)}
    idx = 0

    for (cluster, node_list) in orderedc2n:
        print(f"Allocating cluster {cluster} with {len(node_list)} nodes to a client.")
        attempts = 0
        max_attempts = client_num
        while len(node_list) + len(client_node_idx[idx]) > max_len_client + delta:
            print(f"Client {idx} is full. Trying next client.")
            idx = (idx + 1) % client_num
            attempts += 1
            if attempts >= max_attempts:
                print(f"Warning: Unable to allocate nodes for cluster {cluster}. All clients are full.")
                delta += int(0.05 * max_len_client)
                print(f"Increasing delta by 5%, new delta: {delta}")
                attempts = 0
                idx = 0
        client_node_idx[idx] += node_list
        print(f"Assigning cluster {cluster} to client {idx}.")
        idx = (idx + 1) % client_num

    # 创建每个客户端的子图
    graphs = []
    for owner in client_node_idx:
        nodes = client_node_idx[owner]
        subgraph_nx = G.subgraph(nodes).copy()
        subgraph = from_networkx(subgraph_nx)

        subgraph.x = data.x[nodes].clone()

        # 检查 edge_attr 是否存在，如果不存在则设置为 None
        if hasattr(subgraph, 'edge_attr'):
            subgraph.edge_attr = subgraph.edge_attr
        else:
            subgraph.edge_attr = None

        graphs.append(subgraph)

        print(f"Client {owner} graph.x shape: {subgraph.x.shape}")
        print(f"Client {owner} edge_index shape: {subgraph.edge_index.shape}")
        if subgraph.edge_attr is not None:
            print(f"Client {owner} edge_attr shape: {subgraph.edge_attr.shape}")
        else:
            print(f"Client {owner} edge_attr is None")

    return graphs
