import os
import pickle
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import torch
import numpy as np
import community as community_louvain
from tqdm import tqdm
def split_Random(args,data):

    args['dropout'] = 0
    ovlap = args['overlapping_rate']
    drop_edge = args['dropout']
    client_num = args['num_workers']

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

    client_num = args['num_workers']
    data.index_orig = torch.arange(data.num_nodes)

    graph_file = f"saved_graph_{args['dataset']}.pkl"
    if os.path.exists(graph_file):
        print(f"Loading graph from {graph_file}")
        with open(graph_file, 'rb') as f:
            G = pickle.load(f)
    else:
        print("Graph to Networkx")
        node_attrs = ['x', 'y', 'train_mask', 'val_mask', 'test_mask']
        G = to_networkx(data, node_attrs=node_attrs, to_undirected=True)
        with open(graph_file, 'wb') as f:
            pickle.dump(G, f)
        print(f"Graph saved to {graph_file}")
    Large_data_list = ['Reddit']
    print("Setting node attributes")
    nx.set_node_attributes(G,
                           dict([(nid, nid)
                                 for nid in tqdm(range(nx.number_of_nodes(G)))]),
                           name="index_orig")

    print("Calculating community partition")
    if args['dataset'] in Large_data_list:
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

    max_len_client = len(G) // client_num
    delta = int(0.1 * max_len_client)
    print(f"Maximum nodes per client: {max_len_client}, with initial delta: {delta}")

    tmp_cluster2node = {}
    for cluster in cluster2node:
        while len(cluster2node[cluster]) > max_len_client - delta:
            tmp_cluster = cluster2node[cluster][:max_len_client - delta]
            tmp_cluster2node[len(cluster2node) + len(tmp_cluster2node) + 1] = tmp_cluster
            cluster2node[cluster] = cluster2node[cluster][max_len_client - delta:]
            print(f"Splitting large cluster {cluster} into smaller clusters.")
    cluster2node.update(tmp_cluster2node)
    orderedc2n = (zip(cluster2node.keys(), cluster2node.values()))
    orderedc2n = sorted(orderedc2n, key=lambda x: len(x[1]), reverse=True)
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

    graphs = []
    for owner in client_node_idx:
        nodes = client_node_idx[owner]
        graphs.append(from_networkx(nx.subgraph(G, nodes)))

    return graphs

