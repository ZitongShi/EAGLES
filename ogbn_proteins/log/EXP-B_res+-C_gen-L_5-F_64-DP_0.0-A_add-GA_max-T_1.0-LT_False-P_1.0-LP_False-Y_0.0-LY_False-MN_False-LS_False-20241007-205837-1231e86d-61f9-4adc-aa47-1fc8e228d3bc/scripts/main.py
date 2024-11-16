import os
import pickle

import torch
import torch.optim as optim
import statistics
from dataset import OGBNDataset
from args import ArgsInit
import time
import numpy as np
from ogb.nodeproppred import Evaluator

from split import split_Louvain
from utils import save_ckpt,intersection, process_indexes
import logging
from MoG import MoG
def fed_avg(global_model, client_models):
    """使用 FedAvg 聚合客户端模型参数"""
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

def train(data, dataset, model, optimizer, criterion,temp, device,use_topo= True):

    loss_list = []
    num_edges = 0
    num_masks = 0
    model.train()
    sg_nodes, sg_edges, sg_edges_index, _ = data

    train_y = dataset.y[dataset.train_idx]
    idx_clusters = np.arange(len(sg_nodes))
    np.random.shuffle(idx_clusters)

    for idx in idx_clusters:

        x = dataset.x[sg_nodes[idx]].float().to(device)
        sg_nodes_idx = torch.LongTensor(sg_nodes[idx]).to(device)

        sg_edges_ = sg_edges[idx].to(device)
        sg_edges_attr = dataset.edge_attr[sg_edges_index[idx]].to(device)

        mapper = {node: idx for idx, node in enumerate(sg_nodes[idx])}

        inter_idx = intersection(sg_nodes[idx], dataset.train_idx.tolist())
        training_idx = [mapper[t_idx] for t_idx in inter_idx]

        optimizer.zero_grad()
        if use_topo:
            model.learner.get_topo_val(sg_edges_)
        mask,add_loss = model.learner(x, sg_edges_, temp,sg_edges_attr,True)
        pred = model.gnn(x, sg_nodes_idx, sg_edges_, sg_edges_attr,mask)

        target = train_y[inter_idx].to(device)
        
        loss = criterion(pred[training_idx].to(torch.float32), target.to(torch.float32))+add_loss*0.1
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        num_edges+=mask.numel()
        num_masks+=mask.nonzero().size(0)

    print(f"pruning ratio:{num_masks/num_edges}")
    return statistics.mean(loss_list)


@torch.no_grad()
def multi_evaluate(valid_data_list, dataset, model, evaluator,temp, device,use_topo=True):
    model.eval()
    target = dataset.y.detach().numpy()

    train_pre_ordered_list = []
    valid_pre_ordered_list = []
    test_pre_ordered_list = []

    test_idx = dataset.test_idx.tolist()
    train_idx = dataset.train_idx.tolist()
    valid_idx = dataset.valid_idx.tolist()

    for valid_data_item in valid_data_list:
        sg_nodes, sg_edges, sg_edges_index, _ = valid_data_item
        idx_clusters = np.arange(len(sg_nodes))

        test_predict = []
        test_target_idx = []

        train_predict = []
        valid_predict = []

        train_target_idx = []
        valid_target_idx = []

        for idx in idx_clusters:
            x = dataset.x[sg_nodes[idx]].float().to(device)
            sg_nodes_idx = torch.LongTensor(sg_nodes[idx]).to(device)

            mapper = {node: idx for idx, node in enumerate(sg_nodes[idx])}
            sg_edges_attr = dataset.edge_attr[sg_edges_index[idx]].to(device)

            inter_tr_idx = intersection(sg_nodes[idx], train_idx)
            inter_v_idx = intersection(sg_nodes[idx], valid_idx)

            train_target_idx += inter_tr_idx
            valid_target_idx += inter_v_idx

            tr_idx = [mapper[tr_idx] for tr_idx in inter_tr_idx]
            v_idx = [mapper[v_idx] for v_idx in inter_v_idx]
            
            sg_edges_ = sg_edges[idx].to(device)
            if use_topo:
                model.learner.get_topo_val(sg_edges_)
            mask,add_loss = model.learner(x, sg_edges_, temp,sg_edges_attr,False)
            pred = model.gnn(x, sg_nodes_idx,sg_edges_ , sg_edges_attr,mask).cpu().detach()

            train_predict.append(pred[tr_idx])
            valid_predict.append(pred[v_idx])

            inter_te_idx = intersection(sg_nodes[idx], test_idx)
            test_target_idx += inter_te_idx

            te_idx = [mapper[te_idx] for te_idx in inter_te_idx]
            test_predict.append(pred[te_idx])

        train_pre = torch.cat(train_predict, 0).numpy()
        valid_pre = torch.cat(valid_predict, 0).numpy()
        test_pre = torch.cat(test_predict, 0).numpy()

        train_pre_ordered = train_pre[process_indexes(train_target_idx)]
        valid_pre_ordered = valid_pre[process_indexes(valid_target_idx)]
        test_pre_ordered = test_pre[process_indexes(test_target_idx)]

        train_pre_ordered_list.append(train_pre_ordered)
        valid_pre_ordered_list.append(valid_pre_ordered)
        test_pre_ordered_list.append(test_pre_ordered)

    train_pre_final = torch.mean(torch.Tensor(np.array(train_pre_ordered_list)), dim=0)
    valid_pre_final = torch.mean(torch.Tensor(np.array(valid_pre_ordered_list)), dim=0)
    test_pre_final = torch.mean(torch.Tensor(np.array(test_pre_ordered_list)), dim=0)

    eval_result = {}

    input_dict = {"y_true": target[train_idx], "y_pred": train_pre_final}
    eval_result["train"] = evaluator.eval(input_dict)

    input_dict = {"y_true": target[valid_idx], "y_pred": valid_pre_final}
    eval_result["valid"] = evaluator.eval(input_dict)

    input_dict = {"y_true": target[test_idx], "y_pred": test_pre_final}
    eval_result["test"] = evaluator.eval(input_dict)

    return eval_result

def cache_split_data(args, client_data):
    file_name = f"{args.is_iid}_{args.num_workers}_split.pkl"
    file_path = os.path.join("cache", file_name)
    if not os.path.exists("cache"):
        os.makedirs("cache")
    with open(file_path, 'wb') as f:
        pickle.dump(client_data, f)
    print(f"Data split saved to {file_path}")


def load_cached_data(args):
    # 构建文件名
    file_name = f"{args.is_iid}_{args.num_workers}_split.pkl"
    file_path = os.path.join("cache", file_name)

    # 如果文件存在，加载数据
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            client_data = pickle.load(f)
        print(f"Loaded cached data from {file_path}")
        return client_data
    return None

def main():
    args = ArgsInit().save_exp()

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")

    logging.info('%s' % device)

    # 加载数据集
    dataset = OGBNDataset(dataset_name=args.dataset)
    nf_path = dataset.extract_node_features(args.aggr)
    args.num_tasks = dataset.num_tasks
    args.nf_path = nf_path

    logging.info('%s' % args)

    evaluator = Evaluator(args.dataset)
    criterion = torch.nn.BCEWithLogitsLoss()

    # 尝试加载缓存的客户端数据集
    client_data_list = load_cached_data(args)

    # 如果缓存不可用，生成客户端数据集并保存
    if client_data_list is None:
        if args.is_iid == "iid":
            client_data_list = []
            for i in range(args.num_workers):
                parts = dataset.random_partition_graph(dataset.total_no_of_nodes, cluster_number=args.cluster_number)
                client_data = dataset.generate_sub_graphs(parts, cluster_number=args.cluster_number)
                client_data_list.append(client_data)
        elif args.is_iid == "non-iid-louvain":
            client_data_list = split_Louvain(args, dataset.whole_graph)
        cache_split_data(args, client_data_list)

    sub_dir = 'random-train_{}-test_{}-num_workers_{}'.format(args.cluster_number,
                                                              args.valid_cluster_number,
                                                              args.num_workers)
    logging.info(sub_dir)

    # 初始化客户端模型和全局模型
    model_list = []
    for i in range(args.num_workers):
        client_model = MoG(dataset.x.size(1), dataset.num_tasks, args, device).to(device)
        model_list.append(client_model)
    global_model = MoG(dataset.x.size(1), dataset.num_tasks, args, device).to(device)

    # 创建 results 目录（如果不存在）
    os.makedirs('results/acc', exist_ok=True)
    os.makedirs('results/sparsity', exist_ok=True)

    # 清空或创建文件，并写入表头
    acc_file_path = 'results/acc/acc.txt'
    sparsity_file_path = 'results/sparsity/sparsity.txt'

    with open(acc_file_path, 'w') as acc_file, open(sparsity_file_path, 'w') as sparsity_file:
        acc_file.write("Epoch,Average_Test_ROCAUC\n")
        sparsity_file.write("Epoch,Average_Sparsity\n")

    # 训练和评估循环
    results = {'highest_valid': 0, 'final_train': 0, 'final_test': 0, 'highest_train': 0}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # 计算稀疏学习器的温度
        if (epoch - 1) % args.temp_N == 0:
            decay_temp = np.exp(-1 * args.temp_r * epoch)
            temp = max(0.05, decay_temp)
        else:
            temp = 1.0

        # 将全局模型参数分发到每个客户端
        for client_model in model_list:
            client_model.load_state_dict(global_model.state_dict())
            client_model.to(device)  # 确保模型在设备上

        client_eval_results = []
        client_sparsity_results = []

        for client_idx, client_model in enumerate(model_list):
            client_data = client_data_list[client_idx]

            optimizer = torch.optim.Adam(client_model.parameters(), lr=args.lr)

            for inner_epoch in range(1, args.inner_epochs + 1):
                epoch_loss = train(client_data, dataset, client_model, optimizer, criterion, temp, device, args.use_topo)
                logging.info('Client {}, Epoch {}, training loss {:.4f}'.format(client_idx, epoch, epoch_loss))

            # 客户端评估
            with torch.no_grad():
                client_model.eval()
                client_result = multi_evaluate([client_data], dataset, client_model, evaluator, temp, device, args.use_topo)

                client_train_rocauc = client_result['train']['rocauc']
                client_test_rocauc = client_result['test']['rocauc']
                client_sparsity = client_result['sparsity']

                client_eval_results.append(client_test_rocauc)
                client_sparsity_results.append(client_sparsity)

                logging.info(f'Client {client_idx}, Test ROCAUC: {100 * client_test_rocauc:.2f}%, Sparsity: {100 * client_sparsity:.2f}%')

        # FedAvg 聚合
        fed_avg(global_model, model_list)
        logging.info("FedAvg aggregation has been completed")

        # 计算平均测试准确率和稀疏性
        if len(client_eval_results) > 0:
            average_test_rocauc = sum(client_eval_results) / len(client_eval_results)
            average_sparsity = sum(client_sparsity_results) / len(client_sparsity_results)
        else:
            average_test_rocauc = 0.0
            average_sparsity = 0.0
            logging.warning("No client evaluation results to average.")

        logging.info(f'Average Test ROCAUC across all clients: {100 * average_test_rocauc:.2f}%')
        logging.info(f'Average Sparsity across all clients: {100 * average_sparsity:.2f}%')

        # 记录结果
        with open(acc_file_path, 'a') as acc_file, open(sparsity_file_path, 'a') as sparsity_file:
            acc_file.write(f"Epoch {epoch},{average_test_rocauc}\n")
            sparsity_file.write(f"Epoch {epoch},{average_sparsity}\n")

        # 更新最佳验证和测试结果
        if average_test_rocauc > results['highest_valid']:
            results['highest_valid'] = average_test_rocauc
            save_ckpt(global_model, optimizer, round(epoch_loss, 4), epoch, args.model_save_path, sub_dir, name_post='valid_best')

    end_time = time.time()
    total_time = end_time - start_time
    logging.info('Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))

if __name__ == "__main__":
    main()
