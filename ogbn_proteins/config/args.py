import argparse
import uuid
import logging
import time
import os
import sys
from utils import create_exp_dir
import glob


class ArgsInit(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='MoG(proteins)')
        # dataset
        parser.add_argument('--dataset', type=str, default='ogbn-proteins',
                            help='dataset name (default: ogbn-proteins)')
        parser.add_argument('--cluster_number', type=int, default=10,
                            help='the number of sub-graphs for training')
        parser.add_argument('--valid_cluster_number', type=int, default=5,
                            help='the number of sub-graphs for evaluation')
        parser.add_argument('--aggr', type=str, default='add',
                            help='the aggregation operator to obtain nodes\' initial features [mean, max, add]')
        parser.add_argument('--nf_path', type=str, default='init_node_features_add.pt',
                            help='the file path of extracted node features saved.')
        # training & eval settings
        parser.add_argument('--use_gpu', action='store_true')
        parser.add_argument('--device', type=int, default=0,
                            help='which gpu to use if any (default: 0)')

        parser.add_argument('--num_evals', type=int, default=1,
                            help='The number of evaluation times')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='learning rate set for optimizer.')
        parser.add_argument('--dropout', type=float, default=0.0)
        parser.add_argument('--temp_N',type=int,default=50)
        parser.add_argument('--temp_r',type=float,default=1e-3)
        parser.add_argument('--seed',type=int,default=5)
        # model
        parser.add_argument('--num_layers', type=int, default=3,
                            help='the number of layers of the networks')
        parser.add_argument('--mlp_layers', type=int, default=2,
                            help='the number of layers of mlp in conv')
        parser.add_argument('--hidden_channels', type=int, default=64,
                            help='the dimension of embeddings of nodes and edges')
        parser.add_argument('--block', default='plain', type=str,
                            help='graph backbone block type {res+, res, dense, plain}')
        parser.add_argument('--conv', type=str, default='gen',
                            help='the type of GCNs')
        parser.add_argument('--gcn_aggr', type=str, default='max',
                            help='the aggregator of GENConv [mean, max, add, softmax, softmax_sg, softmax_sum, power, power_sum]')
        parser.add_argument('--norm', type=str, default='layer',
                            help='the type of normalization layer')
        parser.add_argument('--num_tasks', type=int, default=1,
                            help='the number of prediction tasks')
        # learnable parameters
        parser.add_argument('--t', type=float, default=1.0,
                            help='the temperature of SoftMax')
        parser.add_argument('--p', type=float, default=1.0,
                            help='the power of PowerMean')
        parser.add_argument('--y', type=float, default=0.0,
                            help='the power of degrees')
        parser.add_argument('--learn_t', action='store_true')
        parser.add_argument('--learn_p', action='store_true')
        parser.add_argument('--learn_y', action='store_true')
        # message norm
        parser.add_argument('--msg_norm', action='store_true')
        parser.add_argument('--learn_msg_scale', action='store_true')
        # encode edge in conv
        parser.add_argument('--conv_encode_edge', action='store_true')
        # if use one-hot-encoding node feature
        parser.add_argument('--use_one_hot_encoding', action='store_true')
        # save model
        parser.add_argument('--model_save_path', type=str, default='model_ckpt',
                            help='the directory used to save models')
        parser.add_argument('--save', type=str, default='EXP', help='experiment name')
        # load pre-trained model
        parser.add_argument('--model_load_path', type=str, default='ogbn_proteins_pretrained_model.pth',
                            help='the path of pre-trained model')
        # args about SpLearner expert
        parser.add_argument('--hidden_spl',type=float,default=32)
        parser.add_argument('--num_layers_spl',type=int,default=2)
        
        # args about MoE
        parser.add_argument('--expert_select',type=int,default=3)
        parser.add_argument('--k_list', nargs='+', type=float)
        parser.add_argument('--lam',type=float,default=1e-1)  
        parser.add_argument('--use_topo',default=False,action="store_true")


        parser.add_argument('--overlapping_rate', type=float, default=0.0, choices=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                            help="Additional samples of overlapping data")
        parser.add_argument("--num_workers", type=int, default=10, help="number of clients")
        parser.add_argument("--is_iid", type=str, default="iid", choices=["iid", "non-iid-louvain", 'non-iid-Metis'],
                            help="split the graph into the clients: random is randomly split, louvain is the community detection method")
        parser.add_argument('--epochs', type=int, default=200, help='number of communication epochs')
        parser.add_argument('--inner_epochs', type=int, default=10, help='number of inner epochs')
        parser.add_argument('--spar_wei', type=int, default=0, help='Whether weight sparsification is enabled or not')
        parser.add_argument('--w2loss', type=float, default=1e-3, help='weight of weight loss')
        parser.add_argument('--load_spar_wei', type=int, default=0, help='')
        parser.add_argument('--save_spar_wei', type=int, default=0, help='')
        parser.add_argument('--lambda2', type=float, default=0.3, help='')
        parser.add_argument('--selected_pruning_rate', type=int, default=90, help='')
        self.args = parser.parse_args()
        assert len(self.args.k_list),"The sparsity of each sparsifier must be specified"


    def save_exp(self):
        self.args.save = '{}-B_{}-C_{}-L_{}-F_{}-DP_{}' \
                    '-A_{}-GA_{}-T_{}-LT_{}-P_{}-LP_{}-Y_{}-LY_{}' \
                    '-MN_{}-LS_{}'.format(self.args.save, self.args.block, self.args.conv,
                                          self.args.num_layers, self.args.hidden_channels, self.args.dropout,
                                          self.args.aggr, self.args.gcn_aggr,
                                          self.args.t, self.args.learn_t,
                                          self.args.p, self.args.learn_p,
                                          self.args.y, self.args.learn_y,
                                          self.args.msg_norm, self.args.learn_msg_scale)

        self.args.save = 'log/{}-{}-{}'.format(self.args.save, time.strftime("%Y%m%d-%H%M%S"), str(uuid.uuid4()))
        self.args.model_save_path = os.path.join(self.args.save, self.args.model_save_path)
        create_exp_dir(self.args.save, scripts_to_save=glob.glob('*.py'))
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout,
                            level=logging.INFO,
                            format=log_format,
                            datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(self.args.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

        return self.args
