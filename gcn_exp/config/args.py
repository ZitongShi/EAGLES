import argparse

def parser_loader():
    parser = argparse.ArgumentParser(description='MoG(arxiv)')
    # experiment settings
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--temp_N',type=int,default=20)
    parser.add_argument('--temp_r',type=float,default=1e-3)#0.001
    parser.add_argument('--seed',type=int,default=5)


    # args about optim
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay',type=float,default=1e-4)
    
    # args about gnn
    parser.add_argument('--hidden_channels',type=float,default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)    
    
    # args about SpLearner expert
    parser.add_argument('--hidden_spl',type=float,default=128)
    parser.add_argument('--num_layers_spl',type=int,default=2)
    
    # args about MoE
    parser.add_argument('--expert_select',type=int,default=3)
    parser.add_argument('--k_list', nargs='+', type=float)
    parser.add_argument('--lam',type=float,default=1e-1)
    parser.add_argument('--use_topo',default=True,action="store_true")
    parser.add_argument('--dataset', type=str, default='Reddit',
                        help='Dataset',
                        choices=['Cora', 'Citeseer', 'Pubmed', 'Flickr', 'ogbn-arxiv', 'Reddit', 'Reddit2',
                                 'Yelp',"Cs","Physics","computers","photo",'ogbn-products','ogbn-proteins'])
    parser.add_argument('--overlapping_rate', type=float, default=0.0, choices=[0.0,0.1,0.2,0.3,0.4,0.5],
                        help="Additional samples of overlapping data")
    parser.add_argument("--num_workers", type=int, default=10, help="number of clients")
    parser.add_argument("--is_iid", type=str, default="iid", choices=["iid", "non-iid-louvain",'non-iid-Metis'],
                        help="split the graph into the clients: random is randomly split, louvain is the community detection method")
    parser.add_argument('--lottery_epochs', type=int, default=100,help='number of communication epochs to find lottery tickets')
    parser.add_argument('--epochs', type=int, default=200,help='number of communication epochs to final train')
    parser.add_argument('--inner_epochs', type=int, default=10,help='number of inner epochs')
    parser.add_argument('--spar_wei', type=int, default=0,help='Whether weight sparsification is enabled or not')
    parser.add_argument('--w2loss', type=float, default=1e-3,help='weight of weight loss')
    parser.add_argument('--load_spar_wei', type=int, default=0,help='Whether to load the para sparsifier')
    parser.add_argument('--save_spar_wei', type=int, default=0,help='Whether to save the para sparsifier')
    parser.add_argument('--fix_mask', type=int, default=0,help='Whether to fix the para')
    parser.add_argument('--selected_pruning_rate', type=int, default=30,help='Whether to fix the para')
    parser.add_argument('--agg_method', type=str, default="FedAvg",
                        help='Federated Algorithms')
    parser.add_argument('--num_clusters', type=int, default=5, help='Number of clusters for FedCP')
    parser.add_argument('--lambda2', type=float, default=0.1, help='Number of clusters for FedCP')



    args = vars(parser.parse_args())
    assert len(args['k_list']),"The sparsity of each sparsifier must be specified"
    
    return args

