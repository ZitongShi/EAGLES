import pdb
from operator import index
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero_index = torch.abs(input) > 1
        middle_index = (torch.abs(input) <= 1) * (torch.abs(input) > 0.4)
        additional = 2-4*torch.abs(input)
        additional[zero_index] = 0.
        additional[middle_index] = 0.4
        return grad_input*additional

eps = 1e-8
class Expert(nn.Module):
    def __init__(self, nlayers, in_dim, hidden, activation, k, weight=True, metric=None, processors=None):
        super().__init__()

        self.nlayers = nlayers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden))
        for _ in range(nlayers - 2):
            self.layers.append(nn.Linear(hidden, hidden))
        self.layers.append(nn.Linear(hidden, 1))

        self.param_init()
        self.activation = activation
        self.k = k
        self.weight = weight

    def param_init(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)

    def hard_concrete_sample(self, indices, values, temperature, training):

        r = self.sample_uniform_noise(values.shape)
        if training is not None:
            values = torch.log(values + eps) + r.to(indices.device)
        else:
            values = torch.log(values + eps)
        values /= temperature
        stretched_values = torch.sigmoid(values)
        hard_values = torch.clamp(stretched_values, min=0, max=1)
        y = torch.sparse_coo_tensor(indices=indices, values=hard_values, requires_grad=True)
        return torch.sparse.softmax(y, dim=1)

    def sample_uniform_noise(shape):
        U = torch.rand(shape)
        return torch.log(U + eps) - torch.log(1 - U + eps)
    def forward(self, features, indices,values, temperature,training=None):
        # print("indices shape:",indices.shape)
        #pdb.set_trace()
        f1_features = torch.index_select(features, 0, indices[0, :])
        f2_features = torch.index_select(features, 0, indices[1, :])
        auv = values
        #print("f1_features:", f1_features)
        #print("f2_features:", f2_features)
        #print("auv:", auv)
        temp = torch.cat([f1_features,f2_features,auv],-1)
        temp = self.internal_forward(temp)
        z = torch.reshape(temp, [-1])
        z_matrix = torch.sparse_coo_tensor(indices=indices, values=z,requires_grad=True)
        pi = torch.sparse.softmax(z_matrix, dim=1) 
        pi_values = pi.coalesce().values()
        y = self.hard_concrete_sample(indices, pi_values, temperature, training)
        sparse_indices = y.coalesce().indices()
        sparse_values = y.coalesce().values()
        num_edges_per_node = sparse_indices[0].unique(return_counts=True)
        k_edges_per_node = (num_edges_per_node.float() * self.k).round().long()
        val_sort_idx = sparse_values.sort(descending=True)
        sparse_idx0 = sparse_indices[0].index_select(dim = -1,index = val_sort_idx)  
        idx_sort_idx = sparse_idx0.argsort(stable=True,dim=-1,descending = False) 
        scores_sorted = sparse_values.index_select(dim=-1,index=idx_sort_idx) 

        edge_start_indices = torch.cat((torch.tensor([0],device=y.device), torch.cumsum(num_edges_per_node[:-1], dim=0)))
        edge_end_indices = torch.abs(torch.add(edge_start_indices,k_edges_per_node)-1).long()  # (num_nodes)
        node_keep_thre_cal = torch.index_select(scores_sorted,dim=-1,index=edge_end_indices) 
        node_keep_thre_augmented = node_keep_thre_cal.repeat_interleave(num_edges_per_node) # num_edges
        mask = BinaryStep.apply(scores_sorted-node_keep_thre_augmented+1e-7) 
        masked_scores = mask*scores_sorted
        
        idx_resort_idx = idx_sort_idx.argsort()
        val_resort_idx = val_sort_idx.argsort()
        masked_scores = masked_scores.index_select(dim=-1,index = idx_resort_idx)
        masked_scores = masked_scores.index_select(dim=-1,index = val_resort_idx)
        return masked_scores
        

        
        
        