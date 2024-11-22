
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
from expert import Expert
import networkit as nk
import networkx as nx


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



class moe(nn.Module):

    def __init__(self, input_size,hidden_size, num_experts,nlayers,activation,k_list,expert_select ,noisy_gating=True, coef=1e-2,lam=1):
        super(moe, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.k = expert_select
        self.loss_coef = coef
        self.k_list = k_list
        self.experts = nn.ModuleList([Expert(nlayers=nlayers, in_dim=input_size * 2 + 8, hidden=hidden_size, activation=activation, k=k) for k in k_list])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.lam = lam
        self.topo_val =None
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):

        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
    
    
    def noisy_top_k_gating(self, x, edge_index,train, noise_epsilon=1e-2):
        #print(f"Input x shape: {x.shape}")
        #print(f"Input edge_index shape: {edge_index.shape}")
        clean_logits = x @ self.w_gate # size:(nums_node,nums_expert)
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k+1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, edge_index, temp, edge_attr=None, training=None):

        node_gates, load = self.noisy_top_k_gating(x, edge_index, self.training)
        importance = node_gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= self.loss_coef
        edge_gates = torch.index_select(node_gates, dim=0, index=edge_index[0])
        expert_outputs = []
        for i in range(self.num_experts):
            expert_i_output = self.experts[i](x, edge_index, edge_attr, temp, training)
            if self.topo_val is not None:
                expert_i_output = expert_i_output * self.lam + self.topo_val[:, i % 4]
            expert_outputs.append(expert_i_output)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        gated_output = edge_gates * expert_outputs
        gated_output = gated_output.mean(dim=1)
        node_idx, num_edges_per_node = edge_index[0].unique(return_counts=True)
        k_per_node = torch.sum(node_gates * torch.unsqueeze(self.k_list, 0),
                               dim=1)
        k_edges_per_node = k_per_node[node_idx] * num_edges_per_node
        k_edges_per_node_long = k_edges_per_node.round().long()
        k_edges_per_node_long = torch.where(
            k_edges_per_node_long > 0,
            k_edges_per_node_long,
            torch.ones_like(k_edges_per_node_long, device=k_edges_per_node.device)
        )
        sparse_values, val_sort_idx = gated_output.sort(descending=True)
        sparse_idx0 = edge_index[0].index_select(dim=-1, index=val_sort_idx)
        idx_sort_idx = sparse_idx0.argsort(stable=True, dim=-1, descending=False)
        scores_sorted = sparse_values.index_select(dim=-1, index=idx_sort_idx)
        edge_start_indices = torch.cat(
            (torch.tensor([0], device=edge_index.device), torch.cumsum(num_edges_per_node[:-1], dim=0))
        )
        edge_end_indices = torch.abs(torch.add(edge_start_indices, k_edges_per_node_long) - 1).long()
        node_keep_thre_cal = torch.index_select(scores_sorted, dim=-1, index=edge_end_indices)
        node_keep_thre_augmented = node_keep_thre_cal.repeat_interleave(num_edges_per_node)
        mask = BinaryStep.apply(scores_sorted - node_keep_thre_augmented + 1e-12)
        idx_resort_idx = idx_sort_idx.argsort()
        mask = mask.index_select(dim=-1, index=idx_resort_idx)
        mask = mask.index_select(dim=-1, index=val_sort_idx)
        D_complete = torch.zeros(edge_index.max() + 1, device=mask.device)
        P_complete = torch.zeros(edge_index.max() + 1, device=mask.device)
        D_complete[node_idx] = num_edges_per_node.float()
        P_complete[node_idx] = num_edges_per_node.float() - k_edges_per_node
        self.num_edges_per_node = num_edges_per_node
        self.k_edges_per_node = k_edges_per_node
        self.k_per_node = k_per_node

        return mask, loss

    def get_topo_val(self,edge_index):
        G=nx.DiGraph()
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)
        G = nk.nxadapter.nx2nk(G)
        G.indexEdges()
        lds = nk.sparsification.LocalDegreeScore(G).run().scores()
        ffs = nk.sparsification.ForestFireScore(G, 0.6, 5.0).run().scores()
        triangles = nk.sparsification.TriangleEdgeScore(G).run().scores()
        lss = nk.sparsification.LocalSimilarityScore(G, triangles).run().scores()
        scan = nk.sparsification.SCANStructuralSimilarityScore(G, triangles).run().scores()
        topo_val = torch.tensor([lds,ffs,lss,scan],device = edge_index.device).t()
        normalized_features = F.normalize(topo_val,dim=0)
        self.topo_val = normalized_features

    
def min_max_normalization(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor
