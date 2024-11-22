import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from SpLearner import SpLearner

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
        self.experts = nn.ModuleList([SpLearner(nlayers=nlayers, in_dim=input_size*2+1, hidden=hidden_size, activation=activation, k=k) for k in k_list])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.lam = lam
        self.topo_val = None
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
        prob = torch.where(is_in)
        return prob

    def noisy_top_k_gating(self, x, edge_index,train, noise_epsilon=1e-2):
        #pdb.set_trace()
        clean_logits = x @ self.w_gate
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

    def forward(self, input_nodes, adjacency_index, temperature, data_shape, adjacency_attr=None, is_training=None):
        node_selections, workload = self.noisy_top_k_gating(input_nodes, adjacency_index, self.training)
        relevance = node_selections.sum(0)
        loss_val = self.cv_squared(relevance) + self.cv_squared(workload)
        loss_val *= self.loss_coef
        edge_selections = torch.index_select(node_selections, dim=0,
                                             index=adjacency_index[0])
        expert_outputs = []
        for expert_index in range(self.num_experts):
            expert_output = self.experts[expert_index](input_nodes, adjacency_index, adjacency_attr, data_shape,
                                                       temperature, is_training)
            if self.topo_val is not None:
                expert_output = expert_output * self.lam + self.topo_val[:, expert_index % 4]
            expert_outputs.append(expert_output)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        gated_output = edge_selections * expert_outputs
        gated_output = gated_output.mean(dim=1)
        node_indices, edges_per_node = adjacency_index[0].unique(return_counts=True)
        selected_k_per_node = torch.sum(node_selections * torch.unsqueeze(self.k_list, 0),
                                        dim=1)
        edges_selected_per_node = selected_k_per_node[
                                      node_indices] * edges_per_node

        edges_selected_per_node_long = edges_selected_per_node.round().long()
        edges_selected_per_node_long = torch.where(
            edges_selected_per_node_long > 0,
            edges_selected_per_node_long,
            torch.ones_like(edges_selected_per_node_long, device=edges_selected_per_node.device)
        )
        sorted_sparse_values, sorted_idx = gated_output.sort(descending=True)
        sparse_idx0 = adjacency_index[0].index_select(dim=-1, index=sorted_idx)
        sorted_idx_reorder = sparse_idx0.argsort(stable=True, dim=-1, descending=False)
        reordered_scores = sorted_sparse_values.index_select(dim=-1, index=sorted_idx_reorder)
        start_edge_indices = torch.cat((torch.tensor([0], device=adjacency_index.device), torch.cumsum(edges_per_node[:-1], dim=0)))
        end_edge_indices = torch.abs(torch.add(start_edge_indices, edges_selected_per_node_long) - 1).long()
        node_thresholds = torch.index_select(reordered_scores, dim=-1, index=end_edge_indices)
        augmented_node_thresholds = node_thresholds.repeat_interleave(edges_per_node)
        mask = BinaryStep.apply(reordered_scores - augmented_node_thresholds + 1e-12)
        final_sort_idx = sorted_idx_reorder.argsort()
        mask = mask.index_select(dim=-1, index=final_sort_idx)
        mask = mask.index_select(dim=-1, index=sorted_idx)
        return mask, loss_val


def min_max_normalization(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor
