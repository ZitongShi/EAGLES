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
    """
        Apply a sparsely gated mixture of experts layer, where each expert is a single-layer feed-forward network.

        Args:
            input_size: Integer representing the size of the input.
            num_experts: Integer specifying the number of experts.
            hidden_size: Integer defining the hidden size of each expert.
            noisy_gating: Boolean indicating whether to apply noisy gating.
            k: Integer specifying the number of experts to select for each batch element.
    """

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
        """
            Calculate the squared coefficient of variation for a given sample.
            This measure is useful as a loss to encourage a more uniform positive distribution.
            Small epsilons are added for numerical stability.

            For an empty tensor, the function returns 0.

            Args:
                x: A tensor representing the sample.

            Returns:
                A scalar representing the squared coefficient of variation.
        """

        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """
            Compute the actual load per expert based on gate activations.
            The load represents the count of examples for which the corresponding gate activation is greater than zero.

            Args:
                gates: Tensor of shape [batch_size, n] representing gate activations.

            Returns:
                Tensor of shape [n], dtype float32, indicating the load for each expert.
        """

        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """
            Helper function for NoisyTopKGating.
            Calculates the probability of a value being in the top k when random noise is added.
            This approach enables backpropagation from a loss that encourages balanced selection
            of experts within the top k for each example.

            If no noise is added (by setting noise_stddev to None), the result will be non-differentiable.

            Args:
                clean_values: Tensor of shape [batch, n] representing the original values.
                noisy_values: Tensor of shape [batch, n], equal to clean values plus random noise
                              with standard deviation defined by noise_stddev.
                noise_stddev: Tensor of shape [batch, n], or None if no noise is added.
                noisy_top_values: Tensor of shape [batch, m] containing the top m values
                                  from the noisy values (m >= k+1).

            Returns:
                Tensor of shape [batch, n] representing the probability of each value being in the top k.
        """

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, edge_index,train, noise_epsilon=1e-2):
        """
            Noisy Top-k Gating.
            Refer to the paper: https://arxiv.org/abs/1701.06538.

            Args:
                x: Input tensor with shape [batch_size, input_size].
                train: Boolean indicating whether noise should be added (applied only during training).
                noise_epsilon: Float value controlling the level of added noise.

            Returns:
                gates: Tensor with shape [batch_size, num_experts] representing gate activations.
                load: Tensor with shape [num_experts] representing expert load distribution.
        """

        #print("the shape of w_gate is:",self.w_gate.shape)
        #pdb.set_trace()
        clean_logits = x @ self.w_gate # size:(nums_node,nums_expert)
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k+1, self.num_experts), dim=1) 
        top_k_logits = top_logits[:, :self.k]# size:(batch_size,self.k)
        top_k_indices = top_indices[:, :self.k]# size:(batch_size,self.k)
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)  # size:(batch_size,num_experts)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, input_nodes, adjacency_index, temperature, data_shape, adjacency_attr=None, is_training=None):
        """
        Args:
            input_nodes: tensor shape [num_nodes, input_size]
            adjacency_index: tensor shape [2, num_edges]
            temperature: additional tensor
            data_shape: additional argument
            adjacency_attr: tensor shape [num_edges, edge_attr_size] or None
            is_training: boolean indicating training mode
        Returns:
            mask: tensor shape [num_edges]
            loss: scalar
        """
        node_selections, workload = self.noisy_top_k_gating(input_nodes, adjacency_index, self.training)
        relevance = node_selections.sum(0)  # size: [num_experts]
        loss_val = self.cv_squared(relevance) + self.cv_squared(workload)
        loss_val *= self.loss_coef
        edge_selections = torch.index_select(node_selections, dim=0,
                                             index=adjacency_index[0])  # [num_edges, num_experts]
        expert_outputs = []
        for expert_index in range(self.num_experts):
            expert_output = self.experts[expert_index](input_nodes, adjacency_index, adjacency_attr, data_shape,
                                                       temperature, is_training)  # size:(num_edge)
            if self.topo_val is not None:
                expert_output = expert_output * self.lam + self.topo_val[:, expert_index % 4]
            expert_outputs.append(expert_output)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # shape: [num_edges, num_experts]
        gated_output = edge_selections * expert_outputs  # [num_edges, num_experts]
        gated_output = gated_output.mean(dim=1)  # [num_edges]
        node_indices, edges_per_node = adjacency_index[0].unique(return_counts=True)
        selected_k_per_node = torch.sum(node_selections * torch.unsqueeze(self.k_list, 0),
                                        dim=1)  # [num_nodes, num_experts] -> [num_nodes]
        edges_selected_per_node = selected_k_per_node[
                                      node_indices] * edges_per_node  # [num_nodes] * [num_nodes] -> [num_nodes]

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

        start_edge_indices = torch.cat(
            (torch.tensor([0], device=adjacency_index.device), torch.cumsum(edges_per_node[:-1], dim=0))
        )  # [num_nodes]
        end_edge_indices = torch.abs(
            torch.add(start_edge_indices, edges_selected_per_node_long) - 1).long()  # [num_nodes]
        node_thresholds = torch.index_select(reordered_scores, dim=-1, index=end_edge_indices)
        augmented_node_thresholds = node_thresholds.repeat_interleave(edges_per_node)  # [num_edges]
        mask = BinaryStep.apply(reordered_scores - augmented_node_thresholds + 1e-12)

        final_sort_idx = sorted_idx_reorder.argsort()
        mask = mask.index_select(dim=-1, index=final_sort_idx)
        mask = mask.index_select(dim=-1, index=sorted_idx)

        self.edges_per_node = edges_per_node
        self.edges_selected_per_node = edges_selected_per_node
        self.selected_k_per_node = selected_k_per_node

        return mask, loss_val


def min_max_normalization(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor
