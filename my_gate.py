r"""
Naive gate
"""
from fmoe.gates.base_gate import BaseGate

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyGate(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, gate_bias=True):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert, bias = gate_bias)
        self.top_k = top_k 

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_expert = 1
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def raw_forward(self, inp, return_all_scores=True):
        gate_out = self.gate(inp)
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate_out, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)

        # (BxL) x 1 x top_k
        gate_score = F.softmax(gate_top_k_val, dim=-1)

        # Calculate load balancing loss
        # Get full softmax for importance calculation
        full_gates = F.softmax(gate_out, dim=-1)
        
        # Calculate importance (how much each expert is used)
        importance = full_gates.sum(0)
        
        # Calculate load balancing loss using coefficient of variation
        balance_loss = self.cv_squared(importance)
        
        self.set_loss(balance_loss)

        if return_all_scores:
            return gate_top_k_idx, gate_score, F.softmax(gate_out, dim=-1)
        return gate_top_k_idx, gate_score

    def forward(self, inp, return_all_scores=False):
        return self.raw_forward(inp, return_all_scores)
