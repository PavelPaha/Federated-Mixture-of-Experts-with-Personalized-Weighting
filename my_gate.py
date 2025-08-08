import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmoe.fmoe.gates.naive_gate import NaiveGate


class MyGate(NaiveGate):
    def __init__(self, d_model, num_expert, world_size, topk=1,
                 switch_eps=0.1, capacity=(1.2, 2.4), gate_bias=True):
        assert topk == 1, 'topk should be 1 in switch'
        super().__init__(d_model, num_expert, world_size, top_k=1, gate_bias=gate_bias)
        self.switch_eps = switch_eps
        self.capacity = capacity

    def forward(self, inp, return_all_scores: bool = False):
        gate_logits = self.gate(inp)
        self.last_gate_logits = gate_logits.detach()
        score = gate_logits

        if self.training:
            noise = torch.rand_like(score)
            noise = noise * 2 * self.switch_eps + 1.0 - self.switch_eps
            score += noise

        score = F.softmax(score.float(), dim=-1)
        top1_score, top1_idx = torch.topk(score, k=1, dim=-1)  # [N, 1]
        top1_score = top1_score.to(dtype=inp.dtype)

        valid_idx = top1_idx[top1_idx > -1]

        if valid_idx.numel() > 0:
            fraction_expert = torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            ) / valid_idx.numel()

            prob_expert = score.sum(dim=0) / valid_idx.numel()
            loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        else:
            loss = torch.tensor(0.0, device=inp.device, dtype=inp.dtype)

        self.set_loss(loss)

        if return_all_scores:
            return top1_idx, top1_score, gate_logits
        return top1_idx, top1_score
