
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmoe.fmoe.gates.base_gate import BaseGate


# r"""
# Gumbel-Softmax gate
# """
# from .base_gate import BaseGate

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class MyGate(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2,
                 gate_bias=True, temperature=1.0, hard=False):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert, bias=gate_bias)
        self.top_k = top_k
        self.temperature = temperature
        self.hard = hard
        # last_* будут заполнены в forward для логов/диагностики и градиента
        self.last_gate_logits = None
        self.last_gumbel_scores = None

    def forward(self, inp, return_all_scores=False):
        gate_logits = self.gate(inp)                 # [B*L, E]
        self.last_gate_logits = gate_logits          # <-- сохраняем (без detach)

        gumbel_scores = F.gumbel_softmax(
            gate_logits, tau=self.temperature, hard=self.hard, dim=-1
        )  # [B*L, E]

        self.last_gumbel_scores = gumbel_scores     # <-- сохраняем

        gate_top_k_val, gate_top_k_idx = torch.topk(
            gumbel_scores, k=self.top_k, dim=-1, largest=True, sorted=False
        )

        gate_score = gate_top_k_val / (gate_top_k_val.sum(dim=-1, keepdim=True) + 1e-9)

        probs = F.softmax(gate_logits, dim=-1)          # [N, E]
        importance = probs.mean(dim=0)                  # [E]
        load = probs.sum(dim=0)                         # [E]
        load = load / (load.sum() + 1e-12)

        loss_raw = self.tot_expert * (importance * load).sum()
        loss = loss_raw - 1.0

        # set_loss должен принимать тензор, который является частью графа
        self.set_loss(loss)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gumbel_scores
        return gate_top_k_idx, gate_score
