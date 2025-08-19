import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_gate import BaseGate


class GShardGate(BaseGate):
    """GShard-style Top-k Gate with load balancing loss"""
    def __init__(self, d_model, num_expert, world_size=1, top_k=2):
        super().__init__(num_expert, world_size)
        self.top_k = top_k
        self.w_gating = nn.Linear(d_model, self.tot_expert)

    def forward(self, x):
        """
        x: [N, d_model]
        returns:
          top_scores: [N, top_k]   (normalized weights)
          top_indices: [N, top_k]  (chosen expert indices in [0, tot_expert-1])
        """
        N = x.size(0)
        logits = self.w_gating(x)                  # [N, tot_expert]
        scores = F.softmax(logits, dim=-1)         # [N, tot_expert]

        # top-k экспертов
        top_scores, top_indices = torch.topk(scores, self.top_k, dim=-1)  # both [N, top_k]

        # нормировка внутри top-k (по строке)
        top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-12)

        # ---------- balance loss ----------
        # prob_per_expert: avg softmax prob per expert
        prob_per_expert = scores.mean(dim=0)  # [tot_expert]

        # actual counts: сколько раз каждый эксперт был выбран (суммируем по N и top_k)
        one_hot = F.one_hot(top_indices, num_classes=self.tot_expert).float()  # [N, top_k, tot_expert]
        counts = one_hot.sum(dim=(0, 1)).float()  # [tot_expert]  (sum over N and top_k)

        # load: доля назначений на эксперта
        load = counts / (N * float(self.top_k) + 1e-12)  # [tot_expert], в диапазоне [0,1]

        # L_balance (GShard-like): tot_expert * sum(prob * load)
        balance_loss = self.tot_expert * torch.dot(prob_per_expert, load)

        # сохраняем лосс в гейте
        self.set_loss(balance_loss)
        
        # сохраняем выход гейта для логирования
        self.save_gate_output(scores)

        return top_scores, top_indices
