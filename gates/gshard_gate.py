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

    def forward(self, x, capacity=None):
      """
      x: [N, d_model]
      returns:
        top_scores: [N, top_k]
        top_indices: [N, top_k]
      """
      N = x.size(0)
      logits = self.w_gating(x)                  # [N, tot_expert]
      scores = F.softmax(logits, dim=-1)         # [N, tot_expert]

      # top-k экспертов
      top_scores, top_indices = torch.topk(scores, self.top_k, dim=-1)  # [N, top_k]
      top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-12)

      # ---------------- AUX LOSS ----------------
      # mean gates per expert (m_e)
      mean_gates = scores.mean(dim=0)  # [E]

      # фактические назначения (c_e)
      # берём только top-1 (в картинке баланс считается на основе e1)
      one_hot_e1 = F.one_hot(top_indices[:, 0], num_classes=self.tot_expert).float()  # [N, E]
      counts = one_hot_e1.sum(dim=0)  # [E]

      # нормализуем по размеру группы
      frac_assigned = counts / float(N)  # c_e / S

      # формула из статьи
      balance_loss = (frac_assigned * mean_gates).mean() * self.tot_expert

      # сохранить
      self.set_loss(balance_loss)
      self.save_gate_output(mean_gates)

      return top_scores, top_indices
