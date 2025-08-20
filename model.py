import torch
import torch.nn as nn
import torch.nn.functional as F

from gates.gshard_gate import GShardGate


class Expert(nn.Module):
    """Простой эксперт: двухслойный MLP"""
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model)
        )

    def forward(self, x):
        return self.net(x)

class MoELayer(nn.Module):
    """Mixture of Experts с GShard Gate"""
    def __init__(self, d_model, num_experts, top_k=2, world_size=1):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(d_model, d_model * 4) for _ in range(num_experts)])
        # ВАЖНО: явно передаём world_size и top_k по именам
        self.gate = GShardGate(d_model, num_experts, world_size=world_size, top_k=top_k)

    def forward(self, x):
        """
        x: [batch*seq_len, d_model]
        """
        device = x.device
        top_scores, top_indices = self.gate(x)   # top_scores: [N, top_k], top_indices: [N, top_k]
        N, k = top_indices.shape
        d_model = x.size(-1)

        # Проверка: все индексы должны быть в диапазоне 0..num_experts-1 (при world_size==1)
        if top_indices.max().item() >= self.num_experts or top_indices.min().item() < 0:
            raise IndexError(f"Gate produced expert index out of local range: max {top_indices.max().item()}, num_experts {self.num_experts}")

        out = torch.zeros_like(x, device=device)  # [N, d_model]

        # Для простоты: делаем по k, а для каждого эксперта применяем маску
        for i in range(k):
            expert_idx = top_indices[:, i]                      # [N]
            # one-hot по локальному числу экспертов (world_size assumed 1)
            mask = F.one_hot(expert_idx, num_classes=self.num_experts).float().to(device)  # [N, num_experts]

            # применяем каждому эксперту соответствующие элементы
            for e_idx, expert in enumerate(self.experts):
                expert_mask = mask[:, e_idx].unsqueeze(-1)  # [N,1]
                if expert_mask.sum().item() == 0:
                    continue
                x_expert = x * expert_mask                  # zero out non-selected
                out_expert = expert(x_expert)               # [N, d_model]
                out = out + out_expert * top_scores[:, i].unsqueeze(-1)

        return out


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_experts, top_k):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, num_experts, top_k)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention
        res = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = res + attn_out

        # MoE
        res = x
        x = self.norm2(x)
        bsz, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        moe_out = self.moe(x_flat)
        moe_out = moe_out.view(bsz, seq_len, d_model)
        return res + moe_out


class TransformerWithMoE(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=3, num_experts=10, top_k=2, padding_idx=None):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Parameter(torch.randn(1024, d_model) * 0.02)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_experts, top_k)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        bsz, seq_len = x.size()
        x = self.token_emb(x) + self.pos_emb[:seq_len]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)  # [batch, seq_len, vocab_size]
