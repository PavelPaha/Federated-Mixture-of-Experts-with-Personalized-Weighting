import os
import pickle
import torch
import torch.nn as nn

from fmoe.transformer import FMoETransformerMLP
# from my_gshard import GShardGate
from my_gate import MyGate
from gumbel_gate import GumbelGate


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_experts, world_size, top_k, gate_hook=None):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.moe = FMoETransformerMLP(
            num_expert=num_experts,
            world_size=world_size,
            d_model=d_model,
            d_hidden=d_model * 4,
            top_k=top_k,
            activation=nn.GELU(),
            expert_dp_comm="none",
            expert_rank=0,
            gate=GumbelGate,
            gate_hook=gate_hook
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        res = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = res + attn_out

        res = x
        x = self.norm2(x)
        bsz, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        moe_out = self.moe(x_flat)
        moe_out = moe_out.view(bsz, seq_len, d_model)
        return res + moe_out


class TransformerWithMoE(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=3, num_experts=10, wolrd_size=2,top_k=2, padding_idx=None, gate_hook=None):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model,
                                      padding_idx=padding_idx)
        self.pos_emb = nn.Parameter(torch.randn(1024, d_model) * 0.02)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_experts, wolrd_size, top_k, gate_hook=gate_hook)
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
        return self.head(x)

