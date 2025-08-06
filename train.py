import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.utils.data.dataset import IterableDataset
from tqdm import tqdm
from config import TrainConfig
from copy import deepcopy

from fmoe.transformer import FMoETransformerMLP
from fmoe.gates import NaiveGate

from model import TransformerWithMoE
from data import WikiText103LMIterable, create_wikitext_dataloader, tokenizer

import os
import datetime
import torch
import torch.nn as nn
from tqdm import tqdm
from config import TrainConfig, Metrics

vocab_size = tokenizer.vocab_size + 1

gate_scores = None
def get_scores(module, inp, out, idx=0):
    global gate_scores
    gate_scores = module.raw_forward(inp[0], return_all_scores=True)
    print(gate_scores[-1].shape)


def train(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    config: TrainConfig,
    device="cuda"
):
    """
    Запускает тренировку модели с учётом параметров из config,
    накапливает метрики в config.metrics и по завершении
    сохраняет config вместе с метриками в человекочитаемый JSON.
    """
    model.to(device)
    model.train()

    # Сброс метрик перед стартом
    config.metrics = Metrics()
    global_iter = 0

    for epoch in range(1, config.epochs + 1):
        loop = tqdm(enumerate(train_loader, 1), desc=f"Epoch {epoch}/{config.epochs}")
        for batch_idx, (inputs, targets) in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            target_loss = criterion(
                outputs.view(-1, outputs.size(-1)),
                targets.view(-1)
            )

            balance_loss = 0.0
            cnt = 0
            for m in model.modules():
                if hasattr(m, "gate") and getattr(m.gate, "has_loss", False):
                    balance_loss += m.gate.get_loss()
                    cnt += 1
            if cnt > 0:
                balance_loss = balance_loss / cnt
                total_loss = target_loss + config.alpha * balance_loss
            else:
                total_loss = target_loss

            gate_top_k_idx, gate_score, expert_distr = gate_scores
            seq_len = expert_distr.shape[0] # на самом деле это batch_size * seq_len
            raise Exception(expert_distr.shape)

            B, S = config.batch_size, config.seq_len

            expert_distr_by_device = expert_distr.view(B, S, config.world_size, config.num_experts_per_device) # seq_len x world_size x num_experts_per_device

            fashions = expert_distr_by_device.max(dim=-1).values     # (batch, seq, world)
            dist2   = (expert_distr_by_device**2).mean(dim=-1)        # (batch, seq, world)
            fash2   = (fashions**2).mean(dim=-1)                     # (batch, seq)
            loss_dist    = dist2.mean()      # scalar
            loss_fashion = fash2.mean()      # scalar

            total_loss = total_loss + config.lambda_2 * loss_dist - config.lambda_1 * loss_fashion

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_iter += 1

            config.metrics.train_losses.target_loss.append(target_loss.item())
            config.metrics.train_losses.balance_loss.append(balance_loss.item() if cnt > 0 else 0.0)

            config.metrics.train_losses.distribution_loss.append(loss_dist.item())
            config.metrics.train_losses.fashion_loss.append(loss_fashion.item())
 
            if global_iter % config.log_interval == 0:
                
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for t_in, t_tgt in test_loader:
                        t_in, t_tgt = t_in.to(device), t_tgt.to(device)
                        t_out = model(t_in)
                        v_loss = criterion(
                            t_out.view(-1, t_out.size(-1)),
                            t_tgt.view(-1)
                        )
                        val_losses.append(v_loss.item())
                avg_val = sum(val_losses) / len(val_losses)
                config.metrics.val_losses.target_loss.append(avg_val)

                out_name = f"traincfg_alpha{config.alpha:.2f}_{timestamp}.json"
                config.to_json(out_name)

                tqdm.write(f"[Iter {global_iter}] Val loss: {avg_val:.4f}")
                model.train()

        ckpt_name = f"ckpt_alpha{config.alpha:.2f}_epoch{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, ckpt_name)
        print(f"🔖 Saved checkpoint: {ckpt_name}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"traincfg_alpha{config.alpha:.2f}_{timestamp}.json"
    config.to_json(out_name)
    print(f"✅ Training finished. Config with metrics saved to {out_name}")




def run_with_alphas(alphas, config: TrainConfig):
    train_loader = create_wikitext_dataloader(config.batch_size, config.seq_len, split="train")
    test_loader  = create_wikitext_dataloader(config.batch_size, config.seq_len, split="validation")
    
    for alpha in alphas:
        print(f"\n=== Training with alpha={alpha} ===\n")
        config = deepcopy(config)
        config.alpha = alpha
        model = TransformerWithMoE(
            vocab_size,
            config.d_model,
            config.num_layers,
            config.num_experts_per_device,
            config.world_size,
            config.top_k,
            padding_idx=tokenizer.pad_token_id
        )

        for i, layer in enumerate(model.layers):
            layer.moe.gate.register_forward_hook(get_scores)

        optimizer = optim.Adam(model.parameters(), lr=4e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        train(
            model,
            train_loader,
            test_loader,
            optimizer,
            criterion,
            config=config,
            device='cuda'
        )


config = TrainConfig()
alphas = [0.0, 0.2, 0.5, 0.8, 1.0]
run_with_alphas(alphas, config)
