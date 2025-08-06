import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from torch.utils.data.dataset import IterableDataset
from tqdm import tqdm

from fmoe.transformer import FMoETransformerMLP
from fmoe.gates import GShardGate

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

class WikiText103LMIterable(IterableDataset):
    def __init__(self, split="train", seq_len=40):
        super().__init__()
        self.seq_len = seq_len
        self.raw_ds = load_dataset(
            "wikitext", "wikitext-103-raw-v1", split=split, streaming=True
        )

    def __iter__(self):
        buffer = []
        for example in self.raw_ds:
            text = example["text"]
            if not text or text.isspace():
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(ids)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len + 1 :]
                inp = torch.tensor(chunk[:-1], dtype=torch.long)
                tgt = torch.tensor(chunk[1:], dtype=torch.long)
                yield inp, tgt


def create_wikitext_dataloader(batch_size=6, seq_len=40, split="train", num_workers=4):
    ds = WikiText103LMIterable(split=split, seq_len=seq_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        prefetch_factor=2,
    )

class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_experts, top_k):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.moe = FMoETransformerMLP(
            num_expert=num_experts,
            d_model=d_model,
            d_hidden=d_model * 4,
            top_k=top_k,
            activation=nn.GELU(),
            expert_dp_comm="none",
            expert_rank=0,
            gate=GShardGate
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
    def __init__(self, vocab_size, d_model=256, num_layers=3, num_experts=10, top_k=2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model,
                                      padding_idx=tokenizer.pad_token_id)
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
        return self.head(x)

def train(
    model, 
    train_loader, 
    test_loader, 
    optimizer, 
    criterion, 
    load_balance_weight,
    log_interval=2000,
    num_epochs=2,
    device="cuda"
):
    model.to(device)
    model.train()


    metrics = {
        'iter': [],
        'train_main': [],
        'train_balance': [],
        'train_total': [],
        'test_loss': []
    }
    global_iter = 0

    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}")
        for batch_idx, (inputs, targets) in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            main_loss = criterion(
                outputs.view(-1, outputs.size(-1)),
                targets.view(-1)
            )
            balance_loss = 0.0
            cnt = 0
            for m in model.modules():
                if isinstance(m, FMoETransformerMLP) and hasattr(m.gate, 'get_loss') and m.gate.has_loss:
                    balance_loss += m.gate.get_loss()
                    
                    cnt += 1
            if cnt > 0:
                balance_loss = balance_loss / cnt
                loss = main_loss + load_balance_weight * balance_loss
            else:
                loss = main_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_iter += 1
            metrics['iter'].append(global_iter)
            metrics['train_main'].append(main_loss.item())
            metrics['train_balance'].append(balance_loss.item() if cnt>0 else 0.0)
            metrics['train_total'].append(loss.item())

            loop.set_postfix({
                'main': f"{main_loss.item():.4f}",
                'bal': f"{balance_loss.item():.6f}",
                'total': f"{loss.item():.4f}"
            })

            if global_iter % log_interval == 0:
                model.eval()
                test_losses = []
                with torch.no_grad():
                    for i, (t_in, t_tgt) in enumerate(test_loader):
                        t_in, t_tgt = t_in.to(device), t_tgt.to(device)
                        t_out = model(t_in)
                        t_loss = criterion(
                            t_out.view(-1, t_out.size(-1)),
                            t_tgt.view(-1)
                        )
                        test_losses.append(t_loss.item())
                avg_test = sum(test_losses) / len(test_losses)
                metrics['test_loss'].append(avg_test)
                print(f"\n[Iter {global_iter}] Test loss: {avg_test:.4f}")
                with open('metrics.pkl', 'wb') as f:
                    pickle.dump(metrics, f)
                model.train()

        ckpt_path = f'checkpoint_epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch+1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'metrics': metrics
        }, ckpt_path)
        print(f"🔖 Checkpoint saved: {ckpt_path}")

    with open('metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    print("Training complete. Metrics saved to metrics.pkl")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_experts_per_device, world_size = 5, 2
    total_experts = num_experts_per_device * world_size
    d_model, num_layers, top_k = 256, 3, 2
    seq_len, batch_size = 256, 32
    vocab_size = tokenizer.vocab_size + 1

    print(f"Config: {total_experts} experts, vocab_size={vocab_size}")

    train_loader = create_wikitext_dataloader(batch_size, seq_len, split="train")
    test_loader  = create_wikitext_dataloader(batch_size, seq_len, split="validation")

    model = TransformerWithMoE(vocab_size, d_model, num_layers, total_experts, top_k)
    optimizer = optim.Adam(model.parameters(), lr=4e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train(
        model,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        load_balance_weight=0.015,
        log_interval=5000,
        num_epochs=5,
        device=device
    )

if __name__ == "__main__":
    main()
