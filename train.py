import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from datasets import load_dataset
from data import create_wikitext_dataloader, tokenizer
from model import TransformerWithMoE
from gates import BaseGate
from hydra.utils import instantiate


def calc_balance_loss(model, device):
    balance_loss = torch.tensor(0.0, device=device)

    for m in model.modules():
        if hasattr(m, "gate") and isinstance(getattr(m, "gate"), BaseGate) and m.gate.has_loss:
            gl = m.gate.get_loss(clear=True)
            if isinstance(gl, torch.Tensor):
                balance_loss = balance_loss + gl.to(device)
            else:
                raise Exception
    return balance_loss


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name=cfg.experiment.name):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        raw_dataset = load_dataset("wikitext", cfg.training.dataset_name, split="train")
        dataloader = create_wikitext_dataloader(
            raw_dataset,
            batch_size=cfg.training.batch_size,
            seq_len=cfg.training.seq_len,
            num_workers=cfg.training.num_workers,
        )

        alpha_sched = instantiate(cfg.alpha_schedule, total_steps=cfg.training.total_steps)

        vocab_size = len(tokenizer)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TransformerWithMoE(
            vocab_size=vocab_size,
            d_model=cfg.model.d_model,
            num_layers=cfg.model.num_layers,
            num_experts=cfg.model.num_experts,
            top_k=cfg.model.top_k,
            padding_idx=cfg.model.padding_idx,
        ).to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=cfg.model.padding_idx)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.training.lr)

        global_step = 0
        total_steps = cfg.training.total_steps

        pbar = tqdm(total=total_steps, desc="Training", unit="step")

        model.train()
        while global_step < total_steps:
            for batch in dataloader:
                inp, tgt = batch
                # quick sanity check (debugging): убедиться, что id < vocab_size
                if inp.max().item() >= vocab_size or tgt.max().item() >= vocab_size:
                    raise ValueError(f"Token id exceeds vocab size: max(inp)={inp.max().item()}, vocab_size={vocab_size}")

                inp, tgt = inp.to(device), tgt.to(device)

                optimizer.zero_grad()
                logits = model(inp)
                ce_loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
                
                balance_loss = calc_balance_loss(model, f'cuda:{cfg.training.gpu_id}')    
                loss = ce_loss + alpha_sched.get_value() * balance_loss

                loss.backward()
                optimizer.step()

                if global_step % cfg.training.log_interval == 0:
                    mean_balance_loss = balance_loss / cfg.model.num_layers
                    ppl = torch.exp(ce_loss).item()
                    mlflow.log_metric("loss", loss.item(), step=global_step)
                    mlflow.log_metric("ce_loss", ce_loss.item(), step=global_step)
                    mlflow.log_metric("mean_balance_loss", mean_balance_loss.item(), step=global_step)
                    mlflow.log_metric("perplexity", ppl, step=global_step)
                    mlflow.log_metric("alpha_sched", alpha_sched.get_value(), step=global_step)
                    pbar.set_postfix(loss=loss.item(), ce=ce_loss.item(), mean_balance=mean_balance_loss.item(), ppl=ppl)

                global_step += 1
                alpha_sched.step()
                pbar.update(1)
                if global_step >= total_steps:
                    break

            mlflow.pytorch.log_model(model, artifact_path="model")


if __name__ == "__main__":
    main()
