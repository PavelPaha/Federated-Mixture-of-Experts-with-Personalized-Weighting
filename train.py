import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data import get_dataloaders, tokenizer
from model import TransformerWithMoE
from gates import BaseGate
from hydra.utils import instantiate
import matplotlib.pyplot as plt
import io


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

def evaluate(model, dataloader, criterion, device, alpha_sched=None, num_layers=None):
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            inp, tgt = batch
            inp, tgt = inp.to(device), tgt.to(device)

            logits = model(inp)
            ce_loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
            
            balance_loss = calc_balance_loss(model, device) if alpha_sched else 0.0
            loss = ce_loss + (alpha_sched.get_value() * balance_loss if alpha_sched else 0.0)

            total_loss += loss.item() * inp.size(0)
            total_ce_loss += ce_loss.item() * inp.size(0)
            total_tokens += inp.size(0)

    mean_loss = total_loss / total_tokens
    mean_ce_loss = total_ce_loss / total_tokens
    ppl = torch.exp(torch.tensor(mean_ce_loss)).item()
    mean_balance_loss = (balance_loss / num_layers).item() if num_layers else 0.0

    return mean_loss, mean_ce_loss, mean_balance_loss, ppl



@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg_: DictConfig):
    cfg = cfg_
    print(OmegaConf.to_yaml(cfg))
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name=cfg.experiment.name):
        # Log full resolved Hydra config as an artifact
        resolved_cfg_yaml = OmegaConf.to_yaml(cfg)
        mlflow.log_text(resolved_cfg_yaml, artifact_file="hydra_config.yaml")
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        dataloader, val_dataloader, _ = get_dataloaders(
            cfg.training.dataset_name,
            seq_len=cfg.training.seq_len, 
            batch_size=cfg.training.batch_size, 
            num_workers=cfg.training.num_workers
        )

        alpha_sched = instantiate(cfg.alpha_schedule, total_steps=cfg.training.total_steps)

        vocab_size = len(tokenizer)
        device = f'cuda:{cfg.training.gpu_id}'
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
                
                balance_loss = calc_balance_loss(model, device)    
                loss = ce_loss + alpha_sched.get_value() * balance_loss

                loss.backward()
                optimizer.step()

                if global_step % 50 == 0:
                    mean_balance_loss = balance_loss / cfg.model.num_layers
                    ppl = torch.exp(ce_loss).item()
                    mlflow.log_metric("loss", loss.item(), step=global_step)
                    mlflow.log_metric("ce_loss", ce_loss.item(), step=global_step)
                    mlflow.log_metric("mean_balance_loss", mean_balance_loss.item(), step=global_step)
                    mlflow.log_metric("perplexity", ppl, step=global_step)
                    mlflow.log_metric("alpha_sched", alpha_sched.get_value(), step=global_step)

                    with torch.no_grad():
                        dist_payload = {"step": int(global_step), "layers": {}}
                        for layer_idx, layer in enumerate(model.layers):
                            if hasattr(layer, "moe") and hasattr(layer.moe, "gate"):
                                gate = layer.moe.gate

                                # print(gate.has_distributions)
                                if getattr(gate, "has_distributions", False) and gate.has_distributions:
                                    probs, loads = gate.get_distributions(clear=True)
                                    # print(probs, loads)
                                    dist_payload["layers"][str(layer_idx)] = {
                                        "probs": [float(v) for v in probs.view(-1).tolist()],
                                        "loads": [float(v) for v in loads.view(-1).tolist()],
                                    }

                                    if global_step % 500 == 0:
                                        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

                                        probs_values = probs.numpy()
                                        bins_probs = range(len(probs_values))
                                        axes[0].bar(bins_probs, probs_values, color='tab:blue', alpha=0.8)
                                        axes[0].set_title(f"Layer {layer_idx} probs")
                                        axes[0].set_xlabel("bin")
                                        axes[0].set_ylabel("count")
                                        loads_values = loads.numpy()
                                        bins_loads = range(len(loads_values))
                                        axes[1].bar(bins_loads, loads_values, color='tab:orange', alpha=0.8)
                                        axes[1].set_title(f"Layer {layer_idx} loads")
                                        axes[1].set_xlabel("bin")
                                        axes[1].set_ylabel("count")

                                        plt.tight_layout()
                                        mlflow.log_figure(fig, f"gate_histograms/step_{global_step}_layer_{layer_idx}.png")

                        if global_step % cfg.training.log_interval == 0 and len(dist_payload["layers"]) > 0:
                            try:
                                mlflow.log_dict(dist_payload, artifact_file=f"gate_distributions/step_{global_step}.json")
                            except Exception:
                                pass

                if global_step % cfg.training.log_interval == 0:
                    val_loss, val_ce_loss, val_mean_balance_loss, val_ppl = evaluate(
                        model, val_dataloader, criterion, device, alpha_sched, cfg.model.num_layers
                    )
                    mlflow.log_metric("val_loss", val_loss, step=global_step)
                    mlflow.log_metric("val_ce_loss", val_ce_loss, step=global_step)
                    mlflow.log_metric("val_mean_balance_loss", val_mean_balance_loss, step=global_step)
                    mlflow.log_metric("val_perplexity", val_ppl, step=global_step)
                    model.train()
                    pbar.set_postfix(loss=loss.item(), ce=ce_loss.item(), mean_balance=mean_balance_loss.item(), ppl=ppl)

                global_step += 1
                alpha_sched.step()
                pbar.update(1)
                if global_step >= total_steps:
                    break

            mlflow.pytorch.log_model(model, artifact_path="model")


if __name__ == "__main__":
    main()
