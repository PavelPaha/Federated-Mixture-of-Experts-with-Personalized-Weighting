import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import io

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


def evaluate_model(model, dataloader, criterion, device):
    """Функция для валидации модели"""
    model.eval()
    total_loss = 0.0
    total_balance_loss = 0.0
    num_batches = 0

    max_steps = 400
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if i >= max_steps:
                break
            
            inp, tgt = batch
            inp, tgt = inp.to(device), tgt.to(device)
            
            logits = model(inp)
            ce_loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
            balance_loss = calc_balance_loss(model, device)
            
            total_loss += ce_loss.item()
            total_balance_loss += balance_loss.item()
            num_batches += 1
            
            # Очищаем память от промежуточных тензоров
            del inp, tgt, logits, ce_loss, balance_loss
    
    model.train()
    
    # Принудительная очистка памяти GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return total_loss / num_batches, total_balance_loss / num_batches


def log_gate_distributions(model, step, mlflow):
    """Логирует распределение использования экспертов из всех гейтов модели"""
    for layer_idx, layer in enumerate(model.layers):
        if hasattr(layer, 'moe') and hasattr(layer.moe, 'gate'):
            gate = layer.moe.gate
            if hasattr(gate, 'get_gate_distribution'):
                expert_usage, gate_weights = gate.get_gate_distribution(clear=True)
                if expert_usage is not None and gate_weights is not None:
                    # Создаем график распределения
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # График количества использований экспертов
                    expert_indices = range(len(expert_usage))
                    ax1.bar(expert_indices, expert_usage.numpy(), color='skyblue', alpha=0.7)
                    ax1.set_title(f'Layer {layer_idx}: Expert Usage Probability (Mean)')
                    ax1.set_xlabel('Expert Index')
                    ax1.set_ylabel('Mean Probability')
                    ax1.grid(True, alpha=0.3)
                    
                    # График средних весов экспертов
                    ax2.bar(expert_indices, gate_weights.numpy(), color='lightcoral', alpha=0.7)
                    ax2.set_title(f'Layer {layer_idx}: Expert Weight Standard Deviation')
                    ax2.set_xlabel('Expert Index')
                    ax2.set_ylabel('Standard Deviation')
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    
                    # Логируем график в MLflow
                    mlflow.log_figure(fig, f"gate_distributions/layer_{layer_idx}_step_{step}.png")                
                    plt.close(fig)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name=cfg.experiment.name):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        # Загружаем тренировочные данные
        raw_train_dataset = load_dataset("wikitext", cfg.training.dataset_name, split="train")
        train_dataloader = create_wikitext_dataloader(
            raw_train_dataset,
            batch_size=cfg.training.batch_size,
            seq_len=cfg.training.seq_len,
            num_workers=cfg.training.num_workers,
            shuffle=True,
            streaming=cfg.training.streaming,  # Используем параметр из конфигурации
        )
        
        # Загружаем валидационные данные
        raw_val_dataset = load_dataset("wikitext", cfg.training.dataset_name, split="validation")
        val_dataloader = create_wikitext_dataloader(
            raw_val_dataset,
            batch_size=cfg.validation.batch_size,
            seq_len=cfg.training.seq_len,
            num_workers=cfg.validation.num_workers,
            shuffle=False,  # Валидация без перемешивания
            streaming=False,  # Используем параметр из конфигурации
        )

        alpha_sched = instantiate(cfg.alpha_schedule, total_steps=cfg.training.total_steps)

        vocab_size = len(tokenizer)
        # Используем указанный в конфигурации GPU ID
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{cfg.training.gpu_id}')
            # Устанавливаем текущий GPU
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        
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

        # Переменные для накопления loss за log_interval шагов
        accumulated_loss = 0.0
        accumulated_ce_loss = 0.0
        accumulated_balance_loss = 0.0
        accumulated_steps = 0
        
        # Списки для хранения всех метрик за log_interval шагов
        batch_losses = []
        batch_ce_losses = []
        batch_balance_losses = []
        batch_steps = []

        model.train()
        while global_step < total_steps:
            for batch in train_dataloader:
                if global_step >= total_steps:
                    # print("global_step >= total_steps")
                    break
                    
                # print(f"Processing step {global_step}")
                inp, tgt = batch
                inp, tgt = inp.to(device), tgt.to(device)

                optimizer.zero_grad()
                logits = model(inp)
                ce_loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
                # print('aboba')
                
                balance_loss = calc_balance_loss(model, device)    
                loss = ce_loss + alpha_sched.get_value() * balance_loss

                loss.backward()
                optimizer.step()

                # Вычисляем mean_balance_loss на каждом шаге для tqdm
                mean_balance_loss = balance_loss / cfg.model.num_layers

                # Накопление loss для среднего (оставляем для tqdm)
                accumulated_loss += loss.item()
                accumulated_ce_loss += ce_loss.item()
                accumulated_balance_loss += mean_balance_loss.item()
                accumulated_steps += 1
                
                # Добавляем батч метрики в списки
                batch_losses.append(loss.item())
                batch_ce_losses.append(ce_loss.item())
                batch_balance_losses.append(mean_balance_loss.item())
                batch_steps.append(global_step)

                # print('asdfasdf')

                # Логгирование каждые log_interval шагов
                if global_step % cfg.training.log_interval == 0:
                    # Вычисляем средние значения за log_interval шагов (для tqdm)
                    avg_loss = accumulated_loss / accumulated_steps
                    avg_ce_loss = accumulated_ce_loss / accumulated_steps
                    avg_balance_loss = accumulated_balance_loss / accumulated_steps
                    avg_ppl = torch.exp(torch.tensor(avg_ce_loss)).item()
                    
                    # Логируем все батч метрики за период log_interval
                    for i, (step, loss_val, ce_loss_val, balance_loss_val) in enumerate(zip(batch_steps, batch_losses, batch_ce_losses, batch_balance_losses)):
                        mlflow.log_metric("train_loss", loss_val, step=step)
                        mlflow.log_metric("train_ce_loss", ce_loss_val, step=step)
                        mlflow.log_metric("train_mean_balance_loss", balance_loss_val, step=step)
                        mlflow.log_metric("train_perplexity", torch.exp(torch.tensor(ce_loss_val)).item(), step=step)
                    
                    # Логируем средние значения для сравнения
                    mlflow.log_metric("train_loss_avg", avg_loss, step=global_step)
                    mlflow.log_metric("train_ce_loss_avg", avg_ce_loss, step=global_step)
                    mlflow.log_metric("train_mean_balance_loss_avg", avg_balance_loss, step=global_step)
                    mlflow.log_metric("train_perplexity_avg", avg_ppl, step=global_step)
                    mlflow.log_metric("alpha_sched", alpha_sched.get_value(), step=global_step)
                    
                    # Сбрасываем накопленные значения
                    accumulated_loss = 0.0
                    accumulated_ce_loss = 0.0
                    accumulated_balance_loss = 0.0
                    accumulated_steps = 0
                    
                    # Очищаем списки батч метрик
                    batch_losses.clear()
                    batch_ce_losses.clear()
                    batch_balance_losses.clear()
                    batch_steps.clear()

                # Обновляем tqdm на каждом шаге
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", 
                    ce=f"{ce_loss.item():.4f}", 
                    mean_balance=f"{mean_balance_loss.item():.4f}"
                )

                # print('12323')
                # Валидация каждые 1000 шагов
                if global_step % cfg.training.eval_interval == 0:
                    val_ce_loss, val_balance_loss = evaluate_model(model, val_dataloader, criterion, device)
                    val_ppl = torch.exp(torch.tensor(val_ce_loss)).item()
                    mean_val_balance_loss = val_balance_loss / cfg.model.num_layers
                    
                    mlflow.log_metric("val_ce_loss", val_ce_loss, step=global_step)
                    mlflow.log_metric("val_mean_balance_loss", mean_val_balance_loss, step=global_step)
                    mlflow.log_metric("val_perplexity", val_ppl, step=global_step)
                    
                    # Дополнительная очистка памяти после валидации
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Синхронизируем GPU

                if global_step % 2500 == 0:
                    log_gate_distributions(model, global_step, mlflow)
                    
                    # Очистка памяти после создания графиков
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                global_step += 1
                alpha_sched.step()
                pbar.update(1)

        mlflow.pytorch.log_model(model, artifact_path="model")


if __name__ == "__main__":
    main()
