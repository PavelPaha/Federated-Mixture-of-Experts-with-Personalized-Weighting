import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import TrainConfig, Metrics
from copy import deepcopy
from multiprocessing import Process, Queue
import datetime
import time
from multiprocessing import Queue
from datasets import load_dataset
import math
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from model import TransformerWithMoE
from data import create_wikitext_dataloader, tokenizer

vocab_size = tokenizer.vocab_size + 1

import math
import torch
from typing import Optional

def make_cosine_after_warmup_scheduler(optimizer,
                                       warmup_steps: int = 1000,
                                       decay_steps: int = 9000,
                                       min_lr: float = 0.0):
    """
    Возвращает LambdaLR, который:
      - держит lr = base_lr первые warmup_steps итераций;
      - потом применяет косинусный спад от base_lr до min_lr в течение decay_steps;
      - если итерации > warmup_steps + decay_steps, lr = min_lr.
    Поддерживает несколько param_groups (каждому своя шкала min_ratio).
    """
    lr_lambdas = []
    for group in optimizer.param_groups:
        base_lr = float(group.get("lr", 0.0))
        # защита от деления на 0
        if base_lr <= 0.0:
            min_ratio = 0.0
        else:
            min_ratio = float(min_lr) / base_lr

        # закрытие значений, чтобы не захватить изменяемые переменные в цикле
        def _make_fn(warmup=warmup_steps, decay=decay_steps, min_r=min_ratio):
            def lr_multiplier(step: int):
                if step < 0:
                    step = 0
                if step < warmup:
                    return 1.0
                # прогресс в [0,1]
                prog = (step - warmup) / float(max(1, decay))
                if prog >= 1.0:
                    return min_r
                # косинусная интерполяция от 1.0 -> min_r
                return min_r + 0.5 * (1.0 - min_r) * (1.0 + math.cos(math.pi * prog))
            return lr_multiplier

        lr_lambdas.append(_make_fn())

    return LambdaLR(optimizer, lr_lambdas)


def get_alpha_scheduler(
    schedule_type: str,
    base_alpha: float,
    total_steps: int,
    warmup_steps: int = 0,
    period: Optional[int] = None,
    hold_steps: Optional[int] = None,
    warmup_strategy: str = "linear",
    rise_fraction: float = 0.5,
):
    aliases = {
        "exponential": "exp",
        "exp_decay": "exp",
        "linear_decay": "linear",
        "cosine_decay": "cosine",
        "cosine_increase": "cosine_increase",
        "exp_increase": "exp_increase",
        "cosine_rise_decay": "cosine_rise_decay",
        "saw": "sawtooth",
    }
    schedule_type = aliases.get(schedule_type, schedule_type)

    def scheduler(step: int):
        # локальные копии параметров, чтобы не переписывать внешние переменные
        period_local = period
        hold_local = hold_steps

        if schedule_type == 'cosine_increase':
            if step < warmup_steps:
                return 0.0
            adjusted_step = step - warmup_steps
            rise_steps = max(1, warmup_steps)
            if adjusted_step < rise_steps:
                return base_alpha * (float(adjusted_step) / float(rise_steps))
            decay_total = max(1, total_steps - warmup_steps - rise_steps)
            decay_step = min(max(0, adjusted_step - rise_steps), decay_total)
            decay_progress = float(decay_step) / float(decay_total)
            return base_alpha * 0.5 * (1 + math.cos(math.pi * decay_progress))

        if step < warmup_steps:
            if warmup_strategy == "zero":
                return 0.0
            return base_alpha * float(step) / float(max(1, warmup_steps))

        adjusted_step = step - warmup_steps
        adjusted_total = max(1, total_steps - warmup_steps)
        progress = float(adjusted_step) / float(adjusted_total)

        if schedule_type == 'constant':
            return base_alpha

        elif schedule_type == 'linear':
            return base_alpha * (1 - progress)

        elif schedule_type == 'cosine':
            return base_alpha * 0.5 * (1 + math.cos(math.pi * progress))

        elif schedule_type == 'exp':
            return base_alpha * (0.1 ** progress)

        elif schedule_type == 'exp_increase':
            return base_alpha * (1.0 - (0.1 ** progress))

        elif schedule_type == 'cosine_rise_decay':
            rf = max(1e-6, min(1.0 - 1e-6, rise_fraction))
            if progress < rf:
                local = progress / rf
                return base_alpha * 0.5 * (1 - math.cos(math.pi * local))
            else:
                local = (progress - rf) / (1.0 - rf)
                return base_alpha * 0.5 * (1 + math.cos(math.pi * local))

        elif schedule_type == 'sawtooth':
            if period_local is None:
                period_local = max(1, adjusted_total // 4)
            cycle_pos = adjusted_step % period_local
            return base_alpha * (1 - cycle_pos / float(period_local))

        elif schedule_type == 'cosine_hold':
            if hold_local is None:
                hold_local = max(1, adjusted_total // 3)
            if adjusted_step < hold_local:
                return base_alpha
            else:
                hold_progress = (adjusted_step - hold_local) / float(max(1, adjusted_total - hold_local))
                return base_alpha * 0.5 * (1 + math.cos(math.pi * hold_progress))

        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

    return scheduler


gate_scores = None
def get_scores(module, inp, out):
    # print(f'module = {module}, inp = {inp}, out = {out}')
    # global gate_scores
    pass
    # gate_scores = module.last_gate_logits

    

def collect_gate_distribution(model, global_iter, metrics):
    try:
        for m in model.modules():
            if hasattr(m, "gate") and hasattr(m.gate, "last_gate_logits"):
                with torch.no_grad():
                    probs = torch.softmax(m.gate.last_gate_logits.float(), dim=-1)
                    distr = probs.mean(dim=0).detach().cpu().tolist()
                metrics.gate_distr.append({'iter': global_iter, 'distr': distr})
                break
    except Exception:
        pass

def train_single_alpha(gpu_id, ds_train, ds_validation, config: TrainConfig, result_queue):
    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'

    warmup_steps = 1000          # старт по твоему запросу
    decay_steps = 12000           # сколько итераций занимает спад (настрой)
    min_lr = 0.0                 # минимальный LR в конце спада
    train_loader = create_wikitext_dataloader(ds_train, config.batch_size, config.seq_len)
    test_loader = create_wikitext_dataloader(ds_validation, config.batch_size, config.seq_len)

    model = TransformerWithMoE(
        tokenizer.vocab_size + 1,
        config.d_model,
        config.num_layers,
        config.num_experts_per_device,
        config.world_size,
        config.top_k,
        padding_idx=tokenizer.pad_token_id,
        gate_hook=get_scores
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    scheduler = make_cosine_after_warmup_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        min_lr=min_lr
    )

    tokens_total = 117920140.
    # tokens_total = 2391884
    len_loader = tokens_total / (config.batch_size * config.seq_len)
    total_steps = config.epochs * len_loader


    warmup_steps = getattr(config, 'warmup_steps', 0)

    alpha_sched = get_alpha_scheduler(
        config.schedule_type,
        config.alpha,
        total_steps,
        warmup_steps,
        period=getattr(config, 'period', None),
        hold_steps=getattr(config, 'hold_steps', None),
        warmup_strategy=getattr(config, 'warmup_strategy', 'linear'),
        rise_fraction=getattr(config, 'rise_fraction', 0.5),
    )

    config.metrics = Metrics()
    global_iter = 0

    ckpt_dir = f"metrics_gumbel/{config.schedule_type}_alpha{config.alpha:.2f}_gpu{gpu_id}"

    for epoch in range(1, config.epochs + 1):
        loop = tqdm(enumerate(train_loader, 1),
                   desc=f"Sched={config.schedule_type} GPU={gpu_id} Ep {epoch}/{config.epochs}",
                   position=gpu_id)
        model.train()

        for batch_idx, (inputs, targets) in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            current_alpha = alpha_sched(global_iter)

            outputs = model(inputs)
            target_loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            balance_loss = 0.0
            cnt = 0
            for m in model.modules():
                if hasattr(m, "gate") and getattr(m.gate, "has_loss", False):
                    balance_loss += m.gate.get_loss()
                    cnt += 1
            balance_loss = (balance_loss / cnt) if cnt > 0 else 0.0

            # gate_top_k_idx, gate_score, expert_distr = gate_scores
            # seq_len = expert_distr.shape[0]
            # expert_distr_by_device = expert_distr.reshape(seq_len, config.world_size, config.num_experts_per_device)

            # fashions_on_experts = expert_distr_by_device.max(dim=-1).values
            # norm2_expert_distr_by_device = torch.mean(expert_distr_by_device**2, dim=-1)
            # norm2_fashions_on_experts = torch.mean(fashions_on_experts**2, dim=-1)
            
            # loss_dist = torch.mean(norm2_expert_distr_by_device)
            # loss_fashion = torch.mean(norm2_fashions_on_experts)

            # total_loss = total_loss + config.lambda_2 * loss_dist - config.lambda_1 * loss_fashion

            total_loss = target_loss + current_alpha * balance_loss
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()

            scheduler.step() 
            config.metrics.train_losses.target_loss.append(target_loss.item())
            config.metrics.train_losses.balance_loss.append(
                balance_loss if isinstance(balance_loss, float) else balance_loss.item())
            config.metrics.hyperparams.append({'iter': global_iter, 'alpha': current_alpha})
            collect_gate_distribution(model, global_iter, config.metrics)
            config.metrics.schedule_hist.append(current_alpha)
            

            loop.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                't_loss': f"{target_loss.item():.4f}",
                'b_loss': f"{balance_loss.item():.4f}",
                'alpha': f"{current_alpha:.4f}" 
            })

            if global_iter and global_iter % config.log_interval == 0:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs(ckpt_dir, exist_ok=True)
                out_name = os.path.join(ckpt_dir, f"traincfg_alpha{config.alpha:.2f}_{timestamp}.json")
                config.to_json(out_name)


                model.eval()
                val_target_losses = []
                val_balance_losses = []
                with torch.no_grad():
                    for t_in, t_tgt in test_loader:
                        t_in, t_tgt = t_in.to(device), t_tgt.to(device)
                        v_out = model(t_in)
                        v_loss = criterion(v_out.view(-1, v_out.size(-1)), t_tgt.view(-1))

                        v_balance_loss = 0.0
                        v_cnt = 0
                        for m in model.modules():
                            if hasattr(m, "gate") and getattr(m.gate, "has_loss", False):
                                v_balance_loss += m.gate.get_loss(clear=False)
                                v_cnt += 1
                        v_balance_loss = (v_balance_loss / v_cnt) if v_cnt > 0 else 0.0

                        val_target_losses.append(v_loss.item())
                        val_balance_losses.append(
                            v_balance_loss if isinstance(v_balance_loss, float) else float(v_balance_loss.item())
                        )
                avg_val_target = sum(val_target_losses) / len(val_target_losses)
                avg_val_balance = sum(val_balance_losses) / len(val_balance_losses) if val_balance_losses else 0.0
                config.metrics.val_losses.target_loss.append(avg_val_target)
                config.metrics.val_losses.balance_loss.append(avg_val_balance)

                model.train()

            global_iter += 1

       
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(
            ckpt_dir,
            f"ckpt_{config.schedule_type}_alpha{config.alpha:.2f}_ep{epoch}.pt"
        )
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
        }, ckpt_path)
        print(f"🔖 Checkpoint saved: {ckpt_path}")

    result_queue.put({
        'schedule': config.schedule_type,
        'alpha_base': config.alpha,
        'gpu_id': gpu_id,
        'metrics': config.metrics.to_dict()
    })


from multiprocessing import Process, Queue
from datasets import load_dataset

def run_experiments_on_gpu(gpu_id, cfg_list, ds_train, ds_validation, result_queue):
    for cfg in cfg_list:
        cfg.gpu_id = gpu_id
        print(f"▶ started: schedule={cfg.schedule_type}, alpha={cfg.alpha} on GPU {gpu_id}")
        train_single_alpha(gpu_id, ds_train, ds_validation, cfg, result_queue)
        print(f"✔ finished: schedule={cfg.schedule_type}, alpha={cfg.alpha} on GPU {gpu_id}")


def run_parallel_training(configs, gpu_ids: list[int]):
    ds_train = load_dataset("wikitext", "wikitext-103-v1", split='train')
    ds_validation = load_dataset("wikitext", "wikitext-103-v1", split='validation')

    result_queue = Queue()

    # распределяем конфиги по GPU
    configs_per_gpu = {gpu: [] for gpu in gpu_ids}
    for i, cfg in enumerate(configs):
        gpu = gpu_ids[i % len(gpu_ids)]
        configs_per_gpu[gpu].append(cfg)

    procs = []
    for gpu_id, cfg_list in configs_per_gpu.items():
        if not cfg_list:
            continue
        p = Process(
            target=run_experiments_on_gpu,
            args=(gpu_id, cfg_list, ds_train, ds_validation, result_queue)
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())

    print("✅ Все эксперименты завершены")
    return results


if __name__ == "__main__":
    gpu_ids = [0]   
    # base_alphas = [round(x * 0.2, 2) for x in range(0, 6)]
    # base_alphas = [1, 5, 10, 25, 50, 100, 200]
    base_alphas = [0.1, 0.3, 0.5, 0.7]

    c = TrainConfig()
    c.alpha = 1
    c.schedule_type = 'exp'
    configs: list[TrainConfig] = [c]

    # for i, a in enumerate(base_alphas):
    #     c = TrainConfig()
    #     c.alpha = a
    #     c.schedule_type = 'constant'
    #     c.warmup_steps = 0
    #     c.gpu_id = gpu_ids[i % len(gpu_ids)]
    #     configs.append(c)

    # # 2) Для alpha в {0.2, 0.5, 0.7} — все расписания КРОМЕ 'constant'
    # test_alphas = [1]
    # all_schedules = ['constant']
    # lambdas1 = [0.1, 0.5, 0.7]
    # lambdas2 = [0.1, 0.5, 0.7]

    # for j, a in enumerate(test_alphas):
    #     for k, sch in enumerate(all_schedules):
    #         for l1 in lambdas1:
    #             for l2 in lambdas2:
    #                 c = TrainConfig()
    #                 c.alpha = a
    #                 c.schedule_type = sch
    #                 c.warmup_steps = 1000
    #                 c.lambda_1 = l1
    #                 c.lambda_2 = l2
    #                 if sch == 'cosine_hold':
    #                     c.hold_steps = 1000
    #                 if sch == 'sawtooth':
    #                     c.period = 1000
    #                 c.gpu_id = gpu_ids[(j * len(all_schedules) + k) % len(gpu_ids)]
    #                 configs.append(c)

    run_parallel_training(configs, gpu_ids)
