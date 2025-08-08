from dataclasses import dataclass
from typing import Optional
import torch
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from metrics import Metrics

@dataclass
class TrainConfig:
    alpha: float = 0.2
    lambda_1: float = 0.0
    lambda_2: float = 0.0
    log_interval: int = 1000
    epochs: int = 15
    num_experts_per_device: int = 5
    world_size: int = 2
    d_model: int = 256
    num_layers: int = 3
    top_k: int = 1
    seq_len: int = 256
    batch_size: int = 16
    lr: float = 4e-4
    metrics: Metrics = field(default_factory=Metrics)
    schedule_type: str = 'constant'  # 'constant', 'linear', 'cosine', 'exponential', 'cosine_rise', 'exp_rise', 'sawtooth', 'cosine_hold', 'cosine_increase', 'exp_increase', 'cosine_rise_decay'
    warmup_steps: int = 1000
    warmup_strategy: str = 'linear'
    period: Optional[int] = None     # для 'sawtooth'
    hold_steps: Optional[int] = None # для 'cosine_hold'
    rise_fraction: float = 0.5       # доля фазы роста для комбинированных стратегий
    gpu_id: int | None = None
    scheduling_hist: list = field(default_factory=list)

    @classmethod
    def from_json(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path):
        d = self.__dict__.copy()
        if hasattr(self, "metrics") and hasattr(self.metrics, "to_dict"):
            d["metrics"] = self.metrics.to_dict()
        with open(path, "w") as f:
            json.dump(d, f, indent=4)

    @staticmethod
    def visualize_metrics(metrics_path="metrics.pkl"):
        import pickle
        with open(metrics_path, 'rb') as f:
            metrics = pickle.load(f)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(metrics['iter'], metrics['train_main'], label="Main Loss")
        plt.plot(metrics['iter'], metrics['train_total'], label="Total Loss")
        plt.plot(metrics['iter'], metrics['train_balance'], label="Balance Loss")
        plt.legend()
        plt.title("Train Losses")

        plt.subplot(1, 2, 2)
        plt.plot(range(len(metrics['test_loss'])), metrics['test_loss'], label="Test Loss", color="orange")
        plt.legend()
        plt.title("Test Loss")

        plt.tight_layout()
        plt.show()
