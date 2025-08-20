r"""
Base gate with standard interface
"""
import torch.nn as nn
import torch


class BaseGate(nn.Module):
    def __init__(self, num_expert, world_size):
        super().__init__()
        self.world_size = world_size
        self.num_expert = num_expert
        self.tot_expert = world_size * num_expert
        self.loss = None
        # Сохраняем последний выход гейта для логирования
        self.last_gate_output = None

    def forward(self, x):
        raise NotImplementedError('Base gate cannot be directly used for fwd')

    def set_loss(self, loss):
        self.loss = loss

    def get_loss(self, clear=True):
        loss = self.loss
        if clear:
            self.loss = None
        return loss

    def save_gate_output(self, expert_distribution):
        """
        Сохраняет только распределение по экспертам для последующего логирования
        Args:
            expert_distribution: тензор формы [tot_expert] — средние вероятности по экспертам
        """
        # Сохраняем как 1D CPU-тензор
        self.last_gate_output = expert_distribution.detach().cpu().view(-1)

    def get_gate_distribution(self, clear=True):
        """
        Возвращает сохранённое распределение использования экспертов [tot_expert]
        """
        if self.last_gate_output is None:
            return None, None

        expert_usage = self.last_gate_output

        if clear:
            self.last_gate_output = None

        return expert_usage, None

    @property
    def has_loss(self):
        return self.loss is not None