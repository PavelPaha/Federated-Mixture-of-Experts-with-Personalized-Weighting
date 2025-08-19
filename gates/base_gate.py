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

    def save_gate_output(self, gate_scores):
        """
        Сохраняет выход гейта для последующего логирования
        Args:
            gate_scores: оригинальные softmax вероятности для всех экспертов [batch_size, tot_expert]
        """
        # Сохраняем как CPU тензор для логирования
        self.last_gate_output = gate_scores.detach().cpu()

    def get_gate_distribution(self, clear=True):
        """
        Возвращает распределение использования экспертов
        Returns:
            expert_usage: средняя вероятность использования каждого эксперта
            gate_weights: стандартное отклонение весов для каждого эксперта
        """
        if self.last_gate_output is None:
            return None, None
        
        # Вычисляем среднюю вероятность для каждого эксперта по батчу
        expert_usage = self.last_gate_output.mean(dim=0)  # [tot_expert]
        
        # Вычисляем стандартное отклонение весов для каждого эксперта по батчу
        gate_weights = self.last_gate_output.std(dim=0)  # [tot_expert]
        
        if clear:
            self.last_gate_output = None
            
        return expert_usage, gate_weights

    @property
    def has_loss(self):
        return self.loss is not None