r"""
Base gate with standard interface
"""
import torch.nn as nn
from typing import Optional, Tuple


class BaseGate(nn.Module):
    def __init__(self, num_expert, world_size):
        super().__init__()
        self.world_size = world_size
        self.num_expert = num_expert
        self.tot_expert = world_size * num_expert
        self.loss = None
        # Last computed distributions (per forward call)
        self._last_prob_per_expert = None
        self._last_load = None

    def forward(self, x):
        raise NotImplementedError('Base gate cannot be directly used for fwd')

    def set_loss(self, loss):
        self.loss = loss

    def get_loss(self, clear=True):
        loss = self.loss
        if clear:
            self.loss = None
        return loss

    @property
    def has_loss(self):
        return self.loss is not None

    # --------- distributions (for logging) ---------
    def set_distributions(self, prob_per_expert, load):
        """
        Save per-expert probability (mean softmax) and load (assignment fraction).
        Tensors are expected to be 1-D of shape [tot_expert].
        """
        # Store detached cpu copies to avoid creating graph retention
        try:
            self._last_prob_per_expert = prob_per_expert.detach().to("cpu")
        except Exception:
            self._last_prob_per_expert = prob_per_expert
        try:
            self._last_load = load.detach().to("cpu")
        except Exception:
            self._last_load = load

    def get_distributions(self, clear: bool = True) -> Tuple[Optional[object], Optional[object]]:
        probs = self._last_prob_per_expert
        loads = self._last_load
        if clear:
            self._last_prob_per_expert = None
            self._last_load = None
        return probs, loads

    @property
    def has_distributions(self) -> bool:
        return self._last_prob_per_expert is not None and self._last_load is not None