import json
from dataclasses import dataclass, field
import datetime

@dataclass
class Losses:
    target_loss: list[float] = field(default_factory=list)
    balance_loss: list[float] = field(default_factory=list)
    distribution_loss: list[float] = field(default_factory=list)
    fashion_loss: list[float] = field(default_factory=list)

@dataclass
class Metrics:
    train_losses: Losses = field(default_factory=Losses)
    val_losses: Losses = field(default_factory=Losses)