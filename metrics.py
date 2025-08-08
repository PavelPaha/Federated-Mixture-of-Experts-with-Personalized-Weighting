import json
from dataclasses import dataclass, field, asdict
import datetime


@dataclass
class Losses:
    target_loss: list[float] = field(default_factory=list)
    balance_loss: list[float] = field(default_factory=list)
    distribution_loss: list[float] = field(default_factory=list)
    fashion_loss: list[float] = field(default_factory=list)
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class Metrics:
    train_losses: Losses = field(default_factory=Losses)
    val_losses: Losses = field(default_factory=Losses)
    hyperparams: list = field(default_factory=list)
    gate_distr: list = field(default_factory=list)
    schedule_hist: list = field(default_factory=list)
    
    def to_dict(self):
        return {
            'train_losses': self.train_losses.to_dict(),
            'val_losses': self.val_losses.to_dict(),
            'gate_distr': self.gate_distr,
            'schedule_hist': self.schedule_hist
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            train_losses=Losses.from_dict(data['train_losses']),
            val_losses=Losses.from_dict(data['val_losses']),
            gate_distr=data.get('gate_distr', []),
        )
    
    def to_json(self):
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls.from_dict(data)