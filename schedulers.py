import torch

class AlphaScheduler:
    def __init__(self, total_steps, use_warmup, warmup_steps=None, initial_value=0.01, final_value=0.01):
        self.total_steps = total_steps
        self.use_warmup = use_warmup
        # Only use warmup if both use_warmup is True and warmup_steps is provided and > 0
        self.warmup_steps = warmup_steps if (self.use_warmup and warmup_steps is not None and warmup_steps > 0) else 0
        self.initial_value = initial_value
        self.final_value = final_value
        self.step_num = 0


    def step(self):
        self.step_num += 1
        alpha = self.get_value()
        return alpha

    def get_value(self):
        raise NotImplementedError


class ConstantScheduler(AlphaScheduler):
    def get_value(self):
        if self.use_warmup and self.warmup_steps > 0 and self.step_num <= self.warmup_steps:
            return self.initial_value * self.step_num / self.warmup_steps
        return self.initial_value


class CosineScheduler(AlphaScheduler):
    def get_value(self):
        if self.use_warmup and self.warmup_steps > 0 and self.step_num <= self.warmup_steps:
            return self.initial_value * self.step_num / self.warmup_steps
        progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.final_value + 0.5 * (self.initial_value - self.final_value) * (1 + torch.cos(torch.tensor(progress * 3.14159265)))


class ExponentialScheduler(AlphaScheduler):
    def get_value(self):
        if self.use_warmup and self.warmup_steps > 0 and self.step_num <= self.warmup_steps:
            return self.initial_value * self.step_num / self.warmup_steps
        progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.initial_value * (self.final_value / self.initial_value) ** progress


class LinearScheduler(AlphaScheduler):
    def get_value(self):
        if self.use_warmup and self.warmup_steps > 0 and self.step_num <= self.warmup_steps:
            return self.initial_value * self.step_num / self.warmup_steps
        progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.initial_value + (self.final_value - self.initial_value) * progress


class PeriodicLinearDecayScheduler(AlphaScheduler):
    def __init__(self, total_steps, use_warmup, warmup_steps=None, initial_value=1.0, final_value=0.1, period_steps=1000):
        super().__init__(
            total_steps=total_steps,
            use_warmup=use_warmup,
            warmup_steps=warmup_steps,
            initial_value=initial_value,
            final_value=final_value,
        )
        self.period_steps = max(1, int(period_steps))

    def get_value(self):
        if self.use_warmup and self.warmup_steps > 0 and self.step_num <= self.warmup_steps:
            return self.initial_value * self.step_num / self.warmup_steps
        # After warmup, linearly decay from initial_value to final_value over each period, then jump back to initial_value
        steps_after_warmup = max(0, self.step_num - self.warmup_steps)
        phase_step = steps_after_warmup % self.period_steps
        phase_progress = phase_step / self.period_steps
        return self.initial_value + (self.final_value - self.initial_value) * phase_progress
