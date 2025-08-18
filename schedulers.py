class AlphaScheduler:
    def __init__(self, total_steps, use_warmup, warmup_steps, initial_value, final_value):
        self.total_steps = total_steps
        self.use_warmup = use_warmup
        self.warmup_steps = warmup_steps if self.use_warmup else 0
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
        if self.use_warmup and self.step_num <= self.warmup_steps:
            return self.initial_value * self.step_num / self.warmup_steps
        return self.initial_value


class CosineScheduler(AlphaScheduler):
    def get_value(self):
        if self.use_warmup and self.step_num <= self.warmup_steps:
            return self.initial_value * self.step_num / self.warmup_steps
        progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.final_value + 0.5 * (self.initial_value - self.final_value) * (1 + torch.cos(torch.tensor(progress * 3.14159265)))


class ExponentialScheduler(AlphaScheduler):
    def get_value(self):
        if self.use_warmup and self.step_num <= self.warmup_steps:
            return self.initial_value * self.step_num / self.warmup_steps
        progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.initial_value * (self.final_value / self.initial_value) ** progress
