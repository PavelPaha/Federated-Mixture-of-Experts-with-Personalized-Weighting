import torch

class AlphaScheduler:
    def __init__(self, total_steps, use_warmup, warmup_steps, initial_value, final_value):
        self.total_steps = total_steps
        self.use_warmup = use_warmup
        # Если use_warmup=True, но warmup_steps не указан или None, отключаем warmup
        if self.use_warmup and (warmup_steps is None or warmup_steps <= 0):
            self.use_warmup = False
            self.warmup_steps = 0
        else:
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


class LinearScheduler(AlphaScheduler):
    def get_value(self):
        if self.use_warmup and self.step_num <= self.warmup_steps:
            return self.initial_value * self.step_num / self.warmup_steps
        progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.initial_value + (self.final_value - self.initial_value) * progress


class PeriodicLinearDecayScheduler(AlphaScheduler):
    def __init__(self, total_steps, use_warmup, warmup_steps, initial_value, final_value, period_steps):
        super().__init__(total_steps, use_warmup, warmup_steps, initial_value, final_value)
        self.period_steps = period_steps
    
    def get_value(self):
        if self.use_warmup and self.step_num <= self.warmup_steps:
            return self.initial_value * self.step_num / self.warmup_steps
        
        # Вычисляем текущий период
        steps_after_warmup = self.step_num - self.warmup_steps
        current_period = steps_after_warmup // self.period_steps
        steps_in_current_period = steps_after_warmup % self.period_steps
        
        # Внутри периода делаем линейный decay
        period_progress = steps_in_current_period / self.period_steps
        current_period_value = self.initial_value + (self.final_value - self.initial_value) * period_progress
        
        return current_period_value
