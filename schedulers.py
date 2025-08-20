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
        
        # Вычисляем прогресс после warmup (от 0 до 1)
        steps_after_warmup = self.step_num - self.warmup_steps
        total_steps_after_warmup = self.total_steps - self.warmup_steps
        progress = steps_after_warmup / max(1, total_steps_after_warmup)
        
        # Косинусный decay: плавное изменение по косинусу
        # progress = 0 -> alpha = initial_value, progress = 1 -> alpha = final_value
        return self.final_value + (self.initial_value - self.final_value) * (1 + torch.cos(torch.tensor(progress * 3.14159265))) / 2


class ExponentialScheduler(AlphaScheduler):
    def get_value(self):
        if self.use_warmup and self.step_num <= self.warmup_steps:
            return self.initial_value * self.step_num / self.warmup_steps
        
        # Вычисляем прогресс после warmup (от 0 до 1)
        steps_after_warmup = self.step_num - self.warmup_steps
        total_steps_after_warmup = self.total_steps - self.warmup_steps
        progress = steps_after_warmup / max(1, total_steps_after_warmup)
        
        # Экспоненциальный decay с защитой от final_value = 0
        if self.final_value == 0:
            # Если final_value = 0, используем формулу: initial * exp(-k * progress)
            # где k подбирается так, чтобы alpha был близок к 0 в конце
            k = 5.0  # коэффициент затухания (можно настроить)
            return self.initial_value * torch.exp(torch.tensor(-k * progress))
        else:
            # Если final_value != 0, используем стандартную формулу
            return self.initial_value * (self.final_value / self.initial_value) ** progress


class LinearScheduler(AlphaScheduler):
    def get_value(self):
        if self.use_warmup and self.step_num <= self.warmup_steps:
            return self.initial_value * self.step_num / self.warmup_steps
        
        # Вычисляем прогресс после warmup (от 0 до 1)
        steps_after_warmup = self.step_num - self.warmup_steps
        total_steps_after_warmup = self.total_steps - self.warmup_steps
        progress = steps_after_warmup / max(1, total_steps_after_warmup)
        
        # Линейный decay: плавное линейное изменение
        # progress = 0 -> alpha = initial_value, progress = 1 -> alpha = final_value
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
