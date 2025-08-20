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
        
        return current_period_value * (0.9 ** current_period)

class SmoothResumeScheduler(AlphaScheduler):
    """
    Шедулер для гладкого продолжения с любого значения alpha.
    Используется при возобновлении с чекпоинта для обеспечения гладкости.
    """
    def __init__(self, total_steps, current_step, resume_alpha_value, final_value, 
                 use_warmup=False, warmup_steps=0):
        # Инициализируем базовый класс
        super().__init__(total_steps, use_warmup, warmup_steps, resume_alpha_value, final_value)
        
        self.resume_alpha_value = resume_alpha_value
        self.current_step = current_step
        self.step_num = current_step
        
        # Вычисляем оставшиеся шаги
        self.remaining_steps = max(1, total_steps - current_step)
        
        print(f"🔄 SmoothResumeScheduler: resume_alpha={resume_alpha_value:.6f}, "
              f"final={final_value:.6f}, remaining_steps={self.remaining_steps}")

    def get_value(self):
        # Если еще не начали обучение с чекпоинта
        if self.step_num <= self.current_step:
            return self.resume_alpha_value
        
        # Если достигли конца обучения    
        if self.step_num >= self.total_steps:
            return self.final_value
            
        # Линейная интерполяция от текущего значения к финальному
        steps_since_resume = self.step_num - self.current_step
        progress = steps_since_resume / self.remaining_steps
        
        return self.resume_alpha_value + (self.final_value - self.resume_alpha_value) * progress


class CosineAnnealingWarmupScheduler:
    """
    Learning rate scheduler with warmup and cosine annealing.
    Compatible with PyTorch's optimizer interface.
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.last_epoch = last_epoch
        self.base_lr = optimizer.param_groups[0]['lr']
        
        # Initialize step counter
        self.step_num = 0
        
    def step(self):
        """Update learning rate and step counter"""
        self.step_num += 1
        lr = self.get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr
    
    def get_lr(self):
        """Get current learning rate"""
        if self.step_num <= self.warmup_steps:
            # Linear warmup
            return self.base_lr * self.step_num / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, progress))  # Clamp to [0, 1]
            
            # Cosine decay from base_lr to min_lr
            cos_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)))
            return self.min_lr + (self.base_lr - self.min_lr) * cos_decay
    
    def state_dict(self):
        """Return state dict for checkpointing"""
        return {
            'step_num': self.step_num,
            'base_lr': self.base_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint"""
        self.step_num = state_dict['step_num']
        self.base_lr = state_dict['base_lr']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.min_lr = state_dict['min_lr']
