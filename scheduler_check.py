import os
import sys
from pathlib import Path
from hydra import initialize, compose
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import torch
from schedulers import (
    ConstantScheduler, 
    CosineScheduler, 
    ExponentialScheduler, 
    LinearScheduler, 
    PeriodicLinearDecayScheduler
)

# Словарь всех доступных шедулеров
SCHEDULERS = {
    "ConstantScheduler": ConstantScheduler,
    "CosineScheduler": CosineScheduler,
    "ExponentialScheduler": ExponentialScheduler,
    "LinearScheduler": LinearScheduler,
    "PeriodicLinearDecayScheduler": PeriodicLinearDecayScheduler,
}

def get_scheduler_configs(experiments_dir):
    """Получает все конфигурации шедулеров из указанной папки experiments"""
    configs = []
    
    if not os.path.exists(experiments_dir):
        print(f"Папка {experiments_dir} не существует!")
        return configs
    
    # Ищем все .yaml файлы в папке
    for config_file in Path(experiments_dir).glob("*.yaml"):
        try:
            # Используем правильный config_path для поиска base
            with initialize(config_path="configs", version_base=None):
                cfg = compose(config_name=str(config_file).replace("configs/", "").replace(".yaml", ""))
            
            if hasattr(cfg, 'alpha_schedule') and hasattr(cfg.alpha_schedule, '_target_'):
                scheduler_name = cfg.alpha_schedule._target_.split(".")[-1]
                configs.append({
                    'file': config_file.name,
                    'scheduler': scheduler_name,
                    'config': cfg,
                    'alpha_config': cfg.alpha_schedule
                })
                print(f"✓ Загружен: {config_file.name} -> {scheduler_name}")
        except Exception as e:
            print(f"✗ Ошибка загрузки {config_file.name}: {e}")
    
    return configs

def plot_scheduler(scheduler_config, total_steps, ax):
    """Строит график для одного шедулера"""
    try:
        scheduler_name = scheduler_config['scheduler']
        alpha_config = scheduler_config['alpha_config']
        
        if scheduler_name not in SCHEDULERS:
            ax.text(0.5, 0.5, f"Неизвестный шедулер: {scheduler_name}", 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        SchedulerCls = SCHEDULERS[scheduler_name]
        
        # Создаем экземпляр шедулера
        if scheduler_name == "PeriodicLinearDecayScheduler":
            scheduler = SchedulerCls(
                total_steps=total_steps,
                use_warmup=alpha_config.use_warmup,
                warmup_steps=alpha_config.warmup_steps,
                initial_value=alpha_config.initial_value,
                final_value=alpha_config.final_value,
                period_steps=alpha_config.period_steps
            )
        else:
            scheduler = SchedulerCls(
                total_steps=total_steps,
                use_warmup=alpha_config.use_warmup,
                warmup_steps=alpha_config.warmup_steps,
                initial_value=alpha_config.initial_value,
                final_value=alpha_config.final_value,
            )
        
        # Генерируем значения
        values = []
        for _ in range(total_steps):
            values.append(scheduler.step())
        
        # Строим график
        ax.plot(values, linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Alpha")
        ax.set_title(f"{scheduler_name}\n{os.path.splitext(scheduler_config['file'])[0]}")
        ax.grid(True, alpha=0.3)
        
        # Добавляем информацию о параметрах
        params_text = f"warmup: {alpha_config.warmup_steps}\n"
        params_text += f"initial: {alpha_config.initial_value}\n"
        params_text += f"final: {alpha_config.final_value}"
        if scheduler_name == "PeriodicLinearDecayScheduler":
            params_text += f"\nperiod: {alpha_config.period_steps}"
        
        ax.text(0.02, 0.98, params_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
    except Exception as e:
        ax.text(0.5, 0.5, f"Ошибка: {str(e)}", 
               ha='center', va='center', transform=ax.transAxes, color='red')

def main():
    if len(sys.argv) != 2:
        print("Использование: python scheduler_check.py <папка_experiments>")
        print("Пример: python scheduler_check.py configs/experiments/wikitext2")
        sys.exit(1)
    
    experiments_dir = sys.argv[1]
    print(f"Загружаю конфигурации из: {experiments_dir}")
    
    # Получаем все конфигурации
    configs = get_scheduler_configs(experiments_dir)
    
    if not configs:
        print("Не найдено ни одной конфигурации!")
        sys.exit(1)
    
    print(f"\nНайдено {len(configs)} конфигураций шедулеров")
    
    # Определяем размер сетки для графиков
    n_configs = len(configs)
    cols = min(3, n_configs)  # Максимум 3 колонки
    rows = (n_configs + cols - 1) // cols
    
    # Создаем график
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_configs == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Скрываем лишние оси
    for i in range(n_configs, len(axes)):
        axes[i].set_visible(False)
    
    # Строим графики для каждого шедулера
    total_steps = 30000  # Фиксированное количество шагов для сравнения
    
    for i, config in enumerate(configs):
        plot_scheduler(config, total_steps, axes[i])
    
    plt.tight_layout()
    plt.suptitle(f"Сравнение шедулеров Alpha ({experiments_dir})", fontsize=16, y=1.02)
    
    # Сохраняем изображение
    output_filename = f"schedulers_comparison_{Path(experiments_dir).name}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nГрафик сохранен в файл: {output_filename}")
    
    # Показываем график (опционально)
    plt.show()

if __name__ == "__main__":
    main()
