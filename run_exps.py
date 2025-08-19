#!/usr/bin/env python3
"""
Улучшенная версия скрипта для запуска экспериментов на 3 GPU
с выводом всех процессов в реальном времени
"""

import os
import subprocess
import time
from pathlib import Path
import logging
import threading
import queue

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Конфигурация GPU
AVAILABLE_GPUS = [5, 6, 7]
EXCLUDED_DATASETS = ["wikitext2"]

def get_all_experiments():
    """Получает список всех экспериментов, исключая wikitext103"""
    experiments_dir = Path("configs/experiments")
    experiment_paths = []
    
    for dataset_dir in experiments_dir.iterdir():
        if dataset_dir.is_dir() and dataset_dir.name in EXCLUDED_DATASETS:
            for config_file in dataset_dir.glob("*.yaml"):
                experiment_paths.append(str(config_file))
    
    return sorted(experiment_paths)

def run_experiment_on_gpu(experiment_path: str, gpu_id: int):
    """Запускает эксперимент на указанной GPU с выводом в реальном времени"""
    # Получаем имя конфига для Hydra
    config_name = experiment_path.replace("configs/", "").replace(".yaml", "")
    
    cmd = [
        "python", "train.py", 
        "--config-name", config_name,
        f"training.gpu_id={gpu_id}"
    ]
    
    logging.info(f"🚀 Запускаю эксперимент {config_name} на GPU {gpu_id}")
    logging.info(f"Команда: {' '.join(cmd)}")
    
    try:
        # Запускаем процесс БЕЗ перенаправления вывода - он будет показываться в консоли
        process = subprocess.Popen(
            cmd,
            # stdout=None,  # Вывод в консоль
            # stderr=None,  # Ошибки в консоль
            # text=True
        )
        return process
    except Exception as e:
        logging.error(f"❌ Ошибка запуска эксперимента {config_name}: {e}")
        return None

def monitor_process_output(process, gpu_id, experiment_name):
    """Мониторит вывод процесса (если нужно)"""
    # В этой версии вывод уже идет в консоль, поэтому просто ждем завершения
    process.wait()
    return process.returncode

def main():
    """Основная функция"""
    print("=" * 60)
    print("🚀 ЗАПУСК СИСТЕМЫ УПРАВЛЕНИЯ ЭКСПЕРИМЕНТАМИ")
    print("=" * 60)
    logging.info(f"Доступные GPU: {AVAILABLE_GPUS}")
    
    # Получаем список всех экспериментов
    experiment_paths = get_all_experiments()
    
    if not experiment_paths:
        logging.error("❌ Не найдено ни одного эксперимента!")
        return
    
    logging.info(f"📊 Найдено экспериментов: {len(experiment_paths)}")
    
    # Распределяем эксперименты по GPU равномерно
    gpu_experiments = {gpu_id: [] for gpu_id in AVAILABLE_GPUS}
    
    for i, exp_path in enumerate(experiment_paths):
        gpu_id = AVAILABLE_GPUS[i % len(AVAILABLE_GPUS)]
        gpu_experiments[gpu_id].append(exp_path)
    
    # Выводим план распределения
    print("\n📋 ПЛАН РАСПРЕДЕЛЕНИЯ ЭКСПЕРИМЕНТОВ:")
    print("-" * 50)
    for gpu_id, exps in gpu_experiments.items():
        print(f"🎯 GPU {gpu_id}: {len(exps)} экспериментов")
        for exp in exps:
            exp_name = exp.replace("configs/experiments/", "").replace(".yaml", "")
            print(f"   • {exp_name}")
    
    print(f"\n💡 Всего экспериментов: {len(experiment_paths)}")
    print(f"💡 GPU будет загружено: {len([gpu for gpu, exps in gpu_experiments.items() if exps])}")
    
    # Спрашиваем подтверждение
    # response = input("\n❓ Продолжить запуск? (y/n): ").lower().strip()
    # if response not in ['y', 'yes', 'да', 'д']:
    #     print("❌ Запуск отменен")
    #     return
    
    print("\n🚀 ЗАПУСКАЮ ЭКСПЕРИМЕНТЫ...")
    print("=" * 60)
    
    # Запускаем эксперименты на каждой GPU
    processes = {}
    
    for gpu_id, experiments in gpu_experiments.items():
        if experiments:
            # Запускаем первый эксперимент на этой GPU
            first_exp = experiments[0]
            process = run_experiment_on_gpu(first_exp, gpu_id)
            if process:
                processes[gpu_id] = {
                    'process': process,
                    'experiment': first_exp,
                    'remaining': experiments[1:],
                    'completed': []
                }
                print(f"✅ GPU {gpu_id}: запущен эксперимент {first_exp.replace('configs/experiments/', '').replace('.yaml', '')}")
    
    print(f"\n🎯 Запущено экспериментов: {len(processes)}")
    print("📺 Теперь вы увидите вывод всех процессов в реальном времени!")
    print("⏳ Ожидайте завершения экспериментов...")
    print("=" * 60)
    
    # Мониторим завершение и запускаем следующие эксперименты
    while any(processes.values()):
        for gpu_id in list(processes.keys()):
            if gpu_id in processes:
                gpu_data = processes[gpu_id]
                process = gpu_data['process']
                
                # Проверяем, завершился ли процесс
                if process.poll() is not None:
                    return_code = process.returncode
                    exp_name = gpu_data['experiment'].replace("configs/experiments/", "").replace(".yaml", "")
                    
                    if return_code == 0:
                        print(f"✅ GPU {gpu_id}: эксперимент {exp_name} завершился успешно")
                        gpu_data['completed'].append(gpu_data['experiment'])
                    else:
                        print(f"❌ GPU {gpu_id}: эксперимент {exp_name} завершился с ошибкой (код: {return_code})")
                    
                    # Проверяем, есть ли еще эксперименты для этой GPU
                    if gpu_data['remaining']:
                        next_exp = gpu_data['remaining'].pop(0)
                        next_exp_name = next_exp.replace("configs/experiments/", "").replace(".yaml", "")
                        print(f"🔄 GPU {gpu_id}: запускаю следующий эксперимент {next_exp_name}")
                        
                        new_process = run_experiment_on_gpu(next_exp, gpu_id)
                        if new_process:
                            gpu_data['process'] = new_process
                            gpu_data['experiment'] = next_exp
                        else:
                            # Если не удалось запустить, убираем GPU из списка
                            print(f"❌ GPU {gpu_id}: не удалось запустить следующий эксперимент")
                            del processes[gpu_id]
                    else:
                        # Все эксперименты на этой GPU завершены
                        print(f"🎉 GPU {gpu_id}: все эксперименты завершены!")
                        del processes[gpu_id]
        
        time.sleep(1)  # Проверяем каждые 30 секунд
        
        # Выводим текущий статус
        active_count = len(processes)
        if active_count > 0:
            print(f"📊 Активных GPU: {active_count}")
    
    print("\n" + "=" * 60)
    print("🎉 ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print("=" * 60)

if __name__ == "__main__":
    main()
