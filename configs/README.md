# Конфигурации для экспериментов

## Структура конфигов

### Базовый конфиг: `base.yaml`
Содержит все основные параметры модели, обучения и инфраструктуры.

### Конфиги alpha_schedule
- `constant_0.0.yaml` - α = 0.0 (только cross-entropy loss)
- `constant_0.2.yaml` - α = 0.2
- `constant_0.4.yaml` - α = 0.4
- `constant_0.6.yaml` - α = 0.6
- `constant_0.8.yaml` - α = 0.8
- `constant_1.0.yaml` - α = 1.0 (только balance loss)

### Конфиги экспериментов
Готовые конфиги для разных комбинаций датасетов и alpha значений.

## Использование

### Запуск с базовым конфигом
```bash
python train.py --config-name=base
```

### Запуск с готовым экспериментом
```bash
python train.py --config-name=experiments/wikitext2_alpha_0.4
```

### Запуск с переопределением параметров
```bash
python train.py --config-name=base dataset_name=ptb alpha_schedule=constant_0.8
```

### Создание нового эксперимента
Создайте новый файл в `configs/experiments/`:

```yaml
# configs/experiments/my_experiment.yaml
defaults:
  - base
  - override dataset_name: wikitext-2-v1
  - override alpha_schedule: constant_0.6

experiment:
  name: "my_experiment_name"
```

## Доступные датасеты
- `wikitext-2-v1` - WikiText-2 (малый)
- `wikitext-103-raw-v1` - WikiText-103 (большой)
- `ptb` - Penn Treebank
- `openwebtext` - OpenWebText
- `c4` - C4 (Common Crawl)
- `lm1b` - 1B Word Benchmark

## Принцип работы
1. `base.yaml` загружается как основа
2. `alpha_schedule` конфиг добавляет параметры шедулера
3. Экспериментальные конфиги переопределяют нужные параметры
4. Hydra автоматически объединяет все конфиги
