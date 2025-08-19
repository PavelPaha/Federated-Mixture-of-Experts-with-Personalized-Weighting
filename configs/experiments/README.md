# Конфигурации экспериментов

## Структура

Все конфиги организованы по датасетам:

### wikitext2/
Датасет: `wikitext-2-v1`

- `alpha_0_0.yaml` - α = 0.0 (moe_wikitext2_alpha_0_0)
- `alpha_0_2.yaml` - α = 0.2 (moe_wikitext2_alpha_0_2)
- `alpha_0_4.yaml` - α = 0.4 (moe_wikitext2_alpha_0_4)
- `alpha_0_6.yaml` - α = 0.6 (moe_wikitext2_alpha_0_6)
- `alpha_0_8.yaml` - α = 0.8 (moe_wikitext2_alpha_0_8)
- `alpha_1_0.yaml` - α = 1.0 (moe_wikitext2_alpha_1_0)

### wikitext103/
Датасет: `wikitext-103-raw-v1`

- `alpha_0_0.yaml` - α = 0.0 (moe_wikitext103_alpha_0_0)
- `alpha_0_2.yaml` - α = 0.2 (moe_wikitext103_alpha_0_2)
- `alpha_0_4.yaml` - α = 0.4 (moe_wikitext103_alpha_0_4)
- `alpha_0_6.yaml` - α = 0.6 (moe_wikitext103_alpha_0_6)
- `alpha_0_8.yaml` - α = 0.8 (moe_wikitext103_alpha_0_8)
- `alpha_1_0.yaml` - α = 1.0 (moe_wikitext103_alpha_1_0)

### ptb/
Датасет: `ptb`

- `alpha_0_0.yaml` - α = 0.0 (moe_ptb_alpha_0_0)
- `alpha_0_2.yaml` - α = 0.2 (moe_ptb_alpha_0_2)
- `alpha_0_4.yaml` - α = 0.4 (moe_ptb_alpha_0_4)
- `alpha_0_6.yaml` - α = 0.6 (moe_ptb_alpha_0_6)
- `alpha_0_8.yaml` - α = 0.8 (moe_ptb_alpha_0_8)
- `alpha_1_0.yaml` - α = 1.0 (moe_ptb_alpha_1_0)

### openwebtext/
Датасет: `openwebtext`

- `alpha_0_0.yaml` - α = 0.0 (moe_openwebtext_alpha_0_0)
- `alpha_0_2.yaml` - α = 0.2 (moe_openwebtext_alpha_0_2)
- `alpha_0_4.yaml` - α = 0.4 (moe_openwebtext_alpha_0_4)
- `alpha_0_6.yaml` - α = 0.6 (moe_openwebtext_alpha_0_6)
- `alpha_0_8.yaml` - α = 0.8 (moe_openwebtext_alpha_0_8)
- `alpha_1_0.yaml` - α = 1.0 (moe_openwebtext_alpha_1_0)

### c4/
Датасет: `c4`

- `alpha_0_0.yaml` - α = 0.0 (moe_c4_alpha_0_0)
- `alpha_0_2.yaml` - α = 0.2 (moe_c4_alpha_0_2)
- `alpha_0_4.yaml` - α = 0.4 (moe_c4_alpha_0_4)
- `alpha_0_6.yaml` - α = 0.6 (moe_c4_alpha_0_6)
- `alpha_0_8.yaml` - α = 0.8 (moe_c4_alpha_0_8)
- `alpha_1_0.yaml` - α = 1.0 (moe_c4_alpha_1_0)

### lm1b/
Датасет: `lm1b`

- `alpha_0_0.yaml` - α = 0.0 (moe_lm1b_alpha_0_0)
- `alpha_0_2.yaml` - α = 0.2 (moe_lm1b_alpha_0_2)
- `alpha_0_4.yaml` - α = 0.4 (moe_lm1b_alpha_0_4)
- `alpha_0_6.yaml` - α = 0.6 (moe_lm1b_alpha_0_6)
- `alpha_0_8.yaml` - α = 0.8 (moe_lm1b_alpha_0_8)
- `alpha_1_0.yaml` - α = 1.0 (moe_lm1b_alpha_1_0)

## Использование

### Запуск конкретного эксперимента:
```bash
python train.py --config-name=experiments/wikitext2/alpha_0_4
python train.py --config-name=experiments/ptb/alpha_1_0
```

### Запуск с переопределением параметров:
```bash
python train.py --config-name=base training.dataset_name=wikitext-2-v1 alpha_schedule=constant_0.8
```

## Доступные датасеты
- `wikitext2/` - wikitext-2-v1
- `wikitext103/` - wikitext-103-raw-v1
- `ptb/` - ptb
- `openwebtext/` - openwebtext
- `c4/` - c4
- `lm1b/` - lm1b

## Alpha значения
- 0.0 - только cross-entropy loss
- 0.2 - легкий баланс
- 0.4 - умеренный баланс  
- 0.6 - сильный баланс
- 0.8 - очень сильный баланс
- 1.0 - только balance loss

## Всего конфигураций: 36
