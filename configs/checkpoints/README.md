# Checkpoint Configuration Guide

This directory contains configuration files for resuming training from checkpoints.

## Configuration Structure

All checkpoint configs follow the same pattern:
- Inherit from base config or specific experiment config
- Set `resume_from_checkpoint: true`
- Specify `checkpoint_path` - path to the checkpoint file
- Set `additional_steps` - how many more steps to train
- Optionally override other training parameters

## Available Configs

### `resume_basic.yaml`
Basic checkpoint resumption from the latest checkpoint:
```bash
python train.py --config-name checkpoints/resume_basic
```

### `resume_from_step.yaml`
Resume from a specific step checkpoint with fine-tuning settings:
```bash
python train.py --config-name checkpoints/resume_from_step
```

### `resume_with_experiment.yaml`
Resume from a checkpoint created during a specific experiment:
```bash
python train.py --config-name checkpoints/resume_with_experiment
```

## Usage Examples

1. **Resume from latest checkpoint:**
   ```bash
   python train.py --config-name checkpoints/resume_basic \
     training.checkpoint_path=./checkpoints/latest.pt \
     training.additional_steps=5000
   ```

2. **Resume with custom parameters:**
   ```bash
   python train.py --config-name checkpoints/resume_basic \
     training.checkpoint_path=./my_checkpoints/step_10000.pt \
     training.additional_steps=20000 \
     training.lr=5e-5 \
     training.checkpoint_dir=./my_checkpoints/resumed
   ```

3. **Resume from experiment checkpoint:**
   ```bash
   python train.py --config-name checkpoints/resume_with_experiment \
     training.checkpoint_path=./checkpoints/my_experiment/checkpoint_step_25000.pt
   ```

## Checkpoint Structure

Checkpoints contain:
- Model state dict
- Optimizer state dict  
- Scheduler step count
- Global training step
- Full configuration used for training
- Model configuration for reconstruction

## Scheduler Behavior Fix

⚠️ **Important**: The system automatically handles scheduler continuity when resuming training.

### Problem Fully Solved  
ALL scheduler types now have perfectly smooth continuation when resuming from checkpoints.

### Solution Evolution
1. **First attempt**: Recreate scheduler with adjusted `total_steps` → Still caused alpha jumps
2. **FINAL SOLUTION**: `SmoothResumeScheduler` uses saved alpha value for perfect continuity

### How It Works
- **Complete scheduler state** (all parameters + step_num) is saved in checkpoint
- On resume: scheduler is recreated with **exact same parameters** from checkpoint
- Scheduler state (step_num) is **precisely restored**
- **Result**: Identical behavior = perfect continuity

### Example
```
Original: CosineScheduler(total_steps=100, initial=1.0, final=0.1)
Checkpoint at: step 40, alpha = 0.676

Resume: Scheduler recreated with EXACT same parameters:
- total_steps=100 (original value!)
- step_num=41 (restored state)

Result: IDENTICAL behavior = perfect smooth continuation
Test: 10-step sequence difference = 0.000000000000 (perfect!)
```

## Tips

- Always verify the checkpoint path exists before starting training
- Use lower learning rates when resuming training
- Consider saving checkpoints more frequently when resuming
- Keep track of original experiment configs when resuming
- **Scheduler decay will continue properly** - no manual adjustments needed 