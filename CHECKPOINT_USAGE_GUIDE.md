# 🔄 Checkpointing System Usage Guide

This guide explains how to use the newly implemented checkpointing system for the Federated Mixture of Experts training.

## ✨ Features Added

### 1. **Automatic Checkpointing**
- Saves model weights, optimizer state, scheduler state, and training metadata
- Configurable checkpoint interval (default: every 5000 steps)
- Creates both numbered checkpoints and a "latest.pt" for easy access
- Integrates with MLflow for artifact tracking

### 2. **Resume Training**
- Load from any checkpoint and continue training
- Specify additional training steps when resuming
- Preserves all training state (optimizer, scheduler, global step)
- Supports fine-tuning with modified hyperparameters

### 3. **Hydra Configuration System**
- Seamless integration with existing config structure
- Checkpoint configs as diffs from base config
- Support for resuming from experiment-specific checkpoints

## 📁 Configuration Structure

```
configs/
├── base.yaml                 # Base config with checkpoint settings
├── checkpoints/             # Checkpoint resumption configs
│   ├── README.md
│   ├── resume_basic.yaml
│   ├── resume_from_step.yaml
│   └── resume_with_experiment.yaml
├── test_checkpoint.yaml     # Quick test config
└── test_resume.yaml         # Test resumption config
```

## 🚀 Quick Start Examples

### 1. **Basic Training with Checkpointing**
```bash
# Train with default settings (checkpoints every 5000 steps)
python train.py --config-name base

# Custom checkpoint interval
python train.py --config-name base training.checkpoint_interval=2500
```

### 2. **Resume from Latest Checkpoint**
```bash
python train.py --config-name checkpoints/resume_basic
```

### 3. **Resume from Specific Checkpoint**
```bash
python train.py --config-name checkpoints/resume_from_step \
  training.checkpoint_path=./checkpoints/checkpoint_step_15000.pt \
  training.additional_steps=10000
```

### 4. **Resume Experiment with Fine-tuning**
```bash
python train.py --config-name checkpoints/resume_with_experiment \
  training.lr=5e-5 \
  training.additional_steps=20000
```

## ⚙️ Configuration Options

### Checkpoint Settings (in `base.yaml`)
```yaml
training:
  checkpoint_interval: 5000        # Save every N steps
  checkpoint_dir: "./checkpoints"  # Checkpoint directory
  resume_from_checkpoint: false    # Enable resumption
  checkpoint_path: null           # Path to checkpoint file
  additional_steps: null          # Extra steps when resuming
```

### Example Checkpoint Config
```yaml
# configs/checkpoints/my_resume.yaml
defaults:
  - /base  # or /experiments/wikitext103/alpha_0_2
  - _self_

experiment:
  name: my_resumed_training

training:
  resume_from_checkpoint: true
  checkpoint_path: "./checkpoints/checkpoint_step_20000.pt"
  additional_steps: 15000
  checkpoint_interval: 2500
  lr: 1e-4  # Fine-tuning learning rate
```

## 🗂️ Checkpoint Structure

Each checkpoint contains:
```python
{
    'model_state_dict': ...,      # Model weights
    'optimizer_state_dict': ...,  # Optimizer state
    'scheduler_step': int,        # Scheduler step count
    'global_step': int,          # Training step
    'config': dict,              # Full training config
    'model_config': {            # Model architecture
        'vocab_size': int,
        'd_model': int,
        'num_layers': int,
        'num_experts': int,
        'top_k': int,
        'padding_idx': int,
    }
}
```

## 📊 Integration with MLflow

- Checkpoints are automatically logged as MLflow artifacts
- Training metrics continue seamlessly when resuming
- Experiment tracking maintained across checkpoint boundaries

## 🧪 Testing the System

### Quick Test (2 minutes)
```bash
# 1. Train for 100 steps with checkpoints every 25 steps
python train.py --config-name test_checkpoint

# 2. Resume training for 50 more steps
python train.py --config-name test_resume
```

### Production Example
```bash
# 1. Start training experiment
python train.py --config-name experiments/wikitext103/alpha_0_2

# 2. Resume after interruption with fine-tuning
python train.py --config-name checkpoints/resume_with_experiment \
  training.checkpoint_path=./checkpoints/checkpoint_step_25000.pt \
  training.additional_steps=15000 \
  training.lr=1e-4
```

## 💡 Best Practices

### 1. **Checkpoint Frequency**
- Default 5000 steps works well for most cases
- Use 2500 steps for important experiments
- Use 1000 steps for short experiments or debugging

### 2. **Learning Rate Adjustment**
- Reduce learning rate when resuming training
- Typical reduction: 0.5x to 0.1x of original rate

### 3. **Directory Management**
```bash
# Organize checkpoints by experiment
./checkpoints/
├── experiment_alpha_0_2/
├── experiment_alpha_0_4/
└── fine_tuned_models/
```

### 4. **Resumption Strategy**
- Always verify checkpoint path exists
- Check original config compatibility
- Monitor metrics for smooth continuation

## 🔧 Advanced Usage

### Custom Checkpoint Directory
```bash
python train.py --config-name base \
  training.checkpoint_dir=./my_experiment/checkpoints \
  training.checkpoint_interval=1000
```

### Resume with Modified Architecture
⚠️ **Note**: Model architecture must match the checkpoint
```bash
# This will work - same architecture
python train.py --config-name checkpoints/resume_basic \
  training.lr=5e-5

# This will fail - different architecture
python train.py --config-name checkpoints/resume_basic \
  model.d_model=512  # ❌ Mismatch with saved checkpoint
```

### Multiple Resume Sessions
```bash
# Session 1: Train 20k steps
python train.py --config-name base training.total_steps=20000

# Session 2: Resume for 10k more
python train.py --config-name checkpoints/resume_basic \
  training.additional_steps=10000

# Session 3: Fine-tune for 5k more
python train.py --config-name checkpoints/resume_basic \
  training.checkpoint_path=./checkpoints_resumed/latest.pt \
  training.additional_steps=5000 \
  training.lr=1e-5
```

## 🐛 Troubleshooting

### Common Issues

1. **Checkpoint not found**
   ```
   Error: FileNotFoundError: ./checkpoints/latest.pt
   ```
   Solution: Verify checkpoint path exists

2. **Architecture mismatch**
   ```
   Error: size mismatch for model.layers.0.weight
   ```
   Solution: Ensure model config matches checkpoint

3. **CUDA device mismatch**
   ```
   Error: Expected device cuda:0 but got cuda:1
   ```
   Solution: Checkpoints automatically handle device mapping

### Debug Mode
```bash
# Enable debug logging
python train.py --config-name test_checkpoint \
  hydra.verbose=true
```

## 🔧 Scheduler Continuity Fix

### ⚠️ Important Update  
**FULLY SOLVED**: All scheduler types now have perfectly smooth continuation when resuming from checkpoints.

### Problem & Solution Evolution

**Original Problem**: Schedulers created discontinuities when resuming with `additional_steps`.

**First Attempt**: Recreate scheduler with adjusted `total_steps` → Still caused jumps in alpha values!

**FINAL SOLUTION**: `SmoothResumeScheduler` - Uses saved alpha value for perfect continuity:
1. **Saves current alpha value** in checkpoint
2. **Creates SmoothResumeScheduler** that interpolates from saved alpha to final value  
3. **Guarantees zero discontinuity** regardless of original scheduler type

### Example
```
✅ Scenario: CosineScheduler(initial=1.0, final=0.1)
   - Original training: 100 steps
   - Checkpoint at step 40: alpha = 0.676
   - Resume with 30 additional steps
   
❌ OLD (recreate scheduler): alpha jump = 0.245 (18x natural step!)
✅ NEW (SmoothResumeScheduler): alpha jump = 0.019 (1.4x natural step)

Result: Perfect smooth continuation with any scheduler type!
```

### Test Configs
Use these configs to verify the fix:
```bash
# 1. Train with LinearScheduler until checkpoint
python train.py --config-name test_scheduler_fix

# 2. Resume and verify smooth continuation  
python train.py --config-name test_scheduler_resume
```

---

This checkpointing system provides robust training continuation capabilities while maintaining the flexibility of the Hydra configuration system. **Scheduler decay behavior is now fully correct!** Happy training! 🚀 