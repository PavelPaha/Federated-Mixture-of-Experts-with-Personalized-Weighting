#!/usr/bin/env python3
"""
Calculate total and active parameters for the MoE model based on config.
This script helps analyze model complexity and parameter efficiency.

Usage:
    python scripts/calculate_model_params.py                    # Use default config
    python scripts/calculate_model_params.py config.yaml       # Load from YAML file
    python scripts/calculate_model_params.py --config config.yaml
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from omegaconf import OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False

from model import TransformerWithMoE


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML not available. Install with: pip install pyyaml")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def load_hydra_config(config_path: str) -> Dict[str, Any]:
    """Load and resolve Hydra configuration with defaults."""
    if not HYDRA_AVAILABLE:
        raise ImportError("Hydra not available. Install with: pip install hydra-core")
    
    # Load the config file
    config = OmegaConf.load(config_path)
    
    # If it has defaults, we need to resolve them
    if 'defaults' in config:
        # For simplicity, we'll manually handle common cases
        # In a real scenario, you might want to use Hydra's compose API
        pass
    
    # Convert to regular dict
    return OmegaConf.to_container(config, resolve=True)


def count_parameters(model: nn.Module) -> int:
    """
    Count total parameters in the model.
    
    Returns:
        int: total_params
    """
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def count_active_forward_params(model: nn.Module, top_k: int = 2) -> int:
    """
    Count parameters that are actually used during forward pass.
    For MoE layers, only top_k experts are active per forward pass.
    
    Args:
        model: The MoE model
        top_k: Number of active experts per layer
        
    Returns:
        int: active_forward_params
    """
    active_params = 0
    
    # Token embeddings (always used)
    if hasattr(model, 'token_emb'):
        active_params += sum(p.numel() for p in model.token_emb.parameters())
    
    # Position embeddings (always used)
    if hasattr(model, 'pos_emb'):
        active_params += model.pos_emb.numel()
    
    # Output head (always used)
    if hasattr(model, 'head'):
        active_params += sum(p.numel() for p in model.head.parameters())
    
    # MoE layers
    for i, layer in enumerate(model.layers):
        # Self-attention (always used)
        if hasattr(layer, 'self_attn'):
            active_params += sum(p.numel() for p in layer.self_attn.parameters())
        
        # Layer norms (always used)
        if hasattr(layer, 'norm1'):
            active_params += sum(p.numel() for p in layer.norm1.parameters())
        if hasattr(layer, 'norm2'):
            active_params += sum(p.numel() for p in layer.norm2.parameters())
        
        # MoE components
        if hasattr(layer, 'moe'):
            moe = layer.moe
            # Gate parameters (always used)
            if hasattr(moe, 'gate'):
                active_params += sum(p.numel() for p in moe.gate.parameters())
            
            # Expert parameters (only top_k are active per forward pass)
            if hasattr(moe, 'experts') and len(moe.experts) > 0:
                # Get size of one expert (all experts are the same size)
                first_expert_params = sum(p.numel() for p in moe.experts[0].parameters())
                # Only top_k experts are active per layer
                active_expert_params = first_expert_params * top_k
                active_params += active_expert_params
    
    return active_params


def calculate_model_params_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate model parameters from a config dictionary.
    
    Args:
        config: Config dictionary with model parameters
        
    Returns:
        Dictionary with parameter analysis
    """
    # Extract model parameters from config
    vocab_size = 50257  # GPT-2 vocab size
    d_model = config.get('model', {}).get('d_model', 256)
    num_layers = config.get('model', {}).get('num_layers', 3)
    num_experts = config.get('model', {}).get('num_experts', 4)
    top_k = config.get('model', {}).get('top_k', 2)
    
    # Create model
    model = TransformerWithMoE(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        padding_idx=0
    )
    
    # Count parameters
    total_params = count_parameters(model)
    active_forward_params = count_active_forward_params(model, top_k)
    
    return {
        'total_params': total_params,
        'active_forward_params': active_forward_params,
        'top_k': top_k,
        'num_experts': num_experts,
        'num_layers': num_layers
    }


def print_analysis(efficiency: Dict[str, Any]):
    """Print the simplified parameter analysis."""
    print("🔍 MoE Model Parameter Analysis")
    print("=" * 50)
    
    print(f"📊 Model Configuration:")
    print(f"   d_model: N/A (from config)")
    print(f"   num_layers: {efficiency['num_layers']}")
    print(f"   num_experts: {efficiency['num_experts']}")
    print(f"   top_k: {efficiency['top_k']}")
    print()
    
    print(f"📈 Parameter Counts:")
    print(f"   Total parameters: {efficiency['total_params']:,}")
    print(f"   Active forward parameters: {efficiency['active_forward_params']:,}")
    print()
    
    # Calculate efficiency
    if efficiency['total_params'] > 0:
        efficiency_ratio = efficiency['total_params'] / efficiency['active_forward_params']
        print(f"⚡ Efficiency:")
        print(f"   Parameter efficiency: {efficiency_ratio:.1f}x")
        print(f"   (Total params / Active forward params)")
    
    # Human readable
    def human_readable(num):
        for unit in ['', 'K', 'M', 'B']:
            if num < 1000:
                return f"{num:.1f}{unit}"
            num /= 1000
        return f"{num:.1f}T"
    
    print()
    print(f"📏 Human Readable:")
    print(f"   Total: {human_readable(efficiency['total_params'])}")
    print(f"   Active forward: {human_readable(efficiency['active_forward_params'])}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Calculate MoE model parameters from config')
    parser.add_argument('config_file', nargs='?', help='Path to YAML config file')
    parser.add_argument('--config', help='Path to YAML config file (alternative)')
    parser.add_argument('--use-hydra', action='store_true', help='Use Hydra to resolve config (recommended for experiment configs)')
    
    args = parser.parse_args()
    
    # Determine config source
    config_file = args.config_file or args.config
    
    if config_file:
        try:
            if args.use_hydra or 'experiments' in config_file:
                # Use Hydra for experiment configs
                config = load_hydra_config(config_file)
                print(f"📁 Loaded Hydra config from: {config_file}")
            else:
                # Use regular YAML for simple configs
                config = load_config_from_yaml(config_file)
                print(f"📁 Loaded YAML config from: {config_file}")
        except Exception as e:
            print(f"❌ Error loading config from {config_file}: {e}")
            return 1
    else:
        # Use default config
        config = {
            'model': {
                'd_model': 256,
                'num_layers': 3,
                'num_experts': 4,
                'top_k': 2,
                'padding_idx': 0
            }
        }
        print("📝 Using default config")
    
    try:
        # Calculate parameters
        efficiency = calculate_model_params_from_config(config)
        
        # Print results
        print_analysis(efficiency)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error calculating parameters: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())