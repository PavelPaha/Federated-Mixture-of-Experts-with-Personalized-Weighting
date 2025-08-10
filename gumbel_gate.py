"""
Gumbel-Softmax Gate for Federated Mixture of Experts

A theoretically grounded gate implementation using Gumbel-Softmax for differentiable
discrete expert selection in federated learning environments.

Based on:
- Gumbel-Softmax: Jang et al. (2017) "Categorical Reparameterization with Gumbel-Softmax"
- Recent federated MoE research (2024-2025)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmoe.fmoe.gates.naive_gate import NaiveGate


class GumbelGate(NaiveGate):
    """
    Gumbel-Softmax Gate for Federated MoE
    
    Key improvements over standard gates:
    1. Uses Gumbel-Softmax for differentiable discrete sampling
    2. Temperature annealing for exploration-exploitation trade-off
    3. Proper load balancing with entropy regularization
    4. Federated-aware capacity management
    """
    
    def __init__(self, d_model, num_expert, world_size, topk=1,
                 temperature_init=1.0, temperature_min=0.1, temperature_decay=0.999,
                 capacity=(1.2, 2.4), load_balance_weight=0.01, entropy_weight=0.001,
                 gate_bias=True, use_noise_for_balance=True):
        
        assert topk == 1, 'Gumbel gate currently supports top-1 routing only'
        super().__init__(d_model, num_expert, world_size, top_k=1, gate_bias=gate_bias)
        
        # Gumbel-Softmax parameters
        self.temperature_init = temperature_init
        self.temperature_min = temperature_min  
        self.temperature_decay = temperature_decay
        self.current_temperature = temperature_init
        
        # Training step counter for temperature annealing
        self.register_buffer('training_step', torch.tensor(0))
        
        # Capacity and load balancing
        self.capacity = capacity
        self.load_balance_weight = load_balance_weight
        self.entropy_weight = entropy_weight
        self.use_noise_for_balance = use_noise_for_balance
        
        # Expert utilization tracking (for debugging/monitoring)
        self.register_buffer('expert_counts', torch.zeros(self.tot_expert))
        
    def sample_gumbel(self, shape, device, eps=1e-20):
        """Sample from Gumbel(0, 1) distribution"""
        u = torch.rand(shape, device=device)
        return -torch.log(-torch.log(u + eps) + eps)
    
    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        Gumbel-Softmax sampling
        
        Args:
            logits: [batch_size, num_experts] unnormalized log probabilities
            temperature: temperature parameter
            hard: if True, use straight-through estimator for discrete output
        
        Returns:
            [batch_size, num_experts] sampled probabilities
        """
        gumbel_noise = self.sample_gumbel(logits.shape, logits.device)
        y = logits + gumbel_noise
        
        # Apply softmax with temperature
        y_soft = F.softmax(y / temperature, dim=-1)
        
        if hard:
            # Straight-through estimator: discrete in forward, soft in backward
            y_hard = F.one_hot(y_soft.argmax(dim=-1), num_classes=logits.size(-1)).float()
            y = (y_hard - y_soft).detach() + y_soft
        else:
            y = y_soft
            
        return y
    
    def update_temperature(self):
        """Anneal temperature during training"""
        if self.training:
            self.training_step += 1
            self.current_temperature = max(
                self.temperature_min,
                self.temperature_init * (self.temperature_decay ** self.training_step)
            )
    
    def compute_load_balance_loss(self, gate_logits, expert_weights):
        """
        Compute load balancing loss using multiple strategies
        
        Args:
            gate_logits: [batch_size, num_experts] raw gate outputs
            expert_weights: [batch_size, num_experts] expert selection weights
        
        Returns:
            load_balance_loss: scalar tensor
        """
        # Strategy 1: Switch Transformer style load balancing
        expert_probs = F.softmax(gate_logits, dim=-1)  # [B, E]
        expert_usage = expert_weights.mean(dim=0)      # [E] - actual usage
        expert_importance = expert_probs.mean(dim=0)   # [E] - probability mass
        
        # Switch loss: minimize correlation between usage and importance
        switch_loss = (expert_usage * expert_importance).sum() * self.tot_expert
        
        # Strategy 2: Entropy regularization for exploration
        entropy_loss = -torch.sum(expert_probs * torch.log(expert_probs + 1e-8), dim=-1).mean()
        entropy_reg = -self.entropy_weight * entropy_loss  # Negative because we want high entropy
        
        # Strategy 3: CV² loss (coefficient of variation squared)
        cv_squared = expert_usage.var() / (expert_usage.mean() ** 2 + 1e-8)
        
        return self.load_balance_weight * (switch_loss + cv_squared) + entropy_reg
    
    def apply_capacity_constraints(self, expert_weights, capacity_factor=None):
        """
        Apply capacity constraints to prevent expert overload
        
        Args:
            expert_weights: [batch_size, num_experts] expert selection weights
            capacity_factor: optional override for capacity
        
        Returns:
            constrained_weights: [batch_size, num_experts] capacity-constrained weights
        """
        if capacity_factor is None:
            capacity_factor = self.capacity[0 if self.training else 1]
        
        batch_size = expert_weights.size(0)
        max_capacity = math.ceil(capacity_factor * batch_size / self.tot_expert)
        
        # For top-1 routing, we can use a simple constraint
        expert_loads = expert_weights.sum(dim=0)  # [num_experts]
        
        # Mask out overloaded experts
        overloaded = expert_loads > max_capacity
        if overloaded.any():
            # Zero out weights for overloaded experts and renormalize
            expert_weights = expert_weights.clone()
            expert_weights[:, overloaded] = 0
            
            # Renormalize (ensure each token goes somewhere)
            row_sums = expert_weights.sum(dim=-1, keepdim=True)
            expert_weights = expert_weights / (row_sums + 1e-8)
        
        return expert_weights
    
    def forward(self, inp, return_all_scores=False):
        """
        Forward pass with Gumbel-Softmax routing
        
        Args:
            inp: [batch_size, d_model] input features
            return_all_scores: if True, return gate logits
        
        Returns:
            expert_indices: [batch_size, 1] selected expert indices  
            expert_weights: [batch_size, 1] expert weights (or [batch_size, num_experts] if soft)
            gate_logits: [batch_size, num_experts] (if return_all_scores=True)
        """
        # Update temperature for annealing
        self.update_temperature()
        
        # Compute gate logits
        gate_logits = self.gate(inp)  # [batch_size, tot_expert]
        self.last_gate_logits = gate_logits.detach()
        
        # Add noise for load balancing during training (if enabled)
        if self.training and self.use_noise_for_balance:
            # Use uniform noise instead of Gaussian for better discrete distribution
            noise = torch.rand_like(gate_logits) * 0.1
            gate_logits = gate_logits + noise
        
        # Apply Gumbel-Softmax sampling
        if self.training:
            # Soft routing during training for differentiability
            expert_weights = self.gumbel_softmax(
                gate_logits, 
                self.current_temperature, 
                hard=False
            )
        else:
            # Hard routing during inference for efficiency
            expert_weights = self.gumbel_softmax(
                gate_logits,
                self.current_temperature,
                hard=True
            )
        
        # Apply capacity constraints
        expert_weights = self.apply_capacity_constraints(expert_weights)
        
        # For top-1 routing, extract the selected expert
        expert_indices = expert_weights.argmax(dim=-1, keepdim=True)  # [batch_size, 1]
        top1_weights = expert_weights.gather(1, expert_indices)       # [batch_size, 1]
        
        # Convert to input dtype
        top1_weights = top1_weights.to(dtype=inp.dtype)
        
        # Compute load balancing loss
        load_balance_loss = self.compute_load_balance_loss(gate_logits, expert_weights)
        self.set_loss(load_balance_loss)
        
        # Update expert usage statistics (for monitoring)
        if self.training:
            with torch.no_grad():
                expert_usage = F.one_hot(expert_indices.squeeze(-1), num_classes=self.tot_expert).float()
                self.expert_counts += expert_usage.sum(dim=0)
        
        if return_all_scores:
            return expert_indices, top1_weights, gate_logits
        return expert_indices, top1_weights
    
    def get_expert_utilization(self):
        """Get expert utilization statistics for monitoring"""
        if self.expert_counts.sum() > 0:
            utilization = self.expert_counts / self.expert_counts.sum()
            return {
                'utilization': utilization.cpu().numpy(),
                'entropy': -(utilization * torch.log(utilization + 1e-8)).sum().item(),
                'max_util': utilization.max().item(),
                'min_util': utilization.min().item(),
                'std_util': utilization.std().item()
            }
        return None
    
    def reset_statistics(self):
        """Reset expert utilization statistics"""
        self.expert_counts.zero_()
        self.training_step.zero_()
        self.current_temperature = self.temperature_init


class AdaptiveGumbelGate(GumbelGate):
    """
    Adaptive version that adjusts parameters based on federated learning dynamics
    """
    
    def __init__(self, d_model, num_expert, world_size, **kwargs):
        super().__init__(d_model, num_expert, world_size, **kwargs)
        
        # Adaptive parameters
        self.round_counter = 0
        self.performance_history = []
        
    def federated_round_update(self, round_num, performance_metrics=None):
        """
        Update gate parameters based on federated learning round
        
        Args:
            round_num: current federated round number
            performance_metrics: dict with 'accuracy', 'loss', etc.
        """
        self.round_counter = round_num
        
        if performance_metrics:
            self.performance_history.append(performance_metrics)
            
            # Adaptive temperature based on performance
            if len(self.performance_history) > 1:
                current_perf = performance_metrics.get('accuracy', 0)
                prev_perf = self.performance_history[-2].get('accuracy', 0)
                
                # If performance is improving, reduce temperature faster
                if current_perf > prev_perf:
                    self.temperature_decay *= 0.99  # Accelerate decay
                else:
                    self.temperature_decay = min(0.999, self.temperature_decay / 0.99)  # Slow decay
    
    def get_adaptive_capacity(self, client_data_size, total_clients):
        """
        Compute adaptive capacity based on federated learning context
        
        Args:
            client_data_size: number of samples on current client
            total_clients: total number of clients in federation
        
        Returns:
            adaptive_capacity: tuple of (train_capacity, eval_capacity)
        """
        # Smaller clients get higher capacity to ensure participation
        size_factor = max(0.8, min(2.0, 1000 / client_data_size))
        base_capacity = self.capacity[0]
        
        adaptive_train_capacity = base_capacity * size_factor
        adaptive_eval_capacity = self.capacity[1] * size_factor
        
        return (adaptive_train_capacity, adaptive_eval_capacity)


# Factory function for easy instantiation
def create_federated_gate(gate_type="gumbel", d_model=256, num_expert=10, world_size=2, **kwargs):
    """
    Factory function to create federated-optimized gates
    
    Args:
        gate_type: "gumbel", "adaptive_gumbel", or "switch" (fallback)
        **kwargs: additional parameters for gate initialization
    
    Returns:
        gate: instantiated gate object
    """
    if gate_type == "gumbel":
        return GumbelGate(d_model, num_expert, world_size, **kwargs)
    elif gate_type == "adaptive_gumbel":
        return AdaptiveGumbelGate(d_model, num_expert, world_size, **kwargs)
    elif gate_type == "switch":
        # Fallback to improved switch gate
        from my_gate import MyGate
        return MyGate(d_model, num_expert, world_size, **kwargs)
    else:
        raise ValueError(f"Unknown gate type: {gate_type}")


if __name__ == "__main__":
    # Example usage and testing
    d_model, num_expert, world_size = 256, 8, 2
    batch_size = 32
    
    # Create gate
    gate = GumbelGate(
        d_model=d_model,
        num_expert=num_expert, 
        world_size=world_size,
        temperature_init=2.0,
        temperature_min=0.5,
        load_balance_weight=0.01
    )
    
    # Test forward pass
    inp = torch.randn(batch_size, d_model)
    
    # Training mode
    gate.train()
    expert_idx, expert_weights, gate_logits = gate(inp, return_all_scores=True)
    
    print(f"Expert indices shape: {expert_idx.shape}")
    print(f"Expert weights shape: {expert_weights.shape}")
    print(f"Gate logits shape: {gate_logits.shape}")
    print(f"Current temperature: {gate.current_temperature:.4f}")
    print(f"Load balance loss: {gate.get_loss():.6f}")
    
    # Check expert utilization
    util_stats = gate.get_expert_utilization()
    if util_stats:
        print(f"Expert utilization entropy: {util_stats['entropy']:.4f}")
        print(f"Expert utilization std: {util_stats['std_util']:.4f}") 