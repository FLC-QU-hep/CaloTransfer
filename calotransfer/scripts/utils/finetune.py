import torch
import torch.nn as nn
import torch.nn.init as init
import math
from . import gen_utils_CaloChallenge as gen_utils
from models.epicVAE_nflows_kDiffusion import epicVAE_nFlow_kDiffusion
import numpy as np
import matplotlib.pyplot as plt

class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=8, alpha=1.0):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha

        # Free parameters of original linear layer
        for param in self.original_linear.parameters():
            param.requires_grad = False

        # Get dimensions and device
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.device = original_linear.weight.device
        
        # Initialize A with Kaiming normal
        self.A = nn.Parameter(torch.empty(rank, in_features, device=self.device))
        init.kaiming_uniform_(self.A, a=math.sqrt(5))
        
        # This ensures LoRA starts with zero contribution
        self.B = nn.Parameter(torch.zeros(out_features, rank, device=self.device))

    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device)
            
        # Original output
        output_original = self.original_linear(x)
        
        # LoRA contribution
        lora_output = (x @ self.A.T) @ self.B.T
        lora_output *= (self.alpha / self.rank)

        return output_original + lora_output
    
def print_model_analysis(model):
    for name, param in model.named_parameters():
        print(f'{name:<60} | Trainable: {param.requires_grad} | Device: {param.device} | Shape: {param.shape}')
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal Parameters: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M')

# Function to apply LoRA to a layer
def apply_lora(layer, rank, alpha=1.0):
    # If the layer is a ConcatSquashLinear instance, extract the base linear layer
    if hasattr(layer, '_layer'):
        if isinstance(layer._layer, LoRALinear):
            # print(f"Layer already has LoRA applied. Skipping reapplication.")
            return layer  # Skip reapplying LoRA
        elif isinstance(layer._layer, nn.Linear):
            original_layer = layer._layer
        else:
            raise ValueError(f"Layer {layer} has an unsupported _layer type: {type(layer._layer)}")
    # If the layer is a standard nn.Linear instance, use it directly
    elif isinstance(layer, nn.Linear):
        original_layer = layer
    else:
        raise ValueError(f"Layer {layer} is not a ConcatSquashLinear instance or a nn.Linear layer.")

    # Apply LoRA to the base linear layer
    lora_layer = LoRALinear(original_layer, rank=rank, alpha=alpha)

    # If the input was a ConcatSquashLinear, replace its _layer attribute
    if hasattr(layer, '_layer'):
        layer._layer = lora_layer
        return layer
    # If the input was a nn.Linear, return the LoRALinear layer directly
    else:
        return lora_layer
    
# Function to update EMA for LoRA parameters
def update_ema_lora(model, model_ema, ema_decay):
    """
    Update EMA for models with LoRA parameters.
    """
    with torch.no_grad():
        # Create a mapping of parameter names for safety
        ema_params = {name: param for name, param in model_ema.named_parameters()}
        
        for name, param in model.named_parameters():
            if name in ema_params:
                if param.requires_grad:  # This includes LoRA params (A, B) and any other trainable params
                    # Apply EMA update
                    ema_params[name].mul_(ema_decay).add_(param, alpha=1 - ema_decay)
                # Note: We don't need to copy non-trainable params as they don't change
            else:
                print(f"Warning: Parameter {name} not found in EMA model")

def apply_lora_to_model(cfg, model, model_ema, checkpoint):
    """Apply LoRA to specified layers in the model"""
    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    for param in model_ema.parameters():
        param.requires_grad = False
    
    # Apply LoRA to specified layers
    for i, layer in enumerate(model.diffusion.inner_model.layers):
        if i in cfg.train_pwise_layers:
            print(f"Applying LoRA to layer {i}")

            # Apply LoRA with specified rank
            model.diffusion.inner_model.layers[i] = apply_lora(layer, rank=cfg.lora_rank, alpha=cfg.lora_alpha)
            
            # Also apply to EMA model
            ema_layer = model_ema.diffusion.inner_model.layers[i]
            model_ema.diffusion.inner_model.layers[i] = apply_lora(ema_layer, rank=cfg.lora_rank, alpha=cfg.lora_alpha)
            
            # Load LoRA weights from checkpoint if available
            lora_prefix = f"diffusion.inner_model.layers.{i}._layer."
            if lora_prefix + "A" in checkpoint['state_dict']:
                model.diffusion.inner_model.layers[i]._layer.A.data.copy_(
                    checkpoint['state_dict'][lora_prefix + "A"]
                )
                model.diffusion.inner_model.layers[i]._layer.B.data.copy_(
                    checkpoint['state_dict'][lora_prefix + "B"]
                )
                model_ema.diffusion.inner_model.layers[i]._layer.A.data.copy_(
                    checkpoint['state_dict'][lora_prefix + "A"]
                )
                model_ema.diffusion.inner_model.layers[i]._layer.B.data.copy_(
                    checkpoint['state_dict'][lora_prefix + "B"]
                )
                print(f"  Loaded LoRA weights from checkpoint for layer {i}")
            else:
                print(f"  No LoRA weights in checkpoint for layer {i}, using fresh initialization")
                
            print(f"  LoRA configuration: Rank {cfg.lora_rank}, Alpha {cfg.lora_alpha}")

            # Set gradients for LoRA parameters in training model only
            model.diffusion.inner_model.layers[i]._layer.A.requires_grad = True
            model.diffusion.inner_model.layers[i]._layer.B.requires_grad = True
            
            # Verify the parameters are set correctly
            print(f"  Layer {i} LoRA A requires_grad: {model.diffusion.inner_model.layers[i]._layer.A.requires_grad}")
            print(f"  Layer {i} LoRA B requires_grad: {model.diffusion.inner_model.layers[i]._layer.B.requires_grad}")
    
    print(f"\n=== LoRA applied to layers {cfg.train_pwise_layers} ===\n")

def debug_lora_training(model, step, experiment=None):
    """
    Comprehensive LoRA debugging to identify training issues.
    Call this every 100-1000 steps during training.
    """
    diagnostics = {
        'step': step,
        'lora_layers': [],
        'issues': []
    }
    
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, '_layer') and isinstance(module._layer, LoRALinear):
                lora_module = module._layer
                layer_info = {
                    'name': name,
                    'A_norm': torch.norm(lora_module.A).item(),
                    'B_norm': torch.norm(lora_module.B).item(),
                    'A_grad_norm': torch.norm(lora_module.A.grad).item() if lora_module.A.grad is not None else 0,
                    'B_grad_norm': torch.norm(lora_module.B.grad).item() if lora_module.B.grad is not None else 0,
                }
                
                # Calculate effective LoRA weight
                lora_weight = (lora_module.B @ lora_module.A) * (lora_module.alpha / lora_module.rank)
                layer_info['lora_weight_norm'] = torch.norm(lora_weight).item()
                layer_info['original_weight_norm'] = torch.norm(lora_module.original_linear.weight).item()
                layer_info['lora_contribution_ratio'] = layer_info['lora_weight_norm'] / layer_info['original_weight_norm']
                
                # Check for potential issues
                if layer_info['B_norm'] < 1e-6:
                    diagnostics['issues'].append(f"{name}: B matrix near zero")
                if layer_info['lora_contribution_ratio'] < 1e-3:
                    diagnostics['issues'].append(f"{name}: LoRA contribution too small")
                if layer_info['lora_contribution_ratio'] > 1.0:
                    diagnostics['issues'].append(f"{name}: LoRA contribution too large")
                if layer_info['A_grad_norm'] < 1e-8 or layer_info['B_grad_norm'] < 1e-8:
                    diagnostics['issues'].append(f"{name}: Gradients vanishing")
                
                diagnostics['lora_layers'].append(layer_info)
                
                # Log to experiment if provided
                if experiment:
                    experiment.log_metric(f'lora_debug/{name}_contribution_ratio', 
                                        layer_info['lora_contribution_ratio'], step)
    
    # Print summary
    print(f"\n=== LoRA Debug Step {step} ===")
    for layer in diagnostics['lora_layers']:
        print(f"{layer['name']}: contrib_ratio={layer['lora_contribution_ratio']:.4f}, "
              f"A_grad={layer['A_grad_norm']:.2e}, B_grad={layer['B_grad_norm']:.2e}")
    
    if diagnostics['issues']:
        print("\nISSUES DETECTED:")
        for issue in diagnostics['issues']:
            print(f"  - {issue}")
    
    return diagnostics

def monitor_lora_experiment(model, it, experiment):
    if it % 1000 != 0:
        return

    import re
    layer_contributions, layer_param_norms, layer_grad_norms = {}, {}, {}

    for name, module in model.named_modules():
        if hasattr(module, '_layer'):
            lora = module._layer
            if not all(hasattr(lora, attr) for attr in ('A', 'B', 'alpha', 'rank', 'original_linear')):
                continue

            match = re.search(r'layers\.(\d+)\.', name)
            if not match:
                continue
            layer = int(match.group(1))

            lora_weight = (lora.B @ lora.A) * (lora.alpha / lora.rank)
            contrib = torch.norm(lora_weight.data) / (torch.norm(lora.original_linear.weight.data) + 1e-8)
            a_norm, b_norm = torch.norm(lora.A).item(), torch.norm(lora.B).item()
            a_grad = torch.norm(lora.A.grad).item() if lora.A.grad is not None else 0.0
            b_grad = torch.norm(lora.B.grad).item() if lora.B.grad is not None else 0.0

            layer_contributions[layer] = contrib.item()
            layer_param_norms[layer] = (a_norm, b_norm)
            layer_grad_norms[layer] = (a_grad, b_grad)

    for layer in sorted(layer_contributions):
        contrib = layer_contributions[layer]
        a_norm, b_norm = layer_param_norms[layer]
        a_grad, b_grad = layer_grad_norms[layer]
        prefix = f'exp/layer_{layer}'
        experiment.log_metric(f'{prefix}_contribution', contrib, it)
        experiment.log_metric(f'{prefix}_A_norm', a_norm, it)
        experiment.log_metric(f'{prefix}_B_norm', b_norm, it)
        experiment.log_metric(f'{prefix}_A_grad', a_grad, it)
        experiment.log_metric(f'{prefix}_B_grad', b_grad, it)

    print(f"\n=== LoRA Status Step {it} ===")
    problematic_layers = []

    for layer in sorted(layer_contributions):
        c = layer_contributions[layer]
        a_g, b_g = layer_grad_norms[layer]
        status = "✅" if c > 0.01 else "⚠️" if c > 0.005 else "❌"
        print(f"Layer {layer}: contrib={c:.4f} ({c*100:.1f}%), A_grad={a_g:.2e}, B_grad={b_g:.2e} {status}")

        # Log general health metrics
        experiment.log_metric(f'exp/layer{layer}_health', c, it)

        # Trigger alerts if thresholds are exceeded
        if c < 0.005:
            message = f"ALERT: Layer {layer} dead or inactive (contribution={c:.4f}) at step {it}"
            experiment.log_text(message)
            problematic_layers.append(message)
        elif c > 0.25:
            message = f"ALERT: Layer {layer} overactive (contribution={c:.4f}) at step {it}"
            experiment.log_text(message)
            problematic_layers.append(message)

    # Global stats
    if layer_contributions:
        vals = list(layer_contributions.values())
        avg = sum(vals) / len(vals)
        std = (sum((v - avg) ** 2 for v in vals) / len(vals)) ** 0.5
        balance = std / (avg + 1e-8)
        min_c, max_c = min(vals), max(vals)

        experiment.log_metric('exp/avg_contribution', avg, it)
        experiment.log_metric('exp/min_contribution', min_c, it)
        experiment.log_metric('exp/max_contribution', max_c, it)
        experiment.log_metric('exp/std_contribution', std, it)
        experiment.log_metric('exp/balance_score', balance, it)

        # Success criteria
        success_metrics = {
            'no_dead_layers': min_c > 0.005,
            'all_healthy': min_c > 0.01,
            'balanced': balance < 0.08,
            'no_overactive': max_c < 0.25
        }

        for metric, success in success_metrics.items():
            experiment.log_metric(f'exp/success_{metric}', int(success), it)

        overall_success = all(success_metrics.values())
        experiment.log_metric('exp/overall_success', int(overall_success), it)

        print(f"Health: Min={min_c:.3f}, Max={max_c:.3f}, Balance={balance:.3f} {'✅' if overall_success else '❌'}")
        if problematic_layers:
            print("Issues detected:")
            for msg in problematic_layers:
                print(f"  - {msg}")

def get_lora_specific_lr(dataset_size, rank):
    """
    Adjust learning rate based on both dataset size and LoRA rank.
    Higher rank needs lower LR for stability.
    """
    # Your existing adaptive LR
    if dataset_size <= 100:
        lr = 1e-3
    elif dataset_size <= 500:
        lr = 5e-4
    elif dataset_size <= 1000:
        lr = 2e-4
    elif dataset_size <= 5000:
        lr = 1e-4
    else:
        lr = 5e-5
    
    # Adjust for rank
    if rank >= 16:
        lr = lr * 0.5  # Reduce LR for higher ranks
    elif rank >= 32:
        lr = lr * 0.25
    
    return lr

def get_lora_adaptive_lr(dataset_size, use_lora=False, lora_rank=8):
    """
    Adaptive learning rate based on dataset size.
    Larger datasets -> more stable gradients -> lower learning rate.
    
    Args:
        dataset_size: Size of the training dataset
        use_lora: Whether using LoRA fine-tuning
        lora_rank: LoRA rank (for rank-dependent reduction)
    """
    # Base learning rates
    if dataset_size <= 100:
        base_lr = 1e-3
    elif dataset_size <= 500:
        base_lr = 5e-4
    elif dataset_size <= 1000:
        base_lr = 2e-4
    elif dataset_size <= 5000:
        base_lr = 1e-4
    elif dataset_size <= 10000:
        base_lr = 5e-5
    elif dataset_size <= 50000:
        base_lr = 2e-5
    else:  # 100k
        base_lr = 1e-5
    return base_lr

### for generation

def load_lora_model_for_generation(pretrained_path, lora_checkpoint_path, cfg):
    """
    Fixed version that handles common parameter naming issues.
    """
    
    print("=== Loading LoRA Model for Generation (Fixed Version) ===")
    
    # Step 1: Create models
    print("\nCreating models...")
    model = epicVAE_nFlow_kDiffusion(cfg).to(cfg.device)
    model_ema = epicVAE_nFlow_kDiffusion(cfg).to(cfg.device)
    
    # Step 2: Load pretrained weights
    print("Loading pretrained weights...")
    pretrained_checkpoint = torch.load(pretrained_path, map_location=cfg.device, weights_only=False)
    
    model.load_state_dict(pretrained_checkpoint['state_dict'], strict=True)
    
    if 'others' in pretrained_checkpoint and 'model_ema' in pretrained_checkpoint['others']:
        model_ema.load_state_dict(pretrained_checkpoint['others']['model_ema'], strict=True)
    else:
        model_ema.load_state_dict(pretrained_checkpoint['state_dict'], strict=True)
    
    # Step 3: Apply LoRA architecture
    print("\nApplying LoRA architecture...")
    
    # Detect if layer-specific ranks were used
    try:
        actual_ranks = detect_lora_ranks_from_checkpoint(lora_checkpoint_path, use_ema=True)
        use_layer_specific_rank = len(set(actual_ranks.values())) > 1
    except:
        print("Could not detect ranks from checkpoint, using uniform rank")
        use_layer_specific_rank = False
    
    for i in cfg.train_pwise_layers:
        if use_layer_specific_rank:
            layer_rank = get_layer_specific_rank(i, cfg.lora_rank)
        else:
            layer_rank = cfg.lora_rank
        
        print(f"  Layer {i}: rank={layer_rank}")
        
        # Apply to both models
        model.diffusion.inner_model.layers[i] = apply_lora(
            model.diffusion.inner_model.layers[i], rank=layer_rank, alpha=cfg.lora_alpha
        )
        model_ema.diffusion.inner_model.layers[i] = apply_lora(
            model_ema.diffusion.inner_model.layers[i], rank=layer_rank, alpha=cfg.lora_alpha
        )
    
    # Step 4: Load LoRA weights with flexible name matching
    print("\nLoading LoRA weights...")
    lora_checkpoint = torch.load(lora_checkpoint_path, map_location=cfg.device, weights_only=False)
    
    # Choose which state dict to use
    if 'others' in lora_checkpoint and 'model_ema' in lora_checkpoint['others']:
        lora_state = lora_checkpoint['others']['model_ema']
        print("  Using EMA weights from checkpoint")
    else:
        lora_state = lora_checkpoint['state_dict']
        print("  Using main weights from checkpoint")
    
    # Create a mapping of simplified names to handle different import paths
    def simplify_param_name(name):
        """Remove module prefixes that might differ between training and inference"""
        # Remove common prefixes
        simplified = name
        for prefix in ['module.', 'model.', '_orig_mod.']:
            if simplified.startswith(prefix):
                simplified = simplified[len(prefix):]
        return simplified
    
    # Build mapping of simplified names to actual checkpoint names
    checkpoint_name_map = {}
    for name in lora_state.keys():
        if '_layer.A' in name or '_layer.B' in name:
            simplified = simplify_param_name(name)
            checkpoint_name_map[simplified] = name
    
    print(f"\n  Found {len(checkpoint_name_map)} LoRA parameters in checkpoint")
    
    # Load parameters with flexible matching
    lora_params_loaded = 0
    lora_params_not_found = []
    
    for name, param in model.named_parameters():
        if '_layer.A' in name or '_layer.B' in name:
            # Try exact match first
            loaded = False
            if name in lora_state:
                param.data.copy_(lora_state[name])
                ema_param = dict(model_ema.named_parameters())[name]
                ema_param.data.copy_(lora_state[name])
                loaded = True
                lora_params_loaded += 1
            else:
                # Try simplified name matching
                simplified = simplify_param_name(name)
                if simplified in checkpoint_name_map:
                    checkpoint_name = checkpoint_name_map[simplified]
                    if lora_state[checkpoint_name].shape == param.shape:
                        param.data.copy_(lora_state[checkpoint_name])
                        ema_param = dict(model_ema.named_parameters())[name]
                        ema_param.data.copy_(lora_state[checkpoint_name])
                        loaded = True
                        lora_params_loaded += 1
                        print(f"    Matched {name} -> {checkpoint_name}")
            
            if loaded:
                # Verify the loaded weights are non-zero
                weight_norm = torch.norm(param).item()
                if weight_norm < 1e-8:
                    print(f"    WARNING: {name} has near-zero weights (norm={weight_norm:.2e})")
            else:
                lora_params_not_found.append(name)
    
    print(f"\n  Loaded {lora_params_loaded} LoRA parameters")
    if lora_params_not_found:
        print(f"  Could not find {len(lora_params_not_found)} parameters:")
        for name in lora_params_not_found[:5]:  # Show first 5
            print(f"    - {name}")
    
    # Step 5: Verify LoRA contributions
    print("\nVerifying LoRA contributions:")
    total_contribution = 0
    
    for layer_idx in cfg.train_pwise_layers:
        lora_layer = model_ema.diffusion.inner_model.layers[layer_idx]._layer
        
        with torch.no_grad():
            # Check if B is non-zero (critical for LoRA to work)
            B_norm = torch.norm(lora_layer.B).item()
            A_norm = torch.norm(lora_layer.A).item()
            
            if B_norm < 1e-8:
                print(f"  ⚠️  Layer {layer_idx}: B matrix is zero! LoRA won't contribute.")
                continue
            
            # Calculate contribution
            test_input = torch.randn(1, lora_layer.original_linear.in_features).to(cfg.device)
            original_output = lora_layer.original_linear(test_input)
            lora_contribution = (test_input @ lora_layer.A.T) @ lora_layer.B.T
            lora_contribution *= (lora_layer.alpha / lora_layer.rank)
            
            original_norm = torch.norm(original_output).item()
            lora_norm = torch.norm(lora_contribution).item()
            ratio = lora_norm / (original_norm + 1e-8)
            total_contribution += ratio
            
            print(f"  Layer {layer_idx}: contribution={ratio:.1%}, A_norm={A_norm:.3f}, B_norm={B_norm:.3f}")
    
    avg_contribution = total_contribution / len(cfg.train_pwise_layers)
    print(f"\n  Average LoRA contribution: {avg_contribution:.1%}")
    
    if avg_contribution < 0.001:
        print("\n⚠️  CRITICAL WARNING: LoRA is not contributing to the model!")
        print("  The model will behave exactly like the pretrained model.")
        print("  Possible causes:")
        print("  1. LoRA weights were not saved properly during training")
        print("  2. Parameter names don't match between training and inference")
        print("  3. The checkpoint path is incorrect")
    
    # Step 6: Final preparation
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False
    
    print("\n=== Model loading complete ===")
    return model_ema

def generate_shower(model, cfg, energy=70, num_points=4000):
    """Generate a shower using the LoRA fine-tuned model."""
    print(f"\\nGenerating shower (energy={energy}, points={num_points})...")
    
    model.eval()
    with torch.no_grad():
        generated = gen_utils.get_shower(
            model=model,
            config=cfg,
            num_points=num_points,
            energy=energy,
            cond_N=num_points,
        ).cpu().numpy()
    
    print("Shower generated successfully!")
    print(f"Generated shower shape: {generated.shape}")
    print(f"Generated shower min/max: {generated.min():.4f}, {generated.max():.4f}")
    print(f"Generated shower mean/std: {generated.mean():.4f}, {generated.std():.4f}")
    
    return generated

def plot_shower_3d(shower_data, title="Generated Shower", save_path=None):
    """Plot shower in 3D."""
    # Ensure data is in the right format [N_points, 4] where columns are [x, y, z, energy]
    if shower_data.shape[1] != 4:
        shower_data = np.moveaxis(shower_data, 1, -1)
    
    # Extract coordinates and energy
    x, y, z, energy = shower_data[:, 0], shower_data[:, 1], shower_data[:, 2], shower_data[:, 3]
    
    # Create 3D plot
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color points by energy
    scatter = ax.scatter(x, y, z, c=energy, cmap='viridis', s=5, alpha=0.3)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Energy')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    return fig

def plot_shower_projections(shower_data, title="Generated Shower Projections", save_path=None):
    """Plot shower projections in 2D."""
    # Ensure data is in the right format
    if shower_data.shape[1] != 4:
        shower_data = np.moveaxis(shower_data, 1, -1)
    
    x, y, z, energy = shower_data[:, 0], shower_data[:, 1], shower_data[:, 2], shower_data[:, 3]
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=16)
    
    # XY projection
    scatter1 = axes[0].scatter(x, y, c=energy, cmap='viridis', s=2, alpha=0.3)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('XY Projection')
    
    # XZ projection
    scatter2 = axes[1].scatter(x, z, c=energy, cmap='viridis', s=2, alpha=0.3)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    axes[1].set_title('XZ Projection')
    
    # YZ projection
    scatter3 = axes[2].scatter(y, z, c=energy, cmap='viridis', s=2, alpha=0.3)
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    axes[2].set_title('YZ Projection')
    plt.colorbar(scatter3, ax=axes[2])
    
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Projections saved to {save_path}")
    
    plt.show()
    return fig

### BitFit


def apply_bitfit(model):
    """
    Apply BitFit to a model by freezing all parameters except biases.
    
    Args:
        model: PyTorch model to apply BitFit to
        
    Returns:
        List of trainable bias parameters
    """
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze bias parameters
    trainable_params = []
    for name, param in model.named_parameters():
        if 'bias' in name:
            param.requires_grad = True
            trainable_params.append({'params': param, 'name': name})
    
    return trainable_params


def count_bitfit_params(model):
    """
    Count the number of trainable parameters when using BitFit.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    bias_params = sum(p.numel() for n, p in model.named_parameters() if 'bias' in n)
    trainable_bias_params = sum(p.numel() for n, p in model.named_parameters() if 'bias' in n and p.requires_grad)
    
    return {
        'total_params': total_params,
        'total_bias_params': bias_params,
        'trainable_bias_params': trainable_bias_params,
        'percentage': (trainable_bias_params / total_params) * 100 if total_params > 0 else 0
    }


def update_ema_bitfit(model, model_ema, ema_decay):
    """
    Update EMA for BitFit parameters (bias only).
    
    Args:
        model: Source model
        model_ema: EMA model
        ema_decay: EMA decay rate
    """
    with torch.no_grad():
        for (name_ema, param_ema), (name, param) in zip(model_ema.named_parameters(), model.named_parameters()):
            if 'bias' in name and param.requires_grad:
                # Update only trainable bias parameters
                param_ema.copy_(param_ema * ema_decay + param * (1 - ema_decay))
            else:
                # Keep other parameters unchanged
                param_ema.copy_(param)