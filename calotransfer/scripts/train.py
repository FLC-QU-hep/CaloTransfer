from comet_ml import Experiment

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import argparse
import time

from utils.dataset import PointCloudDataset, PointCloudDatasetGH, CaloChallangeDataset
from utils.misc import seed_all
from models.vae_flow import FlowVAE
from models.allCond_epicVAE_nflow_PointDiff import AllCond_epicVAE_nFlow_PointDiff
from models.epicVAE_nflows_kDiffusion import epicVAE_nFlow_kDiffusion
import configs
import utils.finetune as ft
from utils.logging import initialize_experiment, setup_logging, log_training_metrics, generate_and_save_shower, log_ema_differences

import k_diffusion as K
import numpy as np

cfg = configs.Configs()

def get_adaptive_lr(dataset_size):
    """
    Adaptive learning rate based on dataset size.
    Larger datasets -> more stable gradients -> lower learning rate.

    old params lr:  e-5 for fine-tuning, to e−4 
    """
    if dataset_size <= 100:
        return 1e-3
    elif dataset_size <= 500:
        return 5e-4
    elif dataset_size <= 1000:
        return 2e-4
    elif dataset_size <= 5000:
        return 1e-4
    elif dataset_size <= 10000:
        return 5e-5
    elif dataset_size <= 50000:
        return 2e-5
    else:  # 100k
        return 1e-5

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_key', type=str, required=True)
    parser.add_argument('--dataset_size', type=int, required=True)
    parser.add_argument('--use_pretrain', type=lambda x: x.lower() == 'true', required=True)
    parser.add_argument('--use_lora', type=lambda x: x.lower() == 'true', required=True)
    parser.add_argument('--lora_rank', type=int, required=False, default=8)
    parser.add_argument('--lora_alpha', type=float, required=False, default=8.0)
    parser.add_argument('--use_bitfit', type=lambda x: x.lower() == 'true', required=False, default=False)
    parser.add_argument('--seed', type=int, required=True)
    return parser.parse_args()

def load_dataset(cfg):
    """Load and configure the dataset"""
    print('\n=== Dataset loading ===')
    print(f"Using dataset: {cfg.dataset_size}")
    print(f"Using dataset key: {cfg.dataset_key}")       
    
   
    # Select appropriate dataset class based on configuration
    if cfg.dataset == 'x36_grid' or cfg.dataset == 'clustered':
        train_dataset = PointCloudDataset(
            file_path=cfg.dataset_path,
            bs=cfg.train_bs,
            quantized_pos=cfg.quantized_pos
        )
    elif cfg.dataset == 'gettig_high':
        train_dataset = PointCloudDatasetGH(
            file_path=cfg.dataset_path,
            bs=cfg.train_bs,
            quantized_pos=cfg.quantized_pos
        )
    elif cfg.dataset == 'calo-challange':
        base_ds_path = '/data/dust/user/valentel/maxwell.merged/MyCaloTransfer/CaloTransfer/data/calo-challenge/preprocessing/reduced_datasets/'
        if cfg.val_ds_type == '10-90GeV':
            ds_path = base_ds_path + '10-90GeV/47k_dset1-2-3_prep_10-90GeV.hdf5'
        elif cfg.val_ds_type == '1-1000GeV':
            ds_path = base_ds_path + '1-1000GeV/100k_train_dset1-2-3_prep_1-1000GeV.hdf5'
        print(f"Using dataset path: {ds_path}")
        train_dataset = CaloChallangeDataset(
            file_path=ds_path,
            bs=cfg.train_bs,
            cfg=cfg,
            seed=cfg.seed,
            dataset_size=cfg.dataset_size
        )
    
    # Create dataloader
    dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=cfg.workers,
        shuffle=cfg.shuffle
    )
    
    print(f'Dataset size: {len(train_dataset)}')
    print('\n=== Dataset loaded ===\n')
    
    return train_dataset, dataloader

def initialize_model(cfg):
    """Initialize and configure the model based on configuration"""
    print('\n=== Model loading ===')
    print(f'Model name: {cfg.model_name}')
    print(f'LoRA: {cfg.lora}')
    print(f'BitFit: {cfg.bitfit}')
    print(f'Use pretrained: {cfg.use_pretrained}')
    
    # Initialize model based on model_name
    if cfg.model_name == 'flow':
        model = FlowVAE(cfg).to(cfg.device)
        model_ema = None
    elif cfg.model_name == 'AllCond_epicVAE_nFlow_PointDiff':
        model = AllCond_epicVAE_nFlow_PointDiff(cfg).to(cfg.device)
        model_ema = AllCond_epicVAE_nFlow_PointDiff(cfg).to(cfg.device)
    elif cfg.model_name == 'epicVAE_nFlow_kDiffusion':
        model = epicVAE_nFlow_kDiffusion(cfg).to(cfg.device)
        model_ema = epicVAE_nFlow_kDiffusion(cfg).to(cfg.device)
    else:
        raise ValueError(f"Unknown model_name: {cfg.model_name}")
    
    # Initialize EMA model if applicable
    if model_ema is not None:
        model_ema.load_state_dict(model.state_dict())
        model_ema.eval().requires_grad_(False)
        
        # Verify EMA type
        assert cfg.ema_type == 'inverse', "Only 'inverse' EMA type is supported"
        ema_sched = K.utils.EMAWarmup(
            power=cfg.ema_power,
            max_value=cfg.ema_max_value
        )
    else:
        ema_sched = None
    # Configure sigma distribution for diffusion models
    sample_density = None
    if 'diffusion' in cfg.model_name.lower():
        sample_density = K.config.make_sample_density(cfg.__dict__["model"])
    
    print(f'\n=== Model loaded: {cfg.model_name} ===\n')
    
    return model, model_ema, ema_sched, sample_density

def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    """
    Creates a learning rate scheduler that linearly decreases the learning rate
    from start_lr to end_lr over the course of start_epoch to end_epoch.
    """
    def lr_lambda(epoch):
        if epoch < start_epoch:
            return 1.0  # Keep initial LR until start_epoch
        elif epoch > end_epoch:
            return end_lr / start_lr  # Maintain end_lr after end_epoch
        else:
            # Linear interpolation between start_epoch and end_epoch
            t = (epoch - start_epoch) / (end_epoch - start_epoch)
            return (1.0 - t) + t * (end_lr / start_lr)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def apply_pretrained_weights(cfg, model, model_ema):
    """Apply pretrained weights and configure fine-tuning (LoRA, BitFit, or selective)"""
    if not cfg.use_pretrained:
        return model, model_ema, None, None, None, None
    
    print("\n=== Loading pretrained weights ===")
    
    # Load checkpoint
    checkpoint_path = cfg.diffusion_pretrained_model_path
    checkpoint = torch.load(
        checkpoint_path, 
        map_location=torch.device(cfg.device), 
        weights_only=False
    )
    
    # Load model and EMA state dicts
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model_ema.load_state_dict(checkpoint['others']['model_ema'], strict=False)
    
    # Check for conflicting fine-tuning methods
    if sum([cfg.lora, cfg.bitfit, (not cfg.lora and not cfg.bitfit)]) > 1:
        raise ValueError("Only one fine-tuning method can be active: LoRA, BitFit, or selective layer training")
    
    # Apply fine-tuning strategy
    if cfg.lora:
        # Apply LoRA
        ft.apply_lora_to_model(cfg, model, model_ema, checkpoint)
        
    elif cfg.bitfit:
        # Apply BitFit - only train bias parameters
        print("Applying BitFit - training only bias parameters")
        
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze only bias parameters
        trainable_params = []
        for name, param in model.named_parameters():
            if 'bias' in name:
                param.requires_grad = True
                trainable_params.append(param)
                print(f"Training bias: {name}")
                
    else:
        # Selective layer fine-tuning (original behavior)
        # Freeze all layers except those in train_pwise_layers
        for i, layer in enumerate(model.diffusion.inner_model.layers):
            for param in layer.parameters():
                param.requires_grad = (i in cfg.train_pwise_layers)
    
    # Freeze timestep embedding if not used (applies to all strategies)
    # BUT skip this for LoRA parameters
    if not cfg.time_embedded:
        for layer in model.diffusion.inner_model.timestep_embed:
            for param in layer.parameters():
                param.requires_grad = False
    
    # Print model summary and parameter counts
    print(f'\nModel loaded from checkpoint: {checkpoint_path}\n')
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)')
    
    # Debug: Print which parameters are trainable
    if cfg.lora:
        print("\nTrainable LoRA parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad and ("A" in name or "B" in name):
                print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")
    
    if cfg.lora:
        print(f'LoRA configuration: Layers {cfg.train_pwise_layers}, Rank {cfg.lora_rank}, Alpha {cfg.lora_alpha}')
    elif cfg.bitfit:
        print('BitFit configuration: Training only bias parameters')
    else:
        print(f'Selective layer configuration: Training layers {cfg.train_pwise_layers}')

    # Create optimizer based on fine-tuning method
    if cfg.lora:
        # LoRA-specific optimizer
        trainable_params = [p for name, p in model.named_parameters() if p.requires_grad and ("A" in name or "B" in name)]
        
        if not trainable_params:
            raise ValueError("No LoRA parameters found for training! Check if LoRA was applied correctly.")
            
        print(f"\nNumber of LoRA parameters in optimizer: {len(trainable_params)}")
        
        optimizer = torch.optim.RAdam(
            trainable_params, 
            lr=cfg.lr, 
            weight_decay=cfg.weight_decay
        )
        scheduler = get_linear_scheduler(
            optimizer, 
            start_epoch=cfg.sched_start_epoch, 
            end_epoch=cfg.sched_end_epoch, 
            start_lr=cfg.lr, 
            end_lr=cfg.end_lr
        )
        return model, model_ema, optimizer, None, scheduler, None
        
    elif cfg.bitfit:
        # BitFit-specific optimizer (only bias parameters)
        bias_params = [p for name, p in model.named_parameters() if p.requires_grad and 'bias' in name]
        
        if not bias_params:
            raise ValueError("No bias parameters found for BitFit training!")
            
        optimizer = torch.optim.RAdam(
            bias_params, 
            lr=cfg.lr, 
            weight_decay=cfg.weight_decay
        )
        scheduler = get_linear_scheduler(
            optimizer, 
            start_epoch=cfg.sched_start_epoch, 
            end_epoch=cfg.sched_end_epoch, 
            start_lr=cfg.lr, 
            end_lr=cfg.end_lr
        )
        return model, model_ema, optimizer, None, scheduler, None
    
    # For selective layer training, return None to use the standard optimizer configuration
    return model, model_ema, None, None, None, None

def configure_optimizer(cfg, model):
    """Configure optimizer and learning rate scheduler"""
    optimizer = None
    optimizer_flow = None
    scheduler = None
    scheduler_flow = None
    
    # Apply adaptive learning rate based on dataset size
    if cfg.use_pretrained:
        # Get adaptive learning rate
        adaptive_lr = get_adaptive_lr(cfg.dataset_size)
        
        if cfg.bitfit:
            # BitFit can use slightly higher LR
            cfg.lr = adaptive_lr * 2
            cfg.end_lr = cfg.lr / 10
            cfg.optimizer = 'AdamW'
        elif cfg.lora:
            # LoRA uses adaptive LR directly
            adaptive_lr = ft.get_lora_adaptive_lr(cfg.dataset_size, cfg.lora_rank)
            cfg.lr = adaptive_lr
            cfg.end_lr = cfg.lr / 10
            cfg.optimizer = 'AdamW'
        else:
            # Selective layer training uses lower LR
            cfg.lr = adaptive_lr / 2
            cfg.end_lr = cfg.lr / 10
        
        print(f'\n=== Adaptive Learning Rate Configuration ===')
        print(f'Dataset size: {cfg.dataset_size}')
        print(f'Base adaptive LR: {adaptive_lr:.2e}')
        print(f'Actual LR: {cfg.lr:.2e}')
        print(f'End LR: {cfg.end_lr:.2e}')
        print(f'Fine-tuning method: {"LoRA" if cfg.lora else "BitFit" if cfg.bitfit else "Selective"}')
        print('==========================================\n')
    else:
        # Non-pretrained uses fixed LR
        cfg.lr = 2e-4
        cfg.end_lr = 1e-4

    print(f'Learning rate: {cfg.lr}')
    
    # FlowVAE model
    if cfg.model_name == 'flow':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=cfg.lr, 
            weight_decay=cfg.weight_decay
        )
        scheduler = get_linear_scheduler(
            optimizer,
            start_epoch=cfg.sched_start_epoch,
            end_epoch=cfg.sched_end_epoch,
            start_lr=cfg.lr,
            end_lr=cfg.end_lr
        )
        return optimizer, None, scheduler, None
    
    # Diffusion-based models
    if cfg.model_name in ['AllCond_epicVAE_nFlow_PointDiff', 'epicVAE_nFlow_kDiffusion']:
        # Select optimizer type
        optim_class = torch.optim.Adam if cfg.optimizer == 'Adam' else torch.optim.RAdam
        
        # Different parameter groups based on latent dimension
        if cfg.latent_dim > 0:
            optimizer = optim_class(
                [
                    {'params': model.encoder.parameters()}, 
                    {'params': model.diffusion.parameters()},
                ], 
                lr=cfg.lr,  
                weight_decay=cfg.weight_decay
            )
            optimizer_flow = optim_class(
                [
                    {'params': model.flow.parameters()}, 
                ], 
                lr=cfg.lr,  
                weight_decay=cfg.weight_decay
            )
        else:
            optimizer = optim_class(
                [
                    {'params': model.diffusion.parameters()},
                ], 
                lr=cfg.lr,  
                weight_decay=cfg.weight_decay
            )
        
        # Create schedulers
        scheduler = get_linear_scheduler(
            optimizer, 
            start_epoch=cfg.sched_start_epoch, 
            end_epoch=cfg.sched_end_epoch, 
            start_lr=cfg.lr, 
            end_lr=cfg.end_lr
        )
        
        if cfg.latent_dim > 0:
            scheduler_flow = get_linear_scheduler(
                optimizer_flow, 
                start_epoch=cfg.sched_start_epoch, 
                end_epoch=cfg.sched_end_epoch, 
                start_lr=cfg.lr, 
                end_lr=cfg.end_lr
            )
    
    return optimizer, optimizer_flow, scheduler, scheduler_flow

def train_step(model, model_ema, batch, it, cfg, optimizer, optimizer_flow, scheduler, scheduler_flow, ema_sched, sample_density, experiment, ckpt_mgr):
    """Execute a single training step"""
    # Load data
    x = batch['event'][0].float().to(cfg.device)  # B, N, 4
    e = batch['energy'][0].float().to(cfg.device)  # B, 1
    n = batch['points'][0].float().to(cfg.device)  # B, 1
    
    # Reset gradients and set model to training mode
    optimizer.zero_grad()
    if cfg.model_name in ['AllCond_epicVAE_nFlow_PointDiff', 'epicVAE_nFlow_kDiffusion'] and cfg.latent_dim > 0:
        optimizer_flow.zero_grad()
    
    model.train()
    
    # Forward pass based on model type
    if cfg.model_name == 'flow':
        # Regular flow model
        loss = model.get_loss(x, kl_weight=cfg.kl_weight, writer=experiment, it=it)
        
        # Backward pass and optimize
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        # No flow loss for this model type
        loss_flow = torch.tensor(0.0)
        ema_decay = 0.0
        
    elif cfg.model_name == 'epicVAE_nFlow_kDiffusion':
        # Diffusion model with flow
        cond_feats = torch.cat([e, n], -1)
        noise = torch.randn_like(x)  # noise for forward diffusion
        sigma = sample_density([x.shape[0]], device=x.device)  # time steps
        
        # Get loss for both main model and flow components
        loss, loss_flow = model.get_loss(
            x, noise, sigma, cond_feats, 
            kl_weight=cfg.kl_weight, it=it, kld_min=cfg.kld_min
        )
    
        # Only proceed with backward pass if loss is finite
        if torch.isfinite(loss):
            # Backward for main loss
            loss.backward()
            
            # Backward for flow loss if using latent dimensions
            if cfg.latent_dim > 0:
                loss_flow.backward()
            
            # Gradient clipping
            orig_grad_norm = clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            
            # Optimizer steps
            optimizer.step()
            scheduler.step()
            
            if cfg.latent_dim > 0:
                optimizer_flow.step()
                scheduler_flow.step()
        else:
            print(f'Warning: Loss is NaN at step {it}')
            orig_grad_norm = torch.tensor(0.0)

        # Update EMA model
        ema_decay = ema_sched.get_value()
        if cfg.lora:
            ft.update_ema_lora(model, model_ema, ema_decay)
        else:
            K.utils.ema_update(model, model_ema, ema_decay)
        ema_sched.step()
    
    # Log LoRA-specific metrics if applicable
    if cfg.lora and it % 1000 == 0:  # Log less frequently to reduce overhead
        # Track norms of LoRA parameters
        ft.monitor_lora_experiment(model, it, experiment)
    
    if cfg.bitfit and it % 1000 == 0:
        # Track norms of bias parameters
        bias_norms = {}
        bias_grad_norms = {}
        
        for name, param in model.named_parameters():
            if 'bias' in name and param.requires_grad:
                bias_norms[name] = torch.norm(param).item()
                if param.grad is not None:
                    bias_grad_norms[name] = torch.norm(param.grad).item()
        
        # Log aggregated statistics
        if bias_norms:
            experiment.log_metric('bitfit/avg_bias_norm', np.mean(list(bias_norms.values())), it)
            experiment.log_metric('bitfit/max_bias_norm', np.max(list(bias_norms.values())), it)
        
        if bias_grad_norms:
            experiment.log_metric('bitfit/avg_bias_grad_norm', np.mean(list(bias_grad_norms.values())), it)
            experiment.log_metric('bitfit/max_bias_grad_norm', np.max(list(bias_grad_norms.values())), it)
        
        # Log number of trainable bias parameters
        num_bias_params = sum(p.numel() for n, p in model.named_parameters() if 'bias' in n and p.requires_grad)
        experiment.log_metric('bitfit/trainable_bias_params', num_bias_params, it)

    # Check for NaN weights and recover if needed
    if check_for_nan_weights(model):
        model, optimizer = recover_from_nan(model, ckpt_mgr)
    
    # Log metrics at regular intervals
    if it % cfg.log_iter == 0:
        log_training_metrics(it, loss, orig_grad_norm, cfg, ema_decay, optimizer, optimizer_flow, loss_flow, experiment, model)

    return loss.item()

def check_for_nan_weights(model):
    """Check if any model weights are NaN"""
    return torch.stack([torch.isnan(p).any() for p in model.parameters()]).any()

def recover_from_nan(model, ckpt_mgr):
    """Recover from NaN weights by loading a previous checkpoint"""
    print("Warning: Weights contain NaN values!")
    
    # Load the latest checkpoint
    latest_ckpt = ckpt_mgr.load_latest()
    if latest_ckpt is None:
        raise RuntimeError("No checkpoint available to recover from NaN weights")
    
    # Load model state from checkpoint
    model.load_state_dict(latest_ckpt['state_dict'])
    
    # Create a fresh optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    if 'optimizer' in latest_ckpt:
        optimizer.load_state_dict(latest_ckpt['optimizer'])
    
    print("Model recovered from latest checkpoint, optimizer reset")
    return model, optimizer

def save_checkpoint(model, model_ema, cfg, ckpt_mgr, ema_sched, optimizer, optimizer_flow, 
                    scheduler, scheduler_flow, it):
    """Save model checkpoint with appropriate optimizer states"""
    # Different checkpoint content based on model type
    if cfg.model_name == 'flow':
        opt_states = {
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
    elif cfg.model_name in ['AllCond_epicVAE_nFlow_PointDiff', 'epicVAE_nFlow_kDiffusion']:
        # Create LoRA state dict if using LoRA
        lora_state = {}
        if cfg.lora:
            for name, param in model.named_parameters():
                if "A" in name or "B" in name:
                    lora_state[name] = param.data.clone()
        
        # Basic optimizer states
        opt_states = {
            'model_ema': model_ema.state_dict(),  # Save EMA model
            'ema_sched': ema_sched.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        
        # Add flow optimizer states if applicable
        if cfg.latent_dim > 0:
            opt_states['optimizer_flow'] = optimizer_flow.state_dict()
            opt_states['scheduler_flow'] = scheduler_flow.state_dict()
        
        # Add LoRA state if applicable
        if cfg.lora and lora_state:
            opt_states['lora_state'] = lora_state
    
    # Save checkpoint
    ckpt_mgr.save(model, cfg, 0, others=opt_states, step=it)

def main():
    """Main function to run training"""
    # Set up basic configuration
    args = parse_arguments()
    cfg = configs.Configs()
    
    # Apply command line arguments to config
    cfg.dataset_key = args.dataset_key
    cfg.dataset_size = args.dataset_size
    cfg.use_pretrained = args.use_pretrain
    cfg.lora = args.use_lora
    cfg.bitfit = args.use_bitfit

    if cfg.lora:
        cfg.lora_rank = args.lora_rank
        cfg.lora_alpha = args.lora_alpha
        
        print(f"Using LoRA with rank {cfg.lora_rank} and alpha {cfg.lora_alpha}")

    cfg.seed = args.seed

    # Validate that only one fine-tuning method is selected
    if cfg.lora and cfg.bitfit:
        raise ValueError("Cannot use both LoRA and BitFit simultaneously. Choose one.")
    
    print(f"DEBUG: cfg.bitfit = {cfg.bitfit}")
    print(f"DEBUG: cfg.lora = {cfg.lora}")

    print(f"min and max energy: {cfg.min_energy} and {cfg.max_energy}")
    # Set random seed
    seed_all(seed=cfg.seed)
    
    # Initialize timing
    start_time = time.localtime()
    training_start_time = time.time()
    
    # Initialize experiment logging
    experiment, name_experiment = initialize_experiment(cfg, start_time)
    # Set experiment-specific tags
   
    experiment.add_tag(f"dataset_size_{cfg.dataset_size}")
    experiment.add_tag(f"seed_{cfg.seed}")
    
    # Log adaptive learning rate configuration
    if cfg.use_pretrained:
        experiment.log_metric('adaptive_lr/dataset_size', cfg.dataset_size)
        if not cfg.lora:
            experiment.log_metric('adaptive_lr/base_lr', get_adaptive_lr(cfg.dataset_size))
        else:
            experiment.log_metric('adaptive_lr/lora_specific_lr', ft.get_lora_specific_lr(cfg.dataset_size, cfg.lora_rank))
    
    # Set up logging directories
    log_dir, ckpt_mgr = setup_logging(cfg, start_time)
    
    # Load dataset
    train_dataset, dataloader = load_dataset(cfg)
    
    # Initialize model
    model, model_ema, ema_sched, sample_density = initialize_model(cfg)
    
    # Configure standard optimizer
    optimizer, optimizer_flow, scheduler, scheduler_flow = configure_optimizer(cfg, model)
    
    # Apply pretrained weights if specified
    if cfg.use_pretrained:
        model, model_ema, opt, opt_flow, sched, sched_flow = apply_pretrained_weights(
            cfg, model, model_ema
        )
        
        # Use the optimizer from pretrained weights if provided
        if opt is not None:
            optimizer = opt
        if opt_flow is not None:
            optimizer_flow = opt_flow
        if sched is not None:
            scheduler = sched
        if sched_flow is not None:
            scheduler_flow = sched_flow
    
    if cfg.lora:
        # Debug: Check if LoRA parameters have requires_grad=True
        print("\n=== Checking LoRA parameters ===")
        experiment.add_tag(f"rank_{cfg.lora_rank}")
        experiment.add_tag(f"alpha_{cfg.lora_alpha}")
        lora_params_found = False
        for name, param in model.named_parameters():
            if "A" in name or "B" in name:
                print(f"{name}: requires_grad={param.requires_grad}")
                lora_params_found = True
                
        if not lora_params_found:
            print("WARNING: No LoRA parameters found!")
        print("=== End LoRA check ===\n")

    print(f'model: \n{model}')
    print(f'model_ema: \n{model_ema}')

    # Main training loop
    print('\n\n===Starting training...===\n\n')
    it = 0
    stop = False
    
    while not stop:
        for batch in dataloader:
            it += 1
            
            # Training step
            train_step(
                model, model_ema, batch, it, cfg, 
                optimizer, optimizer_flow, scheduler, scheduler_flow,
                ema_sched, sample_density, experiment, ckpt_mgr
            )
            
            # Log time periodically
            if it % 10000 == 0:
                elapsed_time = time.time() - training_start_time
                print(f'Elapsed Time: {elapsed_time:.2f} seconds')
                
                # Log EMA differences
                if model_ema is not None:
                    log_ema_differences(model, model_ema, it, experiment, cfg)
            
            # Save checkpoint periodically
            if it % cfg.val_freq == 0 or it == cfg.max_iters:
                save_checkpoint(
                    model, model_ema, cfg, ckpt_mgr, ema_sched,
                    optimizer, optimizer_flow, scheduler, scheduler_flow, it
                )
            ## lora debugging
            if cfg.lora and it % 1_000 == 0: 
                ft.debug_lora_training(model, it, experiment)
            # Generate sample periodically
            if it % 10_000 == 0:
                generate_and_save_shower(model, cfg, it, experiment, name_experiment)
            
            # Stop after maximum iterations
            if it >= cfg.max_iters:
                stop = True
                break
    
    total_time = time.time() - training_start_time
    print(f"\n=== Training completed in {total_time:.2f} seconds ===\n")

if __name__ == "__main__":
    main()