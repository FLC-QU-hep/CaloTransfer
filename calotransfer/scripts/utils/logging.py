#logging for the diffusion model training
import os
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from comet_ml import Experiment

from . import gen_utils_CaloChallenge as gen_utils
from . import plot_evaluate as plot
from .misc import get_new_log_dir, CheckpointManager


def initialize_experiment(cfg, start_time):
    """Initialize and configure the Comet experiment"""
    if not cfg.log_comet:
        return None
        
    with open('./utils/comet_api_key.txt', 'r') as file:
        key = file.read().strip()
    
    experiment = Experiment(
        api_key=key,
        project_name=cfg.comet_project, 
        auto_metric_logging=False,
        workspace="lorenzovalente3",
    )
    
    # Log configuration parameters
    experiment.log_parameters(cfg.__dict__)
    
    # Set experiment name
    name_experiment = cfg.name + time.strftime('%Y_%m_%d__%H_%M_%S', start_time)
    experiment.set_name(name_experiment)
    
    # Log source code files
    code_files = [
        'train.py', 'configs.py', 'utils/dataset.py', 
        'utils/misc.py', 'utils/finetune.py', 'job_slurm/train.sh'
    ]
    for file in code_files:
        experiment.log_code(file_name=file)
    
    experiment.log_parameter('random_seed', cfg.seed)
    
    return experiment, name_experiment

def setup_logging(cfg, start_time):

    """Set up logging directory and checkpoint manager"""
    # Determine training type for directory structure

    if cfg.use_pretrained:
        base_dir = '/finetune/'
        if cfg.lora:
            base_dir += 'lora/'
    else:
        base_dir = '/vanilla/'
    
    # Create log directory
    postfix = f'_{cfg.tag}' if cfg.tag is not None else ''
    log_dir = get_new_log_dir(cfg.logdir + base_dir, prefix=cfg.name, postfix=postfix, start_time=start_time)
    
    print("\n=== Logging ===")
    print(f'Logging: {log_dir}')
    name_experiment = cfg.name + time.strftime('%Y_%m_%d__%H_%M_%S', start_time)

    # Create directories for generated samples if needed
    if cfg.log_comet:
        os.makedirs(f'./train-score/train-showers/{name_experiment}', exist_ok=True)
    
    return log_dir, CheckpointManager(log_dir)

def log_training_metrics(it, loss, grad_norm, cfg, ema_decay, optimizer, optimizer_flow, loss_flow, experiment, model):
    """Log training metrics to console and experiment tracker"""
    # Console logging
    print(f'[Train] Iter {it:04d} | Loss {loss.item():.6f} | Grad {grad_norm:.4f} | '
        f'KLWeight {cfg.kl_weight:.4f} | EMAdecay {ema_decay:.4f}')
    
    # Experiment logging
    if cfg.log_comet and experiment is not None:
        # Log basic metrics
        experiment.log_metric('train/loss', loss.item(), it)
        experiment.log_metric('train/kl_weight', cfg.kl_weight, it)
        experiment.log_metric('train/lr', optimizer.param_groups[0]['lr'], it)
        experiment.log_metric('train/grad_norm', grad_norm, it)
        experiment.log_metric('train/ema_decay', ema_decay, it)
        
        # Log flow-specific metrics if applicable
        if cfg.latent_dim > 0:
            experiment.log_metric('train/loss_flow', loss_flow.item(), it)
            experiment.log_metric('train/lr_flow', optimizer_flow.param_groups[0]['lr'], it)
        
        # Log LoRA-specific metrics if applicable
        if cfg.lora:
            for name, param in model.named_parameters():
                if "A" in name or "B" in name:
                    experiment.log_metric(f'lora/{name}_norm', torch.norm(param), it)
                    if param.grad is not None:
                        experiment.log_metric(f'lora/{name}_grad_norm', torch.norm(param.grad), it)

def generate_and_save_shower(model, cfg, it, experiment, name_experiment):

    """Generate a shower sample and save/log it"""
    with torch.no_grad():
        print('Generating shower sample...')
        
        # Generate shower
        generated = gen_utils.get_shower(
            model=model, 
            config=cfg, 
            num_points=10000, 
            energy=500, 
            cond_N=10000
        ).cpu().numpy()
        
        # Reshape for plotting
        generated = np.moveaxis(generated, 1, -1)
        
        # Save plot
        save_dir = f'./train-score/train-showers/{name_experiment}'
        os.makedirs(save_dir, exist_ok=True)
        save_path = f'{save_dir}/shower_{it}.png'
        
        plot.plt_scatter(
            generated[-1], 
            title=f'Generated Shower iteration: {it}', 
            save_plot=True,
            save_path=save_path
        )
        
        # Log image and metrics to experiment
        if cfg.log_comet and experiment is not None:
            experiment.log_image(save_path, name=f"Generated Shower {it}")
            experiment.log_metric("iteration", it)

def log_ema_differences(model, model_ema, it, experiment, cfg):
    """Log differences between model and EMA model parameters"""
    if not cfg.log_comet or experiment is None:
        return
        
    for name, param in model.named_parameters():
        if "A" in name or "B" in name:
            ema_param = model_ema.state_dict()[name]
            diff = torch.norm(param - ema_param).item()
            experiment.log_metric(f'ema_diff/{name}', diff, it)