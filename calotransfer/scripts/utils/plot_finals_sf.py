import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import warnings
from scipy import stats
from scipy.stats import bootstrap

@dataclass
class ExperimentConfig:
    """Configuration for experiment paths and parameters."""
    base_path: str = '/data/dust/user/valentel/maxwell.merged/MyCaloTransfer/CaloTransfer/results/shower_flow'
    energy_range: str = '1-1000GeV'
    
    # Experiment folders for different seeds - to be set by specific experiments
    vanilla_folders: Dict[int, str] = None
    finetune_folders: Dict[int, str] = None


def create_showerflow_config(vanilla_folders: Dict[int, str], 
                            finetune_folders: Dict[int, str],
                            base_path: str = None,
                            energy_range: str = None) -> ExperimentConfig:
    """
    Create configuration for specific ShowerFlow experiments.
    
    Args:
        vanilla_folders: Dictionary mapping seed -> folder name for vanilla experiments
        finetune_folders: Dictionary mapping seed -> folder name for finetune experiments
        base_path: Optional override for base path
        energy_range: Optional override for energy range
    
    Returns:
        ExperimentConfig with specified folder mappings
    """
    config = ExperimentConfig()
    
    if base_path is not None:
        config.base_path = base_path
    if energy_range is not None:
        config.energy_range = energy_range
        
    config.vanilla_folders = vanilla_folders
    config.finetune_folders = finetune_folders
    
    return config


def validate_config(config: ExperimentConfig) -> bool:
    """
    Validate that the configuration contains required folder mappings.
    
    Args:
        config: ExperimentConfig to validate
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValueError: If critical validation fails
    """
    if config.vanilla_folders is None or config.finetune_folders is None:
        raise ValueError("Configuration must specify both vanilla_folders and finetune_folders")
    
    if len(config.vanilla_folders) == 0 or len(config.finetune_folders) == 0:
        raise ValueError("Configuration must contain at least one experiment folder for each type")
    
    # Check for duplicate folder names (common mistake)
    vanilla_folders = list(config.vanilla_folders.values())
    if len(vanilla_folders) != len(set(vanilla_folders)):
        raise ValueError("Duplicate folder names detected in vanilla_folders configuration")
    
    finetune_folders = list(config.finetune_folders.values()) 
    if len(finetune_folders) != len(set(finetune_folders)):
        raise ValueError("Duplicate folder names detected in finetune_folders configuration")
    
    # Check for overlapping seeds
    vanilla_seeds = set(config.vanilla_folders.keys())
    finetune_seeds = set(config.finetune_folders.keys())
    if not vanilla_seeds == finetune_seeds:
        missing_vanilla = finetune_seeds - vanilla_seeds
        missing_finetune = vanilla_seeds - finetune_seeds
        if missing_vanilla:
            print(f"Warning: Seeds {missing_vanilla} present in finetune but missing in vanilla")
        if missing_finetune:
            print(f"Warning: Seeds {missing_finetune} present in vanilla but missing in finetune")
    
    return True


class ExperimentDataLoader:
    """Robust loader for experiment data with error handling and validation."""
    
    def __init__(self, config: ExperimentConfig):
        validate_config(config)  # Validate configuration on initialization
        self.config = config
        self.base_path = Path(config.base_path)
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {config.base_path}")
    
    def load_experiment_data(self, 
                           model_type: str,
                           experiment_folder: str,
                           filename: str) -> Optional[Dict[str, Any]]:
        """Load experiment data with validation."""
        full_path = self.base_path / model_type / self.config.energy_range / experiment_folder / filename
        
        if not full_path.exists():
            warnings.warn(f"File not found: {full_path}")
            return None
        
        try:
            with open(full_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            warnings.warn(f"JSON decode error in {full_path}: {e}")
            return None
    
    def load_all_seeds(self, model_type: str, metric_file: str) -> Dict[int, Dict]:
        """Load data for all seeds of a given model type."""
        folders = self.config.vanilla_folders if model_type == 'vanilla' else self.config.finetune_folders
        results = {}
        
        for seed, folder in folders.items():
            data = self.load_experiment_data(model_type, folder, metric_file)
            if data is not None:
                results[seed] = data
            else:
                print(f"Warning: No data for {model_type} seed {seed}")
        
        return results


def extract_metric_value(data_dict: Dict, metric_name: str = 'Clusters per Layer') -> Dict:
    """Extract specific metric from the data dictionary."""
    result = {}
    for step, metrics in data_dict.items():
        if metric_name in metrics:
            result[int(step)] = metrics[metric_name]
    return result


def calculate_statistics(data_by_seed: Dict[int, Dict], 
                         metric_name: str = 'Clusters per Layer') -> Tuple[Dict, Dict, Dict]:
    """
    Calculate mean and standard deviation across seeds.
    
    Returns:
        means: Dictionary of mean values per step/D
        stds: Dictionary of standard deviations per step/D
        counts: Dictionary of sample counts per step/D
    """
    # First, collect all unique steps and D values
    all_steps = set()
    all_d_values = set()
    
    for seed_data in data_by_seed.values():
        for step, metrics in seed_data.items():
            all_steps.add(int(step))
            if metric_name in metrics:
                all_d_values.update(metrics[metric_name].keys())
    
    # Calculate statistics
    means = {}
    stds = {}
    counts = {}
    
    for step in all_steps:
        means[step] = {}
        stds[step] = {}
        counts[step] = {}
        
        for d_val in all_d_values:
            values = []
            for seed_data in data_by_seed.values():
                if str(step) in seed_data and metric_name in seed_data[str(step)]:
                    if d_val in seed_data[str(step)][metric_name]:
                        cluster_list = seed_data[str(step)][metric_name][d_val]
                        values.append(np.mean(cluster_list))
            
            if values:
                means[step][d_val] = np.mean(values)
                stds[step][d_val] = np.std(values, ddof=1) if len(values) > 1 else 0
                counts[step][d_val] = len(values)
    
    return means, stds, counts


def calculate_bootstrap_statistics(data_by_seed: Dict[int, Dict],
                                  metric_name: str = 'Clusters per Layer',
                                  n_bootstrap: int = 10000,
                                  confidence_level: float = 0.95) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Calculate bootstrap confidence intervals for the data.
    
    Args:
        data_by_seed: Dictionary of seed -> data
        metric_name: Name of metric to analyze
        n_bootstrap: Number of bootstrap samples (default 10000)
        confidence_level: Confidence level for intervals (default 0.95)
    
    Returns:
        means: Dictionary of mean values
        lower_bounds: Dictionary of lower confidence bounds
        upper_bounds: Dictionary of upper confidence bounds
        raw_values: Dictionary of raw values for each step/D
    """
    # Collect all unique steps and D values
    all_steps = set()
    all_d_values = set()
    
    for seed_data in data_by_seed.values():
        for step, metrics in seed_data.items():
            all_steps.add(int(step))
            if metric_name in metrics:
                all_d_values.update(metrics[metric_name].keys())
    
    means = {}
    lower_bounds = {}
    upper_bounds = {}
    raw_values = {}
    
    for step in all_steps:
        means[step] = {}
        lower_bounds[step] = {}
        upper_bounds[step] = {}
        raw_values[step] = {}
        
        for d_val in all_d_values:
            values = []
            for seed_data in data_by_seed.values():
                if str(step) in seed_data and metric_name in seed_data[str(step)]:
                    if d_val in seed_data[str(step)][metric_name]:
                        cluster_list = seed_data[str(step)][metric_name][d_val]
                        values.append(np.mean(cluster_list))
            
            if values:
                values = np.array(values)
                raw_values[step][d_val] = values
                
                if len(values) == 1:
                    # Single value, no uncertainty
                    means[step][d_val] = values[0]
                    lower_bounds[step][d_val] = values[0]
                    upper_bounds[step][d_val] = values[0]
                elif len(values) < 3:
                    # Too few for bootstrap, use simple mean ± std
                    means[step][d_val] = np.mean(values)
                    std = np.std(values, ddof=1)
                    lower_bounds[step][d_val] = means[step][d_val] - std
                    upper_bounds[step][d_val] = means[step][d_val] + std
                else:
                    # Perform bootstrap
                    def mean_statistic(x, axis):
                        return np.mean(x, axis=axis)
                    
                    # Use scipy's bootstrap if available (scipy >= 1.7.0)
                    try:
                        res = bootstrap((values,), mean_statistic, 
                                      n_resamples=n_bootstrap,
                                      confidence_level=confidence_level,
                                      method='percentile',
                                      random_state=42)
                        means[step][d_val] = np.mean(values)
                        lower_bounds[step][d_val] = res.confidence_interval.low
                        upper_bounds[step][d_val] = res.confidence_interval.high
                    except:
                        # Fallback to manual bootstrap
                        bootstrap_means = []
                        rng = np.random.RandomState(42)
                        for _ in range(n_bootstrap):
                            resample = rng.choice(values, size=len(values), replace=True)
                            bootstrap_means.append(np.mean(resample))
                        
                        bootstrap_means = np.array(bootstrap_means)
                        means[step][d_val] = np.mean(values)
                        alpha = 1 - confidence_level
                        lower_bounds[step][d_val] = np.percentile(bootstrap_means, 100 * alpha/2)
                        upper_bounds[step][d_val] = np.percentile(bootstrap_means, 100 * (1 - alpha/2))
    
    return means, lower_bounds, upper_bounds, raw_values


def d_key_to_value(d_key: str) -> float:
    """Convert D key string to numerical value."""
    parts = d_key.split(' x 10^')
    base = float(parts[0].split(' = ')[1].strip())
    exponent = int(parts[1].strip())
    return base * (10 ** exponent)


def plot_convergence_curves(vanilla_data: Dict[int, Dict], 
                           finetune_data: Dict[int, Dict],
                           metric_name: str = 'Clusters per Layer',
                           metric_file: str = '',
                           save_fig: bool = False,
                           save_path: Optional[str] = None):
    """
    Plot convergence curves with standard error bands.
    Clean version with minimal grid lines and matched y-axis ranges for comparison.
    
    Args:
        vanilla_data: Vanilla experiment data
        finetune_data: Fine-tuned experiment data
        metric_name: Name of metric to extract from data
        metric_file: Metric file name to determine y-axis label
        save_fig: Whether to save the figure
        save_path: Path to save the figure
    """
    # Calculate statistics using standard error
    vanilla_means, vanilla_stds, vanilla_counts = calculate_statistics(vanilla_data, metric_name)
    finetune_means, finetune_stds, finetune_counts = calculate_statistics(finetune_data, metric_name)
    
    # Get sorted D keys
    all_d_keys = set()
    for step_data in vanilla_means.values():
        all_d_keys.update(step_data.keys())
    sorted_d_keys = sorted(all_d_keys, key=d_key_to_value)
    
    # Determine y-axis label based on metric file
    if 'KL' in metric_file.upper():
        y_label = 'Avg KL Points per Layer ' #+ metric_name
    elif 'WD' in metric_file.upper() or 'WASSERSTEIN' in metric_file.upper():
        y_label = 'Avg WD Points per Layer ' #+ metric_name
    else:
        y_label = 'Avg Metric Value'  # Fallback
    
    # Calculate global min and max for y-axis alignment
    all_values = []
    
    # Collect all values from vanilla
    for step_dict in vanilla_means.values():
        for d_key in sorted_d_keys:
            if d_key in step_dict:
                mean_val = step_dict[d_key]
                all_values.append(mean_val)
                # Also include error bounds for complete range
                step_num = list(vanilla_means.keys())[list(vanilla_means.values()).index(step_dict)]
                if step_num in vanilla_stds and d_key in vanilla_stds[step_num]:
                    std = vanilla_stds[step_num][d_key]
                    n = vanilla_counts[step_num][d_key]
                    se = std / np.sqrt(n) if n > 1 else std
                    all_values.extend([mean_val - se, mean_val + se])
    
    # Collect all values from finetune
    for step_dict in finetune_means.values():
        for d_key in sorted_d_keys:
            if d_key in step_dict:
                mean_val = step_dict[d_key]
                all_values.append(mean_val)
                # Also include error bounds for complete range
                step_num = list(finetune_means.keys())[list(finetune_means.values()).index(step_dict)]
                if step_num in finetune_stds and d_key in finetune_stds[step_num]:
                    std = finetune_stds[step_num][d_key]
                    n = finetune_counts[step_num][d_key]
                    se = std / np.sqrt(n) if n > 1 else std
                    all_values.extend([mean_val - se, mean_val + se])
    
    # Calculate y-axis limits with some padding for log scale
    if all_values:
        y_min = min(all_values) * 0.8  # 20% padding below
        y_max = max(all_values) * 1.2  # 20% padding above
    else:
        y_min, y_max = 1e-3, 1e0  # Fallback defaults
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.get_cmap('viridis')(np.linspace(0.1, 0.9, len(sorted_d_keys)))
    
    # Plot function with standard error bands
    def plot_with_error_bands(ax, means_dict, stds_dict, counts_dict, title, sorted_keys, colors_list):
        for d_key, color in zip(sorted_keys, reversed(colors_list)):
            steps = []
            means = []
            lower = []
            upper = []
            
            for step in sorted(means_dict.keys()):
                if d_key in means_dict[step]:
                    steps.append(step)
                    mean_val = means_dict[step][d_key]
                    means.append(mean_val)
                    
                    # Calculate standard error
                    std = stds_dict[step][d_key]
                    n = counts_dict[step][d_key]
                    se = std / np.sqrt(n) if n > 1 else std
                    lower.append(mean_val - se)
                    upper.append(mean_val + se)
            
            if steps:
                steps = np.array(steps)
                means = np.array(means)
                lower = np.array(lower)
                upper = np.array(upper)
                
                # Plot mean line only
                ax.plot(steps, means, '-o', label=d_key, color=color, markersize=4, linewidth=2)
                
                # Add standard error band
                ax.fill_between(steps, lower, upper, color=color, alpha=0.15)
        
        ax.set_xlabel('Epochs', fontsize=18)
        ax.set_ylabel(y_label, fontsize=18)
        ax.set_title(title, fontsize=24)
        ax.set_yscale('log')
        
        # Set the same y-axis limits for both plots
        ax.set_ylim(y_min, y_max)
        
        # Minimal grid - only major gridlines
        # ax.grid(False, which="major", ls="-", alpha=0.2)
        # ax.grid(False, which="minor")
        ax.grid(False)

    
    # Plot both datasets
    plot_with_error_bands(ax1, vanilla_means, vanilla_stds, vanilla_counts,
                         'From scratch', sorted_d_keys, colors)
    plot_with_error_bands(ax2, finetune_means, finetune_stds, finetune_counts,
                         'Fine-tuned', sorted_d_keys, colors)
    
    # Add legend
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], 
              title='Number of training showers', title_fontsize=20,
              loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=4, frameon=False, edgecolor='black', fontsize=16)
    
    plt.tight_layout()
    
    if save_fig and save_path:
        save_figure(fig, save_path)
    
    plt.show()


def _get_metric_ylabel(metric_file: str) -> str:
    """
    Get appropriate y-axis label based on metric file type.
    
    Args:
        metric_file: Name of the metric file being analyzed
        
    Returns:
        Appropriate y-axis label string
    """
    if 'KL' in metric_file.upper():
        return 'KL Divergence'
    elif 'WD' in metric_file.upper() or 'WASSERSTEIN' in metric_file.upper():
        return 'Normalised WD'
    else:
        return 'Metric Value'  # Fallback


def plot_final_performance_comparison(vanilla_data: Dict[int, Dict],
                                     finetune_data: Dict[int, Dict],
                                     metric_name: str = 'Clusters per Layer',
                                     metric_file: str = '',
                                     save_fig: bool = False,
                                     save_path: Optional[str] = None,
                                     show_average_lines: bool = False,
                                     use_bootstrap: bool = False):
    """Plot final best performance vs number of training showers with uncertainty bands."""
    
    # Find minimum values for each D for each seed
    vanilla_mins_by_seed = {}
    finetune_mins_by_seed = {}
    vanilla_best_epochs = {}
    finetune_best_epochs = {}
    
    # Process each seed to find its minimum for each D
    for seed, seed_data in vanilla_data.items():
        for step, metrics in seed_data.items():
            if metric_name in metrics:
                for d_key, values in metrics[metric_name].items():
                    avg_val = np.mean(values)
                    
                    if d_key not in vanilla_mins_by_seed:
                        vanilla_mins_by_seed[d_key] = {}
                        vanilla_best_epochs[d_key] = {}
                    if seed not in vanilla_mins_by_seed[d_key]:
                        vanilla_mins_by_seed[d_key][seed] = avg_val
                        vanilla_best_epochs[d_key][seed] = step
                    else:
                        if avg_val < vanilla_mins_by_seed[d_key][seed]:
                            vanilla_mins_by_seed[d_key][seed] = avg_val
                            vanilla_best_epochs[d_key][seed] = step
    
    # Similar for finetune
    for seed, seed_data in finetune_data.items():
        for step, metrics in seed_data.items():
            if metric_name in metrics:
                for d_key, values in metrics[metric_name].items():
                    avg_val = np.mean(values)
                    
                    if d_key not in finetune_mins_by_seed:
                        finetune_mins_by_seed[d_key] = {}
                        finetune_best_epochs[d_key] = {}
                    if seed not in finetune_mins_by_seed[d_key]:
                        finetune_mins_by_seed[d_key][seed] = avg_val
                        finetune_best_epochs[d_key][seed] = step
                    else:
                        if avg_val < finetune_mins_by_seed[d_key][seed]:
                            finetune_mins_by_seed[d_key][seed] = avg_val
                            finetune_best_epochs[d_key][seed] = step
    
    # Sort D keys
    sorted_d_keys = sorted(vanilla_mins_by_seed.keys(), key=d_key_to_value)
    x_values = [d_key_to_value(d) for d in sorted_d_keys]
    
    # Calculate statistics with bootstrap
    y_vanilla_mean = []
    y_vanilla_lower = []
    y_vanilla_upper = []
    y_finetune_mean = []
    y_finetune_lower = []
    y_finetune_upper = []
    
    for d_key in sorted_d_keys:
        # Vanilla
        vanilla_vals = list(vanilla_mins_by_seed[d_key].values())
        y_vanilla_mean.append(np.mean(vanilla_vals))
        
        if use_bootstrap and len(vanilla_vals) >= 3:
            # Bootstrap CI
            res = bootstrap((vanilla_vals,), np.mean, confidence_level=0.95)
            y_vanilla_lower.append(res.confidence_interval.low)
            y_vanilla_upper.append(res.confidence_interval.high)
        else:
            # Standard error
            se = np.std(vanilla_vals, ddof=1) / np.sqrt(len(vanilla_vals)) if len(vanilla_vals) > 1 else 0
            y_vanilla_lower.append(y_vanilla_mean[-1] - 1.96*se)  # 95% CI
            y_vanilla_upper.append(y_vanilla_mean[-1] + 1.96*se)
        
        # Finetune
        finetune_vals = list(finetune_mins_by_seed[d_key].values())
        y_finetune_mean.append(np.mean(finetune_vals))
        
        if use_bootstrap and len(finetune_vals) >= 3:
            # Bootstrap CI
            res = bootstrap((finetune_vals,), np.mean, confidence_level=0.95)
            y_finetune_lower.append(res.confidence_interval.low)
            y_finetune_upper.append(res.confidence_interval.high)
        else:
            # Standard error
            se = np.std(finetune_vals, ddof=1) / np.sqrt(len(finetune_vals)) if len(finetune_vals) > 1 else 0
            y_finetune_lower.append(y_finetune_mean[-1] - 1.96*se)
            y_finetune_upper.append(y_finetune_mean[-1] + 1.96*se)
    
    # Calculate overall average across all D values
    all_vanilla_vals = [val for sublist in vanilla_mins_by_seed.values() for val in sublist.values()]
    all_finetune_vals = [val for sublist in finetune_mins_by_seed.values() for val in sublist.values()]
    
    vanilla_avg = np.mean(all_vanilla_vals)
    finetune_avg = np.mean(all_finetune_vals)
    
    if use_bootstrap and len(all_vanilla_vals) >= 3:
        res = bootstrap((all_vanilla_vals,), np.mean, confidence_level=0.95)
        vanilla_avg_lower = res.confidence_interval.low
        vanilla_avg_upper = res.confidence_interval.high
    else:
        se = np.std(all_vanilla_vals, ddof=1) / np.sqrt(len(all_vanilla_vals)) if len(all_vanilla_vals) > 1 else 0
        vanilla_avg_lower = vanilla_avg - 1.96*se
        vanilla_avg_upper = vanilla_avg + 1.96*se
    
    if use_bootstrap and len(all_finetune_vals) >= 3:
        res = bootstrap((all_finetune_vals,), np.mean, confidence_level=0.95)
        finetune_avg_lower = res.confidence_interval.low
        finetune_avg_upper = res.confidence_interval.high
    else:
        se = np.std(all_finetune_vals, ddof=1) / np.sqrt(len(all_finetune_vals)) if len(all_finetune_vals) > 1 else 0
        finetune_avg_lower = finetune_avg - 1.96*se
        finetune_avg_upper = finetune_avg + 1.96*se
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define colors (using your specified style)
    colors = {
        "From scratch": "#0D3B66",
        "Fine-tuned": "#C03221"
    }
    
    # Plot with uncertainty bands using your requested style
    ax.plot(x_values, y_vanilla_mean, 'o--', color=colors["From scratch"], 
            label='From scratch', linewidth=2, markersize=6)
    ax.fill_between(x_values, y_vanilla_lower, y_vanilla_upper, 
                   color=colors["From scratch"], alpha=0.2)
    
    ax.plot(x_values, y_finetune_mean, 'o--', color=colors["Fine-tuned"], 
            label='Fine-tuned', linewidth=2, markersize=6)
    ax.fill_between(x_values, y_finetune_lower, y_finetune_upper, 
                   color=colors["Fine-tuned"], alpha=0.2)
    
    if show_average_lines:
        # Add overall average lines
        ax.axhline(vanilla_avg, color=colors["From scratch"], linestyle=':', linewidth=2, alpha=0.7, 
                label='From scratch Avg')
        ax.fill_between(x_values, [vanilla_avg_lower]*len(x_values), [vanilla_avg_upper]*len(x_values), 
                    color=colors["From scratch"], alpha=0.1)
        
        ax.axhline(finetune_avg, color=colors["Fine-tuned"], linestyle=':', linewidth=2, alpha=0.7,
                label='Fine-tuned Avg')
        ax.fill_between(x_values, [finetune_avg_lower]*len(x_values), [finetune_avg_upper]*len(x_values),
                    color=colors["Fine-tuned"], alpha=0.1)
    print (f"Vanilla Avg: {vanilla_avg:.2f} ({vanilla_avg_lower:.2f} - {vanilla_avg_upper:.2f})")
    print (f"Fine-tuned Avg: {finetune_avg:.2f} ({finetune_avg_lower:.2f} - {finetune_avg_upper:.2f})")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Training Showers', fontsize=18)
    
    # Use metric-specific y-axis label
    ylabel = _get_metric_ylabel(metric_file)
    ax.set_ylabel(f'{ylabel}', fontsize=18)

    ax.set_xticks([100, 1000, 10000, 100000])
    ax.set_xticklabels(['$10^2$', '$10^3$', '$10^4$', '$10^5$'])
    # ax.grid(False, which="major", ls="-", alpha=0.3)
    ax.grid(False)

    ax.legend(frameon=False, edgecolor='black', loc='upper right', fontsize=18)

    plt.tight_layout()
    
    if save_fig and save_path:
        save_figure(fig, save_path)
    
    plt.show()


def save_figure(fig, save_path: str):
    """Save figure in appropriate format based on file extension."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    if save_path.endswith('.pdf'):
        fig.savefig(save_path, format='pdf', bbox_inches='tight')
    elif save_path.endswith('.png'):
        fig.savefig(save_path, format='png', dpi=600, bbox_inches='tight')
    else:
        # Default to PDF
        if '.' not in save_path:
            save_path += '.pdf'
        fig.savefig(save_path, format='pdf', bbox_inches='tight')
    
    print(f"Figure saved to: {save_path}")


def run_complete_analysis(config: ExperimentConfig,
                         metric_files: List[str] = None,
                         output_dir: str = "./results/for_paper/"):
    """
    Run complete analysis for all metrics.
    Creates exactly 4 plots: convergence curves and final performance for both KL and WD.
    
    Args:
        config: Experiment configuration (from experiment_config.py)
        metric_files: List of metric files to analyze
        output_dir: Directory for saving figures
    """
    if metric_files is None:
        metric_files = ['KL_Features_all_epochs.json', 'WD_Features_all_epochs.json']
    
    loader = ExperimentDataLoader(config)
    
    # Track what plots have been created to avoid duplicates
    plots_created = set()
    
    for metric_file in metric_files:
        metric_type = 'KL' if 'KL' in metric_file else 'WD'
        
        # Skip if we already created plots for this metric type
        if metric_type in plots_created:
            print(f"Plots for {metric_type} already created, skipping...")
            continue
            
        print(f"\n{'='*60}")
        print(f"Analyzing {metric_type} metrics from {metric_file}")
        print('='*60)
        
        # Load all seeds
        vanilla_data = loader.load_all_seeds('vanilla', metric_file)
        finetune_data = loader.load_all_seeds('finetune', metric_file)
        
        if not vanilla_data or not finetune_data:
            print(f"Insufficient data for {metric_type} analysis")
            continue
        
        print(f"\nLoaded {len(vanilla_data)} vanilla seeds and {len(finetune_data)} fine-tune seeds")
        
        # Generate exactly one convergence curves plot for this metric
        print(f"Creating convergence curves plot for {metric_type}...")
        plot_convergence_curves(
            vanilla_data, finetune_data,
            metric_name='Clusters per Layer',
            metric_file=metric_file,  # Pass metric_file for y-axis label
            save_fig=True,
            save_path=f"{output_dir}/{metric_type}_convergence_curves.pdf"
        )
        
        # Generate exactly one final performance plot for this metric
        print(f"Creating final performance plot for {metric_type}...")
        plot_final_performance_comparison(
            vanilla_data, finetune_data,
            metric_name='Clusters per Layer',
            metric_file=metric_file,
            save_fig=True,
            save_path=f"{output_dir}/{metric_type}_final_performance.pdf"
        )
        
        # Mark this metric type as completed
        plots_created.add(metric_type)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Created {len(plots_created)*2} plots:")
    print('='*60)


if __name__ == "__main__":
    # This would be imported in your notebook, so remove direct execution
    print("Import this module and use create_showerflow_config() to get your configuration")