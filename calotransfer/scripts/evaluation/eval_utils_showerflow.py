import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from numpy.typing import ArrayLike
from scipy.stats import lognorm, entropy

# Handle imports based on how the module is being run
try:
    # Try absolute imports first (when running as part of package)
    from configs import Configs
    from models.shower_flow import compile_HybridTanH_model, compile_HybridTanH_model_CaloC
    import utils.plot_evaluate as plot
except ImportError:
    # Try relative imports (when module is in a package)
    try:
        from ..configs import Configs
        from ..models.shower_flow import compile_HybridTanH_model, compile_HybridTanH_model_CaloC
        from . import plot_evaluate as plot
    except ImportError:
        # If both fail, try adding parent directory to path
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from configs import Configs
        from models.shower_flow import compile_HybridTanH_model, compile_HybridTanH_model_CaloC
        import plot_evaluate as plot

# Initialize config
config = Configs()


def load_models_and_distributions(trainings_paths, device, num_blocks=5, num_inputs=95):
    models = {}
    distributions = []
    model_name_to_index = {}

    for model_name, model_path in tqdm(trainings_paths.items(), desc='Loading models', total=len(trainings_paths)):
        try:
            # Compile the model
            model, distribution = compile_HybridTanH_model_CaloC(num_blocks=num_blocks, num_inputs=num_inputs, num_cond_inputs=1, device=device)
            
            # Load the checkpoint
            checkpoint = torch.load(model_path, weights_only=True)
            
            # Load the model state dict
            model.load_state_dict(checkpoint['model'])
            
            # Set the model to evaluation mode and move it to the appropriate device
            model.eval().to(device)
            
            # Store the model in the models dictionary
            models[model_name] = model
            
            # Store the distribution in the list and map the model name to the index
            distributions.append(distribution)
            model_name_to_index[model_name] = len(distributions) - 1
            # print(f'{model_name} model loaded')
        except Exception as e:
            print(f'Error loading {model_name} model from {model_path}: {e}')

    return models, distributions, model_name_to_index

def generate_samples(distributions, model_name_to_index, cfg, energy, device, min_energy=1, max_energy=1000):
    """
    Generate samples for each distribution and store them in a dictionary.
    
    Parameters:
    distributions (list): List of distributions.
    model_name_to_index (dict): Dictionary mapping model names to indices in the distributions list.
    cfg (object): Configuration object containing min_energy and max_energy.
    energy (np.ndarray): Array of energy values.
    device (torch.device): Device to run the computations on.
    
    Returns:
    dict: Dictionary containing generated samples for each model.
    """
    samples_dict = {}
    print('\n Energy range:', min_energy, '-', max_energy)
    for model_name, distribution in tqdm(zip(model_name_to_index.keys(), distributions), desc='Generating samples', total=len(model_name_to_index)):
        low_log = np.log10(min_energy)  # convert to log space
        high_log = np.log10(max_energy)  # convert to log space
        uniform_samples = np.random.uniform(low_log, high_log, len(energy))

        # Apply exponential function (base 10)
        log_uniform_samples = np.power(10, uniform_samples)
        log_uniform_samples = (np.log(log_uniform_samples / log_uniform_samples.min()) / np.log(log_uniform_samples.max() / log_uniform_samples.min())).reshape(-1)
        cond_E = torch.tensor(log_uniform_samples).view(len(energy), 1).to(device).float()

        # Generate samples
        with torch.no_grad():
            samples = distribution.condition(cond_E).sample(torch.Size([len(energy), ])).cpu().numpy()
        
        # Store the samples in the samples dictionary
        samples_dict[model_name] = samples
    
    return samples_dict

### Plotting functions kl and wasserstein distance calculation

def plot_averaged_per_layers(e_per_layer, clusters_per_layer, samples_dict, wasserstein_distances=None, title='Averaged per layers', points_range=(2, 47), cfg=config):

    if wasserstein_distances is None:
        wasserstein_distances = {}

    # Ensure the keys exist in the dictionary
    if 'E_mean' not in wasserstein_distances:
        wasserstein_distances['E_mean'] = {}
    if '# points_mean' not in wasserstein_distances:
        wasserstein_distances['# points_mean'] = {}

    # Create a figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Define colors for different samples
    colors = plt.get_cmap('plasma')(np.linspace(0.1, .9, len(samples_dict)))

    # First subplot (E_mean [GeV])
    h1 = axs[0].hist(np.arange(45), weights=e_per_layer.mean(axis=0), bins=45, color='lightgray', label='Original')

    for i, (model_name, samples) in enumerate(samples_dict.items()):
        axs[0].hist(np.arange(45), weights=(samples[:, -45:]*cfg.sf_norm_energy).mean(axis=0), bins=h1[1], histtype='step', lw=1.5, color=colors[i % len(colors)], label=model_name)
        wd = plot.wasserstein_d(e_per_layer.mean(axis=0), (samples[:, -45:]*cfg.sf_norm_energy).mean(axis=0))
        wasserstein_distances['E_mean'][model_name] = wd

    axs[0].set_yscale('log')
    axs[0].set_xlabel('layer', fontsize=15)
    axs[0].set_ylabel('E_mean [GeV]', fontsize=15)
    axs[0].legend()

    # Second subplot (# points_mean)
    h2 = axs[1].hist(np.arange(45), weights=clusters_per_layer.mean(axis=0), bins=45, color='lightgray', label='Original')

    # Compute Wasserstein distances for # points_mean
    for i, (model_name, samples) in enumerate(samples_dict.items()):
        axs[1].hist(np.arange(45), weights=(samples[:, points_range[0]:points_range[1]]*cfg.sf_norm_points).mean(axis=0), bins=h2[1], histtype='step', lw=1.5, color=colors[i % len(colors)], label=model_name)
        wd = plot.wasserstein_d(clusters_per_layer.mean(axis=0), (samples[:, points_range[0]:points_range[1]]*cfg.sf_norm_points).mean(axis=0))
        wasserstein_distances['# points_mean'][model_name] = wd

    axs[1].set_yscale('log')
    axs[1].set_xlabel('layer', fontsize=15)
    axs[1].set_ylabel('# points_mean', fontsize=15)
    axs[1].legend()

    plt.suptitle(title, fontsize=20)

    # Display Wasserstein distances
    wd_text_e_mean = "\n".join([f"{model_name}: {wd:.4f}" for model_name, wd in wasserstein_distances['E_mean'].items()])
    wd_text_points_mean = "\n".join([f"{model_name}: {wd:.4f}" for model_name, wd in wasserstein_distances['# points_mean'].items()])
    plt.gcf().text(0.015, 0.95, f"Wasserstein Distances (E_mean):\n{wd_text_e_mean}", fontsize=12, verticalalignment='top')
    plt.gcf().text(0.62, 0.95, f"Wasserstein Distances (# points_mean):\n{wd_text_points_mean}", fontsize=12, verticalalignment='top')

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return wasserstein_distances

def plot_cog(
    df: 'DataFrame',
    df_cc: 'DataFrame',
    samples_dict: Dict[str, ArrayLike],
    coordinates: List[str] = ['x', 'z'],
    wasserstein_distances: Optional[Dict[str, Dict[str, float]]] = None,
    kl_distances: Optional[Dict[str, Dict[str, float]]] = None,
    coord_range: Tuple[int, int] = (0, 1),  # Rinominato da 'range' a 'coord_range'
    save_plots: bool = False,
    save_dir: Optional[str] = None,
    file_prefix: str = 'cog',
    epoch: int = 0,
    dpi: int = 300,
    pretrained: bool = False  # Add this parameter
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Plot and optionally save center of gravity distributions.
    """

    # Input validation
    valid_coordinates = ['x', 'y', 'z']
    if not all(coord in valid_coordinates for coord in coordinates):
        raise ValueError(f"Invalid coordinates. Must be one of {valid_coordinates}")

    # Initialize metrics dictionaries
    wasserstein_distances = wasserstein_distances or {coord: {} for coord in coordinates}
    kl_distances = kl_distances or {coord: {} for coord in coordinates}
    # Create save directory if needed
    if save_plots and save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
        except OSError as e:
            raise IOError(f"Failed to create save directory: {e}")

    # Define colors with a colormap for better scalability
    colors = plt.get_cmap('plasma')(np.linspace(0.1, .9, len(samples_dict)))

    for coord in coordinates:
        try:
            df_values = df[f'cog_{coord}'].values
            df_values_cc = df_cc[f'cog_{coord}'].values
        except KeyError as e:
            raise ValueError(f"Missing required column: {e}")

        fig = plt.figure(figsize=(12, 6))

        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

        # Create subplots
        axes = []
        for i in range(2):
            axes.append(fig.add_subplot(gs[i]))

        for ax, scale in zip(axes, ['linear', 'log']):
            h1 = ax.hist(df_values, bins=100, color='lightgrey', label='CaloChallenge', alpha=0.7)
            ax.hist(df_values_cc, bins=h1[1], color='black', label='CaloClouds',
                    histtype='step', edgecolor='black', linestyle='--', linewidth=2)

            # Plot generated samples
            for i, (model_name, samples) in enumerate(samples_dict.items()):
                sampled_values = samples[:, {'x': coord_range[0], 'z': coord_range[1]}[coord]]
                ax.hist(sampled_values, bins=h1[1], histtype='step', lw=2,
                        color=colors[i], label=model_name)

                # Compute Wasserstein distance
                if model_name not in wasserstein_distances[coord]:
                    wd = plot.wasserstein_d(df_values.flatten(), sampled_values.flatten())
                    wasserstein_distances[coord][model_name] = wd
                if model_name not in kl_distances[coord]:
                    kl = plot.quantiled_kl_divergence(df_values.flatten(), sampled_values.flatten())
                    kl_distances[coord][model_name] = kl

            ax.set_title(f'{scale.capitalize()} Scale')
            ax.set_xlim(-10, 10)
            if scale == 'log':
                ax.set_yscale('log')
            ax.legend()

        # Add Wasserstein distances text
        wd_text = "\n".join(f"{model}: {dist:.4f}"
                            for model, dist in wasserstein_distances[coord].items())
        fig.text(0.02, 0.99, f"Wasserstein Distances ({coord.upper()}):\n{wd_text}",
                 fontsize=10, verticalalignment='top')
        
        kl_text = "\n".join(f"{model}: {dist:.4f}"
                            for model, dist in kl_distances[coord].items())
        fig.text(0.82, 0.99, f"KL Distances ({coord.upper()}):\n{kl_text}",
                    fontsize=10, verticalalignment='top')

        subtitle = "Finetune" if pretrained else "Vanilla"
        plt.suptitle(f'Center of Gravity {coord.upper()}, epoch: {epoch}\n{subtitle}', fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save plot if requested
        if save_plots:
            save_path = os.path.join(save_dir or '', f'{file_prefix}_{coord}_epoch_{epoch}.png')
            try:
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                print(f"Saved plot to {save_path}")
            except Exception as e:
                print(f"Failed to save plot: {e}")

        plt.show()

    return wasserstein_distances, kl_distances

def plot_per_layer(
    per_layer: pd.DataFrame,
    samples_dict: Dict[str, np.ndarray],
    sample_range: Tuple[int, int] = (2, 47),
    wasserstein_distances: Optional[Dict[str, Dict[str, float]]] = None,
    kl_distances: Optional[Dict[str, Dict[str, float]]] = None,
    log_scale: bool = True,
    title: str = 'Clusters per Layer',
    norm: float = 1.0,
    clusters: bool = False,
    save_plots: bool = False,
    save_dir: Optional[str] = None,
    epoch: int = 0,
    dpi: int = 300,
    pretrained: bool = False  # Add this parameter
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[float]], 
           Dict[str, Dict[str, float]], Dict[str, List[float]]]:
    """
    Plot and optionally save per-layer distributions.

    Args:
        per_layer: DataFrame containing the original per-layer data
        samples_dict: Dictionary mapping model names to their generated samples
        sample_range: Tuple specifying the range of layers to plot
        wasserstein_distances: Pre-computed Wasserstein distances dictionary
        log_scale: Whether to use log scale for y-axis
        title: Plot title
        norm: Normalization factor for the data
        clusters: Whether to round the data for cluster plotting
        save_plots: Whether to save the plots
        save_dir: Directory to save the plots
        file_prefix: Prefix for saved files
        epoch: Current epoch number for plotting
        dpi: DPI for saved plots
        pretrained: Whether the model is pretrained

    Returns:
        Tuple of (wasserstein_distances, layer_wasserstein_distances)
    
    Raises:
        ValueError: If sample_range is invalid
        IOError: If save_dir is specified but cannot be created
    """
    # Input validation
    if not sample_range[0] < sample_range[1]:
        raise ValueError("Invalid sample_range: start must be less than end")

    # Create save directory if needed
    if save_plots and save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
        except OSError as e:
            raise IOError(f"Failed to create save directory: {e}")

    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(9, 5)

    # Use colormap for better scalability
    colors = plt.get_cmap('plasma')(np.linspace(0.1, .9, len(samples_dict)))

    # Initialize wasserstein_distances
    wasserstein_distances = wasserstein_distances or {}
    if title not in wasserstein_distances:
        wasserstein_distances[title] = {}
    layer_wasserstein_distances = {model_name: [] for model_name in samples_dict}

    kl_distances = kl_distances or {}
    if title not in kl_distances:
        kl_distances[title] = {}
    layer_kl_distances = {model_name: [] for model_name in samples_dict}

    num_layers = min(45, sample_range[1] - sample_range[0])
    for layer in range(num_layers):
        ax = fig.add_subplot(gs[layer])

        # Handle per_layer data: Check if it's a DataFrame or Series
        if isinstance(per_layer, pd.DataFrame):
            original_layer_data = per_layer.apply(lambda x: x[layer])  # Accessing the column for the layer
        elif isinstance(per_layer, pd.Series):
            original_layer_data = per_layer.apply(lambda x: x[layer])  # Accessing the element for the layer
        else:
            raise ValueError("per_layer must be a pandas DataFrame or Series")

        # Plot original data
        h = ax.hist(original_layer_data, bins=100, color='lightgrey', label='CaloChallenge', alpha=0.7)

        # Plot generated sample data for each model
        for j, (model_name, samples) in enumerate(samples_dict.items()):
            layer_sample_data = samples[:, sample_range[0] + layer] * norm
            if clusters:
                layer_sample_data = np.round(layer_sample_data)

            ax.hist(layer_sample_data, bins=h[1], histtype='step', lw=1,
                    color=colors[j], label=model_name)

            # Compute Wasserstein distance
            wd = plot.wasserstein_d(np.array(original_layer_data).flatten(), np.array(layer_sample_data).flatten())
            layer_wasserstein_distances[model_name].append(wd)

            kl = plot.quantiled_kl_divergence(np.array(original_layer_data).flatten(), np.array(layer_sample_data).flatten())
            layer_kl_distances[model_name].append(kl)

        # Layer number annotation
        ax.text(0.95, 0.95, str(layer + 1), transform=ax.transAxes,
                fontsize=12, va='top', ha='right')

        if layer == 2:
            ax.legend()
        if log_scale:
            ax.set_yscale('log')

    # Compute and display mean Wasserstein distances
    mean_wasserstein_distances = {
        model_name: np.mean(wd_list)
        for model_name, wd_list in layer_wasserstein_distances.items()
    }
    wasserstein_distances[title].update(mean_wasserstein_distances)

    wd_text = "\n".join(f"{model}: {wd:.4f}"
                       for model, wd in mean_wasserstein_distances.items())
    fig.text(0.02, 0.95, f"Mean Wasserstein Distances:\n{wd_text}",
             fontsize=15, verticalalignment='top')
    
    mean_kl_distances = {
        model_name: np.mean(kl_list)
        for model_name, kl_list in layer_kl_distances.items()
    }
    kl_distances[title].update(mean_kl_distances)
    kl_text = "\n".join(f"{model}: {kl:.4f}"
                        for model, kl in mean_kl_distances.items())
    fig.text(0.82, 0.95, f"Mean KL Distances:\n{kl_text}",
             fontsize=15, verticalalignment='top')
    
    subtitle = "Finetune" if pretrained else "Vanilla"
    plt.suptitle(f"{title} epoch: {epoch}\n{subtitle}", fontsize=35)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save plot if requested
    if save_plots:
        save_path = os.path.join(save_dir or '', f'{title}_epoch_{epoch}.png')
        try:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        except Exception as e:
            print(f"Failed to save plot: {e}")

    plt.show()

    return wasserstein_distances, layer_wasserstein_distances, kl_distances, layer_kl_distances

def create_epoch_summary(epoch_results: Dict[str, Any], epoch: int) -> Dict[str, Any]:
    """
    Create a summary of metrics for a single epoch.
    
    Args:
        epoch_results: Dictionary containing wasserstein and kl metrics
        epoch: Epoch number
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'epoch': epoch,
        'wasserstein': {},
        'kl': {}
    }
    
    # Calculate average Wasserstein distances
    for metric, values in epoch_results['wasserstein'].items():
        if metric in ['x', 'z']:
            # For CoG, calculate mean across models
            summary['wasserstein'][metric] = {
                'mean': np.mean(list(values.values())),
                'std': np.std(list(values.values()))
            }
        else:
            # For layer-based metrics, calculate mean across layers and models
            all_means = []
            for model_name, layer_values in values.items():
                all_means.append(np.mean(layer_values))
            summary['wasserstein'][metric] = {
                'mean': np.mean(all_means),
                'std': np.std(all_means)
            }
    
    # Calculate average KL divergences
    for metric, values in epoch_results['kl'].items():
        if metric in ['x', 'z']:
            summary['kl'][metric] = {
                'mean': np.mean(list(values.values())),
                'std': np.std(list(values.values()))
            }
        else:
            all_means = []
            for model_name, layer_values in values.items():
                all_means.append(np.mean(layer_values))
            summary['kl'][metric] = {
                'mean': np.mean(all_means),
                'std': np.std(all_means)
            }
    
    return summary

def plot_wasserstein_distances(
    wasserstein_distances, 
    colors=None, 
    title='Wasserstein Distances for Different Trainings', 
    save_plots=False, 
    save_dir=None, 
    epoch=0, 
    dpi=300,
    pretrained=False  # Add this parameter
):
    if colors is None:
        colors = plt.get_cmap('plasma')(np.linspace(0.1, .9, len(wasserstein_distances)))
    
    # Create save directory if needed
    if save_plots and save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
        except OSError as e:
            raise IOError(f"Failed to create save directory: {e}")
    
    plt.figure(figsize=(15, 8))
    
    for i, (model_name, distances) in enumerate(wasserstein_distances.items()):
        color = colors[i % len(colors)]
        
        # Plot full layer distances
        plt.plot(np.arange(1, 46), distances, marker='o', label=model_name, color=color, alpha=0.7)
        
        # Compute mean Wasserstein distance for all layers
        mean_distance = np.mean(distances)
        
        # Compute mean of last 15 values
        last_15_mean = np.mean(distances[-15:])
        
        # Add horizontal line for full mean Wasserstein distance
        plt.axhline(y=mean_distance, color=color, linestyle='--', linewidth=2, xmax=45/47)
        
        # Annotate the end of the line with average value
        plt.text(46, mean_distance, f'{mean_distance:.4f}', color=color, fontsize=16, 
                 verticalalignment='center')

    # Add labels and title with epoch
    plt.xlabel('Layer', fontsize=15)
    plt.ylabel('Wasserstein Distance', fontsize=15)
    subtitle = "Finetune" if pretrained else "Vanilla"
    full_title = f'{title} (Epoch {epoch})\n{subtitle}'
    plt.title(full_title, fontsize=18)
    plt.legend(fontsize=22, loc='upper right')
    plt.yscale('log')
    plt.ylim(1e-2, 1e2)

    # Set x-ticks to display all numbers from 1 to 45
    plt.xticks(np.arange(1, 46))
    plt.xlim(0.5, 47)  # Extended to make room for annotations
    plt.grid(linestyle='--', alpha=0.7)
    
    # Add last 15 layers mean values in the corner
    last_15_text = "Last 15 Layers Mean:\n"
    for model_name, distances in wasserstein_distances.items():
        last_15_mean = np.mean(distances[-15:])
        last_15_text += f"{model_name}: {last_15_mean:.4f}\n"
    
    plt.text(0.02, 0.98, last_15_text.strip(), 
             transform=plt.gca().transAxes, 
             verticalalignment='top', 
             horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
             fontsize=24)
    
    # Save plot if requested
    if save_plots:
        save_path = os.path.join(save_dir or '', f'{title}_epoch_{epoch}.png')
        try:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        except Exception as e:
            print(f"Failed to save plot: {e}")
    
    # Display the plot
    plt.tight_layout()
    plt.show()

def plot_wasserstein_distances_features(
    wd_per_epoch, 
    trainings_paths=None, 
    energy=None, 
    title='Wasserstein Distances per Epoch',
    figsize=(20, 20), 
    metrics=None,
    pretrained=False,
    save_plots=False,
    save_dir=None,
    dpi=300
):
    """
    Create a comprehensive visualization of Wasserstein Distances across epochs for different metrics.
    
    Parameters:
    -----------
    wd_per_epoch : dict
        Dictionary containing Wasserstein Distance metrics for each epoch
    trainings_paths : dict, optional
        Dictionary of training paths/labels
    energy : numpy.ndarray, optional
        Energy array used for title annotation
    figsize : tuple, optional
        Figure size, defaults to (20, 20)
    metrics : list, optional
        List of metrics to plot. If None, uses default metrics or all available metrics
    pretrained : bool, optional
        Flag to indicate if the data is from a pretrained model
    save_plots : bool, optional
        Flag to save the plot as a PNG
    save_dir : str, optional
        Directory to save the plot
    dpi : int, optional
        Resolution of the saved plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object
    """
    # Determine metrics to plot
    if metrics is None:
        default_metrics = ['x', 'z', 'Clusters per Layer', 'Energy per Layer']
        
        # Check if these metrics exist in the data
        available_metrics = list(next(iter(wd_per_epoch.values())).keys())
        metrics = [m for m in default_metrics if m in available_metrics]
        
        if not metrics:
            metrics = available_metrics

    # Prepare data
    epochs = list(wd_per_epoch.keys())
    
    # If no trainings_paths provided, extract from data
    if trainings_paths is None:
        trainings_paths = {label: label for label in next(iter(wd_per_epoch.values()))[metrics[0]].keys()}
    
    labels = list(trainings_paths.keys())

    # Extract data for plotting - handle list or single value
    data = {
        label: {
            metric: [
                float(wd_per_epoch[epoch][metric][label][0] if isinstance(wd_per_epoch[epoch][metric][label], list)
                      else wd_per_epoch[epoch][metric][label])
                for epoch in epochs
            ]
            for metric in metrics
        }
        for label in labels
    }

    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Title with energy information and training type
    subtitle = "Finetuned" if pretrained else "Vanilla"
    if energy is not None:
        e_min, e_max = np.round(energy.min()).astype(int), np.round(energy.max()).astype(int)
        fig.suptitle(f'{title} \n Incident Energies: {e_min}-{e_max} \n {subtitle}', 
                     fontsize=40, weight='bold')
    else:
        fig.suptitle(f'Normalised WD per Epoch \n {subtitle}', fontsize=40, weight='bold')

    # Color cycle
    colors = plt.get_cmap('plasma')(np.linspace(0.1, .9, len(labels)))

    # Individual metric plots
    for i, metric in enumerate(metrics):
        ax = plt.subplot(3, 2, i+1)
        
        # Plot each training path
        for j, (label, label_data) in enumerate(data.items()):
            ax.plot(epochs, label_data[metric], 
                    marker='o', color=colors[j % len(colors)], 
                    label=label, markersize=10, linewidth=2)
        
        ax.set_title(f'{metric}', fontsize=34)
        if i % 2 == 0:
            ax.set_ylabel('Normalised WD', fontsize=30)
        ax.set_yscale('log')
        ax.set_ylim(1e-2, 1e1)
        ax.grid(True, linestyle='--', alpha=1)
        ax.tick_params(axis='y', labelsize=20)
        ax.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))
        ax.tick_params(axis='x', labelsize=20) 
         # Increase the font size for x-axis tick labels        
        # Aggiungi la legenda solo se ci sono artisti con etichette
        handles, labels_legend = ax.get_legend_handles_labels()
        # if handles:
        #     ax.legend(fontsize=26)

    # Average plot
    ax = plt.subplot(3, 2, (5, 6))
    
    # Compute mean across metrics for each label
    mean_data = {
        label: [
            float(np.mean([data[label][metric][i] for metric in metrics]))
            for i in range(len(epochs))
        ]
        for label in labels
    }

    # Plot averaged metrics
    for j, (label, mean_values) in enumerate(mean_data.items()):
        ax.plot(epochs, mean_values, marker='o', 
                color=colors[j % len(colors)], label=label, 
                markersize=10, linewidth=2)

    ax.set_title('Averaged', fontsize=34)
    ax.set_xlabel('Epochs', fontsize=30)
    ax.set_ylabel('Normalised WD', fontsize=26)
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e1)
    
    # Aggiungi la legenda solo se ci sono artisti con etichette
    handles, labels_legend = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=26, loc='center left', bbox_to_anchor=(1, 0.5))
        
    ax.grid(True, linestyle='--', alpha=1)
    ax.tick_params(axis='y', labelsize=20)
    ax.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))
    ax.tick_params(axis='x', labelsize=20) 

    plt.tight_layout()

    # Save plot if requested
    if save_plots:
        if save_dir is None:
            save_dir = '.'  # Current directory if not specified
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate filename
        filename = f'{title}_{subtitle.lower()}.png'
        filepath = os.path.join(save_dir, filename)
        
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {filepath}")

    plt.show()
    
    return fig  # Return the figure object

def compute_layer_distribution(
    per_layer: np.ndarray, 
    max_layers: Optional[int] = None, 
    normalize: Union[bool, float] = True,
    verbose: bool = True
) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Compute distribution parameters for each layer.
    
    Parameters:
    -----------
    per_layer : np.ndarray
        2D array of values per layer
    max_layers : int, optional
        Number of layers to analyze (default: all layers)
    normalize : bool or float, optional
        - If True: normalize by max value of input array
        - If False: use raw data
        - If float: normalize by this specific value
    verbose : bool, optional
        Whether to print detailed layer information (default: True)
    
    Returns:
    --------
    Dict with layer information: {layer_name: (mu, sigma, min_val, max_val)}
    """
    # Set default max_layers if not specified
    max_layers = max_layers or per_layer.shape[1]
    max_layers = min(max_layers, per_layer.shape[1])
    
    # Prepare storage for analysis results
    mu_sigma_values = []
    min_max_values = []
    
    # Determine normalization factor
    if normalize is True:
        data_max = per_layer.max()
    elif normalize is False:
        data_max = 1.0
    elif isinstance(normalize, (int, float)):
        data_max = float(normalize)
    else:
        raise ValueError("normalize must be bool or float")
    
    # Analyze each layer
    for i in range(max_layers):
        # Extract and preprocess data
        data = per_layer[:, i].flatten()
        data = data[data > 0]
        
        # Normalize data
        normalized_data = data / data_max
        
        if len(normalized_data) > 0:
            # Fit log-normal distribution
            shape, loc, scale = lognorm.fit(normalized_data, floc=0)
            mu = np.log(scale)
            sigma = shape
            
            mu_sigma_values.append((mu, sigma))
            
            # Calculate min and max values
            min_val = normalized_data.min()
            max_val = normalized_data.max()
            min_max_values.append((min_val, max_val))
    
    # Create layer dictionary
    mu_sigma_layer = {}
    for i, ((mu, sigma), (min_val, max_val)) in enumerate(zip(mu_sigma_values, min_max_values)):
        layer_name = f'Layer {i+1}'
        mu_sigma_layer[layer_name] = (mu, sigma, min_val, max_val)
        
        if verbose:
            print(f'{layer_name}: mu = {mu:.2f}, sigma = {sigma:.2f}, min = {min_val:.2f}, max = {max_val:.2f}')
    
    return mu_sigma_layer

def plot_layer_distribution(
    per_layer: np.ndarray,
    max_layers: Optional[int] = None,
    title: str = 'Distribution Across Layers'
):
    """
    Plot histograms and mu-sigma trends for layer distribution.
    
    Parameters:
    -----------
    per_layer : np.ndarray
        2D array of original values per layer
    max_layers : int, optional
        Number of layers to analyze (default: all layers)
    title : str, optional
        Title for the plots (default: 'Distribution Across Layers')
    """
    # Set default max_layers if not specified
    max_layers = max_layers or per_layer.shape[1]
    max_layers = min(max_layers, per_layer.shape[1])
    
    # Histogram plot
    fig = plt.figure(figsize=(15, 18))
    gs = gridspec.GridSpec(9, 5)
    
    # Prepare lists for mu and sigma for trend plot
    mu_values = []
    sigma_values = []
    
    # Iterate through layers
    for i in range(max_layers):
        # Extract data
        data = per_layer[:, i].flatten()
        data = data[data > 0]
        
        if len(data) > 0:
            # Fit log-normal distribution
            shape, loc, scale = lognorm.fit(data, floc=0)
            mu = np.log(scale)
            sigma = shape
            
            # Plot histogram
            ax = fig.add_subplot(gs[i])
            plt.hist(data, bins=100, color='lightgrey', density=True)
            
            # Plot fitted distribution
            x = np.linspace(data.min(), data.max(), len(per_layer))
            pdf = lognorm.pdf(x, sigma, loc=0, scale=np.exp(mu))
            plt.plot(x, pdf, 'r-', lw=2)
            
            plt.yscale('log')
            plt.title(rf'Layer {i+1}' + '\n' + rf'$\mu={mu:.2f}$, $\sigma={sigma:.2f}$' + '\n', fontsize=18)
            
            # Store mu and sigma for trend plot
            mu_values.append(mu)
            sigma_values.append(sigma)
    
    plt.suptitle(title, fontsize=35)
    plt.tight_layout()
    plt.show()
    
    # Mu-sigma trend plot
    plt.figure(figsize=(12, 6))
    layers = np.arange(1, len(mu_values) + 1)
    
    plt.plot(layers, mu_values, marker='o', linestyle='-', color='blue', label='mu')
    plt.plot(layers, sigma_values, marker='s', linestyle='--', color='red', label='sigma')
    
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.xlim(1, len(layers))
    plt.title('Mu and Sigma Values for Each Layer', fontsize=20)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

# TODO: new part, cleaned up here in the future 

def plot_metric_generic(
    distances_dict, 
    colors=None, 
    title='Distances for Different Trainings',
    metric_name='Distance',
    save_plots=False, 
    save_dir=None, 
    epoch=0, 
    dpi=300,
    pretrained=False,
    y_limits=(1e-4, 1e2),  # Customizable y-axis limits
    file_suffix=''  # Additional suffix for filename
):
    """
    Generic function to plot distances (Wasserstein or KL) across layers.
    
    Parameters:
    -----------
    distances_dict : dict
        Dictionary of distances for each model
    colors : array-like, optional
        Colors for each model
    title : str
        Plot title
    metric_name : str
        Name of the metric (e.g., 'Wasserstein Distance', 'KL Divergence')
    save_plots : bool
        Whether to save the plot
    save_dir : str
        Directory to save plots
    epoch : int
        Current epoch number
    dpi : int
        Resolution for saved plots
    pretrained : bool
        Whether model is pretrained
    y_limits : tuple
        Y-axis limits for the plot
    file_suffix : str
        Additional suffix for saved filename
    """
    if colors is None:
        colors = plt.get_cmap('plasma')(np.linspace(0.1, .9, len(distances_dict)))
    
    # Create save directory if needed
    if save_plots and save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
        except OSError as e:
            raise IOError(f"Failed to create save directory: {e}")
    
    plt.figure(figsize=(15, 8))
    
    for i, (model_name, distances) in enumerate(distances_dict.items()):
        color = colors[i % len(colors)]
        
        # Plot full layer distances
        plt.plot(np.arange(1, 46), distances, marker='o', label=model_name, color=color, alpha=0.7)
        
        # Compute mean distance for all layers
        mean_distance = np.mean(distances)
        
        # Add horizontal line for full mean distance
        plt.axhline(y=mean_distance, color=color, linestyle='--', linewidth=2, xmax=45/47)
        
        # Annotate the end of the line with average value
        plt.text(46, mean_distance, f'{mean_distance:.4f}', color=color, fontsize=16, 
                 verticalalignment='center')

    # Add labels and title with epoch
    plt.xlabel('Layer', fontsize=15)
    plt.ylabel(metric_name, fontsize=15)
    subtitle = "Finetune" if pretrained else "Vanilla"
    full_title = f'{title} (Epoch {epoch})\n{subtitle}'
    plt.title(full_title, fontsize=18)
    plt.legend(fontsize=22, loc='upper right')
    plt.yscale('log')
    plt.ylim(y_limits[0], y_limits[1])

    # Set x-ticks to display all numbers from 1 to 45
    plt.xticks(np.arange(1, 46))
    plt.xlim(0.5, 47)  # Extended to make room for annotations
    plt.grid(linestyle='--', alpha=0.7)
    
    # Add last 15 layers mean values in the corner
    last_15_text = "Last 15 Layers Mean:\n"
    for model_name, distances in distances_dict.items():
        last_15_mean = np.mean(distances[-15:])
        last_15_text += f"{model_name}: {last_15_mean:.4f}\n"
    
    plt.text(0.02, 0.98, last_15_text.strip(), 
             transform=plt.gca().transAxes, 
             verticalalignment='top', 
             horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
             fontsize=24)
    
    # Save plot if requested
    if save_plots:
        filename = f'{title.replace(" ", "_")}{file_suffix}_epoch_{epoch}.png'
        save_path = os.path.join(save_dir or '', filename)
        try:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        except Exception as e:
            print(f"Failed to save plot: {e}")
    
    # Display the plot
    plt.tight_layout()
    plt.show()

def plot_wasserstein_distances(
    wasserstein_distances, 
    colors=None, 
    title='Wasserstein Distances for Different Trainings', 
    save_plots=False, 
    save_dir=None, 
    epoch=0, 
    dpi=300,
    pretrained=False
):
    """Wrapper function for backward compatibility - plots Wasserstein distances."""
    return plot_metric_generic(
        wasserstein_distances,
        colors=colors,
        title=title,
        metric_name='Wasserstein Distance',
        save_plots=save_plots,
        save_dir=save_dir,
        epoch=epoch,
        dpi=dpi,
        pretrained=pretrained,
        y_limits=(1e-2, 1e2),
        file_suffix='_WD'
    )

# New function for KL distances using the generic plotter
def plot_kl_distances(
    kl_distances, 
    colors=None, 
    title='KL Divergences for Different Trainings', 
    save_plots=False, 
    save_dir=None, 
    epoch=0, 
    dpi=300,
    pretrained=False
):
    """Plots KL divergences using the generic distance plotter."""
    return plot_metric_generic(
        kl_distances,
        colors=colors,
        title=title,
        metric_name='KL Divergence',
        save_plots=save_plots,
        save_dir=save_dir,
        epoch=epoch,
        dpi=dpi,
        pretrained=pretrained,
        y_limits=(1e-4, 1e0),
        file_suffix='_KL'
    )

def plot_distances_features_generic(
    distances_per_epoch, 
    trainings_paths=None, 
    energy=None, 
    title='Distances per Epoch',
    metric_name='Distance',
    figsize=(20, 20), 
    metrics=None,
    pretrained=False,
    save_plots=False,
    save_dir=None,
    dpi=300,
    y_limits=(1e-4, 1e1),
    file_suffix=''
):
    """
    Generic function to create comprehensive visualization of distances across epochs.
    
    Parameters:
    -----------
    distances_per_epoch : dict
        Dictionary containing distance metrics for each epoch
    trainings_paths : dict, optional
        Dictionary of training paths/labels
    energy : numpy.ndarray, optional
        Energy array used for title annotation
    title : str
        Main title for the plot
    metric_name : str
        Name of the metric being plotted
    figsize : tuple
        Figure size
    metrics : list, optional
        List of metrics to plot
    pretrained : bool
        Flag for pretrained models
    save_plots : bool
        Whether to save the plot
    save_dir : str
        Directory to save plots
    dpi : int
        Resolution for saved plots
    y_limits : tuple
        Y-axis limits
    file_suffix : str
        Additional suffix for filename
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object
    """
    # Determine metrics to plot
    if metrics is None:
        default_metrics = ['x', 'z', 'Clusters per Layer', 'Energy per Layer']
        
        # Check if these metrics exist in the data
        available_metrics = list(next(iter(distances_per_epoch.values())).keys())
        metrics = [m for m in default_metrics if m in available_metrics]
        
        if not metrics:
            metrics = available_metrics

    # Prepare data
    epochs = list(distances_per_epoch.keys())
    
    # If no trainings_paths provided, extract from data
    if trainings_paths is None:
        trainings_paths = {label: label for label in next(iter(distances_per_epoch.values()))[metrics[0]].keys()}
    
    labels = list(trainings_paths.keys())

    # Extract data for plotting - handle list or single value
    data = {
        label: {
            metric: [
                float(distances_per_epoch[epoch][metric][label][0] 
                      if isinstance(distances_per_epoch[epoch][metric][label], list)
                      else distances_per_epoch[epoch][metric][label])
                for epoch in epochs
            ]
            for metric in metrics
        }
        for label in labels
    }

    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Title with energy information and training type
    subtitle = "Finetuned" if pretrained else "Vanilla"
    if energy is not None:
        e_min, e_max = np.round(energy.min()).astype(int), np.round(energy.max()).astype(int)
        fig.suptitle(f'{title} \n Incident Energies: {e_min}-{e_max} \n {subtitle}', 
                     fontsize=40, weight='bold')
    else:
        fig.suptitle(f'{metric_name} per Epoch \n {subtitle}', fontsize=40, weight='bold')

    # Color cycle
    colors = plt.get_cmap('plasma')(np.linspace(0.1, .9, len(labels)))

    # Individual metric plots
    for i, metric in enumerate(metrics):
        ax = plt.subplot(3, 2, i+1)
        
        # Plot each training path
        for j, (label, label_data) in enumerate(data.items()):
            ax.plot(epochs, label_data[metric], 
                    marker='o', color=colors[j % len(colors)], 
                    label=label, markersize=10, linewidth=2)
        
        ax.set_title(f'{metric}', fontsize=34)
        if i % 2 == 0:
            ax.set_ylabel(metric_name, fontsize=30)
        ax.set_yscale('log')
        ax.set_ylim(y_limits[0], y_limits[1])
        ax.grid(True, linestyle='--', alpha=1)
        ax.tick_params(axis='y', labelsize=20)
        ax.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))
        ax.tick_params(axis='x', labelsize=20)

    # Average plot
    ax = plt.subplot(3, 2, (5, 6))
    
    # Compute mean across metrics for each label
    mean_data = {
        label: [
            float(np.mean([data[label][metric][i] for metric in metrics]))
            for i in range(len(epochs))
        ]
        for label in labels
    }

    # Plot averaged metrics
    for j, (label, mean_values) in enumerate(mean_data.items()):
        ax.plot(epochs, mean_values, marker='o', 
                color=colors[j % len(colors)], label=label, 
                markersize=10, linewidth=2)

    ax.set_title('Averaged', fontsize=34)
    ax.set_xlabel('Epochs', fontsize=30)
    ax.set_ylabel(metric_name, fontsize=26)
    ax.set_yscale('log')
    ax.set_ylim(y_limits[0], y_limits[1])
    
    # Add legend if there are handles
    handles, labels_legend = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=26, loc='center left', bbox_to_anchor=(1, 0.5))
        
    ax.grid(True, linestyle='--', alpha=1)
    ax.tick_params(axis='y', labelsize=20)
    ax.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))
    ax.tick_params(axis='x', labelsize=20)

    plt.tight_layout()

    # Save plot if requested
    if save_plots:
        if save_dir is None:
            save_dir = '.'
        
        os.makedirs(save_dir, exist_ok=True)
        
        filename = f'{title.replace(" ", "_")}{file_suffix}_{subtitle.lower()}.png'
        filepath = os.path.join(save_dir, filename)
        
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {filepath}")

    plt.show()
    
    return fig

# Backward compatibility wrapper for Wasserstein distances features
def plot_wasserstein_distances_features(
    wd_per_epoch, 
    trainings_paths=None, 
    energy=None, 
    title='Wasserstein Distances per Epoch',
    figsize=(20, 20), 
    metrics=None,
    pretrained=False,
    save_plots=False,
    save_dir=None,
    dpi=300
):
    """Wrapper function for backward compatibility - plots Wasserstein distance features."""
    return plot_distances_features_generic(
        wd_per_epoch,
        trainings_paths=trainings_paths,
        energy=energy,
        title=title,
        metric_name='Normalised WD',
        figsize=figsize,
        metrics=metrics,
        pretrained=pretrained,
        save_plots=save_plots,
        save_dir=save_dir,
        dpi=dpi,
        y_limits=(1e-2, 1e1),
        file_suffix='_WD'
    )

# New function for KL distances features
def plot_kl_distances_features(
    kl_per_epoch, 
    trainings_paths=None, 
    energy=None, 
    title='KL Divergences per Epoch',
    figsize=(20, 20), 
    metrics=None,
    pretrained=False,
    save_plots=False,
    save_dir=None,
    dpi=300
):
    """Plots KL divergence features using the generic distance features plotter."""
    return plot_distances_features_generic(
        kl_per_epoch,
        trainings_paths=trainings_paths,
        energy=energy,
        title=title,
        metric_name='KL Divergence',
        figsize=figsize,
        metrics=metrics,
        pretrained=pretrained,
        save_plots=save_plots,
        save_dir=save_dir,
        dpi=dpi,
        y_limits=(1e-4, 1e0),
        file_suffix='_KL'
    )

# fitting techniques used during the training process
def generate_log_normal_noise(
    size: int, 
    mu_sigma_dict: Dict[str, Tuple[float, float, float, float]],
    layer_range: Optional[Union[slice, tuple]] = None
) -> np.ndarray:
    """
    Generate log-normal noise samples for specified layer range.
    
    Parameters:
    -----------
    size : int
        Number of samples to generate
    mu_sigma_dict : dict
        Distribution parameters for each layer
    layer_range : slice or tuple, optional
        Range of layers to generate noise for
        - slice(start, stop): Python-style slicing
        - tuple(start, stop): Inclusive range
    
    Returns:
    --------
    numpy.ndarray of noise samples
    """
    # Convert layer range to slice if tuple is provided
    if isinstance(layer_range, tuple):
        layer_range = slice(layer_range[0]-1, layer_range[1])
    
    # Select layers based on range or use all layers
    if layer_range is None:
        selected_layers = list(mu_sigma_dict.items())
    else:
        selected_layers = list(mu_sigma_dict.items())[layer_range]
    
    # Extract distribution parameters
    mus = [params[0] for _, params in selected_layers]
    sigmas = [params[1] for _, params in selected_layers]
    min_vals = [params[2] for _, params in selected_layers]
    max_vals = [params[3] for _, params in selected_layers]
    layer_names = [name for name, _ in selected_layers]
    
    # Vectorized noise generation
    def generate_layer_noise(mu, sigma, min_val, max_val):
        dist = stats.lognorm(s=sigma, scale=np.exp(mu))
        cdf_min = dist.cdf(min_val)
        cdf_max = dist.cdf(max_val)
        u = np.random.uniform(cdf_min, cdf_max, size)
        return dist.ppf(u)
    
    # Generate noise for selected layers
    noises = np.array([
        generate_layer_noise(mu, sigma, min_val, max_val)
        for mu, sigma, min_val, max_val in zip(mus, sigmas, min_vals, max_vals)
    ]).T
    
    return noises, layer_names

def plot_log_normal_distributions(
    log_normal_noises: np.ndarray, 
    layer_names: list,
    mu_sigma_dict: Dict[str, Tuple[float, float, float, float]],
    title: str = 'Energy per Layer Noise Distributions'
):
    """
    Create grid plot for log-normal noise distributions.
    
    Parameters:
    -----------
    log_normal_noises : numpy.ndarray
        Noise samples for each layer
    layer_names : list
        Names of layers being plotted
    mu_sigma_dict : dict
        Full distribution parameters
    """
    # Dynamic subplot grid calculation
    num_layers = len(layer_names)
    cols = 5
    rows = (num_layers + cols - 1) // cols
    
    # Create figure with dynamic subplots
    fig, axs = plt.subplots(rows, cols, figsize=(15, 3*rows))
    axs_flat = axs.flatten() if hasattr(axs, 'flatten') else [axs]
    
    # Plot distributions with comprehensive formatting
    for i, (noise, layer) in enumerate(zip(log_normal_noises.T, layer_names)):
        ax = axs_flat[i]
        mu, sigma, min_val, max_val = mu_sigma_dict[layer]
        
        ax.hist(noise, bins=100, color='tab:red', alpha=0.7, log=True)
        ax.set_title(f'{layer}\nμ={mu:.2f}, σ={sigma:.2f}', fontsize=18)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        
        # Add min/max value lines
        ax.axvline(min_val, color='blue', linestyle='--', label='Min Value')
        ax.axvline(max_val, color='green', linestyle='--', label='Max Value')
        ax.set_xlim(min_val, max_val)
        # ax.legend()
    
    # Remove extra subplots
    for j in range(i+1, len(axs_flat)):
        fig.delaxes(axs_flat[j])
    
    fig.suptitle(title, fontsize=26)
    plt.tight_layout()
    plt.show()
