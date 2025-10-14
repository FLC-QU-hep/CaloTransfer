import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import pandas as pd
from typing import List, Dict, Any
import re
import warnings
import pandas as pd
from collections import OrderedDict
import os
import json
import seaborn as sns
import torch

def load_metric(output_dir, strategy, metric_name):
    file_path = os.path.join(output_dir, f'{metric_name}_{strategy}.json')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            metric = json.load(f)
        return metric
    else:
        return None

def plot_wasserstein_distances_features(
    wd_per_epoch, 
    trainings_paths=None, 
    energy=None, 
    title='Wasserstein Distances per Epoch',
    figsize=(20, 20), 
    metrics=None,
    subtitle='',
    save_plots=False,
    save_dir=None,
    dpi=300,
    show_legend=True
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
    save_plots : bool, optional
        Flag to save the plot as a PNG
    save_dir : str, optional
        Directory to save the plot
    dpi : int, optional
        Resolution of the saved plot
    show_legend : bool, optional
        Flag to activate or deactivate the legend
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object
    """
    
    # First, let's check the structure of wd_per_epoch to determine metrics
    first_epoch_data = next(iter(wd_per_epoch.values()))
    
    # Determine metrics to plot based on actual data structure
    if metrics is None:
        # Check if first_epoch_data is already a dictionary of metrics
        if isinstance(first_epoch_data, dict):
            available_metrics = list(first_epoch_data.keys())
            # Try default metrics first, if they exist
            default_metrics = ['x', 'z', 'Clusters per Layer', 'Energy per Layer']
            metrics = [m for m in default_metrics if m in available_metrics]
            # If none of the defaults are available, use all available metrics
            if not metrics:
                metrics = available_metrics
        else:
            # If the data structure is different, create a single default metric
            metrics = ['metric']  # Generic name if no specific metrics exist
    
    # Prepare data
    epochs = list(wd_per_epoch.keys())
    
    # Handle different data structures for trainings_paths
    if trainings_paths is None:
        if metrics and isinstance(first_epoch_data, dict) and metrics[0] in first_epoch_data and isinstance(first_epoch_data[metrics[0]], dict):
            # Original expected structure: epoch -> metric -> label -> value
            trainings_paths = {label: label for label in first_epoch_data[metrics[0]].keys()}
        else:
            # Alternative structure: just use a single default label
            trainings_paths = {'data': 'data'}
    
    labels = list(trainings_paths.keys())

    # Extract data for plotting with more robust handling of different structures
    data = {}
    for label in labels:
        data[label] = {}
        for metric in metrics:
            metric_data = []
            for epoch in epochs:
                # Handle different data structures
                if isinstance(wd_per_epoch[epoch], dict) and metric in wd_per_epoch[epoch]:
                    # Standard structure: epoch -> metric -> label -> value
                    if isinstance(wd_per_epoch[epoch][metric], dict) and label in wd_per_epoch[epoch][metric]:
                        value = wd_per_epoch[epoch][metric][label]
                    else:
                        # If label is not a key, assume the metric value itself is what we want
                        value = wd_per_epoch[epoch][metric]
                else:
                    # Simplest case: the epoch data is the value directly
                    value = wd_per_epoch[epoch]
                
                # Extract the actual float value, handling both list and direct value cases
                if isinstance(value, list):
                    metric_data.append(float(value[0]) if value else 0.0)
                else:
                    metric_data.append(float(value))
            
            data[label][metric] = metric_data

    # Create figure
    fig = plt.figure(figsize=figsize)
    
    if energy is not None:
        e_min, e_max = np.round(energy.min()).astype(int), np.round(energy.max()).astype(int)
        fig.suptitle(f'{title} \n Incident Energies: {e_min}-{e_max} \n {subtitle}', 
                     fontsize=40, weight='bold')
    else:
        fig.suptitle(f'Normalied Wasserstein Distances per Grad Steps \n {subtitle}', fontsize=40, weight='bold')

    # Use viridis colormap
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(labels)))

    # Individual metric plots
    for i, metric in enumerate(metrics):
        ax = plt.subplot(3, 3, i+1)
        
        # Plot each training path
        for j, (label, label_data) in enumerate(data.items()):
            if metric in label_data:
                ax.plot(epochs, label_data[metric], 
                        marker='o', color=colors[j], 
                        label=label, markersize=10, linewidth=2)
        
        ax.set_title(f'{metric}', fontsize=34)
        if i % 3 == 0:
            ax.set_ylabel('Normalised WD', fontsize=30)
        ax.set_yscale('log')
        ax.set_ylim(5e-3, 1e1)
        ax.grid(True, which="both",linestyle='--', alpha=1)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))
        
    # Average plot
    ax = plt.subplot(3, 3, (7, 9))
    
    # Compute mean across metrics for each label
    mean_data = {}
    for label in labels:
        if metrics and all(metric in data[label] for metric in metrics):
            mean_data[label] = [
                float(np.mean([data[label][metric][i] for metric in metrics]))
                for i in range(len(epochs))
            ]
    
    # Plot averaged metrics
    for j, (label, mean_values) in enumerate(mean_data.items()):
        ax.plot(epochs, mean_values, marker='o', 
                color=colors[j], label=label, 
                markersize=10, linewidth=2)

    ax.set_title('Averaged', fontsize=34)
    ax.set_xlabel('Grad Steps', fontsize=30)
    ax.set_ylabel('Normalised WD', fontsize=26)
    ax.set_yscale('log')
    ax.set_ylim(1e-1, 2)
    
    # Add legend only if there are artists with labels and show_legend is True
    if show_legend:
        handles, labels_legend = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=25, bbox_to_anchor=(1.05, 1), loc='upper left', title='# Showers', title_fontsize=30)
        
    ax.grid(True, which="both", linestyle='--', alpha=1)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))

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

    return fig  # Return the figure object
  
def convert_d_key_to_training_showers(d_key):
    """Convert a key like 'training_showers=1x10^n' to a float value."""
    d_str = d_key.split('=')[1].strip()
    if 'x' in d_str:
        base, exp_part = d_str.split('x')
        base = float(base.strip())
        exp = int(exp_part.strip().replace('10^', ''))
        return base * (10 ** exp)
    else:
        return float(d_str)
    
def check_reorganized_data(name: str, data: Dict[str, List[Dict[str, Any]]]):
    print(f"\n=== Sanity Check: {name} ===")
    
    total_series = sum(len(series_list) for series_list in data.values())
    print(f"Total metrics: {len(data)}, total series across all metrics: {total_series}\n")
    
    for metric, series_list in data.items():
        n_series = len(series_list)
        if n_series == 0:
            print(f"  ⚠️  {metric}: no series found!")
            continue
        
        # Gather lengths of x and y for all series
        lengths_x = [len(s['x']) for s in series_list]
        lengths_y = [len(s['y']) for s in series_list]
        labels = [s['label'] for s in series_list]
        
        print(f"  ✅  {metric}:")
        print(f"       • Series count: {n_series}")
        print(f"       • Labels: {', '.join(labels)}")
        print(f"       • x-lengths: min={min(lengths_x)}, max={max(lengths_x)}, mean={sum(lengths_x)/n_series:.1f}")
        print(f"       • y-lengths: min={min(lengths_y)}, max={max(lengths_y)}, mean={sum(lengths_y)/n_series:.1f}")
        
        # Check if each series is sorted in x
        unsorted = []
        for s in series_list:
            if not all(s['x'][i] <= s['x'][i+1] for i in range(len(s['x'])-1)):
                unsorted.append(s['label'])
        if unsorted:
            print(f"       • ⚠️ Unsorted x-values in series: {', '.join(unsorted)}")
        else:
            print("       • All series x-values are sorted")

def reorganize_metrics(
    config_names: List[str],
    results: List[Dict[str, Any]],
    metric_names: List[str],
    name: str = 'Reorganized Data',
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Reorganize metric data for plotting, and assign colors.

    Args:
        config_names: List of strategy/config names.
        results: List of results dictionaries (e.g., from wasserstein or kl).
        metric_names: List of metric names to extract.
        convert_key_fn: Function to convert d_key to x-axis value.

    Returns:
        A dictionary mapping each metric name to a list of dicts with keys:
            - label
            - x
            - y
            - color
    """
    reorganized_data = {metric: [] for metric in metric_names}

    for cfg_name, result in zip(config_names, results):
        for metric_idx, metric in enumerate(metric_names):
            metric_data = {
                'label': cfg_name,
                'x':   [],
                'y':   [],
                'color': None
            }
            for d_key, d_data in result.items():
                x = convert_d_key_to_training_showers(d_key)
                y = d_data['values'][metric_idx]
                metric_data['x'].append(x)
                metric_data['y'].append(y)
            # sort by x
            sorted_idx = np.argsort(metric_data['x'])
            metric_data['x'] = np.array(metric_data['x'])[sorted_idx]
            metric_data['y'] = np.array(metric_data['y'])[sorted_idx]
            reorganized_data[metric].append(metric_data)

    # assign colors
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(config_names)))
    for metric in metric_names:
        for i, data in enumerate(reorganized_data[metric]):
            data['color'] = colors[i]
    
    check_reorganized_data(name, reorganized_data)

    return reorganized_data

def analyze_and_plot_metrics(*wd_per_epoch_dicts,
                             labels=None, colors=None,
                             avg_method='geometric'):
    """
    Accepts one or more dictionaries of metrics, processes them, and produces a comparative plot.

    avg_method: 'simple' for arithmetic mean (default),
                'geometric' for geometric mean
    """

    def process_metrics(wd_per_epoch):
        d_metrics = {}
        for step, metrics_dict in wd_per_epoch.items():
            for _, d_values in metrics_dict.items():
                for d_key, value in d_values.items():
                    d_metrics.setdefault(d_key, {})\
                             .setdefault(step, []).append(value)
        results = {}
        for d_key, steps in d_metrics.items():
            averages = {st: np.mean(vals) for st, vals in steps.items()}
            min_step = min(averages, key=averages.get)
            results[d_key] = {
                'step': min_step,
                'average': averages[min_step],
                'values': steps[min_step]
            }
        return results

    def d_key_to_value(d_key):
        p = d_key.split(' x 10^')
        b = float(p[0].split(' = ')[1])
        e = int(p[1])
        return b * 10**e

    n_models = len(wd_per_epoch_dicts)
    if labels is None:
        labels = [f"Model {i+1}" for i in range(n_models)]
    if colors is None:
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_models))[:n_models]

    # 1) process
    results_list = [process_metrics(m) for m in wd_per_epoch_dicts]

    # 2) collect keys
    all_keys = set()
    for r in results_list:
        all_keys |= set(r.keys())
    sorted_d_keys = sorted(all_keys, key=d_key_to_value)

    # 3) build per-model arrays
    averages_all, steps_all = [], []
    for r in results_list:
        avgs = [r[d]['average'] if d in r else np.nan for d in sorted_d_keys]
        sts  = [r[d]['step']    if d in r else np.nan for d in sorted_d_keys]
        averages_all.append(avgs)
        steps_all.append(sts)

    # 4) compute overall_avgs by chosen method
    overall_avgs = []
    for avgs in averages_all:
        arr = np.array(avgs)
        if avg_method == 'geometric':
            # geometric mean of positive values only
            eps = 1e-12
            safe = np.where(arr>0, arr, eps)
            g = np.exp(np.nanmean(np.log(safe)))
            overall_avgs.append(g)
        else:
            overall_avgs.append(np.nanmean(arr))

    # --- plotting (unchanged) ---
    x = np.arange(len(sorted_d_keys))
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(15,14))

    total_width = 0.8; bar_width = total_width/n_models
    for i in range(n_models):
        off = (i-(n_models-1)/2)*bar_width
        bars = ax1.bar(x+off, averages_all[i], bar_width,
                       label=labels[i], color=colors[i])
        for bar, st in zip(bars, steps_all[i]):
            h = bar.get_height()
            ax1.text(bar.get_x()+bar.get_width()/2, h,
                     f"Step: {int(st):,}" if not np.isnan(st) else "N/A",
                     ha='center', va='bottom', fontsize=8)
        ax1.axhline(y=overall_avgs[i], color=colors[i],
                    linestyle='--', linewidth=0.8,
                    label=f"{labels[i]} Overall ({avg_method})={overall_avgs[i]:.4f}")

    ax1.set_xlabel('# showers', fontsize=26)
    ax1.set_ylabel('Min Avg Normalised WD', fontsize=26)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_d_keys, rotation=45, fontsize=14)
    ax1.set_yscale('log')
    ax1.grid(axis='y', ls='--', alpha=0.7)
    ax1.legend(fontsize=16, loc='upper right')
    ax1.set_title('Average Values by D-key', fontsize=20, pad=20)

    for i in range(n_models):
        st_int = [int(s) if not np.isnan(s) else np.nan for s in steps_all[i]]
        ax2.plot(x, st_int, 'o-', markersize=8, linewidth=2,
                 label=labels[i], color=colors[i])
        for xp, st in zip(x, st_int):
            if not np.isnan(st):
                ax2.text(xp, st, f'{int(st):,}',
                         ha='center', va='bottom', fontsize=9)

    ax2.set_xlabel('# showers', fontsize=26)
    ax2.set_ylabel('Best Training Step', fontsize=26)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sorted_d_keys, rotation=45, fontsize=14)
    ax2.grid(ls='--', alpha=0.7)
    ax2.legend(fontsize=16, loc='best')
    ax2.set_title('Best Training Steps by D-key', fontsize=20, pad=20)

    plt.tight_layout()
    plt.show()

    # 5) print results
    for i, r in enumerate(results_list):
        print(f"\n{'='*50}\n{labels[i]} Results:\n{'='*50}")
        for d in sorted_d_keys:
            if d in r:
                st = r[d]['step']; av = r[d]['average']; vals = r[d]['values']
                print(f"{d}: Step {int(st):,} - Avg: {av:.4f} - Vals: {vals}")
            else:
                print(f"{d}: No data")
        print(f"\nOverall ({avg_method}) {labels[i]}: {overall_avgs[i]:.4f}")
        avg_step = np.mean([int(x['step']) for x in r.values()])
        print(f"Avg epochs for {labels[i]}: {avg_step:.2f}")
        print(f"{'='*50}")

    return results_list

def convert_d_key_to_training_showers(d_key):
    d_str = d_key.split('=')[1].strip()
    if 'x' in d_str:
        base, exp_part = d_str.split('x')
        return float(base) * 10 ** int(exp_part.replace('10^', ''))
    return float(d_str)

# --- Helper: convert key to numeric x-value ---
def convert_d_key_to_training_showers(d_key):
    d_str = d_key.split('=')[1].strip()
    if 'x' in d_str:
        base, exp_part = d_str.split('x')
        return float(base) * 10 ** int(exp_part.replace('10^', ''))
    return float(d_str)

# --- Build reorganized_data with labels ---
def build_reorganized_data(results, strategy_configs, metric_names):
    config_names = list(strategy_configs.keys())
    reorganized = {m: [] for m in metric_names}
    for cfg_name, result in zip(config_names, results):
        for midx, metric in enumerate(metric_names):
            datum = {'label': cfg_name, 'x': [], 'y': [], 'color': None}
            for d_key, d_data in result.items():
                datum['x'].append(convert_d_key_to_training_showers(d_key))
                datum['y'].append(d_data['values'][midx])
            idxs = np.argsort(datum['x'])
            datum['x'] = np.array(datum['x'])[idxs]
            datum['y'] = np.array(datum['y'])[idxs]
            reorganized[metric].append(datum)
    return reorganized

# --- Compute group statistics by indices ---
def compute_group_statistics(indices, data_list):
    valid_indices = [i for i in indices if 0 <= i < len(data_list)]
    x_arrays, y_arrays = [], []
    for i in valid_indices:
        x = np.array(data_list[i]['x'])
        y = np.array(data_list[i]['y'])
        mask = np.isfinite(x) & np.isfinite(y)
        x_arrays.append(x[mask])
        y_arrays.append(y[mask])
    if not x_arrays:
        return None, None, None
    common_x = set(x_arrays[0])
    for arr in x_arrays[1:]:
        common_x &= set(arr)
    common_x = np.array(sorted(common_x))
    y_mat = []
    for x_arr, y_arr in zip(x_arrays, y_arrays):
        idx = [np.where(x_arr == cx)[0][0] for cx in common_x]
        y_mat.append(y_arr[idx])
    Y = np.vstack(y_mat)
    return common_x, Y.mean(axis=0), Y.std(axis=0)

def compute_weighted_overall_average(
    reorganized_data, 
    group_to_indices, 
    selected_group=None,
    weights=None,
    weight_method='Geometric',  # Options: 'Normalized', 'log_scale', 'Geometric', 'Simple'
    use_weighted_average=True    # Flag to control whether to use weighted averaging
):
    """
    Compute overall averages across metrics for specified groups, with optional simple (arithmetic) averaging.
    """
    # Process only selected groups if specified
    groups_to_process = selected_group if selected_group else list(group_to_indices.keys())
    if isinstance(groups_to_process, str):
        groups_to_process = [groups_to_process]
    
    # Initialize data structure for each group
    overall_data = {group: {'x': None, 'metric_data': {}} 
                    for group in groups_to_process if group in group_to_indices}

    # If weights are not provided, use equal weights for all metrics
    if weights is None:
        weights = {metric: 1.0 for metric in reorganized_data.keys()}
    
    # First pass: collect data for each metric and find common x values
    for metric, data_list in reorganized_data.items():
        weight = weights.get(metric, 1.0)
        for group in groups_to_process:
            if group not in group_to_indices:
                continue
            raw = group_to_indices[group]
            if raw and isinstance(raw[0], str):
                indices = [i for i,d in enumerate(data_list) if d['label'] in raw]
            else:
                indices = raw
            if not indices:
                continue
            try:
                x_mean, y_mean, y_std = compute_group_statistics(indices, data_list)
                valid = np.isfinite(y_mean) & np.isfinite(y_std)
                if not np.any(valid):
                    print(f"Warning: No valid data for group '{group}', metric '{metric}'.")
                    continue
                x_f = x_mean[valid]; y_f = y_mean[valid]; s_f = y_std[valid]
                overall_data[group]['metric_data'][metric] = {'x': x_f, 'y': y_f, 'std': s_f, 'weight': weight}
                if overall_data[group]['x'] is None:
                    overall_data[group]['x'] = x_f
                else:
                    overall_data[group]['x'] = np.intersect1d(overall_data[group]['x'], x_f)
            except Exception as e:
                print(f"Error in group '{group}', metric '{metric}': {e}")

    # Second pass: compute averages
    final_data = {}
    for group, info in overall_data.items():
        common_x = info['x']
        if common_x is None or len(common_x)==0:
            continue
        mdata = {}
        for m, d in info['metric_data'].items():
            idx = [np.where(d['x']==x)[0][0] for x in common_x if x in d['x']]
            if idx:
                mdata[m] = {'y': d['y'][idx], 'std': d['std'][idx], 'weight': d['weight']}
        if not mdata:
            continue
        # build arrays
        Ys = np.array([mdata[m]['y'] for m in mdata])
        Ss = np.array([mdata[m]['std'] for m in mdata])
        Ws = np.array([mdata[m]['weight'] for m in mdata])
        if not use_weighted_average:
            Ws = np.ones_like(Ws)
        Ws = Ws / Ws.sum()

        # Simple arithmetic average
        if weight_method == 'Simple':
            y_mean = Ys.mean(axis=0)
            y_std = np.sqrt((Ss**2).sum(axis=0)) / Ys.shape[0]

        # Normalized max-based
        elif weight_method == 'Normalized':
            maxY = Ys.max(axis=1, keepdims=True)
            normY = Ys / (maxY + 1e-10)
            y_mean = (normY.T @ Ws).flatten() * maxY.mean()
            y_std = np.sqrt((Ss**2 / (maxY**2)).T @ (Ws**2)) * maxY.mean()

        # Log-scale normalized
        elif weight_method == 'log_scale':
            eps=1e-10
            logY=np.log10(Ys+eps)
            mins=logY.min(axis=1,keepdims=True)
            rng=logY.max(axis=1,keepdims=True)-mins
            rng[rng==0]=1
            normL=(logY-mins)/rng
            avgL=(normL.T @ Ws).flatten()
            y_mean=10**(avgL*rng.mean()+mins.mean())
            rel=Ss/(Ys+eps)
            y_std=y_mean*np.sqrt((rel**2).T @ (Ws**2))

        # Geometric (log-space) as before
        elif weight_method == 'Geometric':
            eps=1e-10
            logY=np.log10(Ys+eps)
            sigma=np.divide(Ss,(Ys+eps))/np.log(10)
            sum_log=(logY*Ws[:,None]).sum(axis=0)
            wsum=Ws.sum()
            lmean=sum_log/wsum
            slog=np.sqrt((sigma**2 * (Ws[:,None]**2)).sum(axis=0)) / wsum
            y_mean=10**lmean
            y_std=y_mean*np.log(10)*slog

        else:
            # fallback weighted arithmetic
            y_mean=(Ys.T @ Ws).flatten()
            y_var=(Ss**2).T @ (Ws**2)
            y_std=np.sqrt(y_var)

        final_data[group]=(common_x, y_mean, y_std)

    return final_data


# def plot_reorganized_data(
#     reorganized_data, strategy_configs,
#     main_title=None, ylabel='Normalized WD',
#     save_plot=False, save_dir='./results', filename='final.png',
#     x_max=1.05e5, colors=None, selected_group=None,
#     group_prefixes=None, use_weighted_average=True,
#     weights=None, weight_method='normalized'
# ):
#     # Map labels → index per metric
#     label_idx = {
#         m: {d['label']: i for i, d in enumerate(lst)}
#         for m, lst in reorganized_data.items()
#     }

#     # Build group → list of strategy indices
#     group_to_indices = {}
#     for grp, prefs in (group_prefixes or {}).items():
#         for i, cfg in enumerate(strategy_configs):
#             if any(cfg.startswith(p) for p in prefs):
#                 group_to_indices.setdefault(grp, []).append(i)
#     assigned = set(sum(group_to_indices.values(), []))
#     for i, cfg in enumerate(strategy_configs):
#         if i not in assigned:
#             group_to_indices.setdefault(cfg, []).append(i)

#     groups = selected_group or list(group_to_indices.keys())
#     if isinstance(groups, str):
#         groups = [groups]

#     cmap = plt.cm.plasma
#     gcols = {g: (colors[i] if colors else cmap(i / max(1, len(groups) - 1)))
#              for i, g in enumerate(groups)}

#     # Create figure with constrained_layout to avoid tight_layout warning
#     fig = plt.figure(figsize=(16, 13), constrained_layout=False)
#     gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.5],
#                           hspace=0.3, wspace=0.25, left=0.06, right=0.98, 
#                           top=0.95, bottom=0.05)
    
#     # First two rows → individual metrics
#     axs = [fig.add_subplot(gs[r, c]) for r in (0, 1) for c in range(3)]

#     for idx, (metric, lst) in enumerate(reorganized_data.items()):
#         if idx >= len(axs):
#             break
#         ax = axs[idx]
#         ax.set_xscale('log')
#         ax.set_yscale('log')
#         ax.set_title(metric, fontsize=26, fontweight='medium')
#         ax.set_xlim([0.9e2, x_max])
        
#         if idx % 3 == 0:
#             ax.set_ylabel(ylabel, fontsize=24, fontweight='medium')

#         # NO x-axis label for individual plots
            
#         # LARGER TICK LABELS for individual plots
#         ax.tick_params(axis='both', which='major', labelsize=18, length=5, width=1.2)
#         ax.tick_params(axis='both', which='minor', labelsize=16, length=3, width=0.8)

#         # Remove grid completely
#         ax.grid(False)
        
#         # Enhance spines for publication
#         for spine in ax.spines.values():
#             spine.set_linewidth(1.2)
#             spine.set_color('black')
    
#         for g in groups:
#             inds = group_to_indices.get(g, [])
#             x, y, s = compute_group_statistics(inds, lst)
#             if x is None:
#                 continue
#             mask = (y > 0) & np.isfinite(y)
#             xv, yv, sv = x[mask], y[mask], s[mask]
            
#             # Publication-quality plotting
#             ax.plot(xv, yv, 'o-', color=gcols[g], label=g,
#                     linewidth=2, markersize=5, markeredgewidth=0.5,
#                     markeredgecolor='white', alpha=0.95)
#             lb, ub = np.maximum(yv - sv, yv * 0.01), yv + sv
#             ax.fill_between(xv, lb, ub, color=gcols[g], alpha=0.25)

#     # Overall average + error in the bottom row spanning all columns
#     overall = compute_weighted_overall_average(
#         reorganized_data,
#         group_to_indices,
#         selected_group=groups,
#         weights=weights,
#         weight_method=weight_method,
#         use_weighted_average=use_weighted_average
#     )

#     # Create overall plot spanning the third row
#     axov = fig.add_subplot(gs[2, :])
#     axov.set_xscale('log')
#     axov.set_yscale('log')
#     axov.set_xlim([0.9e2, x_max])
    
#     # Remove grid
#     axov.grid(False)
    
#     # Enhance spines
#     for spine in axov.spines.values():
#         spine.set_linewidth(1.5)
#         spine.set_color('black')
    
#     axov.set_title(f'{weight_method} Mean', fontsize=34, fontweight='bold')
#     axov.set_xlabel('Number of Training Showers', fontsize=32, fontweight='medium')
#     axov.set_ylabel(ylabel, fontsize=30, fontweight='medium')
    
#     for g, (x, y, s) in overall.items():
#         mask = (y > 0) & np.isfinite(y)
#         xv, yv, sv = x[mask], y[mask], s[mask]
#         axov.plot(xv, yv, 'o-', color=gcols[g], label=g,
#                   linewidth=2.5, markersize=7, markeredgewidth=0.7,
#                   markeredgecolor='white', alpha=0.95)
#         lb, ub = np.maximum(yv - sv, yv * 0.01), yv + sv
#         axov.fill_between(xv, lb, ub, color=gcols[g], alpha=0.25)
    
#     # LARGER TICK LABELS for overall plot
#     axov.tick_params(axis='both', which='major', labelsize=22, length=6, width=1.5)
#     axov.tick_params(axis='both', which='minor', labelsize=20, length=3, width=1)
    
#     axov.legend(fontsize=25, loc='upper center', frameon=False, 
#                 fancybox=False, edgecolor='black', 
#                 framealpha=0.95, borderpad=0.6, 
#                 columnspacing=1.2, handlelength=2,
#                 ncol=2)  # Added 2-column layout
    
#     if main_title:
#         fig.suptitle(main_title, fontsize=24, fontweight='bold', y=0.98)

#     if save_plot:
#         os.makedirs(save_dir, exist_ok=True)
#         plt.savefig(os.path.join(save_dir, filename), 
#                    dpi=300, bbox_inches='tight', facecolor='white')

#     plt.show()

from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

class HandlerLineWithBand(HandlerBase):
    """Custom legend handler that overlays line on error band"""
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                      width, height, fontsize, trans):
        # Extract the color from the original handle (tuple of patch, line)
        patch, line = orig_handle
        color = line.get_color()
        
        # Create the band (rectangle) first - it will be behind
        band = Rectangle((xdescent, ydescent), width, height,
                        facecolor=color, alpha=0.25, edgecolor='none',
                        transform=trans)
        
        # Create the line with markers in the middle
        xdata = [xdescent + width * 0.2, xdescent + width * 0.5, xdescent + width * 0.8]
        ydata = [ydescent + height / 2] * 3
        line_artist = Line2D(xdata, ydata, color=color, linewidth=2.5,
                            marker='o', markersize=7, markeredgewidth=0.7,
                            markeredgecolor='white', alpha=0.95,
                            transform=trans)
        
        # Return both artists - band first so it renders behind
        return [band, line_artist]

def plot_reorganized_data(
    reorganized_data, strategy_configs,
    main_title=None, ylabel='Normalized WD',
    save_plot=False, save_dir='./results', filename='final.png',
    x_max=1.05e5, colors=None, selected_group=None,
    group_prefixes=None, use_weighted_average=True,
    weights=None, weight_method='normalized'
):
    # Map labels → index per metric
    label_idx = {
        m: {d['label']: i for i, d in enumerate(lst)}
        for m, lst in reorganized_data.items()
    }

    # Build group → list of strategy indices
    group_to_indices = {}
    for grp, prefs in (group_prefixes or {}).items():
        for i, cfg in enumerate(strategy_configs):
            if any(cfg.startswith(p) for p in prefs):
                group_to_indices.setdefault(grp, []).append(i)
    assigned = set(sum(group_to_indices.values(), []))
    for i, cfg in enumerate(strategy_configs):
        if i not in assigned:
            group_to_indices.setdefault(cfg, []).append(i)

    groups = selected_group or list(group_to_indices.keys())
    if isinstance(groups, str):
        groups = [groups]

    cmap = plt.cm.plasma
    gcols = {g: (colors[i] if colors else cmap(i / max(1, len(groups) - 1)))
             for i, g in enumerate(groups)}

    # Create figure with constrained_layout to avoid tight_layout warning
    fig = plt.figure(figsize=(16, 13), constrained_layout=False)
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.5],
                          hspace=0.3, wspace=0.25, left=0.06, right=0.98, 
                          top=0.95, bottom=0.05)
    
    # First two rows → individual metrics
    axs = [fig.add_subplot(gs[r, c]) for r in (0, 1) for c in range(3)]

    for idx, (metric, lst) in enumerate(reorganized_data.items()):
        if idx >= len(axs):
            break
        ax = axs[idx]
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(metric, fontsize=26, fontweight='medium')
        ax.set_xlim([0.9e2, x_max])
        
        if idx % 3 == 0:
            ax.set_ylabel(ylabel, fontsize=24, fontweight='medium')

        # NO x-axis label for individual plots
            
        # LARGER TICK LABELS for individual plots
        ax.tick_params(axis='both', which='major', labelsize=18, length=5, width=1.2)
        ax.tick_params(axis='both', which='minor', labelsize=16, length=3, width=0.8)

        # Remove grid completely
        ax.grid(False)
        
        # Enhance spines for publication
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('black')
    
        for g in groups:
            inds = group_to_indices.get(g, [])
            x, y, s = compute_group_statistics(inds, lst)
            if x is None:
                continue
            mask = (y > 0) & np.isfinite(y)
            xv, yv, sv = x[mask], y[mask], s[mask]
            
            # Publication-quality plotting
            ax.plot(xv, yv, 'o-', color=gcols[g], label=g,
                    linewidth=2, markersize=5, markeredgewidth=0.5,
                    markeredgecolor='white', alpha=0.95)
            lb, ub = np.maximum(yv - sv, yv * 0.01), yv + sv
            ax.fill_between(xv, lb, ub, color=gcols[g], alpha=0.25)

    # Overall average + error in the bottom row spanning all columns
    overall = compute_weighted_overall_average(
        reorganized_data,
        group_to_indices,
        selected_group=groups,
        weights=weights,
        weight_method=weight_method,
        use_weighted_average=use_weighted_average
    )

    # Create overall plot spanning the third row
    axov = fig.add_subplot(gs[2, :])
    axov.set_xscale('log')
    axov.set_yscale('log')
    axov.set_xlim([0.9e2, x_max])
    
    # Remove grid
    axov.grid(False)
    
    # Enhance spines
    for spine in axov.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    axov.set_title(f'{weight_method} Mean', fontsize=34, fontweight='bold')
    axov.set_xlabel('Number of Training Showers', fontsize=32, fontweight='medium')
    axov.set_ylabel(ylabel, fontsize=30, fontweight='medium')
    
    # Create lists to store legend handles and labels
    legend_handles = []
    legend_labels = []
    
    for g, (x, y, s) in overall.items():
        mask = (y > 0) & np.isfinite(y)
        xv, yv, sv = x[mask], y[mask], s[mask]
        
        # Plot the data (no label here since we'll add custom legend)
        axov.plot(xv, yv, 'o-', color=gcols[g],
                  linewidth=2.5, markersize=7, markeredgewidth=0.7,
                  markeredgecolor='white', alpha=0.95)
        lb, ub = np.maximum(yv - sv, yv * 0.01), yv + sv
        axov.fill_between(xv, lb, ub, color=gcols[g], alpha=0.25)
        
        # Create custom legend handle with band BEHIND line
        # Order matters: patch first (background), then line (foreground)
        patch = Patch(facecolor=gcols[g], alpha=0.25, edgecolor='none')
        line = Line2D([0], [0], color=gcols[g], linewidth=2.5, 
                      marker='o', markersize=7, markeredgewidth=0.7,
                      markeredgecolor='white', alpha=0.95)
        
        # Combine patch and line (patch first so it renders behind)
        legend_handles.append((patch, line))
        legend_labels.append(g)
    
    # LARGER TICK LABELS for overall plot
    axov.tick_params(axis='both', which='major', labelsize=22, length=6, width=1.5)
    axov.tick_params(axis='both', which='minor', labelsize=20, length=3, width=1)
    
    # Enhanced legend with perfectly overlapped band and line
    axov.legend(legend_handles, legend_labels,
                fontsize=25, loc='upper center', frameon=False, 
                fancybox=False, edgecolor='black', 
                framealpha=0.95, borderpad=0.6, 
                columnspacing=1.2, handlelength=3,
                ncol=1,
                handler_map={tuple: HandlerLineWithBand()})  # Custom handler for perfect overlay
    
    if main_title:
        fig.suptitle(main_title, fontsize=24, fontweight='bold', y=0.98)

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename), 
                   dpi=300, bbox_inches='tight', facecolor='white')

    plt.show()



def build_group_to_indices(strategy_configs, group_prefixes=None):
    """
    Build a mapping from group label → list of strategy indices,
    using prefix matching. Falls back to the strategy key itself if
    no prefix group matches.
    
    Parameters:
      strategy_configs (dict): your original configs, keys are strategy names.
      group_prefixes (dict): { group_label: [prefix1, prefix2, …] }.
    
    Returns:
      group_to_indices (dict): { group_label: [indices …] }.
    """
    group_to_indices = {}
    for idx, strategy in enumerate(strategy_configs):
        placed = False
        if group_prefixes:
            for grp, prefixes in group_prefixes.items():
                if any(strategy.startswith(pref) for pref in prefixes):
                    group_to_indices.setdefault(grp, []).append(idx)
                    placed = True
                    break
        if not placed:
            # fallback: use the strategy name as its own group
            group_to_indices.setdefault(strategy, []).append(idx)
    return group_to_indices

# final analysis function 
def generate_overall_table(
    reorganized_data: dict,
    strategy_configs: list,
    group_prefixes: dict = None,
    selected_group: list = None,
    weight_method=None,
    cols: list = [100, 1000, 10000, 100000],
    metric_names: list = None,
    use_weighted_average: bool = True
) -> (pd.DataFrame, pd.DataFrame):
    """
    Same as before but preserves the original order of prefixes
    instead of sorting them alphabetically.
    """
    # Build mapping from group→indices
    g2i = build_group_to_indices(strategy_configs, group_prefixes)

    # Auto‑select all groups if none provided
    if selected_group is None:
        selected_group = list(g2i.keys())

    # Prepare weights dict
    if isinstance(weight_method, dict):
        weight_dict = weight_method
    elif isinstance(weight_method, str):
        # Example: equal weights for each metric (you can customize)
        # You could also implement real geometric vs arithmetic logic here.
        weight_dict = {m: 1.0 for m in metric_names}
    else:
        weight_dict = {}
        if weight_method is not None:
            warnings.warn(
                f"Ignoring weight_method={weight_method!r}; defaulting to uniform weights.",
                UserWarning
            )

    # Compute raw per‐group overall averages/stds
    ov = compute_weighted_overall_average(
        reorganized_data,
        g2i,
        selected_group,
        weight_dict,
        use_weighted_average
    )

    # Build per‐group Series, reindexing (warn if missing)
    raw_mean = {}
    raw_std  = {}
    for grp, (xs, ys, ss) in ov.items():
        ser_mean = pd.Series(ys, index=xs)
        ser_std  = pd.Series(ss, index=xs)

        missing = set(cols) - set(ser_mean.index)
        if missing:
            warnings.warn(
                f"Group '{grp}' missing columns {sorted(missing)}; filling with NaN.",
                UserWarning
            )

        raw_mean[grp] = ser_mean.reindex(cols)
        raw_std[grp]  = ser_std.reindex(cols)

    mean_df = pd.DataFrame(raw_mean).T
    std_df  = pd.DataFrame(raw_std).T
    mean_df.index.name = 'Group'
    std_df.index.name  = 'Group'

    # Strip off any variant suffix (_v1, -v2, .v3)
    def strip_variant(name):
        m = re.match(r'(.+?)[_\. -]v\d+', name, flags=re.IGNORECASE)
        return m.group(1) if m else name

    # Compute prefix column in the same order as groups appear
    prefixes = [strip_variant(g) for g in mean_df.index]
    mean_df['prefix'] = prefixes
    std_df['prefix']  = prefixes

    # Determine unique prefixes in appearance order
    prefix_order = list(OrderedDict.fromkeys(prefixes))

    # Group by prefix WITHOUT sorting, then reindex to prefix_order
    agg_mean = (
        mean_df
        .groupby('prefix', sort=False)[cols]
        .mean()
        .reindex(prefix_order)
    )
    agg_std = (
        std_df
        .groupby('prefix', sort=False)[cols]
        .mean()
        .reindex(prefix_order)
    )

    # ADD AVERAGE COLUMN WITH ERROR PROPAGATION
    agg_mean['Average'] = agg_mean[cols].mean(axis=1)
    agg_std['Average'] = np.sqrt((agg_std[cols]**2).sum(axis=1)) / len(cols)

    agg_mean.index.name = 'Group'
    agg_std.index.name  = 'Group'

    return agg_mean, agg_std

def print_overall_table(mean_df: pd.DataFrame, std_df: pd.DataFrame, name_to_params: dict = None,
                        save_latex=False, latex_path='./results/table.tex'):
    """
    Print mean ± std, bolding the lowest mean in each column.
    Now includes trainable parameters if provided.
    LaTeX version uses compact subscript format for errors.
    """
    BOLD = '\033[1m'
    END = '\033[0m'
    
    out = pd.DataFrame(index=mean_df.index)
        
    # Add parameters column if provided
    if name_to_params:
        params_col = []
        for grp in mean_df.index:
            params_col.append(name_to_params.get(grp, "N/A"))
        out['Trainable Parameters'] = params_col
        
    for col in mean_df.columns:
        col_means = mean_df[col]
        col_stds = std_df[col]
        min_val = col_means.min()
        
        formatted = []
        latex_formatted = []  # For LaTeX version with compact subscript errors
        for grp in mean_df.index:
            m = col_means.loc[grp]
            s = col_stds.loc[grp]
            txt = f"{m:.3f} ± {s:.3f}"
            # LaTeX version with subscript errors for compactness
            latex_txt = f"{m:.3f}$_{{\\pm {s:.3f}}}$"
            
            if m == min_val:
                txt = f"{BOLD}{txt}{END}"
                latex_txt = f"\\textbf{{{m:.3f}}}$_{{\\pm {s:.3f}}}$"
            
            
            formatted.append(txt)
            latex_formatted.append(latex_txt)
        
        out[col] = formatted
        if save_latex:
            out[f'{col}_latex'] = latex_formatted
    
    out.index.name = 'Method'
    # print(out.to_markdown())
    
    # Save LaTeX table if requested
    if save_latex:
        import os
        # Create LaTeX version without ANSI codes
        latex_out = out.copy()
        for col in mean_df.columns:
            latex_out[col] = latex_out[f'{col}_latex']
            latex_out.drop(f'{col}_latex', axis=1, inplace=True)
        
        os.makedirs(os.path.dirname(latex_path), exist_ok=True)
        
        # Generate properly formatted LaTeX table with compact layout
        with open(latex_path, 'w') as f:
            f.write("\\begin{table*}[htbp]\n")  # Use table* for two-column layout if needed
            f.write("\\centering\n")
            f.write("\\label{tab:performance}\n")
            f.write("\\small\n")  # Make entire table smaller
            
            # Create column format string with smaller spacing
            col_format = 'l' + 'c' * len(latex_out.columns)
            f.write(f"\\begin{{tabular}}{{@{{}}{col_format}@{{}}}}\n")
            f.write("\\toprule\n")
            
            # Header row
            headers = ['\\textbf{Method}'] + [f'\\textbf{{{col}}}' for col in latex_out.columns]
            f.write(' & '.join(headers) + ' \\\\\n')
            f.write("\\midrule\n")
            
            # Data rows
            for idx, row in latex_out.iterrows():
                row_data = [str(idx)] + [str(val) for val in row.values]
                f.write(' & '.join(row_data) + ' \\\\\n')
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table*}\n")
        
        print(f"LaTeX table saved to: {latex_path}")
        print("Note: Table uses subscript format (mean_std) and \\small font for compactness")

def plot_performance_vs_params(
    agg_mean: pd.DataFrame, 
    agg_std: pd.DataFrame, 
    name_to_params: dict,
    metric_name: str = "Performance",
    save_path: str = None,
    figsize: tuple = (10, 8),
    colors: dict = None,
    markers: dict = None
):
    """
    Plot performance (Average column) vs log(% trainable parameters).
    
    Parameters:
    -----------
    agg_mean : pd.DataFrame
        Mean values from generate_overall_table
    agg_std : pd.DataFrame
        Std values from generate_overall_table
    name_to_params : dict
        Mapping from group name to parameter string e.g. "524.0K (100.0%)"
    metric_name : str
        Y-axis label (e.g., "1 - Wasserstein Distance", "Accuracy")
    save_path : str
        If provided, save figure to this path
    figsize : tuple
        Figure size
    colors : dict
        Optional mapping from group name to color
    markers : dict
        Optional mapping from group name to marker style
    """
    
    # Extract percentage from params string
    def extract_percentage(param_str):
        match = re.search(r'\((\d+\.?\d*)%\)', param_str)
        if match:
            return float(match.group(1))
        return None
    
    # Prepare data
    x_values = []
    y_values = []
    y_errors = []
    labels = []
    
    for group_name in agg_mean.index:
        if group_name in name_to_params:
            param_str = name_to_params[group_name]
            percentage = extract_percentage(param_str)
            
            if percentage is not None and 'Average' in agg_mean.columns:
                x_values.append(percentage)
                y_values.append(agg_mean.loc[group_name, 'Average'])
                y_errors.append(agg_std.loc[group_name, 'Average'])
                labels.append(group_name)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Default colors and markers if not provided
    if colors is None:
        colors = {
            "From Scratch": "#7f7f7f",
            "Full Finetuned": "#2ca02c",
            "BitFit": "#ff7f0e",
            "Top3": "#d62728",
        }
        # LoRA variants get a gradient
        lora_colors = plt.cm.Blues(np.linspace(0.3, 0.9, 8))
        for i, r in enumerate([1, 2, 4, 8, 16, 32, 48, 64]):
            colors[f"LoRA (r={r})"] = lora_colors[i]
    
    if markers is None:
        markers = {
            "From Scratch": 's',
            "Full Finetuned": 'D',
            "BitFit": '^',
            "Top3": 'v',
        }
        # LoRA gets circles
        for r in [1, 2, 4, 8, 16, 32, 48, 64]:
            markers[f"LoRA (r={r})"] = 'o'
    
    # Plot points with error bars
    for i, (x, y, yerr, label) in enumerate(zip(x_values, y_values, y_errors, labels)):
        color = colors.get(label, 'black')
        marker = markers.get(label, 'o')
        
        ax.errorbar(x, y, yerr=yerr, 
                   fmt=marker, 
                   color=color, 
                   markersize=10,
                   capsize=5,
                   capthick=2,
                   label=label,
                   alpha=0.8)
    
    # Set log scale for x-axis
    # ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Labels and title
    ax.set_xlabel('Trainable Parameters (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=14, fontweight='bold')
    ax.set_title('Performance vs Parameter Efficiency', fontsize=16, fontweight='bold')
    
    # Set x-axis limits to show full range
    ax.set_xlim(-5, 105)
    
    # Add annotations for key points
    for i, (x, y, label) in enumerate(zip(x_values, y_values, labels)):
        if label in ["Full Finetuned", "BitFit", "LoRA (r=8)", "LoRA (r=48)"]:
            ax.annotate(label, 
                       xy=(x, y), 
                       xytext=(10, 5),
                       textcoords='offset points',
                       fontsize=9,
                       alpha=0.7)
    
    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return fig, ax

def create_modification_heatmap(model_pretrained, models_finetuned):
    """Create layer-wise modification magnitude heatmap"""
    
    modifications = {}
    
    for method_name, model_ft in models_finetuned.items():
        layer_mods = []
        
        for layer_idx in range(6):  # Your 6 layers
            if method_name == "BitFit":
                # Only bias changes
                bias_pre = model_pretrained.layers[layer_idx].bias
                bias_ft = model_ft.layers[layer_idx].bias
                mod = torch.norm(bias_ft - bias_pre).item()
                
            elif "LoRA" in method_name:
                # LoRA modification magnitude
                lora_layer = model_ft.layers[layer_idx].lora
                mod = torch.norm(lora_layer.B @ lora_layer.A).item()
                
            elif method_name == "Top3":
                # Only last 3 layers
                if layer_idx >= 3:
                    W_pre = model_pretrained.layers[layer_idx].weight
                    W_ft = model_ft.layers[layer_idx].weight
                    mod = torch.norm(W_ft - W_pre).item()
                else:
                    mod = 0
                    
            elif method_name == "Full FT":
                # All parameters
                W_pre = model_pretrained.layers[layer_idx].weight
                W_ft = model_ft.layers[layer_idx].weight
                mod = torch.norm(W_ft - W_pre).item()
                
            layer_mods.append(mod)
            
        modifications[method_name] = layer_mods
    
    # Normalize by row (each method) for better visualization
    df = pd.DataFrame(modifications, index=[f"Layer {i}" for i in range(6)])
    df_norm = df.div(df.max(axis=1), axis=0)  # Normalize by layer
    
    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_norm.T, 
                cmap='Reds', 
                cbar_kws={'label': 'Relative Modification'},
                linewidths=0.5,
                annot=True,
                fmt='.2f')
    plt.xlabel('Layer')
    plt.ylabel('Method')
    plt.title('Layer-wise Modification Magnitude')
    plt.tight_layout()
    
    return df





## finals for ShowerFlow

