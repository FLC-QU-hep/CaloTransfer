from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))

import sys
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnchoredText

import os
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False

from scipy.stats import entropy, wasserstein_distance, gaussian_kde

from matplotlib import colormaps  
from tqdm import tqdm
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec


def get_plot_axes(rows: int, cols: int, size=(12, 15), flatten_axes=False, **kwargs):
    rows = int(rows)
    cols = int(cols)

    assert rows >= 1
    assert cols >= 1

    fig, axes = plt.subplots(nrows=rows, ncols=cols, **kwargs)

    fig.set_figwidth(size[0] * cols)
    fig.set_figheight(size[1] * rows)

    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)

    if flatten_axes:
        return np.reshape(axes, newshape=[-1])

    return axes

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # GRID  # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

ratio_ylim = (0.2, 1.8)
label_fontsize = 26
axis_fontsize = 30
# Configuration class for plot styling
class PlotConfig:
    def __init__(self, colors, alpha=0.3):
        self.color_lines = colors
        self.alpha = alpha

def plot_histograms_with_ratio(data_list, bins, labels, colors, title, xlabel, ylabel, 
                               log_scale_x=False, log_scale_y=False, xlim=None, ylim=None,
                               training_strategy='vanilla', save_plot=False, save_dir='./results/diffusion/',
                               ratio_ylim=(0.2, 1.8), show_error_bands=True, label_fontsize=label_fontsize, axis_fontsize=axis_fontsize,
                               strategy_labels=False, show_legend=False):
    """
    Generic function to plot histograms with ratio plots and error bands.
    First element in data_list is assumed to be the reference (GEANT4).
    """
    # Create figure with GridSpec for main plot and ratio plot
    
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    
    
    ax_main = plt.subplot(gs[0])
    ax_ratio = plt.subplot(gs[1], sharex=ax_main)
    
    # Calculate histograms for all datasets
    hist_values = []
    hist_errors = []
    
    for i, data in enumerate(data_list):
        counts, bin_edges = np.histogram(data, bins=bins)
        hist_values.append(counts)
        # Use Poisson errors (sqrt(n)) - standard for counting statistics
        hist_errors.append(np.sqrt(counts))
    
    # Use the actual bin edges from histogram
    bins_to_use = bin_edges
    
    # Plot main histograms
    for i, (counts, errors, label, color) in enumerate(zip(hist_values, hist_errors, labels, colors)):
        if i == 0:
            # Reference data (GEANT4) - filled histogram
            ax_main.stairs(counts, bins_to_use, color='lightgrey', fill=True, label=label)
            
            # Error band with hatching for reference
            if show_error_bands:
                ax_main.stairs(counts + errors, bins_to_use, baseline=counts - errors,
                                color='dimgrey', linewidth=2, hatch='///', fill=False)
        else:
            # Generated data - step histogram
            ax_main.stairs(counts, bins_to_use, color=color, 
                            linewidth=2, fill=False, label=label)
            
            # Transparent error band for generated data
            if show_error_bands:
                ax_main.stairs(counts + errors, bins_to_use, baseline=counts - errors,
                                color=color, alpha=0.3, fill=True, linewidth=0)
    
    # Configure main plot
    if log_scale_x:
        ax_main.set_xscale('log')
    if log_scale_y:
        ax_main.set_yscale('log')
    if xlim:
        ax_main.set_xlim(xlim)
    if ylim:
        ax_main.set_ylim(ylim)

    ax_main.set_ylabel(ylabel, fontsize=axis_fontsize)
    if show_legend:
        # ax_main.legend(fontsize=label_fontsize, loc='best', frameon=False)
        leg = ax_main.legend(fontsize=label_fontsize-5, loc='best')
        leg.get_frame().set_visible(False)
    ax_main.tick_params(labelbottom=False, labelsize=axis_fontsize-4)  # Hide x-axis labels on main plot
    
    # Add training strategy to title
    if strategy_labels:
        ax_main.set_title(f'{training_strategy}', loc='right', fontsize=axis_fontsize-2, pad=20, weight='bold')
    
    # Plot ratios (skip first which is reference)
    if len(hist_values) > 1:
        bin_centers = (bins_to_use[:-1] + bins_to_use[1:]) / 2
        ratio_plots_histograms(ax_ratio, hist_values[0], hist_values[1:], 
                               hist_errors[0], hist_errors[1:], 
                               bins_to_use, bin_centers, colors[1:], show_error_bands
                               , lims_min=ratio_ylim[0], lims_max=ratio_ylim[1])
    
    # Configure ratio plot
    ax_ratio.set_xlabel(xlabel, fontsize=axis_fontsize)
    ax_ratio.set_ylabel(r'$\frac{\mathrm{Generated}}{\mathrm{Geant4}}$', fontsize=axis_fontsize)
    ax_ratio.set_ylim(ratio_ylim)
    ax_ratio.axhline(1, linestyle='-', lw=1, color='black')
    ax_ratio.tick_params(labelsize=axis_fontsize-4)
    
    if log_scale_x:
        ax_ratio.set_xscale('log')
    if xlim:
        ax_ratio.set_xlim(xlim)
    
    # Save plot if requested
    if save_plot:
        filename = f"{title.replace(' ', '_')}_with_ratio.png"
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=150)
        print(f'Plot saved to {os.path.join(save_dir, filename)}')
    
    plt.show()


def ratio_plots_histograms(ax, geant4_data, gen_data_list, err_geant4, err_gen_list, 
                           bins, pos, colors, show_error_bands=True, lims_min=0.6, lims_max=1.4,):
    """
    Ratio plotting for histograms with proper error propagation.
    Matches the external reference implementation.
    """
    
    eps = 1e-5
    
    # Get current x-axis limits from parent plot
    xlims = ax.get_xlim()
    
    for i, (gen_data, err_gen, color) in enumerate(zip(gen_data_list, err_gen_list, colors)):
        # Calculate ratios with clipping
        ratios = np.clip((gen_data + eps) / (geant4_data + eps), lims_min, lims_max)
        
        # Plot the ratio line using stairs
        ax.stairs(ratios, edges=bins, color=color, lw=2)
        
        # Filter positions to only show markers within x-limits
        mask_in_xlim = (pos >= xlims[0]) & (pos <= xlims[1])
        filtered_pos = pos[mask_in_xlim]
        filtered_ratios = ratios[mask_in_xlim]
        
        # Mark points that are clipped at limits (only within x-limits)
        mask_min = (filtered_ratios == lims_min)
        if np.any(mask_min):
            ax.plot(filtered_pos[mask_min], filtered_ratios[mask_min], linestyle='', lw=2, 
                   marker='v', color=color, clip_on=False)
        
        mask_max = (filtered_ratios == lims_max)
        if np.any(mask_max):
            ax.plot(filtered_pos[mask_max], filtered_ratios[mask_max], linestyle='', lw=2, 
                   marker='^', color=color, clip_on=False)
        
        # Plot error band using proper error propagation
        if show_error_bands:
            # Avoid division by zero in error calculation
            safe_gen = np.maximum(gen_data, eps)
            safe_geant = np.maximum(geant4_data, eps)
            
            # Error propagation for ratio
            ratio_err = ratios * np.sqrt(
                (err_gen / safe_gen) ** 2 + (err_geant4 / safe_geant) ** 2
            )
            
            # Plot error band with transparency
            ax.stairs(
                ratios + ratio_err,
                edges=bins,
                baseline=ratios - ratio_err,
                color=color,
                alpha=0.3,
                fill=True,
                linewidth=0
            )
    
    # Set y-axis limits
    ax.set_ylim(lims_min, lims_max)


def plot_visible_energy(showers, threshold=0.01515, simulation_labels=None, log_scale=True, 
                        kl_divergences=None, wasserstein=None, colors=None, title='Voxel Energy Spectrum', 
                        training_strategy='vanilla', save_plot=False, save_dir='./results/diffusion/', 
                        show_error_bands=True):
    """
    Plots histograms with ratio plots for the given array of showers.
    """
    labels = list(simulation_labels.keys()) if simulation_labels else [f'Shower {i+1}' for i in range(len(showers))]
    colors = plt.cm.turbo(np.linspace(0, 1, len(showers))) if colors is None else colors
    
    mini, maxi = showers.min(), showers.max()
    print(f'Minimum energy: {mini:.2e} MeV - Maximum energy: {maxi:.2e} MeV')
    
    visible_showers = [shower[(shower != 0) & (shower > threshold)] for shower in showers]
    
    # Create bins
    if log_scale:
        bins = np.logspace(np.log10(mini + 1e-7), np.log10(maxi), 150)
    else:
        bins = np.linspace(mini, maxi, 150)

    # Use the new histogram function with ratio and error bands
    plot_histograms_with_ratio(visible_showers, bins, labels, colors, title, title + ' [MeV]', 'Number of cells', 
                               log_scale_x=log_scale, log_scale_y=True, xlim=(threshold+1e-2, 2e3), ylim=(0, 3e6),
                               training_strategy=training_strategy, save_plot=save_plot, save_dir=save_dir,
                               strategy_labels=True, show_error_bands=show_error_bands, show_legend=False)
    
    # Calculate divergences if provided
    if kl_divergences is not None and wasserstein is not None:
        print(f'Calculating divergences for {title} histograms... \n')
        kl_divergences[title] = {}
        wasserstein[title] = {}
        reference_shower = visible_showers[0]
        
        for i, visible_shower in enumerate(visible_showers[1:], start=1):
            wasserstein_value = wasserstein_d(reference_shower, visible_shower)
            kl_value = quantiled_kl_divergence(reference_shower, visible_shower, num_quantiles=30, show=False, 
                                                labels=[labels[0], labels[i]])
            wasserstein[title][labels[i]] = wasserstein_value
            kl_divergences[title][labels[i]] = kl_value
            print(f'{labels[i]} - Wasserstein distance: {wasserstein_value:.2e}, Quantiled KL Divergence: {kl_value:.2e}')
    
    return kl_divergences if kl_divergences is not None else {}, wasserstein if wasserstein is not None else {}

def plot_calibration_histograms(showers_numpy, incidents_numpy, simulation_labels=None, colors=None, 
                                kl_divergences=None, wasserstein=None, title='Sampling Fraction', 
                                training_strategy='vanilla', save_plot=False, save_dir='./results/diffusion/',
                                show_error_bands=True):
    """
    Plots calibration histograms with ratio plots and error bands.
    """
    # Data preparation
    showers = showers_numpy.copy()
    incidents = incidents_numpy.copy() * 1000  # Convert to MeV
    labels = list(simulation_labels.keys()) if simulation_labels else [f'Shower {i+1}' for i in range(len(showers))]
    colors = plt.cm.turbo(np.linspace(0, 1, len(showers))) if colors is None else colors

    if len(labels) != showers.shape[0]:
        raise ValueError("The length of the labels list must match the number of showers.")

    # Calculate calibration values
    calibrations = []
    for i in range(len(showers)):
        shower_sum = showers[i].sum(axis=-1)
        calibration = shower_sum / incidents[i].flatten()
        calibrations.append(calibration)

    # Determine bins based on the first shower's calibration values
    calibration_minimum = calibrations[0].min()
    bins = np.logspace(np.log10(calibration_minimum), np.log10(max([c.max() for c in calibrations])), 200)

    # Plot histograms with ratio and error bands
    plot_histograms_with_ratio(calibrations, bins, labels, colors, title, 'Energy Ratio', '# showers', 
                               log_scale_x=False, log_scale_y=True, xlim=(0.5, 1.4), ylim=(0, 4e3),
                               training_strategy=training_strategy, save_plot=save_plot, save_dir=save_dir,
                               show_error_bands=show_error_bands, show_legend=True, strategy_labels=True, )

    # Calculate divergences if provided
    if kl_divergences is not None and wasserstein is not None:
        print(f'Calculating divergences for {title} histograms... \n')
        kl_divergences[title] = {}
        wasserstein[title] = {}
        reference_calibration = calibrations[0]
        
        for i, sample_calibration in enumerate(calibrations[1:], start=1):
            wasserstein_value = wasserstein_d(reference_calibration, sample_calibration)
            kl_value = quantiled_kl_divergence(reference_calibration, sample_calibration, 
                                              num_quantiles=30, show=False, 
                                              labels=[labels[0], labels[i]])
            wasserstein[title][labels[i]] = wasserstein_value
            kl_divergences[title][labels[i]] = kl_value
            print(f'{labels[i]} - Wasserstein distance: {wasserstein_value:.2e}, Quantiled KL Divergence: {kl_value:.2e}')
    
    return kl_divergences if kl_divergences is not None else {}, wasserstein if wasserstein is not None else {}

def plot_energy_sum(showers, simulation_labels=None, colors=None, kl_divergences=None, wasserstein=None, 
                    title='Visible Energy', training_strategy='vanilla', save_plot=False, save_dir='./results/diffusion/',
                    show_error_bands=True):
    """
    Plots histograms of the energy sum with ratio plots and error bands.
    """
    labels = list(simulation_labels.keys()) if simulation_labels else [f'Shower {i+1}' for i in range(len(showers))]
    colors = plt.cm.turbo(np.linspace(0, 1, len(showers))) if colors is None else colors
    
    # Compute energy sums
    sum_showers = [shower.sum(-1) for shower in showers]

    # Define bins using np.logspace with base 10 for clarity
    max_val = max([s.max() for s in sum_showers])
    min_val = 1e-7
    bins = np.logspace(np.log10(min_val), np.log10(max_val), 200)

    # Call generic plotting function
    plot_histograms_with_ratio(
        sum_showers, bins, labels, colors, title,
        title + ' [MeV]', '# showers',
        log_scale_x=True, log_scale_y=True,
        xlim=(2e2, 9e5), ylim=(0, 4e2),
        training_strategy=training_strategy,
        save_plot=save_plot, save_dir=save_dir,
        show_error_bands=show_error_bands,
        show_legend=False, strategy_labels=True,
    )
    
    # Calculate divergences if provided
    if kl_divergences is not None and wasserstein is not None:
        print(f'Calculating divergences for {title} histograms... \n')
        kl_divergences[title] = {}
        wasserstein[title] = {}
        reference_sum = sum_showers[0]
        
        for i, sample_sum in enumerate(sum_showers[1:], start=1):
            wasserstein_value = wasserstein_d(reference_sum, sample_sum)
            kl_value = quantiled_kl_divergence(
                reference_sum, sample_sum, num_quantiles=30, show=False, 
                labels=[labels[0], labels[i]]
            )
            wasserstein[title][labels[i]] = wasserstein_value
            kl_divergences[title][labels[i]] = kl_value
            print(f'{labels[i]} - Wasserstein distance: {wasserstein_value:.2e}, Quantiled KL Divergence: {kl_value:.2e}')
    
    return (
        kl_divergences if kl_divergences is not None else {},
        wasserstein if wasserstein is not None else {}
    )


def plot_occupancy(showers, simulation_labels=None, kl_divergences=None, wasserstein=None, colors=None, 
                   title='Occupancy', training_strategy='vanilla', save_plot=False, save_dir='./results/diffusion/',
                   show_error_bands=True):
    """
    Plots histograms of occupancy with ratio plots and error bands.
    """
    labels = list(simulation_labels.keys()) if simulation_labels else [f'Shower {i+1}' for i in range(len(showers))]
    colors = plt.cm.turbo(np.linspace(0, 1, len(showers))) if colors is None else colors

    # Calculate occupancy
    hits_shower = np.count_nonzero(showers, axis=-1)
    hits_shower = [hits_shower[i] / 40500 for i in range(len(hits_shower))]  # Normalize by total number of cells
    
    # Create bins for occupancy
    min_occ = min([h.min() for h in hits_shower])
    max_occ = max([h.max() for h in hits_shower])
    bins = np.linspace(min_occ, max_occ, 100)
    
    # Use the histogram function with ratio and error bands
    plot_histograms_with_ratio(hits_shower, bins, labels, colors, title, 'Occupancy', '# showers', 
                               log_scale_x=False, log_scale_y=False, xlim=(0, 0.45), ylim=(0, 1600),
                               training_strategy=training_strategy, save_plot=save_plot, save_dir=save_dir,
                               show_error_bands=show_error_bands, strategy_labels=True, show_legend=False)
    
    # Calculate divergences if provided
    if kl_divergences is not None and wasserstein is not None:
        print(f'Calculating divergences for {title} histograms... \n')
        kl_divergences[title] = {}
        wasserstein[title] = {}
        reference_hits = hits_shower[0]
        
        for i, sample_hits in enumerate(hits_shower[1:], start=1):
            wasserstein_value = wasserstein_d(reference_hits, sample_hits)
            kl_value = quantiled_kl_divergence(reference_hits, sample_hits, num_quantiles=30, show=False, 
                                              labels=[labels[0], labels[i]])
            wasserstein[title][labels[i]] = wasserstein_value
            kl_divergences[title][labels[i]] = kl_value
            print(f'{labels[i]} - Wasserstein distance: {wasserstein_value:.2e}, Quantiled KL Divergence: {kl_value:.2e}')
    
    return kl_divergences if kl_divergences is not None else {}, wasserstein if wasserstein is not None else {}


def plot_energy_layer(showers, simulation_labels=None, colors=None, kl_divergences=None, wasserstein=None, 
                      title='Longitudinal Profile', training_strategy='vanilla', save_plot=False, save_dir='./results/diffusion/',
                      show_error_bands=True, ratio_ylim=ratio_ylim):
    """
    Plots longitudinal profile with ratio plots and proper error bands.
    """
    labels = list(simulation_labels.keys()) if simulation_labels else [f'Shower {i+1}' for i in range(len(showers))]
    colors = plt.cm.turbo(np.linspace(0, 1, len(showers))) if colors is None else colors
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    
    ax_main = plt.subplot(gs[0])
    ax_ratio = plt.subplot(gs[1], sharex=ax_main)
    
    energy_layers = []
    energy_errors = []
    n_showers = showers[0].shape[0]  # Number of showers
    
    for i, shower in tqdm(enumerate(showers), desc='Processing Longitudinal Profile'):
        # Reshape and calculate mean energy per layer
        shower_reshaped = shower.reshape(shower.shape[0], 45, 50, 18)
        energy_sums = np.sum(shower_reshaped, axis=(2, 3))
        mean_energy_per_level = np.mean(energy_sums, axis=0) / 1000
        energy_layers.append(mean_energy_per_level)
        
        # Calculate standard error of the mean (SEM)
        std_energy = np.std(energy_sums, axis=0) / 1000
        sem_energy = std_energy / np.sqrt(n_showers)
        energy_errors.append(sem_energy)
    
    # Create bins for 45 layers
    bins = np.arange(0.5, 45.5 + 1)  # 0.5 to 45.5 for proper histogram bins
    pos = np.arange(1, 46)  # Position centers for layers 1-45
    
    # Plot main profiles
    for i, (mean_energy, errors, label, color) in enumerate(zip(energy_layers, energy_errors, labels, colors)):
        if i == 0:
            # Reference data (GEANT4) - filled stairs
            ax_main.stairs(mean_energy, bins, color='lightgrey', fill=True, label=label)
            
            # Error band with hatching for reference
            if show_error_bands:
                ax_main.stairs(mean_energy + errors, bins, baseline=mean_energy - errors,
                              color='dimgrey', linewidth=2, hatch='///', fill=False)
        else:
            # Generated data - step line using stairs
            ax_main.stairs(mean_energy, bins, color=color, linewidth=3, fill=False, label=label)
            
            # Transparent error band for generated data
            if show_error_bands:
                ax_main.stairs(mean_energy + errors, bins, baseline=mean_energy - errors,
                              color=color, alpha=0.3, fill=True, linewidth=0)
    
    # Configure main plot
    ax_main.set_ylabel('Mean Energy [GeV]', fontsize=axis_fontsize)
    ax_main.legend(fontsize=22, loc='upper right', frameon=False)
    ax_main.set_yscale('log')
    ax_main.set_xlim(0, 45)
    ax_main.set_ylim(1e-2, 1e1)
    ax_main.tick_params(labelbottom=False, labelsize=axis_fontsize-4)
    
    # Add training strategy to title
    ax_main.set_title(f'{training_strategy}', loc='right', fontsize=axis_fontsize-2, pad=20, weight='bold')
    
    # Plot ratios with error bands
    if len(energy_layers) > 1:
        ratio_plots_profile(ax_ratio, energy_layers[0], energy_layers[1:], 
                           energy_errors[0], energy_errors[1:],
                           bins, pos, colors[1:], show_error_bands)
    
    # Configure ratio plot with same x-limits
    ax_ratio.set_xlabel('Layer', fontsize=axis_fontsize)
    ax_ratio.set_ylabel(r'$\frac{\mathrm{Generated}}{\mathrm{Geant4}}$', fontsize=axis_fontsize)
    ax_ratio.set_xlim(0, 45)  # Explicitly set to match main plot
    ax_ratio.set_ylim(ratio_ylim[0], ratio_ylim[1])
    ax_ratio.axhline(1, linestyle='-', lw=1, color='black')
    ax_ratio.tick_params(labelsize=axis_fontsize-4)
    
    # Save plot if requested
    if save_plot:
        filename = f"{title.replace(' ', '_')}_with_ratio.png"
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=150)
        print(f'Plot saved to {os.path.join(save_dir, filename)}')
    
    plt.show()
    
    # Calculate divergences if provided
    if kl_divergences is not None and wasserstein is not None:
        print(f'Calculating divergences for {title} histograms... \n')
        kl_divergences[title] = {}
        wasserstein[title] = {}
        reference_layer = energy_layers[0]
        
        for i in range(1, len(energy_layers)):
            sample_layer = energy_layers[i]
            wasserstein_value = wasserstein_d(reference_layer, sample_layer)
            kl_value = quantiled_kl_divergence(reference_layer, sample_layer, num_quantiles=30, show=False, 
                                               labels=[labels[0], labels[i]])
            wasserstein[title][labels[i]] = wasserstein_value
            kl_divergences[title][labels[i]] = kl_value
            print(f'{labels[i]} - Wasserstein distance: {wasserstein_value:.2e}, Quantiled KL Divergence: {kl_value:.2e}')
    
    return kl_divergences if kl_divergences is not None else {}, wasserstein if wasserstein is not None else {}


def plot_radial_energy(showers_numpy, simulation_labels=None, colors=None, title='Radial Profile', 
                       kl_divergences=None, wasserstein=None, training_strategy='vanilla', 
                       save_plot=False, save_dir='./results/diffusion/', show_error_bands=True,
                       label_fontsize=22, ratio_ylim=ratio_ylim):
    """
    Plots radial profile with ratio plots and proper error bands.
    """
    labels = list(simulation_labels.keys()) if simulation_labels else [f'Shower {i+1}' for i in range(len(showers_numpy))]
    colors = plt.cm.turbo(np.linspace(0, 1, len(showers_numpy))) if colors is None else colors
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    
    ax_main = plt.subplot(gs[0])
    ax_ratio = plt.subplot(gs[1], sharex=ax_main)
    
    energy_per_level_list = []
    energy_errors = []
    n_showers = showers_numpy[0].shape[0]  # Number of showers
    
    for i, shower in tqdm(enumerate(showers_numpy), desc='Processing Radial Profile'):
        # Reshape and calculate mean energy per radius
        shower_reshaped = shower.reshape(shower.shape[0], 45, 50, 18)
        energy_sums = np.sum(shower_reshaped, axis=(1, 2))
        mean_energy_per_level = np.mean(energy_sums, axis=0) / 1000
        energy_per_level_list.append(mean_energy_per_level)
        
        # Calculate standard error of the mean (SEM)
        std_energy = np.std(energy_sums, axis=0) / 1000
        sem_energy = std_energy / np.sqrt(n_showers)
        energy_errors.append(sem_energy)
    
    # Create bins for 18 radial bins
    bins = np.linspace(0, 18, 19)
    
    # Plot main profiles
    for i, (mean_energy, errors, label, color) in enumerate(zip(energy_per_level_list, energy_errors, labels, colors)):
        if i == 0:
            # Reference data (GEANT4) - filled stairs
            ax_main.stairs(mean_energy, bins, color='lightgrey', fill=True, label=label)
            
            # Error band with hatching for reference
            if show_error_bands:
                ax_main.stairs(mean_energy + errors, bins, baseline=mean_energy - errors,
                              color='dimgrey', linewidth=3, hatch='///', fill=False)
        else:
            # Generated data - step line
            ax_main.stairs(mean_energy, bins, linestyle='-', linewidth=3, 
                          color=color, fill=False, label=label)
            
            # Transparent error band for generated data
            if show_error_bands:
                ax_main.stairs(mean_energy + errors, bins, baseline=mean_energy - errors,
                              color=color, alpha=0.3, fill=True, linewidth=0)
    
    # Configure main plot
    ax_main.set_xlim(0, 18)
    ax_main.set_ylabel('Mean Energy [GeV]', fontsize=axis_fontsize)
    ax_main.set_yscale('log')
    ax_main.tick_params(labelbottom=False, labelsize=axis_fontsize-4)
    
    # Add training strategy to title
    ax_main.set_title(f'{training_strategy}', loc='right', fontsize=axis_fontsize-2, pad=20, weight='bold')
    
    
    # Plot ratios with error bands
    if len(energy_per_level_list) > 1:
        pos = (bins[:-1] + bins[1:]) / 2  # Bin centers
        ratio_plots_profile(ax_ratio, energy_per_level_list[0], energy_per_level_list[1:], 
                           energy_errors[0], energy_errors[1:],
                           bins, pos, colors[1:], show_error_bands, )

    # Configure ratio plot with same x-limits
    ax_ratio.set_xlabel('Radius [bins]', fontsize=axis_fontsize)
    ax_ratio.set_ylabel(r'$\frac{\mathrm{Generated}}{\mathrm{Geant4}}$', fontsize=axis_fontsize)
    ax_ratio.set_xlim(0, 18)  # Explicitly set to match main plot
    ax_ratio.set_ylim(ratio_ylim[0], ratio_ylim[1])
    ax_ratio.axhline(1, linestyle='-', lw=1, color='black')
    ax_ratio.tick_params(labelsize=axis_fontsize-4)
    
    # Save plot if requested
    if save_plot:
        filename = f"{title.replace(' ', '_')}_with_ratio.png"
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=150)
        print(f'Plot saved to {os.path.join(save_dir, filename)}')
    
    plt.show()
    
    # Calculate divergences if provided
    if kl_divergences is not None and wasserstein is not None:
        print(f'Calculating divergences for {title} histograms... \n')
        kl_divergences[title] = {}
        wasserstein[title] = {}
        
        levels = np.arange(18)  # For wasserstein distance calculation
        for i, sample in enumerate(energy_per_level_list[1:], start=1):
            # Calculate Wasserstein distance
            wasserstein_value = wasserstein_d(u=levels, v=levels, 
                                             weight_u=energy_per_level_list[0], 
                                             weight_v=sample)
            wasserstein[title][labels[i]] = wasserstein_value
            # KL divergence
            kl_value = entropy(energy_per_level_list[0], sample)
            kl_divergences[title][labels[i]] = kl_value
            print(f'{labels[i]} - Wasserstein distance: {wasserstein_value:.2e}, Quantiled KL Divergence: {kl_value:.2e}')
    
    return kl_divergences if kl_divergences is not None else {}, wasserstein if wasserstein is not None else {}


def ratio_plots_profile(ax, reference_data, comparison_data_list, ref_errors, comp_errors_list,
                        bins, pos, colors, show_error_bands=True, lims_min=ratio_ylim[0], lims_max=ratio_ylim[1]):
    """
    Specialized ratio plotting for profile data with proper error propagation.
    Matches the external reference implementation for spinal/radial plots.
    """

    eps = 1e-5
    
    # Get current x-axis limits from parent plot
    xlims = ax.get_xlim()
    
    for i, (data, errors, color) in enumerate(zip(comparison_data_list, comp_errors_list, colors)):
        # Calculate ratios with clipping
        ratios = np.clip((data + eps) / (reference_data + eps), lims_min, lims_max)
        
        # Plot the ratio line using stairs
        ax.stairs(ratios, edges=bins, color=color, lw=2)
        
        # Filter positions to only show markers within x-limits
        mask_in_xlim = (pos >= xlims[0]) & (pos <= xlims[1])
        filtered_pos = pos[mask_in_xlim]
        filtered_ratios = ratios[mask_in_xlim] if len(ratios) == len(pos) else ratios
        
        # Mark points that are clipped (only within x-limits)
        mask_min = (filtered_ratios == lims_min)
        if np.any(mask_min):
            ax.plot(filtered_pos[mask_min], filtered_ratios[mask_min], linestyle='', lw=2, 
                   marker='v', color=color, clip_on=False)
        
        mask_max = (filtered_ratios == lims_max)
        if np.any(mask_max):
            ax.plot(filtered_pos[mask_max], filtered_ratios[mask_max], linestyle='', lw=2, 
                   marker='^', color=color, clip_on=False)
        
        # Plot error band with proper error propagation
        if show_error_bands:
            # Avoid division by zero in error calculation
            safe_data = np.maximum(data, eps)
            safe_ref = np.maximum(reference_data, eps)
            
            # Error propagation for ratio
            ratio_err = ratios * np.sqrt(
                (errors / safe_data) ** 2 + (ref_errors / safe_ref) ** 2
            )
            
            # Plot error band with transparency
            ax.stairs(
                ratios + ratio_err,
                edges=bins,
                baseline=ratios - ratio_err,
                color=color,
                alpha=0.3,
                fill=True,
                linewidth=0
            )
    
    # Set y-axis limits
    ax.set_ylim(lims_min, lims_max)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def to_point_cloud(data):
    """
    Converts the input data to a point cloud format and returns it as a dictionary.

    Parameters:
    data: list of dict
        List of dictionaries, each containing 'showers' and 'incident' as keys.

    Returns:
    dict
        Dictionary with keys 'showers' and 'incident', containing the processed point cloud data and incident energies.
    """
    npoint2 = np.count_nonzero(data, axis=-1)
    max_npoint =  npoint2.max()
    print(f"Max number of points: {max_npoint} \n\n")
    
    num_data, n_points, _ = data.shape
    showers_xyz = np.zeros((num_data, n_points, 4, max_npoint))  # List to store the converted shower data in Cartesian coordinates
    # energies_all = []  # List to store the energies of all events

    # Find the maximum number of non-zero points across all events in all datasets
    # max_npoint = max(
    #     max(np.count_nonzero(data_entry[i, :]) for i in range(data_entry.shape[0]))
    #     for data_entry in tqdm(data, desc='Finding max npoints in all datasets')
    # )
    

    # Convert each dataset to point cloud format
    for idx, data_entry in enumerate(data):
        
        showers = data_entry
        # energies = data_entry['incident']
        # energies_all.append(energies)
        print(f" \nDataset {idx + 1} over {len(data)} \nShape of showers: {showers.shape} \n")
        # - Shape of incident: {energies.shape} \n")

        # Reshape the showers array to its original form
        showers = showers.reshape(len(showers), 45, 50, 18)
        
        for j in tqdm(range(len(showers)), file=sys.stdout, position=0, leave=True):
            
            # Find the positions (z, phi, r) where the shower values are greater than 0
            z, phi, r = np.where(showers[j] > 0)
            
            r = r + 0.5  # Adjust radial positions
            z = z + 0.5  # Adjust longitudinal positions
            
            # Get the energy values at those positions
            e = showers[j][np.where(showers[j] > 0)]
            
            # Convert to Cartesian coordinates
            phi = np.radians(360/50 * phi + 360/50/2)
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            
            # Pad the shower data to the maximum number of points found across all events
            len_to_pad = max_npoint - len(x)
            shower = np.array([x, z, y, e])
            shower = np.concatenate((shower, np.zeros((4, len_to_pad))), axis=1)
            shower = shower.astype('float32')
            shower = shower.reshape(1, 4, max_npoint)
            
            # Append the converted shower data to the list
            showers_xyz[idx][j] = shower
    
    # Free up memory if needed 
    # free_memory() 

    # Merge the individual lists of shower data and energies into unified arrays
    # showers_xyz = np.array(showers_xyz)
    # energies_all = np.vstack(energies_all)
    return showers_xyz

def print_shower_stats(showers_pc, keys):
    """
    Prints the minimum and maximum values of the showers in each axis.

    Parameters:
    showers_pc (numpy.ndarray): The shower point cloud array.
    keys (list): List of keys corresponding to each shower in the point cloud.
    """
    print('Min and max values of the showers in each axis')
    print('\n', '+ - ' * 20, '\n')
    
    axis_labels = ['X', 'Y', 'Z']
    
    for j in range(showers_pc.shape[0]):
        print(keys[j])
        for i, axis in enumerate(axis_labels):
            min_val = showers_pc[j][:, i, :].min()
            max_val = showers_pc[j][:, i, :].max()
            print(f'events in {axis}: {min_val}, {max_val}')
        print('\n' + '- ' * 20 + '\n')

def plt_scatter(shower, title='Showers', save_plot=False, save_path=None):
    plt.figure(figsize=(15, 5))
    plt.suptitle(title, size=22)  # Add the title above all subplots

    plt.subplot(1, 3, 1)
    plt.scatter(shower[0, :], shower[1, :], s=1)
    plt.xlabel('x', size=18)
    plt.ylabel('y', size=18)

    plt.subplot(1, 3, 2)
    plt.scatter(shower[0, :], shower[2, :], s=1)
    plt.xlabel('x', size=18)
    plt.ylabel('z', size=18)

    plt.subplot(1, 3, 3)
    plt.scatter(shower[1, :], shower[2, :], s=1)
    plt.xlabel('y', size=18)
    plt.ylabel('z', size=18)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title

    # Save plot if requested
    if save_plot and save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f'Plot saved to {save_path}')

    plt.show()

def plot_cog(showers_point_cloud, simulation_labels=None, 
             colors=None, log_scale=True, training_strategy='vanilla', 
             save_plot=False, save_dir='./results', filename='cog_plot'):
    """
    Plots the center of gravity of showers data.

    Parameters:
    - showers_point_cloud: List or array of showers.
    - colors: List of colors for each histogram.
    - simulation_labels: Dictionary of labels for each histogram.
    - log_scale: Boolean to determine if the plot should use a logarithmic scale.
    - training_strategy: Training strategy label to display on the plot.
    - save_plot: bool, optional (default=False). If True, saves the plot to the specified directory.
    - save_dir: str, optional (default='./results'). Directory to save the plot when save_plot is True.
    - filename: str, optional (default='cog_plot.png'). Filename to save the plot.
    """
    if isinstance(simulation_labels, dict):
        labels = list(simulation_labels.keys())
    else:
        labels = [f'Shower {i+1}' for i in range(len(showers_point_cloud))]

    colors = plt.cm.turbo(np.linspace(0, 1, len(showers_point_cloud))) if colors is None else colors    
    _, axs = plt.subplots(1, 3, figsize=(15, 5))
    axis_labels = ['X', 'Y', 'Z']
    print('Log scale:') if log_scale else print('Linear scale:')

    for j, ax in enumerate(axs):
        for i, shower in tqdm(enumerate(showers_point_cloud), desc=f'Plotting histograms for axis {axis_labels[j]}'):
            # shower[:, -1, :] /= 0.033 # post-processing
            energy_sum = np.sum(shower[:, -1, :], axis=-1) + 1e-8

            numerator = np.sum(shower[:, -1, :] * shower[:, j, :], axis=-1)
            cog = numerator / energy_sum 
            bins_min = np.log10(1e-5)
            if axis_labels[j] == 'Y':
                bins_min = np.log10(5*1e0)
                
            ax.hist(cog, bins= np.logspace(bins_min, np.log10(cog.max()), 100) if log_scale else 100,
                    color=colors[i], alpha=0.2 if i == 0 else 1, linewidth=2. if i == 0 else 1.5,
                    histtype='stepfilled' if i == 0 else 'step', label=labels[i])

        ax.set_xscale('log') if log_scale else ax.set_xscale('linear')
        ax.set_yscale('log')
        ax.set_xlabel(f'Center of Gravity {axis_labels[j]}', fontsize=14)
        ax.set_ylabel('# showers', fontsize=14) if j == 0 else ax.set_ylabel('')
        axs[j].set_xlim(-4, 4)  # Set x-axis limit to 0, 18

        if j == 1:
            ax.legend(fontsize=12, frameon=True) 
            axs[j].set_xlim(5, 25)  # Set x-axis limit to 0, 18

        if j==0:
            # Add training strategy label inside the plot
            anchored_text = AnchoredText(f'{training_strategy}', loc='upper right', prop=dict(size=15), frameon=True)
            ax.add_artist(anchored_text)

    plt.tight_layout()

    # Save plot if requested
    if save_plot:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        log_name = '_log' if log_scale else ''
        filename = filename + log_name + '.png'
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
        print(f'Plot saved to {os.path.join(save_dir, filename)}')

    plt.show()

def plot_radial_energy_pc(showers_point_cloud, simulation_labels=None, 
                        kl_divergences=None, 
                        colors=None, title='Radial Energy Histogram'):
    """
    Plots the radial distribution of energy.

    Parameters:
    - showers_point_cloud: List or array of showers.
    - simulation_labels: Dictionary of labels for each histogram.
    - colors: List of colors for each histogram.
    """
    # Set labels and colors
    labels = list(simulation_labels.keys()) if simulation_labels else [f'Shower {i + 1}' for i in range(len(showers_point_cloud))]
    colors = colors if isinstance(colors, (list, np.ndarray)) else plt.cm.turbo(np.linspace(0, 1, len(showers_point_cloud)))
    _, ax = plt.subplots(figsize=(10, 6))

    radial_energy_kl = {} if kl_divergences is not None else None

    for i, shower in tqdm(enumerate(showers_point_cloud), desc='Plotting radial energy'):
        # Preprocess each shower
        mask = (shower[:, -1, :] > 0.01515)
        visible_energy = shower[:, -1, :][mask]

        r_fake = np.round(np.sqrt(shower[:, 0, :][mask]**2 + shower[:, 2, :][mask]**2), decimals=2)
        # Compute histogram
        hist, bins = np.histogram(r_fake, bins=18, weights=visible_energy)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.hist(bin_centers, bins=18, weights=hist, color=colors[i], 
                alpha=0.2 if i == 0 else 1, linewidth=2 if i == 0 else 1.5, 
                histtype='stepfilled' if i == 0 else 'step', label=labels[i % len(labels)])
        
        if i == 0:
            ref_hist = hist
        elif kl_divergences is not None:
            kl_divergence = quantiled_kl_divergence(ref_hist, hist, bins= 100, quantiles_=(0,1))
            radial_energy_kl[labels[i]] = kl_divergence
            print(f'KL Divergence for {labels[i]}: {kl_divergence:.2e}')
    
    
    print(f'Minimum radial energy: {r_fake.min()}  - Maximum radial energy: {r_fake.max()} ')        

    ax.set_xlim(0, 18)  # Set x-axis limit to 0, 18
    ax.set_xlabel('Radial Distance [radial bin]', fontsize=24)
    ax.set_ylabel('Radial Energy [MeV]', fontsize=24)
    ax.set_yscale('log')  # Optional: set y-axis to log scale if required
    ax.legend(fontsize=22)
    ax.set_title('Radial Energy Distribution', fontsize=26)
    plt.tight_layout()
    plt.show()
    if kl_divergences is not None:
        kl_divergences[title] = radial_energy_kl
        return kl_divergences

def plot_coordinates(showers_pc, simulation_labels, colors, training_strategy='vanilla', save_plot=False, save_dir='./results', filename='coordinates_plot.png'):
    """
    Plots histograms of the data in showers_pc with respect to the given simulation_labels.

    Parameters:
    - showers_pc: A 4D NumPy array of shape (datasets, samples, axes, values)
    - simulation_labels: A dictionary of {label_index: label_name} for each dataset in showers_pc
    - training_strategy: Training strategy label to display on the plot.
    - save_plot: bool, optional (default=False). If True, saves the plot to the specified directory.
    - save_dir: str, optional (default='./results'). Directory to save the plot when save_plot is True.
    - filename: str, optional (default='coordinates_plot.png'). Filename to save the plot.
    """
    
    # Create subplots
    _, axs = plt.subplots(1, 4, figsize=(15, 5))
    axis_labels = ['X', 'Y', 'Z', 'E']
    labels = list(simulation_labels.keys())
    
    # Iterate over the subplots (axs) and the datasets
    for j in range(4):
        for i in tqdm(range(showers_pc.shape[0]), desc=f'Plotting coordinates for axis {axis_labels[j]}'):  # Loop through datasets
            data_res = np.reshape(showers_pc[i, :, j, :], -1)  # Flatten the data
            # Filter out zero values
            data_res = data_res[data_res != 0]
            
            bins = 45 if j == 1 else 18 * 2  # Set number of bins for each subplot
            axs[j].hist(data_res, bins=bins, color=colors[i], 
                alpha=0.2 if i == 0 else 1, linewidth=1.5 if i == 0 else 1.5,  # density=True,
                histtype='stepfilled' if i == 0 else 'step', label=labels[i])
        
        # Set labels and titles for each subplot
        axs[j].set_xlabel(f'{axis_labels[j]}', fontsize=20)  # Replace with your relevant label
        axs[j].set_ylabel('Entries', fontsize=20) if j == 0 else axs[j].set_ylabel('')
        axs[j].set_title(f'{axis_labels[j]} Coordinates', fontsize=20)

        if j == 1:
            axs[j].legend(fontsize=12, frameon=True)
        if j == 0:
            # Add training strategy label inside the plot
            anchored_text = AnchoredText(f'{training_strategy}', loc='upper right', prop=dict(size=15), frameon=True)
            axs[j].add_artist(anchored_text)
        
        
    # Optional: Set y-axis and x-axis to log scale if needed
    for ax in axs:
        ax.set_yscale('log')  # Uncomment if you want a log scale on the y-axis
        # ax.set_xscale('log')  # Uncomment if you want a log scale on the x-axis

    plt.tight_layout()  # Improve layout

    # Save plot if requested
    if save_plot:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
        print(f'Plot saved to {os.path.join(save_dir, filename)}')

    plt.show()

# # # # # # # metrics for evaluation # # # # # # #
def wasserstein_d(u: np.ndarray, v: np.ndarray, weight_u: np.ndarray = None, weight_v: np.ndarray = None) -> float:
    """
    Calculate the normalized Wasserstein distance between two samples.
    
    Parameters:
    u (np.ndarray): Array of sample points from the first distribution.
    v (np.ndarray): Array of sample points from the second distribution.
    weight_u (np.ndarray, optional): Array of weights for the first sample. Default is None.
    weight_v (np.ndarray, optional): Array of weights for the second sample. Default is None.
    
    Returns:
    float: Normalized Wasserstein distance between the two distributions.
    """
    
    # Calculate the Wasserstein distance
    wasserstein_value = wasserstein_distance(u, v, weight_u, weight_v)
    
    # Compute standard deviation for normalization
    if weight_u is None:
        std_u = np.std(u, ddof=1)  # Using ddof=1 for unbiased estimate of standard deviation
    else:
        mean_u = np.average(u, weights=weight_u)
        variance_u = np.average((u - mean_u) ** 2, weights=weight_u)
        std_u = np.sqrt(variance_u)
    
    # Normalize Wasserstein distance
    if std_u != 0:
        wasserstein_value /= std_u
    else:
        raise ValueError("Standard deviation of the first distribution is zero, cannot normalize.")
    
    return wasserstein_value

def quantiled_kl_divergence(
    sample_ref: np.ndarray,
    sample_gen: np.ndarray,
    num_quantiles: int = 30,
    return_bin_edges=False,
    show = False,
    labels = None,
):
    """Calculate the KL divergence using quantiles on sample_ref to define the bounds.

    Parameters
    ----------
    sample_ref : np.ndarray
        The first sample to compare (this is the reference, so in the context of
        jet generation, those are the real jets).
    sample_gen : np.ndarray
        The second sample to compare (this is the model/approximation, so in the
        context of jet generation, those are the generated jets).
    num_quantiles : int
        The number of bins to use for the histogram. Those bins are defined by
        equiprobably quantiles of sample_ref.
    return_bin_edges : bool, optional
        If True, return the bins used to calculate the KL divergence.
    """
    bin_edges = np.quantile(sample_ref, np.linspace(0, 1, num_quantiles + 1))
    bin_edges[0] = float("-inf")
    bin_edges[-1] = float("inf")
    pk = np.histogram(sample_ref, bin_edges)[0] / len(sample_ref) + 1e-6
    qk = np.histogram(sample_gen, bin_edges)[0] / len(sample_gen) + 1e-6
    kl = entropy(pk, qk)
    
    if show:
        labels = labels if labels is not None else ['Sample Ref', 'Sample Gen']
        plt.figure(figsize=(10, 5))

        # Plot KL Divergence
        plt.subplot(1, 2, 2)
        plt.bar(range(num_quantiles), pk, alpha=0.5, label=labels[0])
        plt.bar(range(num_quantiles), qk, alpha=0.5, label=labels[1])
        plt.legend(loc='upper right')
        plt.title(f'Quantile KLD: {kl:.4f}')
        plt.xlabel('Bin edges')
        plt.ylabel('Probability')

        plt.tight_layout()
        # plt.show()

    if return_bin_edges:
        return kl, bin_edges
    return kl

# # # # # # # Final Plot for the fine tuning # # # # # # #
def insert_baseline_values(target_dict, baseline_dict):
    # Function to insert baseline values at the first position

    for key, value in baseline_dict.items():
        if key in target_dict:
            target_dict[key] = {**value, **target_dict[key]}
        else:
            target_dict[key] = value

def annotate_bars(bars, ax):
    """Annotate bars with their height."""
    for bar in bars.patches:
        ax.annotate(f'{bar.get_height():.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def plot_dataframe(data: dict, title: str):
    """
    Convert the nested dictionary to a DataFrame, calculate the averaged metrics,
    and visualize it.
    """
    if not data:
        raise ValueError("data cannot be None or empty.")
    
    df = pd.DataFrame(data)
    df['Averaged'] = df.mean(axis=1)
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()
    
    colors = ['red' if index.startswith('Vanilla') else 'blue' if index.startswith('Finetune') else 'gray' for index in df.index]
    
    for i, column in enumerate(df.columns[:-1]):
        bars = df[column].plot(kind='bar', ax=axes[i], color=colors)
        axes[i].set_title(f"{column} - {title}", fontsize=18)
        axes[i].set_xlabel('Sample Labels')
        axes[i].set_ylabel(column)
        axes[i].tick_params(axis='x', rotation=45)
        annotate_bars(bars, axes[i])
    
    # Plot the 'Averaged' column in the center subplot (2, 2)
    ax_center = fig.add_subplot(3, 3, 8)
    bars = df['Averaged'].plot(kind='bar', ax=ax_center, color=colors)
    ax_center.set_title(f"Averaged - {title}", fontsize=26)
    ax_center.set_xlabel('Sample Labels', fontsize=26)
    ax_center.set_ylabel('Averaged', fontsize=26)
    ax_center.tick_params(axis='x', rotation=45)
    annotate_bars(bars, ax_center)
    
    legend_elements = [
        Line2D([0], [0], color='red', lw=4, label='Vanilla'),
        Line2D([0], [0], color='blue', lw=4, label='Finetune')
    ]
    axes[0].legend(handles=legend_elements, loc='upper left')
    
    for j in range(len(df.columns) - 1, len(axes)):
        if axes[j] != ax_center:
            fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
    print(df)
    return df

# # # # # # # Final Plot for the fine tuning # # # # # # #
# To be fixed!!!

def get_plot_axes(rows: int, cols: int, size=(12, 10), flatten_axes=False, **kwargs):
    rows = int(rows)
    cols = int(cols)

    assert rows >= 1
    assert cols >= 1

    fig, axes = plt.subplots(nrows=rows, ncols=cols, **kwargs)

    fig.set_figwidth(size[0] * cols)
    fig.set_figheight(size[1] * rows)

    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)

    if flatten_axes:
        return np.reshape(axes, newshape=[-1])

    return axes

def plot_finals(data: dict, title=None, metric='acc', legend='best', ax=None,
                fontsize=20, color=None):
    if ax is None:
        ax = plt.gca()

    x_ticks = [10**(i + 3) for i in range(6)] + [10 * 10**2]  # last is 2M
    x_labels = [f'$10^{i + 3}$' for i in range(len(x_ticks) - 1)]

    # Initialize colors to 'turbo' colormap if color is None
    if color is None:
        import matplotlib.cm as cm
        turbo_cmap = cm.get_cmap('turbo', len(data))
        color = {key: turbo_cmap(i) for i, key in enumerate(data.keys())}

    for key, data in data.items():
        if len(data) == 2:
            values, default_color = data
            line_style = None
        else:
            values, default_color, line_style = data

        values = np.array(values)

        if len(values.shape) > 1:
            # plot uncertainty bands
            std = values.std(0)
            mean = values.mean(0)
            ax.fill_between(x_ticks[:values.shape[-1]], mean - std,
                            mean + std, alpha=0.4, color=color.get(key, default_color))

            # redefine values to be the average (so that its plot next line)
            values = values.mean(axis=0)

        ax.plot(x_ticks[:len(values)], values, color=color.get(key, default_color), label=key,
                marker='o', markersize=10, linestyle=line_style)

    if isinstance(title, str):
        ax.set_title(title, y=0.92, pad=15, loc='left', fontsize=fontsize + 2)

    ax.set_xlabel('Number of training shower', fontsize=fontsize)
    ax.set_ylabel('Accuracy' if metric.lower() in ['acc', 'accuracy'] else 'AUC',
                  fontsize=fontsize)

    ax.set_xscale('log')
    ax.set_xticks(x_ticks[:-1], x_labels, fontsize=fontsize - 2)
    ax.yaxis.set_tick_params(labelsize=fontsize - 2)

    ax.legend(loc=legend.lower(), fontsize=fontsize - 2)
    return ax

def plot_finals_tmp(data: dict, title=None, metric='acc', legend='best', ax=None,
                    fontsize=20, color=None, legend_added=False):
    if ax is None:
        ax = plt.gca()
    default_color = (0, 117 / 255, 116 / 255)  # Example RGB color
    line_style = 'solid'  # Example line style

    # x_ticks = [10**(i + 3) for i in range(2)] + [ 10**5]  # last is 100k
    # x_labels = [f'$10^{i + 3}$' for i in range(len(x_ticks) - 1)] + ['$ 10^5$']

    # if 10-90
    # x_ticks = [ 10**2, 5*10**2 ]+[10**(i + 2) for i in range(1,3)] + [ 0.5*10**5]  # last is 50k
    x_ticks = [100, 500, 1000, 5000, 10000, 50000, 100000]  # last is 100k
    x_labels = ['$10^2$', '$5 \\times 10^2$', '$10^3$', '$5 \\times 10^3$', '$10^4$', '$5 \\times 10^4$', '$10^5$']  # last is 100k
    # x_labels = [f'$10$', f'$5 \\times 10^1$',f'$10^2$',f'$5 \\times 10^2$' ]+[f'$10^{i + 2}$' for i in range(1,3)] + ['$ 0.5*10^5$']
    # x_ticks = [10**3, 5*10**2, 10**2, 5*10**1, 10]
    # x_labels = [f'$10^3$', f'$5 \\times 10^2$', f'$10^2$', f'$5 \\times 10^1$', f'$10$']

    # Initialize colors to 'turbo' colormap if color is None
    if color is None:
        turbo_cmap = cm.get_cmap('turbo', len(data))
        
        # Define specific colors for baseline, finetune, and vanilla
        specific_colors = {
            'baseline': 'gray',
            'finetune': 'red',
            'vanilla': 'blue'
        }
        
        # Create the color mapping
        color = {key: specific_colors.get(key, turbo_cmap(i)) for i, key in enumerate(data.keys())}
    for key, values in data.items():
        values = np.array(values)

        if len(values.shape) > 1:
            # plot uncertainty bands
            std = values.std(0)
            mean = values.mean(0)
            ax.fill_between(x_ticks[:values.shape[-1]], mean - std,
                            mean + std, alpha=0.4, color=color.get(key, default_color))

            # redefine values to be the average (so that its plot next line)
            values = values.mean(axis=0)

        ax.plot(x_ticks[:len(values)], values, color=color.get(key, default_color), label=key,
                marker='o', markersize=10, linestyle=line_style)

    if isinstance(title, str):
        ax.set_title(title, loc='left', fontsize=fontsize + 4)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=fontsize - 2)
    ax.yaxis.set_tick_params(labelsize=fontsize - 2)

    if not legend_added:
        ax.legend(loc=legend.lower(), fontsize=fontsize - 2)
    return ax