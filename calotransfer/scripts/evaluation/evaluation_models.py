import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from tqdm import tqdm
import os
import utils.plot_evaluate as plot
import json
import argparse

# Custom imports
import utils.plot_evaluate as plot
from utils.preprocessing_utils import read_hdf5_file2
import matplotlib.gridspec as gridspec
# Configure matplotlib
plt.rcParams['text.usetex'] = False
from configs import Configs
configs = Configs()
class ShowerAnalysisConfig:
    """Configuration class for shower analysis parameters."""
    def __init__(self):
        self.BASE_PATH = '/data/dust/user/valentel/beegfs.migration/dust/evaluate/outputsout'
        self.SUBFIX = 'showers.hdf5'

        # if configs.val_ds_type == 'all_10-90':
        #     self.GEANT4_PATH = '/data/dust/user/valentel/maxwell.merged/MyCaloTransfer/CaloTransfer/data/calo-challenge/preprocessing/reduced_datasets/10-90GeV/evaluation/10k_val_dset4_prep_10-90GeV_fixed.hdf5'
        # elif configs.val_ds_type == '1-1000GeV':
        #     self.GEANT4_PATH = '/data/dust/user/valentel/maxwell.merged/MyCaloTransfer/CaloTransfer/data/calo-challenge/preprocessing/reduced_datasets/1-1000GeV/evaluation/10k_val_dset4_prep_1-1000GeV_cylindrical.hdf5'
        
        self.ENERGY_SCALING_FACTOR = 0.033
        

    def get_checkpoint_paths(self, training_strategy, training_step, ema=''):
        """Get checkpoint paths based on training strategy."""
        from utils.paths_trainings_cleaned import showers_ckpt_paths

        paths = showers_ckpt_paths.get(training_strategy, {})
        print(f"Paths for strategy '{training_strategy}': {paths}")  # Debugging

        formatted_paths = {}
        for key, path_template in paths.items():
            if path_template is None:
                print(f"Warning: No valid path for key '{key}'. Skipping.")
                continue  # Or set formatted_paths[key] = None if you want to keep the key
            formatted_path = path_template.format(training_step=training_step, ema=ema)
            formatted_paths[key] = formatted_path  # Alternatively, add the subfix if needed
        print(f"Formatted paths: {formatted_paths}")  # Debugging
        return formatted_paths

class ShowerAnalysis:
    """Main class for shower analysis operations."""
    
    def __init__(self, config: ShowerAnalysisConfig, training_strategy: str = 'finetune', training_step: int = 200_000,output_dir: str = './results/diffusion/'):
        self.config = config
        self.training_strategy = training_strategy
        self.output_dir = output_dir
        self.simulation_labels = config.get_checkpoint_paths(training_strategy = training_strategy, training_step=training_step)
        self.showers = []
        self.incidents = []
        self.colors = None
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_geant4_data(self):
        """Load initial GEANT4 data."""
        dataset_names, incident_energies, shower = read_hdf5_file2(self.simulation_labels['GEANT4'])
        shower /= self.config.ENERGY_SCALING_FACTOR
        self.showers.append(shower)
        self.incidents.append(incident_energies.squeeze())
        return dataset_names, shower.shape, incident_energies.shape

    def load_simulation_data(self):
        """Load simulation data with faster error handling."""
        for key, path in tqdm(self.simulation_labels.items(), 
                            file=sys.stdout, 
                            position=0, 
                            leave=True, 
                            desc='Loading datasets'):
            if key == 'GEANT4':
                continue
                
            complete_path = f"{path}/{self.config.SUBFIX}"
            
            # Basic file check
            if not os.path.exists(complete_path):
                print(f"WARNING: File does not exist: {complete_path}")
                continue
            
            try:
                tmp_a, incidents_var_name, showers_var_name = read_hdf5_file2(complete_path)
                
                # Only add non-empty datasets
                if incidents_var_name.size > 0 and showers_var_name.size > 0:
                    self.showers.append(showers_var_name)
                    self.incidents.append(incidents_var_name)
                    
                    yield {
                        'key': key,
                        'dataset_names': tmp_a,
                        'shower_shape': showers_var_name.shape,
                        'incidents_shape': incidents_var_name.shape,
                        'path': complete_path
                    }
                else:
                    print(f"WARNING: Empty data from {key}, skipping")
            except Exception as e:
                print(f"ERROR loading {key}: {str(e)}")
                continue

    def prepare_data_for_analysis(self):
        """Convert data to numpy arrays and setup colors."""
        self.showers_numpy = np.array(self.showers)
        self.incidents_numpy = np.array(self.incidents)
        
        # Apply energy scaling to GEANT4 data
        # self.showers_numpy[1] /= self.config.ENERGY_SCALING_FACTOR
        
        # Setup color scheme
        cmap = colormaps['turbo']
        self.colors = ['gray'] + [cmap(i / (len(self.showers)-1)) for i in range(1, len(self.showers))]

    def run_analysis(self):
        """Run the main analysis pipeline."""
        kl_divergences = {} 
        wasserstein_dist = {}
        
        # Run calibration analysis
        self._run_plot_analysis(kl_divergences, wasserstein_dist)
        showers_pc = plot.to_point_cloud(self.showers_numpy)
        self._plot_shower_analysis(showers_pc)
        self._print_and_save_shower_stats(showers_pc)
        
        return kl_divergences, wasserstein_dist

    def _run_plot_analysis(self, kl_divergences, wasserstein_dist):
        """Run all plot analysis functions with their required arguments."""
        plot_configurations = [
            
            # (plot_function, requires_incidents)
            (plot.plot_visible_energy, False),
            (plot.plot_calibration_histograms, True),
            (plot.plot_energy_sum, False),
            (plot.plot_occupancy, False),
            (plot.plot_energy_layer, False),
            (plot.plot_radial_energy, False),
        ]
        
        for plot_func, needs_incidents in plot_configurations:
            args = [self.showers_numpy]
            if needs_incidents:
                args.append(self.incidents_numpy)

            kl_divergences, wasserstein_dist = plot_func(
                *args,
                simulation_labels=self.simulation_labels,
                colors=self.colors,
                kl_divergences=kl_divergences,
                wasserstein=wasserstein_dist,
                training_strategy=self.training_strategy,
                save_plot=True,
                save_dir=self.output_dir,
            )

    def _plot_shower_analysis(self, showers_pc):
        """Plot shower analysis."""

        for log in [True, False]:
            plot.plot_cog(showers_pc, 
                        simulation_labels=self.simulation_labels, 
                        colors=self.colors,
                        log_scale=log,
                        training_strategy=self.training_strategy,
                        save_plot=True,
                        save_dir=self.output_dir)
        
        plot.plot_coordinates(showers_pc, 
                            simulation_labels=self.simulation_labels, 
                            colors=self.colors,
                            training_strategy=self.training_strategy,
                            save_plot=True,
                            save_dir=self.output_dir)

    def _print_and_save_shower_stats(self, showers_pc):
        """Print and save shower statistics."""
        keys = list(self.simulation_labels.keys())  # ['GEANT4', 'Vanilla', 'Finetune']
        plot.print_shower_stats(showers_pc, keys)
        
        max_index = min(showers_pc.shape[1], self.incidents_numpy.shape[1]) - 1

        for j in np.linspace(0, max_index, 10).astype(int):
            for i, key in enumerate(keys):
                title = f"{key}_incident_ENERGY_{round(self.incidents_numpy[i, j], 3)}GeV"
                filename = os.path.join(self.output_dir, 'showers/', f"{title.replace(' ', '_')}.png")
                plot.plt_scatter(showers_pc[i, j], title=title, save_plot=True, save_path=filename)

    
    def reorganize_dict(self, data):
        """
        Reorganize dictionary and calculate averages for all training strategies.
        
        Args:
            data (dict): Input dictionary with metrics data
            
        Returns:
            dict: Reorganized dictionary with averages
        """
        new_dict = {}
        strategy_averages = {}

        # Process each metric in the data
        for metric_name, values in data.items():
            # Skip if no strategies are found
            if not values:
                print(f"Warning: No strategies found for metric '{metric_name}'. Skipping...")
                continue

            # Extract dataset sizes (assume all strategies have the same sizes)
            dataset_sizes = list(next(iter(values.values())).keys())  # Get sizes from the first strategy

            # Collect values for each strategy
            metric_data = {}
            for strategy, size_data in values.items():
                metric_data[strategy] = [size_data.get(size, None) for size in reversed(dataset_sizes)]

            # Store values in new dictionary
            new_dict[metric_name] = metric_data

            # Collect values for averaging
            for strategy, values_list in metric_data.items():
                if strategy not in strategy_averages:
                    strategy_averages[strategy] = []
                strategy_averages[strategy].append(values_list)
        
        # Calculate averages across all metrics for each strategy
        if strategy_averages:  # Ensure there is data to average
            print('\n== Averages ==\n')
            for strategy, averages in strategy_averages.items():
                average_values = np.mean(averages, axis=0).tolist()
                print(f"Average {strategy}: {average_values}")

                # Add averages to dictionary
                if 'Average' not in new_dict:
                    new_dict['Average'] = {}
                new_dict['Average'][strategy] = average_values
        else:
            print("Warning: No valid data found for averaging.")
        
        return new_dict

    def plot_reorganized_data(reorganized_data, plot_finals_tmp, main_title='Normalised Wasserstein Distance', save_plot=False, save_dir='./results', filename='final_w_plot.png'):
        """
        Plots the reorganized data in a 3x3 grid layout with proper color handling.
        """
        fig = plt.figure(figsize=(15, 15))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        axs_flat = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
        ax_center = fig.add_subplot(gs[2, 0:3])

        # Plot individual metrics
        legend_added = False
        for i, (key, data) in enumerate(list(reorganized_data.items())[:-1]):
            if i < len(axs_flat):
                ax = axs_flat[i]

                # Assign colors based on strategy names
                colors = {}
                for strategy in data.keys():
                    if strategy.lower().startswith('finetune'):
                        colors[strategy] = 'red'  # Red for finetune
                    elif strategy.lower().startswith('vanilla'):
                        colors[strategy] = 'blue'  # Blue for vanilla
                    else:
                        colors[strategy] = 'gray'  # Default color for others

                # Always pass a valid legend string
                _ = plot_finals_tmp(
                    data, 
                    title=key, 
                    ax=ax, 
                    color=colors,
                    legend='best'
                )
                if legend_added:
                    # Remove the legend for subsequent axes if one exists
                    leg = ax.get_legend()
                    if leg is not None:
                        leg.remove()
                else:
                    legend_added = True
                    
                if i % 3 == 0:
                    ax.set_ylabel(main_title, fontsize=20)
                ax.grid(True)
                ax.set_ylim([1e-2, 10])

        # Plot average
        last_key = list(reorganized_data.keys())[-1]
        avg_data = reorganized_data[last_key]

        # Assign colors for the average plot
        avg_colors = {}
        for strategy in avg_data.keys():
            if strategy.lower().startswith('finetune'):
                avg_colors[strategy] = 'red'
            elif strategy.lower().startswith('vanilla'):
                avg_colors[strategy] = 'blue'
            else:
                avg_colors[strategy] = 'gray'

        _ = plot_finals_tmp(
            avg_data, 
            title='Average', 
            ax=ax_center, 
            color=avg_colors,
            legend='best'
        )

        # Configure final plot
        ax_center.set_xlabel('Number of training showers', fontsize=20)
        ax_center.set_ylabel(main_title, fontsize=20)
        ax_center.set_ylim([1e-1, 1])
        ax_center.grid(True, linestyle='--')

        # Add main title and adjust layout
        fig.suptitle(main_title, fontsize=40)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_plot:
            plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
            print(f'Plot saved to {os.path.join(save_dir, filename)}')

        plt.show()

def load_wasserstein_dist(output_dir, strategy):
    file_path = os.path.join(output_dir, f'wasserstein_dist_{strategy}.json')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            wasserstein_dist = json.load(f)
        return wasserstein_dist
    else:
        return None
def load_kl_divergences(output_dir, strategy):
    file_path = os.path.join(output_dir, f'kl_divergences_{strategy}.json')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            kl_divergences = json.load(f)
        return kl_divergences
    else:
        return None
    
def main():
    """Main execution function."""
    base_output_dir = './results/diffusion'
    config = ShowerAnalysisConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument('--training_step', type=int, required=True)
    parser.add_argument('--training_strategies', type=str, required=True)
    parser.add_argument('--use_ema', type=lambda x: x.lower() == 'true', required=True)

    args = parser.parse_args()

    training_step = args.training_step  # 10_000, 100_000, 200_000, 250_000, 500_000, 750_000, 1_000_000
    training_strategies = args.training_strategies
    use_ema = args.use_ema
    if use_ema:
        ema = ''
    else:
        ema = '_no_ema'

    # Convert the training_strategies string into a list
    training_strategies_list = training_strategies.strip('[]').split(',')
    training_strategies_list = [strategy.strip() for strategy in training_strategies_list]

    merged_dict_w = {}
    merged_dict_kl = {}
    print(f"Training strategies: {training_strategies_list}")
    print(f"\n{'='*50}")
    print(f"Running analysis for training step: {training_step}")
    print(f"{'='*50}")

    for strategy in training_strategies_list:
        print(f"\n{'='*50}")
        print(f"Running analysis for {strategy} strategy with ema {use_ema}...")
        print(f"{'='*50}")

        training_paths = config.get_checkpoint_paths(training_strategy=strategy, 
                                                    training_step=training_step,
                                                    ema=ema)
        training_paths_list = list(training_paths.values())
        print(f"Training paths: {training_paths_list}")

        # Check if there are any paths available
        if not training_paths_list:
            print(f"ERROR: No paths found for strategy {strategy}. Skipping...")
            continue
            
        # Create a folder name based on the first available path or strategy name
        if len(training_paths_list) > 1:
            second_path = training_paths_list[1]
            folder_name = os.path.basename(second_path)
        elif len(training_paths_list) == 1:
            folder_name = os.path.basename(training_paths_list[0])
        else:
            # Fallback if list is somehow empty despite our check above
            folder_name = f"{strategy}_{training_step}"

        output_dir = os.path.join(base_output_dir, strategy+ema, folder_name, )

        print(f'\n== Creating directory: {output_dir}')

        # Initialize analyzer
        analyzer = ShowerAnalysis(config, training_strategy=strategy, training_step=training_step, output_dir=output_dir)

        # Load GEANT4 data FIRST
        analyzer.load_geant4_data()

        # Load simulation data (other datasets)
        for data_info in analyzer.load_simulation_data():
            print(f"\nLoaded {data_info['key']}: {data_info['shower_shape']}")
        
        # Prepare data for analysis
        analyzer.prepare_data_for_analysis()

        # Run analysis or load results
        if strategy != 'vanilla':
            kl_divergences, wasserstein_dist = analyzer.run_analysis()
        else:
            wasserstein_dist = load_wasserstein_dist(output_dir, strategy)
            kl_divergences = load_kl_divergences(output_dir, strategy)

        # Store results
        merged_dict_w[strategy] = wasserstein_dist
        merged_dict_kl[strategy] = kl_divergences
        
        # Save kl_divergences and wasserstein_dist to a file
        with open(os.path.join(output_dir, f'kl_divergences_{strategy}.json'), 'w') as f:
            json.dump(kl_divergences, f, indent=4)
        
        with open(os.path.join(output_dir, f'wasserstein_dist_{strategy}.json'), 'w') as f:
            json.dump(wasserstein_dist, f, indent=4)

    # Restructure merged_dict to group by metrics for Wasserstein distances
    merged_metrics_w = {}
    merged_metrics_kl = {}
    
    for strategy in training_strategies_list:
        if strategy not in merged_dict_w:
            continue
        strategy_data = merged_dict_w[strategy]
        if not strategy_data:  # Skip if empty
            continue
        for metric, data in strategy_data.items():
            if metric not in merged_metrics_w:
                merged_metrics_w[metric] = {}
            merged_metrics_w[metric][strategy] = data

    # Only process further if we have data
    if merged_metrics_w:
        # Reorganize the restructured data for Wasserstein distances
        analyzer = ShowerAnalysis(config, training_step=training_step)  # Temporary instance for method access
        reorganized_data_w = analyzer.reorganize_dict(merged_metrics_w)
        print("Reorganized Data (Wasserstein):", reorganized_data_w)
        
        # Plot the reorganized data for Wasserstein distances
        ShowerAnalysis.plot_reorganized_data(
            reorganized_data_w,
            plot.plot_finals_tmp,
            save_plot=True,
            save_dir=output_dir,
            filename='final_w_plot.png',
            main_title='Normalised WD'
        )

    # Restructure merged_dict to group by metrics for KL divergences
    for strategy in training_strategies_list:
        if strategy not in merged_dict_kl:
            continue
        strategy_data = merged_dict_kl[strategy]
        if not strategy_data:  # Skip if empty
            continue
        for metric, data in strategy_data.items():
            if metric not in merged_metrics_kl:
                merged_metrics_kl[metric] = {}
            merged_metrics_kl[metric][strategy] = data

    # Only process further if we have data
    if merged_metrics_kl:
        # Reorganize the restructured data for KL divergences
        reorganized_data_kl = analyzer.reorganize_dict(merged_metrics_kl)
        print("Reorganized Data (KL Divergence):", reorganized_data_kl)
        
        # Plot the reorganized data for KL divergences
        ShowerAnalysis.plot_reorganized_data(
            reorganized_data_kl,
            plot.plot_finals_tmp,
            save_plot=True,
            save_dir=output_dir,
            filename='final_kl_plot.png',
            main_title='KL Divergence'
        )
if __name__ == "__main__":
    main()