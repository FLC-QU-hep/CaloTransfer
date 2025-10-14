import os
import sys

# Add the parent directory to Python path to handle imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now do the imports
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import json

# Import project modules
from configs import Configs
import utils.eval_utils_showerflow as eval_utils
from utils.preprocessing_utils import read_hdf5_file2
from utils.paths_configs import sf_eval_paths as dataset_paths
from utils.paths_configs import BASE_OUTPUT_PATH

@dataclass
class ShowerConfig:
    x_bounds: Tuple[float, float] = (-18, 18)
    y_bounds: Tuple[float, float] = (0, 45)
    z_bounds: Tuple[float, float] = (-18, 18)
    
class DeviceManager:
    @staticmethod
    def setup_device() -> torch.device:
        return torch.device('cuda:0')

class DataLoader:
    def __init__(self, config: ShowerConfig):
        self.config = config
        
    def read_hdf5_file(self, path: str) -> Tuple[List, np.ndarray, np.ndarray]:
        """Read and preprocess HDF5 file data."""
        from utils.preprocessing_utils import read_hdf5_file2
        keys, energy, events = read_hdf5_file2(path)
        return keys, energy, self._unnormalize_events(events)
    
    def _unnormalize_events(self, events: np.ndarray) -> np.ndarray:
        """Unnormalize event data based on configured bounds."""
        events = events.copy()
        x_min, x_max = self.config.x_bounds
        y_min, y_max = self.config.y_bounds
        z_min, z_max = self.config.z_bounds
        
        events[:, 0, :] = (events[:, 0, :] + 1) * (x_max - x_min) / 2 + x_min
        events[:, 1, :] = (events[:, 1, :] + 1) * (y_max - y_min) / 2 + y_min
        events[:, 2, :] = (events[:, 2, :] + 1) * (z_max - z_min) / 2 + z_min
        events[:, -1, :] /= 1000  # Preprocessing to match CaloClouds
        return events
class EventAnalyzer:
    @staticmethod
    def calculate_event_metrics(events: np.ndarray, energy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate number of points and visible energy for events."""
        num_points = np.sum(events[:][:, -1] > 0, axis=1)
        visible_energy = np.sum(events[:][:, -1], axis=1)
        return num_points, visible_energy
    
    @staticmethod
    def calculate_center_of_gravity(events: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate center of gravity metrics for events."""
        x, y, z, e = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
        epsilon = 1e-10
        e_sum = e.sum(axis=1) + epsilon

        cog_x = np.sum((x * e), axis=1) / e_sum
        cog_y = np.sum((y * e), axis=1) / e_sum
        cog_z = np.sum((z * e), axis=1) / e_sum

        mean_cog = np.mean([cog_x, cog_y, cog_z], axis=-1)
        std_cog = np.std([cog_x, cog_y, cog_z], axis=-1)
        return mean_cog, std_cog, cog_x, cog_z

class DataFrameGenerator:
    @staticmethod
    def generate_dataframe(energy: np.ndarray, cog_x: np.ndarray, cog_z: np.ndarray, 
                          clusters_per_layer: np.ndarray, e_per_layer: np.ndarray, 
                          mean_cog: np.ndarray, std_cog: np.ndarray,) -> pd.DataFrame:
        """Generate pandas DataFrame from analyzed data."""
        from configs import Configs
        cfg = Configs()

        normalized_energy = (np.log(energy / energy.min()) / np.log(energy.max() / energy.min())).reshape(-1)
        
        return pd.DataFrame({
            'energy': normalized_energy * 100,
            'cog_x': ((cog_x - mean_cog[0]) / std_cog[0]).reshape(-1),
            'cog_z': ((cog_z - mean_cog[2]) / std_cog[2]).reshape(-1),
            'clusters_per_layer': clusters_per_layer.tolist(),
            'e_per_layer': e_per_layer.tolist(),
        })

class Visualizer:
    @staticmethod
    def plot_metrics(metrics: Dict[str, Any], output_path: str) -> None:
        """Plot and save metrics visualization."""
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for i, (data, title, color) in enumerate(zip(
            metrics['data'], metrics['titles'], metrics['colors'])):
            axs[i].hist(data, bins=50, color=color)
            axs[i].set_title(title)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

class ShowerAnalysis:
    def __init__(self, config: ShowerConfig, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        self.device_manager = DeviceManager()
        self.data_loader = DataLoader(config)
        self.event_analyzer = EventAnalyzer()
        self.df_generator = DataFrameGenerator()
        self.visualizer = Visualizer()
        
        # Initialize DataFrame attributes
        self.df = None
        self.df_cc = None
        
        os.makedirs(output_dir, exist_ok=True)
    
    def run_analysis(self, eval_ds: str = '10-90GeV', dim: str = '10k', pretrained: bool = True) -> None:
        """Run the complete shower analysis pipeline."""
        # Load configuration and paths
        cfg, paths = self._load_config_and_paths(eval_ds, dim)
        
        # Load and analyze main dataset
        _, energy, events = self.data_loader.read_hdf5_file(paths['data_path'])
        
        # Load additional data
        e_per_layer = np.load(paths['e_per_layer_path'])
        clusters_per_layer = np.load(paths['clusters_per_layer_path'])
        
        # Calculate center of gravity metrics
        mean_cog, std_cog, cog_x, cog_z = self.event_analyzer.calculate_center_of_gravity(events)
        
        # Generate and store main DataFrame
        self.df = self.df_generator.generate_dataframe(
            energy, cog_x, cog_z, clusters_per_layer, e_per_layer, mean_cog, std_cog)
        
        # Process CaloClouds data and store its DataFrame
        self._process_caloclouds_data()
        
        # Run model analysis
        self._run_model_analysis(cfg, energy, pretrained)
    
    def _load_config_and_paths(self, eval_ds: str, dim: str) -> Tuple[Any, Dict]:
        """Load configuration and paths for CaloChallenge analysis."""
        from configs import Configs
        from utils.paths_configs import sf_eval_paths as dataset_paths
        
        cfg = Configs()
        if eval_ds in dataset_paths and dim in dataset_paths[eval_ds] and 'data_path' in dataset_paths[eval_ds][dim]:
            return cfg, dataset_paths[eval_ds][dim]
        raise ValueError(f"Invalid dataset dimension: {eval_ds}, {dim}")
    
    def _process_caloclouds_data(self) -> None:
        """Process CaloClouds dataset and store its DataFrame."""
        caloclouds_path = '/data/dust/user/valentel/maxwell.merged/MyCaloTransfer/CaloTransfer/data/calo-clouds/val_10-90GeV_x36_grid_regular_524k_float32.hdf5'
        keys, energy_cc, events_cc = self.data_loader.read_hdf5_file(caloclouds_path)
        
        mean_cog_cc, std_cog_cc, cog_x_cc, cog_z_cc = self.event_analyzer.calculate_center_of_gravity(events_cc)
        
        # Generate and store CaloClouds DataFrame
        self.df_cc = pd.DataFrame({
            'cog_x': ((cog_x_cc - mean_cog_cc[0]) / std_cog_cc[0]).reshape(-1),
            'cog_z': ((cog_z_cc - mean_cog_cc[2]) / std_cog_cc[2]).reshape(-1),

            # 'clusters_per_layer': clusters_per_layer.tolist(),
            # 'e_per_layer': e_per_layer.tolist(),
        })
    
    def _run_model_analysis(self, cfg: Any, energy: np.ndarray, pretrained: bool) -> None:
        """
        Run analysis for different model epochs.
        
        Args:
            cfg: Configuration object
            energy: Energy array
            pretrained: Whether using pretrained models
        """
        # Initialize epoch tracking dictionaries
        wd_per_epoch = {}
        kl_per_epoch = {}
        
        # Define epochs to analyze
        # epochs_to_analyze = [25, 50]  # For testing
        epochs_to_analyze = list(range(0, 1001, 25))  # Full analysis
        
        # Main analysis loop
        for epoch in tqdm(epochs_to_analyze, desc='Epochs'):
            print(f"\n{'='*60}")
            print(f"Processing Epoch {epoch}")
            print(f"{'='*60}")
            if not pretrained and epoch == 0:
                print("Using vanilla training models.")
                continue
            # Get training paths for current epoch
            trainings_paths = self._get_training_paths(epoch, pretrained)
            print(f"Models to load: {list(trainings_paths.keys())}")
            
            # Load models and generate samples
            try:
                _, distributions, model_name_to_index = eval_utils.load_models_and_distributions(
                    trainings_paths, 
                    self.device_manager.setup_device(), 
                    num_blocks=2, 
                    num_inputs=92
                )
                
                samples_dict = eval_utils.generate_samples(
                    distributions, 
                    model_name_to_index, 
                    cfg, 
                    energy, 
                    self.device_manager.setup_device(), 
                    min_energy=energy.min(), 
                    max_energy=energy.max()
                )
                
                # Verify samples generated
                for model_name, samples in samples_dict.items():
                    print(f"Generated samples for {model_name}: shape {samples.shape}")
                    
            except Exception as e:
                print(f"Error loading models or generating samples: {e}")
                continue
            
            # Initialize metrics dictionaries for this epoch
            epoch_results = self._analyze_epoch(
                samples_dict=samples_dict,
                epoch=epoch,
                cfg=cfg,
                pretrained=pretrained
            )
            
            # Store results
            wd_per_epoch[epoch] = epoch_results['wasserstein']
            kl_per_epoch[epoch] = epoch_results['kl']
            
            # Sort dictionaries by epoch
            wd_per_epoch = dict(sorted(wd_per_epoch.items()))
            kl_per_epoch = dict(sorted(kl_per_epoch.items()))
            
            self._plot_aggregate_features(
                wd_per_epoch=wd_per_epoch,
                kl_per_epoch=kl_per_epoch,
                pretrained=pretrained
            )
            
            # Save intermediate results every 200 epochs
            if epoch % 100 == 0:
                self._save_intermediate_results(
                    wd_per_epoch=wd_per_epoch,
                    kl_per_epoch=kl_per_epoch,
                    epoch=epoch
                )
        
        # Save final results
        self._save_final_results(wd_per_epoch, kl_per_epoch)

    def _analyze_epoch(self, samples_dict: Dict, epoch: int, cfg: Any, pretrained: bool) -> Dict:
        """
        Analyze a single epoch and return metrics.
        
        Args:
            samples_dict: Dictionary of generated samples
            epoch: Current epoch number
            cfg: Configuration object
            pretrained: Whether using pretrained models
            
        Returns:
            Dictionary containing wasserstein and kl metrics
        """
        # Create epoch-specific output directory
        epoch_dir = os.path.join(self.output_dir, f'epoch_{epoch}')
        
        # Initialize distance dictionaries
        wasserstein_distances = {}
        kl_distances = {}
        
        # 1. Center of Gravity Analysis
        print("\nAnalyzing Center of Gravity...")
        cog_wd, cog_kl = eval_utils.plot_cog(
            df=self.df,
            df_cc=self.df_cc,
            samples_dict=samples_dict,
            wasserstein_distances=wasserstein_distances,
            kl_distances=kl_distances,
            coord_range=(0, 1),
            save_plots=True,
            save_dir=epoch_dir,
            epoch=epoch,
            pretrained=pretrained
        )
        
        # 2. Clusters per Layer Analysis
        print("\nAnalyzing Clusters per Layer...")
        wd_clusters, w_clusters_per_layer, kl_clusters, kl_clusters_per_layer = eval_utils.plot_per_layer(
            per_layer=self.df['clusters_per_layer'],
            samples_dict=samples_dict,
            sample_range=(2, 47),
            title='Clusters_per_Layer',
            wasserstein_distances=wasserstein_distances,
            kl_distances=kl_distances,
            save_plots=True,
            save_dir=epoch_dir,
            epoch=epoch,
            norm=cfg.sf_norm_points,
            clusters=True,
            pretrained=pretrained
        )
        
        # Plot distance distributions for clusters
        self._plot_layer_distances(
            w_distances=w_clusters_per_layer,
            kl_distances=kl_clusters_per_layer,
            title_prefix='Clusters per Layer',
            epoch_dir=epoch_dir,
            epoch=epoch,
            pretrained=pretrained
        )
        
        # 3. Energy per Layer Analysis
        print("\nAnalyzing Energy per Layer...")
        wd_energy, w_energy_per_layer, kl_energy, kl_energy_per_layer = eval_utils.plot_per_layer(
            per_layer=self.df['e_per_layer'],
            samples_dict=samples_dict,
            sample_range=(47, 92),
            title='Energy_per_Layer',
            wasserstein_distances=wasserstein_distances,
            kl_distances=kl_distances,
            save_plots=True,
            save_dir=epoch_dir,
            epoch=epoch,
            norm=cfg.sf_norm_energy,
            clusters=False,
            pretrained=pretrained
        )
        
        # Plot distance distributions for energy
        self._plot_layer_distances(
            w_distances=w_energy_per_layer,
            kl_distances=kl_energy_per_layer,
            title_prefix='Energy per Layer',
            epoch_dir=epoch_dir,
            epoch=epoch,
            pretrained=pretrained
        )
        
        # Compile and return results
        return {
            'wasserstein': {
                'x': cog_wd['x'],
                'z': cog_wd['z'],
                'Clusters per Layer': w_clusters_per_layer,
                'Energy per Layer': w_energy_per_layer
            },
            'kl': {
                'x': cog_kl['x'],
                'z': cog_kl['z'],
                'Clusters per Layer': kl_clusters_per_layer,
                'Energy per Layer': kl_energy_per_layer
            }
        }

    def _plot_layer_distances(self, w_distances: Dict, kl_distances: Dict, 
                            title_prefix: str, epoch_dir: str, epoch: int, pretrained: bool):
        """
        Plot both Wasserstein and KL distances for layer analysis.
        
        Args:
            w_distances: Wasserstein distances dictionary
            kl_distances: KL distances dictionary
            title_prefix: Prefix for plot titles
            epoch_dir: Directory to save plots
            epoch: Current epoch
            pretrained: Whether using pretrained models
        """
        # Plot Wasserstein distances
        eval_utils.plot_wasserstein_distances(
            w_distances,
            title=f'WD {title_prefix}',
            save_plots=True,
            save_dir=epoch_dir,
            epoch=epoch,
            pretrained=pretrained
        )
        
        # Plot KL distances
        eval_utils.plot_kl_distances(
            kl_distances,
            title=f'KL {title_prefix}',
            save_plots=True,
            save_dir=epoch_dir,
            epoch=epoch,
            pretrained=pretrained
        )

    def _plot_aggregate_features(self, wd_per_epoch: Dict, kl_per_epoch: Dict, pretrained: bool):
        """
        Plot aggregate features across epochs.
        
        Args:
            wd_per_epoch: Wasserstein distances per epoch
            kl_per_epoch: KL distances per epoch
            pretrained: Whether using pretrained models
        """
        print("\nPlotting aggregate features...")
        
        # Plot Wasserstein features
        eval_utils.plot_wasserstein_distances_features(
            wd_per_epoch,
            title='WD_Features_per_Epoch',
            save_plots=True,
            save_dir=self.output_dir,
            pretrained=pretrained
        )
        
        # Plot KL features
        eval_utils.plot_kl_distances_features(
            kl_per_epoch,
            title='KL_Features_per_Epoch',
            save_plots=True,
            save_dir=self.output_dir,
            pretrained=pretrained
        )

    def _save_intermediate_results(self, wd_per_epoch: Dict, kl_per_epoch: Dict, epoch: int):
        """
        Save intermediate results to JSON files.
        
        Args:
            wd_per_epoch: Wasserstein distances per epoch
            kl_per_epoch: KL distances per epoch
            epoch: Current epoch number
        """
        print(f"\nSaving intermediate results at epoch {epoch}...")
        
        # Create intermediate results directory
        intermediate_dir = os.path.join(self.output_dir, 'intermediate_results')
        os.makedirs(intermediate_dir, exist_ok=True)
        
        # Save Wasserstein distances
        wd_path = os.path.join(intermediate_dir, f'WD_Features_up_to_epoch_{epoch}.json')
        with open(wd_path, 'w') as f:
            json.dump(wd_per_epoch, f, indent=2)
        
        # Save KL distances
        kl_path = os.path.join(intermediate_dir, f'KL_Features_up_to_epoch_{epoch}.json')
        with open(kl_path, 'w') as f:
            json.dump(kl_per_epoch, f, indent=2)
        
        print(f"Intermediate results saved to {intermediate_dir}")

    def _save_final_results(self, wd_per_epoch: Dict, kl_per_epoch: Dict):
        """
        Save final results to JSON files.
        
        Args:
            wd_per_epoch: Wasserstein distances per epoch
            kl_per_epoch: KL distances per epoch
        """
        print("\nSaving final results...")
        
        # Save Wasserstein distances
        wd_path = os.path.join(self.output_dir, 'WD_Features_all_epochs.json')
        with open(wd_path, 'w') as f:
            json.dump(wd_per_epoch, f, indent=2)
        print(f"Wasserstein distances saved to {wd_path}")
        
        # Save KL distances
        kl_path = os.path.join(self.output_dir, 'KL_Features_all_epochs.json')
        with open(kl_path, 'w') as f:
            json.dump(kl_per_epoch, f, indent=2)
        print(f"KL distances saved to {kl_path}")
        
        # Create summary statistics
        self._create_summary_statistics(wd_per_epoch, kl_per_epoch)
            
    def _get_training_paths(self, epoch: int, pretrained: bool) -> Dict[str, str]:
        """
        Get training paths based on epoch and training type.
        
        Args:
            epoch: Current epoch number
            pretrained: Whether to use pretrained models
            
        Returns:
            Dictionary mapping model names to their paths
        """
        from utils.paths_configs import BASE_OUTPUT_PATH as base_path

        if epoch == 0 and pretrained:
            return {
                'D = 1 x 10^5': f'{base_path}/Pretrained_CaloClouds-ShowerFlow/ShowerFlow_2024_11_27__17_40_49/ShowerFlow_3580.pth',
                'D = 5 x 10^4': f'{base_path}/Pretrained_CaloClouds-ShowerFlow/ShowerFlow_2024_11_27__17_40_49/ShowerFlow_3580.pth',
                'D = 1 x 10^4': f'{base_path}/Pretrained_CaloClouds-ShowerFlow/ShowerFlow_2024_11_27__17_40_49/ShowerFlow_3580.pth',
                'D = 5 x 10^3': f'{base_path}/Pretrained_CaloClouds-ShowerFlow/ShowerFlow_2024_11_27__17_40_49/ShowerFlow_3580.pth',
                'D = 1 x 10^3': f'{base_path}/Pretrained_CaloClouds-ShowerFlow/ShowerFlow_2024_11_27__17_40_49/ShowerFlow_3580.pth',
                'D = 5 x 10^2': f'{base_path}/Pretrained_CaloClouds-ShowerFlow/ShowerFlow_2024_11_27__17_40_49/ShowerFlow_3580.pth',
                'D = 1 x 10^2': f'{base_path}/Pretrained_CaloClouds-ShowerFlow/ShowerFlow_2024_11_27__17_40_49/ShowerFlow_3580.pth',
                # 'D = 5 x 10^1': f'{base_path}/Pretrained_CaloClouds-ShowerFlow/ShowerFlow_2024_11_27__17_40_49/ShowerFlow_3580.pth',
                # 'D = 1 x 10^1': f'{base_path}/Pretrained_CaloClouds-ShowerFlow/ShowerFlow_2024_11_27__17_40_49/ShowerFlow_3580.pth',

                # 'EWC= 1new': f'{base_path}/Pretrained_CaloClouds-ShowerFlow/ShowerFlow_2024_11_27__17_40_49/ShowerFlow_3580.pth',
                # 'EWC=100': f'{base_path}/Pretrained_CaloClouds-ShowerFlow/ShowerFlow_2024_11_27__17_40_49/ShowerFlow_3580.pth',
            }
        training_type = 'finetune' if pretrained else 'vanilla'

        if pretrained:
            # Finetune training paths
            return  { # === 1-1000 GeV === seed 46
                'D = 1 x 10^5': f'{base_path}/100k_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__16_37_25/ShowerFlow_{epoch}.pth',
                'D = 5 x 10^4':  f'{base_path}/50k_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__14_44_52/ShowerFlow_{epoch}.pth',
                'D = 1 x 10^4': f'{base_path}/10k_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__14_20_58/ShowerFlow_{epoch}.pth',
                'D = 5 x 10^3': f'{base_path}/5k_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__12_59_41/ShowerFlow_{epoch}.pth',
                'D = 1 x 10^3': f'{base_path}/1k_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__13_47_21/ShowerFlow_{epoch}.pth',
                'D = 5 x 10^2': f'{base_path}/500_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__13_43_18/ShowerFlow_{epoch}.pth',
                'D = 1 x 10^2': f'{base_path}/100_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__13_35_39/ShowerFlow_{epoch}.pth',
            }
        
            # { # === 1-1000 GeV === seed 45
            #     'D = 1 x 10^5': f'{base_path}/100k_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__12_27_34/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^4':  f'{base_path}/50k_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__12_28_01/ShowerFlow_{epoch}.pth',
            #     'D = 1 x 10^4': f'{base_path}/10k_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__12_30_54/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^3': f'{base_path}/5k_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__12_31_00/ShowerFlow_{epoch}.pth',
            #     'D = 1 x 10^3': f'{base_path}/1k_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__12_31_46/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^2': f'{base_path}/500_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__12_33_38/ShowerFlow_{epoch}.pth',
            #     'D = 1 x 10^2': f'{base_path}/100_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__12_36_57/ShowerFlow_{epoch}.pth',
            # }
        
            
            
            # { # === 1-1000 GeV === seed 44
                
            #     'D = 1 x 10^5': f'{base_path}/100k_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__12_34_58/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^4':  f'{base_path}/50k_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__12_37_44/ShowerFlow_{epoch}.pth',
            #     'D = 1 x 10^4': f'{base_path}/10k_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__12_39_36/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^3': f'{base_path}/5k_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__12_43_04/ShowerFlow_{epoch}.pth',
            #     'D = 1 x 10^3': f'{base_path}/1k_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__12_58_36/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^2': f'{base_path}/500_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__13_01_19/ShowerFlow_{epoch}.pth',
            #     'D = 1 x 10^2': f'{base_path}/100_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__13_12_19/ShowerFlow_{epoch}.pth',
            #     }
            
            # { # === 1-1000 GeV === seed 42
                
            #     'D = 1 x 10^5': f'{base_path}/100k_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__11_48_39/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^4':  f'{base_path}/50k_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__11_51_39/ShowerFlow_{epoch}.pth',
            #     'D = 1 x 10^4': f'{base_path}/10k_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__11_53_01/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^3': f'{base_path}/5k_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__11_53_16/ShowerFlow_{epoch}.pth',
            #     'D = 1 x 10^3': f'{base_path}/1k_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__11_53_39/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^2': f'{base_path}/500_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__11_53_45/ShowerFlow_{epoch}.pth',
            #     'D = 1 x 10^2': f'{base_path}/100_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__11_55_04/ShowerFlow_{epoch}.pth',
            #     }
            
            # { # === 1-1000 GeV === seed 43 the one used so far!!
            #     'D = 1 x 10^5': f'{base_path}/100k_1-1000GeV/{training_type}/ShowerFlow_2025_05_06__18_00_43/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^4':  f'{base_path}/50k_1-1000GeV/{training_type}/ShowerFlow_2025_05_06__18_08_37/ShowerFlow_{epoch}.pth',
            #     'D = 1 x 10^4': f'{base_path}/10k_1-1000GeV/{training_type}/ShowerFlow_2025_05_06__18_09_28/ShowerFlow_{epoch}.pth',
                
            #     'D = 5 x 10^3': f'{base_path}/5k_1-1000GeV/{training_type}/ShowerFlow_2025_05_06__18_09_28/ShowerFlow_{epoch}.pth',
            #     'D = 1 x 10^3': f'{base_path}/1k_1-1000GeV/{training_type}/ShowerFlow_2025_05_06__18_09_03/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^2': f'{base_path}/500_1-1000GeV/{training_type}/ShowerFlow_2025_05_07__14_17_56/ShowerFlow_{epoch}.pth',
            #     'D = 1 x 10^2': f'{base_path}/100_1-1000GeV/{training_type}/ShowerFlow_2025_05_07__14_18_28/ShowerFlow_{epoch}.pth',
            #     }   
        
            # { # === 10-90 GeV ===
            #         'D = 5 x 10^4': f'{base_path}/47k_10_90GeV/{training_type}/ShowerFlow_2025_01_26__13_57_28/ShowerFlow_{epoch}.pth',
            #         'D = 1 x 10^4': f'{base_path}/10k_10_90GeV/{training_type}/ShowerFlow_2025_01_26__14_21_53/ShowerFlow_{epoch}.pth',
            #         'D = 5 x 10^3': f'{base_path}/5k_10_90GeV/{training_type}/ShowerFlow_2025_05_13__16_06_15/ShowerFlow_{epoch}.pth',

            #         'D = 1 x 10^3': f'{base_path}/1k_10_90GeV/{training_type}/ShowerFlow_2025_01_24__23_56_18/ShowerFlow_{epoch}.pth',
            #         'D = 5 x 10^2': f'{base_path}/500_10_90GeV/{training_type}/ShowerFlow_2025_02_11__10_14_50/ShowerFlow_{epoch}.pth',
            #         'D = 1 x 10^2': f'{base_path}/100_10_90GeV/{training_type}/ShowerFlow_2025_01_24__23_56_34/ShowerFlow_{epoch}.pth',
            #         # 'D = 5 x 10^1': f'{base_path}/50_10_90GeV/{training_type}/ShowerFlow_2025_02_11__10_18_10/ShowerFlow_{epoch}.pth',
            #         # 'D = 1 x 10^1': f'{base_path}/10_10_90GeV/{training_type}/ShowerFlow_2025_02_11__10_14_33/ShowerFlow_{epoch}.pth',
            #     }
            
             
                    
        else:
            # Vanilla training paths
            return { # === 1-1000 GeV === seed 46
                'D = 1 x 10^5': f'{base_path}/100k_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__17_09_12/ShowerFlow_{epoch}.pth',
                'D = 5 x 10^4': f'{base_path}/50k_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__14_50_47/ShowerFlow_{epoch}.pth',
                'D = 1 x 10^4': f'{base_path}/10k_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__14_40_48/ShowerFlow_{epoch}.pth',
                'D = 5 x 10^3': f'{base_path}/5k_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__13_11_36/ShowerFlow_{epoch}.pth',
                
                'D = 1 x 10^3': f'{base_path}/1k_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__14_02_53/ShowerFlow_{epoch}.pth',
                'D = 5 x 10^2': f'{base_path}/500_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__14_37_47/ShowerFlow_{epoch}.pth',
                'D = 1 x 10^2': f'{base_path}/100_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__14_43_33/ShowerFlow_{epoch}.pth',
            }
            
            # { # === 1-1000 GeV === seed 45
            #     'D = 1 x 10^5': f'{base_path}/100k_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__12_41_00/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^4': f'{base_path}/50k_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__12_40_47/ShowerFlow_{epoch}.pth',
            #     'D = 1 x 10^4': f'{base_path}/10k_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__12_41_52/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^3': f'{base_path}/5k_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__12_42_13/ShowerFlow_{epoch}.pth',
                
            #     'D = 1 x 10^3': f'{base_path}/1k_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__12_45_01/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^2': f'{base_path}/500_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__13_29_14/ShowerFlow_{epoch}.pth',
            #     'D = 1 x 10^2': f'{base_path}/100_1-1000GeV/{training_type}/ShowerFlow_2025_08_09__13_40_57/ShowerFlow_{epoch}.pth',
            #     }

            # { # === 1-1000 GeV === seed 44
            #     'D = 1 x 10^5': f'{base_path}/100k_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__16_37_03/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^4': f'{base_path}/50k_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__16_37_09/ShowerFlow_{epoch}.pth',
            #     'D = 1 x 10^4': f'{base_path}/10k_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__16_37_15/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^3': f'{base_path}/5k_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__16_40_20/ShowerFlow_{epoch}.pth',
                
            #     'D = 1 x 10^3': f'{base_path}/1k_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__16_41_05/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^2': f'{base_path}/500_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__16_43_55/ShowerFlow_{epoch}.pth',
            #     'D = 1 x 10^2': f'{base_path}/100_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__16_44_46/ShowerFlow_{epoch}.pth',
            #     }
            
            
            # { # === 1-1000 GeV === seed 42
            #     'D = 1 x 10^5': f'{base_path}/100k_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__13_30_54/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^4': f'{base_path}/50k_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__13_53_48/ShowerFlow_{epoch}.pth',
            #     'D = 1 x 10^4': f'{base_path}/10k_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__14_05_51/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^3': f'{base_path}/5k_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__14_09_07/ShowerFlow_{epoch}.pth',
                
            #     'D = 1 x 10^3': f'{base_path}/1k_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__14_13_49/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^2': f'{base_path}/500_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__14_15_50/ShowerFlow_{epoch}.pth',
            #     'D = 1 x 10^2': f'{base_path}/100_1-1000GeV/{training_type}/ShowerFlow_2025_08_08__14_31_06/ShowerFlow_{epoch}.pth',
            #     }
        
            # { # === 1-1000 GeV === seed 43
            #     'D = 1 x 10^5': f'{base_path}/100k_1-1000GeV/{training_type}/ShowerFlow_2025_05_06__18_00_45/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^4': f'{base_path}/50k_1-1000GeV/{training_type}/ShowerFlow_2025_05_06__18_08_15/ShowerFlow_{epoch}.pth',
            #     'D = 1 x 10^4': f'{base_path}/10k_1-1000GeV/{training_type}/ShowerFlow_2025_05_06__18_08_22/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^3': f'{base_path}/5k_1-1000GeV/{training_type}/ShowerFlow_2025_05_06__18_08_35/ShowerFlow_{epoch}.pth',
                
            #     'D = 1 x 10^3': f'{base_path}/1k_1-1000GeV/{training_type}/ShowerFlow_2025_05_06__18_08_43/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^2': f'{base_path}/500_1-1000GeV/{training_type}/ShowerFlow_2025_05_07__14_18_23/ShowerFlow_{epoch}.pth',
            #     'D = 1 x 10^2': f'{base_path}/100_1-1000GeV/{training_type}/ShowerFlow_2025_05_07__14_19_04/ShowerFlow_{epoch}.pth',
            #     }

            #     { # === 10-90 GeV ===
            #     'D = 5 x 10^4': f'{base_path}/47k_10_90GeV/{training_type}/ShowerFlow_2025_01_26__13_53_11/ShowerFlow_{epoch}.pth',
            #     'D = 1 x 10^4': f'{base_path}/10k_10_90GeV/{training_type}/ShowerFlow_2025_01_26__14_22_19/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^3': f'{base_path}/5k_10_90GeV/{training_type}/ShowerFlow_2025_05_12__21_55_38/ShowerFlow_{epoch}.pth',
            #     'D = 1 x 10^3': f'{base_path}/1k_10_90GeV//{training_type}/ShowerFlow_2025_01_25__00_00_13/ShowerFlow_{epoch}.pth',
            #     'D = 5 x 10^2': f'{base_path}/500_10_90GeV/{training_type}/ShowerFlow_2025_02_11__10_30_02/ShowerFlow_{epoch}.pth',
            #     'D = 1 x 10^2': f'{base_path}/100_10_90GeV/{training_type}/ShowerFlow_2025_01_26__12_52_32/ShowerFlow_{epoch}.pth',
            # }

    def _create_summary_statistics(self, wd_per_epoch: Dict, kl_per_epoch: Dict):
        """
        Create and save summary statistics from all epochs.
        
        Args:
            wd_per_epoch: Wasserstein distances per epoch
            kl_per_epoch: KL distances per epoch
        """
        print("\nCreating summary statistics...")
        
        summary = {
            'epochs': list(wd_per_epoch.keys()),
            'metrics': ['x', 'z', 'Clusters per Layer', 'Energy per Layer'],
            'wasserstein_summary': {},
            'kl_summary': {}
        }
        
        # Calculate statistics for each metric
        for metric in summary['metrics']:
            # Wasserstein statistics
            wd_values_by_epoch = []
            for epoch in summary['epochs']:
                if metric in ['x', 'z']:
                    # For CoG metrics, average across models
                    epoch_values = list(wd_per_epoch[epoch][metric].values())
                else:
                    # For layer metrics, average across layers for each model, then across models
                    epoch_values = []
                    for model_values in wd_per_epoch[epoch][metric].values():
                        epoch_values.append(np.mean(model_values))
                wd_values_by_epoch.append(np.mean(epoch_values))
            
            summary['wasserstein_summary'][metric] = {
                'mean_across_epochs': np.mean(wd_values_by_epoch),
                'std_across_epochs': np.std(wd_values_by_epoch),
                'min': np.min(wd_values_by_epoch),
                'max': np.max(wd_values_by_epoch),
                'final_epoch_value': wd_values_by_epoch[-1]
            }
            
            # KL statistics
            kl_values_by_epoch = []
            for epoch in summary['epochs']:
                if metric in ['x', 'z']:
                    epoch_values = list(kl_per_epoch[epoch][metric].values())
                else:
                    epoch_values = []
                    for model_values in kl_per_epoch[epoch][metric].values():
                        epoch_values.append(np.mean(model_values))
                kl_values_by_epoch.append(np.mean(epoch_values))
            
            summary['kl_summary'][metric] = {
                'mean_across_epochs': np.mean(kl_values_by_epoch),
                'std_across_epochs': np.std(kl_values_by_epoch),
                'min': np.min(kl_values_by_epoch),
                'max': np.max(kl_values_by_epoch),
                'final_epoch_value': kl_values_by_epoch[-1]
            }

def main():
    config = ShowerConfig()
    pretrained = True # Set to True for pretrained models
    
    base_output_dir = './results/shower_flow'
    output_subdir = 'finetune' if pretrained else 'vanilla'
    print(f' \n== Starting ShowerFlow analysis for {output_subdir} models ==\n')

    analyzer = ShowerAnalysis(config, base_output_dir)
    training_paths = analyzer._get_training_paths(epoch=1, pretrained=pretrained)
    
    # Get the first path
    cfg = Configs()

    first_path = next(iter(training_paths.values()))
    folder_name = os.path.basename(os.path.dirname(first_path))
    output_dir = os.path.join(base_output_dir, output_subdir, cfg.val_ds_type, folder_name )
    print(f'\n== Creating directory: {output_dir}')
    
    analyzer = ShowerAnalysis(config, output_dir)
    analyzer.run_analysis( eval_ds=cfg.val_ds_type, pretrained=pretrained)


if __name__ == "__main__":
    main()