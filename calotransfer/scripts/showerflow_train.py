import os
import time
import logging
from typing import Dict, Tuple, Optional, List, Union, Any

import numpy as np
import pandas as pd
import h5py
import argparse
import torch
import torch.optim as optim
import torch.utils.data
import random

from tqdm import tqdm
import comet_ml

from configs import Configs
from models.shower_flow import compile_HybridTanH_model_CaloC
from utils.preprocessing_utils import free_memory, read_hdf5_file2
import utils.paths_configs as dirs
from utils.paths_configs import sf_paths, dataset_paths, sf_eval_paths
import utils.eval_utils_showerflow as sf_utils


class DataHandler:
    """
    Handles loading and preprocessing of data for the ShowerFlow model.
    """
    def __init__(self, cfg: Configs):
        self.cfg = cfg
        self.epsilon = 1e-10  # Small value to avoid division by zero
    
    def get_cog(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, e: np.ndarray) -> tuple:
        """
        Calculate the center of gravity for coordinates.
        
        Args:
            x: X coordinates
            y: Y coordinates
            z: Z coordinates
            e: Energy values
            
        Returns:
            Tuple of CoG values for each coordinate
        """
        coords = [x, y, z]
        results = []
        
        e_sum = e.sum(axis=1) + self.epsilon
        
        for coord in coords:
            weighted_sum = np.sum((coord * e), axis=1)
            results.append(weighted_sum / e_sum)
        
        return tuple(results)
    
    def load_and_preprocess_calochallenge(self, path: str, 
                                    e_per_layer_path: Optional[str] = None, 
                                    clusters_per_layer_path: Optional[str] = None) -> torch.utils.data.TensorDataset:
        """
        Load and preprocess CaloChallenge dataset.
        """
        print('\n=== CaloChallenge pre processing ===\n')
        with h5py.File(path, 'r') as hdf_file:
            keys = list(hdf_file.keys())
            print("Keys in the HDF5 file:", keys)
            keys, energy, events = read_hdf5_file2(path)

        print('events shape: ', events.shape)
        print('energy array length: ', len(energy))

        # Unnormalization
        Xmin, Xmax = -18, 18
        Ymin, Ymax = 0, 45
        Zmin, Zmax = -18, 18

        events[:, 0, :] = (events[:, 0, :] + 1) * (Xmax - Xmin) / 2 + Xmin
        events[:, 1, :] = (events[:, 1, :] + 1) * (Ymax - Ymin) / 2 + Ymin
        events[:, 2, :] = (events[:, 2, :] + 1) * (Zmax - Zmin) / 2 + Zmin

        events[:, -1, :] /= 1000  # preprocessing to match caloclouds

        if e_per_layer_path is None or clusters_per_layer_path is None:
            print('Computing e_per_layer and clusters_per_layer')
            # Implementation for calculating these values would go here
            raise NotImplementedError("Computing e_per_layer and clusters_per_layer not implemented")
        else:
            e_per_layer = np.load(e_per_layer_path)
            clusters_per_layer = np.load(clusters_per_layer_path)
            
        print('e_per_layer shape: ', e_per_layer.shape)
        print('clusters_per_layer shape: ', clusters_per_layer.shape)

        # Make sure all arrays have the same first dimension
        num_events = min(len(energy), len(e_per_layer), len(clusters_per_layer))
        print(f"Taking the minimum number of events across all arrays: {num_events}")
        
        # Truncate all arrays to match
        energy = energy[:num_events]
        events = events[:num_events]
        e_per_layer = e_per_layer[:num_events]
        clusters_per_layer = clusters_per_layer[:num_events]

        # Normalize e_per_layer and clusters_per_layer
        e_per_layer = e_per_layer / self.cfg.sf_norm_energy
        clusters_per_layer = clusters_per_layer / self.cfg.sf_norm_points

        # Inverse operations (renormalize events)
        events[:, 0, :] = (events[:, 0, :] - Xmin) * 2 / (Xmax - Xmin) - 1
        events[:, 1, :] = (events[:, 1, :] - Ymin) * 2 / (Ymax - Ymin) - 1
        events[:, 2, :] = (events[:, 2, :] - Zmin) * 2 / (Zmax - Zmin) - 1

        # Calculate center of gravity
        cog = self.get_cog(
            x=events[:, 0],
            y=events[:, 1],
            z=events[:, 2],
            e=events[:, 3],
        )

        # Handle NaN values in CoG calculations
        mean_cog = np.nanmean(cog, axis=-1)
        std_cog = np.nanstd(cog, axis=-1)
        print(f"Mean CoG: {mean_cog}")
        print(f"Std CoG: {std_cog}")

        # Ensure energy is not empty
        if len(energy) == 0:
            raise ValueError("Energy arrays are empty")

        # Create dataframe
        df = pd.DataFrame()
        normalized_energy = (np.log(energy / energy.min()) / np.log(energy.max() / energy.min())).reshape(-1)
        df['energy'] = normalized_energy * 100
        
        df['cog_x'] = ((cog[0] - mean_cog[0]) / std_cog[0]).reshape(-1)
        df['cog_z'] = ((cog[2] - mean_cog[2]) / std_cog[2]).reshape(-1)

        df['clusters_per_layer'] = clusters_per_layer.tolist()
        df['e_per_layer'] = e_per_layer.tolist()

        # Display sample data
        pd.set_option('display.max_columns', None)  
        pd.set_option('display.max_colwidth', None)
        print(df.head(10).to_string(index=False))

        # Create TensorDataset
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(df.energy.values), 
            torch.tensor(df.cog_x.values),
            torch.tensor(df.cog_z.values),
            torch.tensor(df.clusters_per_layer),
            torch.tensor(df.e_per_layer),
        )

        return dataset
    
    def load_and_preprocess_caloclouds(self, path: str, 
                                        e_per_layer_path: str, 
                                        clusters_per_layer_path: str) -> torch.utils.data.TensorDataset:
        """
        Load and preprocess CaloClouds dataset.
        
        Args:
            path: Path to the main dataset
            e_per_layer_path: Path to energy per layer data
            clusters_per_layer_path: Path to clusters per layer data
            
        Returns:
            TensorDataset with processed data
        """
        print('\n=== CaloClouds pre processing ===\n')
        with h5py.File(path, 'r') as hdf_file:
            keys = list(hdf_file.keys())
            print("Keys in the HDF5 file:", keys)
            keys, energy, events = read_hdf5_file2(path)

        print('events shape: ', events.shape)

        # Load cluster and energy data
        clusters_per_layer = np.load(clusters_per_layer_path)
        e_per_layer = np.load(e_per_layer_path)
        print('e_per_layer shape: ', e_per_layer.shape)
        print('clusters_per_layer shape: ', clusters_per_layer.shape)
        
        # Pad with log-normal noise if needed
        current_cols = clusters_per_layer.shape[1]
        target_cols = 45
        columns_to_add = target_cols - current_cols

        if columns_to_add > 0:
            print("Padding needed")
            # Generate log-normal noise
            log_normal_noise_e, _ = sf_utils.generate_log_normal_noise(
                len(e_per_layer), dirs.mu_sigma_e_per_layer_dict, layer_range=(31, 45))
            log_normal_noise_clusters, _ = sf_utils.generate_log_normal_noise(
                len(clusters_per_layer), dirs.mu_sigma_clusters_per_layer_dict, layer_range=(31, 45))
            
            # Concatenate the log-normal noise with the existing data
            clusters_per_layer = np.concatenate((clusters_per_layer, log_normal_noise_clusters), axis=1)
            e_per_layer = np.concatenate((e_per_layer, log_normal_noise_e), axis=1)
            print('New Shape:', clusters_per_layer.shape, e_per_layer.shape)
        else:
            print("No padding needed")

        # Normalize
        e_per_layer = e_per_layer / self.cfg.sf_norm_energy
        clusters_per_layer = clusters_per_layer / self.cfg.sf_norm_points

        # Calculate center of gravity
        cog = self.get_cog(
            x=events[:, 0],
            y=events[:, 1],
            z=events[:, 2],
            e=events[:, 3],
        )

        # Handle NaN values in CoG calculations
        mean_cog = np.nanmean(cog, axis=-1)
        std_cog = np.nanstd(cog, axis=-1)
        print(f"Mean CoG: {mean_cog}")
        print(f"Std CoG: {std_cog}")

        # Ensure energy is not empty
        if len(energy) == 0:
            raise ValueError("Energy arrays are empty")

        # Create dataframe
        df = pd.DataFrame()
        df['energy'] = energy.reshape(-1)
        df['cog_x'] = ((cog[0] - mean_cog[0]) / std_cog[0]).reshape(-1)
        df['cog_z'] = ((cog[2] - mean_cog[2]) / std_cog[2]).reshape(-1)

        df['clusters_per_layer'] = clusters_per_layer.tolist()
        df['e_per_layer'] = e_per_layer.tolist()

        print('How the input data looks like: \n', df.head(10))

        # Create TensorDataset
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(df.energy.values), 
            torch.tensor(df.cog_x.values),
            torch.tensor(df.cog_z.values),
            torch.tensor(df.clusters_per_layer),
            torch.tensor(df.e_per_layer),
        )

        return dataset

class ModelManager:
    """
    Handles model setup, training, and validation.
    """
    def __init__(self, cfg: Configs, device: str):
        self.cfg = cfg
        self.device = device
        self.start_time = time.localtime()
    
    def setup_model(self, model: torch.nn.Module, weight_decay: float = None) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        """
        Set up the model and optimizer.
        
        Args:
            model: The model to set up
            weight_decay: Weight decay parameter for the optimizer
            
        Returns:
            Tuple of (model, optimizer)
        """
        if weight_decay is None:
            weight_decay = self.cfg.sf_weight_decay
            
        model.to(self.device)
        print('weight decay value is:', weight_decay)
        optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
        model.train()
        torch.manual_seed(41)
        return model, optimizer
    
    def adjust_learning_rate(self, optimizer: torch.optim.Optimizer, epoch: Optional[int] = None, 
                            total_epochs: int = 10000) -> None:
        """
        Adjust the learning rate based on configuration and training progress.
        
        Args:
            optimizer: The optimizer to adjust
            epoch: Current epoch number
            total_epochs: Total number of epochs for training
        """
        if self.cfg.sf_use_pretrained and self.cfg.dataset_size != 'pretraining_cc2':
            lr = 1e-4 # 1e-6
        else:
            lr = 1e-4 # 1e-5

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    def save_model_weights(self, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                          output_path: str, loss_list: List[float], best_loss: float) -> float:
        """
        Save model weights based on training progress.
        
        Args:
            epoch: Current epoch number
            model: Model to save
            optimizer: Optimizer to save
            output_path: Base path to save model
            loss_list: List of loss values for current epoch
            best_loss: Best loss value seen so far
            
        Returns:
            Updated best loss value
        """
        training_style = 'vanilla/' if self.cfg.sf_use_pretrained is False else 'finetune/'

        output_path = os.path.join(output_path, training_style)
        save_dir = os.path.join(output_path, "ShowerFlow_" + time.strftime('%Y_%m_%d__%H_%M_%S', self.start_time))
        if epoch == 1:
            print('\n=== Saving model weights ===')
            print('Output path:', output_path)
            print('Save dir:', save_dir)
            print('\n')

        # Create the directory if it does not exist
        os.makedirs(save_dir, exist_ok=True)
            
        if torch.stack([torch.isnan(p).any() for p in model.parameters()]).any():
            print('Model not saved due to NaN weights')
        else:
            # Save the latest model
            torch.save(
                {'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                os.path.join(save_dir, 'ShowerFlow_latest.pth')
            )

            # Save model at specific epochs
            if epoch <= 50 or epoch % 25 == 0:
                torch.save(
                    {'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    os.path.join(save_dir, f'ShowerFlow_{epoch}.pth')
                )

            # Save the best model based on loss
            current_loss = np.mean(loss_list)
            if current_loss <= best_loss:
                best_loss = current_loss
                torch.save(
                    {'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    os.path.join(save_dir, 'ShowerFlow_best.pth')
                )
                # Save best loss value to txt file
                with open(os.path.join(save_dir, 'ShowerFlow_best_loss.txt'), 'w') as f:
                    f.write(str(best_loss))
                print('Best model saved, with loss: ', best_loss)
        
        return best_loss
    
    def create_comet_experiment(self, dataset_size: str) -> Optional[comet_ml.Experiment]:
        """
        Creates a Comet ML experiment for tracking training.
        
        Args:
            dataset_size: Size of the dataset
            
        Returns:
            Comet ML experiment object or None if logging is disabled
        """
        if not self.cfg.log_comet:
            return None
            
        try:
            with open('./utils/comet_api_key.txt', 'r') as file:
                key = file.read().strip()
        except FileNotFoundError:
            print("Warning: Comet API key file not found. Comet logging disabled.")
            return None

        logging.getLogger("comet_ml").setLevel(logging.ERROR)
        name = 'showerflow-caloclouds2' if dataset_size == 'pretraining_cc2' else 'showerflow-train'
        experiment = comet_ml.Experiment(
            api_key=key,
            project_name=name,
            auto_metric_logging=False,
            workspace="lorenzovalente3",
        )
        experiment.log_parameters(self.cfg.__dict__)
        experiment.set_name(name + "_" + time.strftime('%Y_%m_%d__%H_%M_%S', self.start_time))
        experiment.log_code(file_name='showerflow_train.py')
        experiment.log_code(file_name='configs.py')

        experiment.add_tags([
            'showerflow',
            'caloclouds2' if dataset_size == 'pretraining_cc2' else 'calochallenge',
            f'dataset_size_{self.cfg.dataset_size}',
            f'pretrained_{self.cfg.sf_use_pretrained}',
            f'seed_{self.cfg.seed}',
        ])
        return experiment
    
    def log_to_comet(self, experiment: comet_ml.Experiment, epoch: Optional[int] = None, 
                    loss: Optional[torch.Tensor] = None, val_loss: Optional[float] = None, 
                    additional_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Log metrics to Comet ML.
        
        Args:
            experiment: Comet ML experiment object
            epoch: Current epoch number
            loss: Loss value to log
            val_loss: Validation loss value to log
            additional_metrics: Dictionary of additional metrics to log
        """
        if not experiment or epoch is None:
            return
            
        try:
            if loss is not None:
                experiment.log_metric("loss", loss.item(), step=epoch)
            if val_loss is not None:
                experiment.log_metric("val_loss", val_loss, step=epoch)
            if additional_metrics:
                for metric_name, metric_value in additional_metrics.items():
                    experiment.log_metric(metric_name, metric_value, step=epoch)
        except Exception as e:
            print(f"Error logging to Comet ML: {e}")
    
    def validate_model(self, model: torch.nn.Module, val_loader: torch.utils.data.DataLoader, 
                        distribution: Any) -> float:
        """
        Validate the model on validation data.
        
        Args:
            model: Model to validate
            val_loader: DataLoader with validation data
            distribution: Distribution object for computing loss
            
        Returns:
            Mean validation loss
        """
        model.eval()
        val_loss_list = []
        with torch.no_grad():
            for batch_idx, (energy, cog_x, cog_z, clusters_per_layer, e_per_layer) in enumerate(val_loader):
                # Prepare input data
                E_true = energy.view(-1, 1).to(self.device).float()
                cog_x = cog_x.view(-1, 1).to(self.device).float()
                cog_z = cog_z.view(-1, 1).to(self.device).float()
                clusters_per_layer = clusters_per_layer.to(self.device).float()
                e_per_layer = e_per_layer.to(self.device).float()

                input_data = torch.cat((cog_x, cog_z, clusters_per_layer, e_per_layer), 1)
                E_true = (E_true / 100).float()
                context = E_true

                # Compute loss
                nll = -distribution.condition(context).log_prob(input_data)
                val_loss = nll.mean()
                val_loss_list.append(val_loss.item())

        model.train()
        return np.mean(val_loss_list)
    
    def train_model(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                    train_loader: torch.utils.data.DataLoader, 
                    val_loader: Optional[torch.utils.data.DataLoader], 
                    distribution: Any, output_path: str, num_epochs: int, 
                    dataset_size: str = 'all') -> Union[List[float], Tuple[List[float], List[float]]]:
        """
        Train the model.
        
        Args:
            model: Model to train
            optimizer: Optimizer for training
            train_loader: DataLoader with training data
            val_loader: DataLoader with validation data (optional)
            distribution: Distribution object for computing loss
            output_path: Path to save model outputs
            num_epochs: Number of epochs to train
            dataset_size: Size of the dataset
            
        Returns:
            List of training losses or tuple of (training losses, validation losses)
        """
        losses = []
        val_losses = []
        print('Start training ...')

        # Load pretrained model weights if the flag is set
        if self.cfg.dataset_size != 'pretraining_cc2' and self.cfg.sf_use_pretrained and self.cfg.sf_pretrained_model_path:
            pretrained_model_path = self.cfg.sf_pretrained_model_path
            if os.path.exists(pretrained_model_path):
                checkpoint = torch.load(pretrained_model_path, weights_only=True)
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(' + -' * 30)
                print(f'Loaded pretrained model weights from {self.cfg.sf_pretrained_model_path}')
                print(' + -' * 30)
            else:
                print(f'Pretrained model path {self.cfg.sf_pretrained_model_path} does not exist. '
                     f'Proceeding without loading pretrained weights.')

        # Print the learning rate after adjustment
        self.adjust_learning_rate(optimizer)
        print('Learning rate after adjustment:', optimizer.param_groups[0]['lr'])
        
        # Create Comet experiment
        experiment = self.create_comet_experiment(dataset_size)

        best_loss = float('inf')  # Initialize best loss to infinity
        for epoch in range(1, num_epochs + 1):
            # Adjust learning rate dynamically based on the current epoch
            self.adjust_learning_rate(optimizer, epoch=epoch, total_epochs=num_epochs)

            input_list = []
            loss_list = []
            additional_metrics = {}

            for batch_idx, (energy, cog_x, cog_z, clusters_per_layer, e_per_layer) in enumerate(tqdm(train_loader)):
                # Prepare input data
                E_true = energy.view(-1, 1).to(self.device).float()
                cog_x = cog_x.view(-1, 1).to(self.device).float()
                cog_z = cog_z.view(-1, 1).to(self.device).float()
                clusters_per_layer = clusters_per_layer.to(self.device).float()
                e_per_layer = e_per_layer.to(self.device).float()

                input_data = torch.cat((cog_x, cog_z, clusters_per_layer, e_per_layer), 1)

                optimizer.zero_grad()
                E_true = (E_true / 100).float()

                context = E_true

                # Check for NaNs in input data
                if np.any(np.isnan(input_data.clone().detach().cpu().numpy())):
                    print('Nans in the training data!')
                    continue

                # Check for NaNs in model weights
                if torch.stack([torch.isnan(p).any() for p in model.parameters()]).any():
                    # Construct the path correctly
                    model_path_fetch = os.path.join(output_path, 
                                                   "ShowerFlow_" + time.strftime('%Y_%m_%d__%H_%M_%S', self.start_time), 
                                                   'ShowerFlow_latest.pth')
                    # Load the model state dict if file exists
                    if os.path.exists(model_path_fetch):
                        model.load_state_dict(torch.load(model_path_fetch)['model'])
                        # Adjust the learning rate again after loading the model state dict
                        self.adjust_learning_rate(optimizer, epoch=epoch, total_epochs=num_epochs)
                        print('Weights are NaN! The latest model has been loaded')
                    else:
                        print('Weights are NaN but no saved model found to load')
                        continue

                # Compute loss and backpropagate
                nll = -distribution.condition(context).log_prob(input_data)
                loss = nll.mean()

                # Skip step if loss is NaN
                if torch.isnan(loss):
                    print(f'Skipping batch {batch_idx} due to NaN loss')
                    continue

                loss.backward()

                # Clip gradients only if using pretrained model and in early epochs
                if self.cfg.sf_use_pretrained:
                    # Define the initial and final max_norm values
                    initial_max_norm = 1e4
                    final_max_norm = 5e5

                    # Calculate the max_norm value based on the current epoch
                    if epoch <= 50:
                        max_norm = initial_max_norm - (initial_max_norm - final_max_norm) * (epoch / 50)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

                # Check for NaNs in gradients
                if torch.stack([torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None]).any():
                    print('NaNs detected in gradients! Skipping step.')
                    continue

                optimizer.step()

                distribution.clear_cache()
                input_list.append(input_data.detach().cpu().numpy())
                loss_list.append(loss.item())

                # Compute additional metrics
                additional_metrics['gradient_norm'] = torch.norm(
                    torch.stack([torch.norm(p.grad) for p in model.parameters() if p.grad is not None])).item()
                additional_metrics['learning_rate'] = optimizer.param_groups[0]['lr']

            # Calculate mean loss for epoch
            epoch_loss = np.mean(loss_list) if loss_list else float('inf')
            losses.append(epoch_loss)
            
            # Initialize best_loss if first epoch
            if epoch == 1:
                best_loss = epoch_loss
            
            # Save model weights
            best_loss = self.save_model_weights(epoch, model, optimizer, output_path, loss_list, best_loss)

            # Validate the model if validation loader is provided
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate_model(model, val_loader, distribution)
                val_losses.append(val_loss)
                print(f'Epoch: {epoch}, Loss: {epoch_loss}, Validation Loss: {val_loss}')
            else:
                print(f'Epoch: {epoch}, Loss: {epoch_loss}')

            # Log to COMET ML
            if experiment:
                self.log_to_comet(experiment, epoch, loss, val_loss, additional_metrics)

        # Close Comet logger at the end
        if experiment:
            experiment.end()
            
        return (losses, val_losses) if val_loader is not None else losses

class ShowerFlowTrainer:
    """
    Main class to orchestrate ShowerFlow model training.
    """
    def __init__(self):
        self.cfg = Configs()
        self.device = self.cfg.device
        self.data_handler = DataHandler(self.cfg)
        self.model_manager = ModelManager(self.cfg, self.device)
        
        # Default params with batch size adjustments by dataset size
        self.default_params = {
            'epochs': 2000,
            'shuffle': True,
            'batch_size': 32*16,
        }
        
    def get_params(self, kwargs: Dict[str, Any], dataset_size: str = 'all') -> Dict[str, Any]:
        """
        Get training parameters based on dataset size.
        
        Args:
            kwargs: Additional parameters to override defaults
            dataset_size: Size of the dataset
            
        Returns:
            Dictionary of training parameters
        """
        params = {}
        
        # Update batch_size based on dataset_size
        if dataset_size == '1k' or dataset_size == '500k_10-90' or dataset_size == '1k_10-90' or dataset_size == '1k_1-1000'or dataset_size == '100_1-1000' or dataset_size == '500_1-1000':
            self.default_params['batch_size'] = 32 * 2
        elif dataset_size == '100_10-90' or dataset_size == '50_10-90' :
            self.default_params['batch_size'] = 16
        elif dataset_size == '10_10-90':
            self.default_params['batch_size'] = 4
        elif dataset_size == '10k' or dataset_size == 'all_10-90' or dataset_size == '10k_10-90' or dataset_size == '10k_1-1000':
            self.default_params['batch_size'] = 32 * 4
        elif dataset_size == 'all' or dataset_size == '100k_1-1000' or dataset_size == '50k_1-1000':
            self.default_params['batch_size'] = 32 * 16
        elif dataset_size == 'pretraining_cc2':
            self.default_params['batch_size'] = 32 * 4 * 16

        for param in self.default_params.keys():
            params[param] = kwargs.get(param, self.default_params[param])
        
        print('Params are:', params)
        print('Batch size is:', self.default_params['batch_size'])
        return params
    
    def setup_dataset_paths(self):
        """
        Set up and validate paths for datasets.
        """
        # Dataset configuration
        dataset_size = self.cfg.dataset_size
        val_ds_type = self.cfg.val_ds_type

        # Print dataset configuration
        print("\n=== Dataset Configuration ===")
        print(f"Training dataset size: {dataset_size}")
        print(f"Validation dataset type: {val_ds_type}")

        # Validate dataset size
        if dataset_size not in sf_paths:
            print(f"ERROR: Dataset size '{dataset_size}' not found in sf_paths.")
            print(f"Available options: {list(sf_paths.keys())}")
            raise ValueError(f"Invalid dataset_size: {dataset_size}. Available options: {list(sf_paths.keys())}")

        # Training paths
        print("\n=== Training Paths ===")
        
        # Debug: Print the entire sf_paths dictionary for the dataset
        print(f"sf_paths['{dataset_size}'] contains:")
        for key, value in sf_paths[dataset_size].items():
            print(f"  {key}: {value}")
        
        data_path = sf_paths[dataset_size].get('data_path')
        output_path = sf_paths[dataset_size].get('output_path')
        e_per_layer_path = sf_paths[dataset_size].get('e_per_layer_path')
        clusters_per_layer_path = sf_paths[dataset_size].get('clusters_per_layer_path')

        # Check if any of the paths are None
        if data_path is None:
            # Try to get from dataset_paths as a fallback
            if dataset_size in dataset_paths:
                data_path = dataset_paths[dataset_size]
                print(f"Found data_path in dataset_paths: {data_path}")
            else:
                print(f"ERROR: Data path for dataset '{dataset_size}' is None and not found in dataset_paths")
                raise ValueError(f"Data path for dataset '{dataset_size}' is None")

        # Print paths for debugging
        print(f"Data path: {data_path}")
        print(f"Output path: {output_path}")
        print(f"E per layer path: {e_per_layer_path}")
        print(f"Clusters per layer path: {clusters_per_layer_path}")
        # Validation paths
        print("\n=== Validation Paths ===")
        val_paths = {
            'data': sf_eval_paths[val_ds_type]['10k'].get('data_path'),
            'e_per_layer': sf_eval_paths[val_ds_type]['10k'].get('e_per_layer_path'),
            'clusters_per_layer': sf_eval_paths[val_ds_type]['10k'].get('clusters_per_layer_path')
        }

        # Check validation paths
        for path_name, path in val_paths.items():
            if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
                val_paths[path_name] = None
                print(f"{path_name.replace('_', ' ').title()} path: Not available or invalid")
            else:
                print(f"{path_name.replace('_', ' ').title()} path: {path}")

        return {
            'data_path': data_path,
            'output_path': output_path,
            'e_per_layer_path': e_per_layer_path,
            'clusters_per_layer_path': clusters_per_layer_path,
            'val_data_path': val_paths['data'],
            'val_e_per_layer_path': val_paths['e_per_layer'],
            'val_clusters_per_layer_path': val_paths['clusters_per_layer']
        }

    def load_datasets(self, paths):
        """
        Load and prepare datasets for training and validation.
        
        Args:
            paths: Dictionary with dataset paths
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Load appropriate dataset based on dataset_size
        if self.cfg.dataset_size == 'pretraining_cc2':
            dataset = self.data_handler.load_and_preprocess_caloclouds(
                paths['data_path'], 
                e_per_layer_path=paths['e_per_layer_path'], 
                clusters_per_layer_path=paths['clusters_per_layer_path']
            )
        else:
            dataset = self.data_handler.load_and_preprocess_calochallenge(
                paths['data_path'], 
                e_per_layer_path=paths['e_per_layer_path'], 
                clusters_per_layer_path=paths['clusters_per_layer_path']
            )

        # Get parameters
        kwargs = {}
        params = self.get_params(kwargs, dataset_size=self.cfg.dataset_size)
        
        # Create train loader
        train_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=params["batch_size"], 
            shuffle=params["shuffle"], 
            pin_memory=True
        )

        # Create validation loader if validation data exists
        val_loader = None
        if paths['val_data_path'] is not None:
            if self.cfg.dataset_size == 'pretraining_cc2':
                val_dataset = self.data_handler.load_and_preprocess_caloclouds(
                    paths['val_data_path'], 
                    e_per_layer_path=paths['val_e_per_layer_path'], 
                    clusters_per_layer_path=paths['val_clusters_per_layer_path']
                )
            else:
                val_dataset = self.data_handler.load_and_preprocess_calochallenge(
                    paths['val_data_path'], 
                    e_per_layer_path=paths['val_e_per_layer_path'], 
                    clusters_per_layer_path=paths['val_clusters_per_layer_path']
                )
            
            val_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=params["batch_size"], 
                shuffle=params["shuffle"], 
                pin_memory=True
            )

        return train_loader, val_loader, params
    
    def run_training(self):
        """
        Main method to run the full training pipeline.
        """
        # Free GPU memory if needed
        free_memory()
        
        # Setup paths
        paths = self.setup_dataset_paths()
        
        # Load datasets
        train_loader, val_loader, params = self.load_datasets(paths)
        
        # Create and setup model
        model, distribution = compile_HybridTanH_model_CaloC(
            num_blocks=2, 
            num_inputs=92, 
            num_cond_inputs=1, 
            device=self.device
        )
        model, optimizer = self.model_manager.setup_model(model)
        
        # Train model
        num_epochs = params['epochs']
        if val_loader is not None:
            train_losses, val_losses = self.model_manager.train_model(
                model, 
                optimizer, 
                train_loader, 
                val_loader, 
                distribution, 
                paths['output_path'], 
                num_epochs, 
                dataset_size=self.cfg.dataset_size
            )
            return train_losses, val_losses
        else:
            train_losses = self.model_manager.train_model(
                model, 
                optimizer, 
                train_loader, 
                None, 
                distribution, 
                paths['output_path'], 
                num_epochs, 
                dataset_size=self.cfg.dataset_size
            )
            return train_losses


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_key', type=str, required=True, help="Dataset key (e.g., 'all_10-90')")
    parser.add_argument('--sf_use_pretrain', type=lambda x: x.lower() == 'true', required=True, 
                       help="Whether to use pretrained model (True/False)")
    parser.add_argument('--seed', type=int, required=True, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Print received arguments for debugging
    print(f"Received arguments:")
    print(f"  dataset_key: '{args.dataset_key}'")
    print(f"  sf_use_pretrain: {args.sf_use_pretrain}")
    print(f"  seed: {args.seed}")
    
    return args


def main():
    """
    Main entry point for the ShowerFlow training script.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print(f"\n=== Training with arguments ===")
    print(f"Dataset key: '{args.dataset_key}'")
    print(f"Use showerflow pretrained model: {args.sf_use_pretrain}")
    print(f"Random seed: {args.seed}")
    print("===============================\n")
    
    # Validate dataset key early
    if not args.dataset_key or args.dataset_key.strip() == "":
        raise ValueError("Dataset key cannot be empty. Please provide a valid dataset key.")
    
    if args.dataset_key not in sf_paths and args.dataset_key not in dataset_paths:
        print(f"Warning: Dataset key '{args.dataset_key}' not found in sf_paths or dataset_paths.")
        print(f"Available options in sf_paths: {list(sf_paths.keys())}")
        print(f"Available options in dataset_paths: {list(dataset_paths.keys())}")
    
    # Fix: Add data_path to sf_paths from dataset_paths
    dataset_key = args.dataset_key
    if dataset_key in dataset_paths:
        if dataset_key in sf_paths:
            # Add the data_path to the existing entry
            sf_paths[dataset_key]['data_path'] = dataset_paths[dataset_key]
            print(f"Added data_path to sf_paths['{dataset_key}']:")
            print(f"  data_path: {sf_paths[dataset_key]['data_path']}")
        else:
            # Create a new entry in sf_paths
            sf_paths[dataset_key] = {
                'data_path': dataset_paths[dataset_key],
                'output_path': f"/data/dust/user/valentel/maxwell.merged/MyCaloTransfer/CaloTransfer/train-score/train-showerflow/{dataset_key}",
                'e_per_layer_path': f"/data/dust/user/valentel/maxwell.merged/MyCaloTransfer/CaloTransfer/paths/e_per_layer_{dataset_key}.npy",
                'clusters_per_layer_path': f"/data/dust/user/valentel/maxwell.merged/MyCaloTransfer/CaloTransfer/paths/clusters_per_layer_{dataset_key}.npy"
            }
            print(f"Created new entry in sf_paths for '{dataset_key}'")
    
    # Create trainer and update its configuration
    trainer = ShowerFlowTrainer()
    trainer.cfg.dataset_size = args.dataset_key.strip()  # Ensure no whitespace
    trainer.cfg.sf_use_pretrained = args.sf_use_pretrain
    
    # Run training
    try:
        losses = trainer.run_training()
        print(f"Training completed with final loss: {losses[0][-1] if isinstance(losses, tuple) else losses[-1]}")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise
if __name__ == "__main__":
    main()