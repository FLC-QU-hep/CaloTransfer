import h5py
import numpy as np
import os
import sys
import traceback
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm

# Configuration
seed = 44  # Fixed seed for reproducibility
ds_type = '1-1000GeV'  # '1-1000GeV' or '10-90GeV'
CHUNK_SIZE = 10000  # Process data in chunks to avoid memory issues

# Set random seed
np.random.seed(seed)

# Define paths
base_path = '/data/dust/user/valentel/maxwell.merged/MyCaloTransfer/CaloTransfer/data/calo-challenge/preprocessing/reduced_datasets/'

if ds_type == '10-90GeV':
    source_data_path = base_path + '10-90GeV/47k_dset1-2-3_prep_10-90GeV.hdf5'
elif ds_type == '1-1000GeV':
    source_data_path = base_path + '1-1000GeV/100k_train_dset1-2-3_prep_1-1000GeV.hdf5'

output_base = base_path + f'{ds_type}/'
per_layer_base = f'/data/dust/user/valentel/maxwell.merged/MyCaloTransfer/CaloTransfer/utils/per_layer/train/{ds_type}/'

results_dir = './results'

# Apply seed to ALL output paths consistently
output_base = f'{output_base}/seed_{seed}/'
per_layer_base = f'{per_layer_base}/seed_{seed}/'
results_dir = f'{results_dir}/seed_{seed}'

# Dataset sizes
sizes = [100, 500, 1_000, 5_000, 10_000, 50_000, 99_000]
size_names = ['100', '500', '1k', '5k', '10k', '50k', '100k']
# sizes = [99_000]
# size_names = ['100k']


def load_original_dataset():
    """Get info about the original dataset without loading it entirely"""
    if not os.path.exists(source_data_path):
        raise FileNotFoundError(f"Source dataset not found: {source_data_path}")
    
    try:
        with h5py.File(source_data_path, 'r') as f:
            events_shape = f['events'].shape
            energy_shape = f['energy'].shape
            print(f"Shape of 'events': {events_shape}, 'energy': {energy_shape}")
            return events_shape[0], events_shape, energy_shape
    except Exception as e:
        raise RuntimeError(f"Failed to read dataset {source_data_path}: {e}")

def create_subdatasets():
    """Create smaller datasets from the original large dataset"""
    total_samples, events_shape, energy_shape = load_original_dataset()
    output_dir = Path(output_base)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset_size, size_name in zip(sizes, size_names):
        data_path = output_dir / f"{size_name}_train_dset1-2-3_prep_{ds_type}.hdf5"
        
        if os.path.exists(data_path):
            # Check if the file is valid
            try:
                with h5py.File(data_path, 'r') as f:
                    if 'events' in f and 'energy' in f:
                        events_shape_check = f['events'].shape
                        energy_shape_check = f['energy'].shape
                        if events_shape_check[0] == dataset_size and energy_shape_check[0] == dataset_size:
                            print(f"Valid dataset already exists at {data_path}, skipping...")
                            continue
                        else:
                            print(f"Dataset at {data_path} has wrong size, recreating...")
            except Exception as e:
                print(f"Dataset at {data_path} is corrupted: {e}")
                print("Removing corrupted file and recreating...")
            
            # Remove the corrupted/invalid file
            os.remove(data_path)
        
        # Generate random indices
        indices = np.random.choice(total_samples, dataset_size, replace=False)
        indices.sort()
        
        # Create output file and datasets with chunked writing
        with h5py.File(source_data_path, 'r') as f_in, h5py.File(data_path, 'w') as f_out:
            events_in, energy_in = f_in['events'], f_in['energy']
            
            # Create output datasets
            events_out = f_out.create_dataset('events', shape=(dataset_size, *events_shape[1:]), 
                                            dtype=events_in.dtype, chunks=True)
            energy_out = f_out.create_dataset('energy', shape=(dataset_size, *energy_shape[1:]), 
                                            dtype=energy_in.dtype, chunks=True)
            
            # Process in chunks
            for i in tqdm(range(0, dataset_size, CHUNK_SIZE), desc=f"Creating {size_name} dataset"):
                chunk_indices = indices[i:i+CHUNK_SIZE]
                chunk_size = len(chunk_indices)
                
                # Read and write chunk
                events_out[i:i+chunk_size] = events_in[chunk_indices]
                energy_out[i:i+chunk_size] = energy_in[chunk_indices]
            
        print(f"Successfully saved dataset of size {dataset_size}")

def calculate_per_layer_metrics():
    """Calculate energy and clusters per layer for each dataset"""
    for size_name in size_names:
        data_path = Path(output_base) / f"{size_name}_train_dset1-2-3_prep_{ds_type}.hdf5"
        output_dir = Path(per_layer_base) / size_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        e_per_layer_path = output_dir / "e_per_layer.npy"
        clusters_per_layer_path = output_dir / "clusters_per_layer.npy"
        
        if os.path.exists(e_per_layer_path) and os.path.exists(clusters_per_layer_path):
            # Check if the files are valid
            try:
                e_test = np.load(e_per_layer_path)
                c_test = np.load(clusters_per_layer_path)
                if e_test.shape[1] == 45 and c_test.shape[1] == 45:
                    print(f"Valid metrics already exist for {size_name}, skipping...")
                    continue
            except Exception as e:
                print(f"Metrics files for {size_name} are corrupted: {e}")
                print("Removing corrupted files and recalculating...")
                if os.path.exists(e_per_layer_path):
                    os.remove(e_per_layer_path)
                if os.path.exists(clusters_per_layer_path):
                    os.remove(clusters_per_layer_path)
        
        if not os.path.exists(data_path):
            print(f"WARNING: Dataset file {data_path} not found. Skipping...")
            continue
        
        print(f"Processing dataset: {data_path}")
        
        # Verify the dataset file is valid before processing
        try:
            with h5py.File(data_path, 'r') as f:
                n_samples = f['events'].shape[0]
                print(f"Dataset shape: {f['events'].shape}")
        except Exception as e:
            print(f"ERROR: Cannot read dataset file {data_path}: {e}")
            print("Please run with --create flag to recreate the dataset.")
            continue
        
        # Unnormalization parameters
        Xmin, Xmax = -18, 18
        Ymin, Ymax = 0, 45
        Zmin, Zmax = -18, 18
        
        num_layers = 45
        
        # Initialize arrays to accumulate results
        clusters_per_layer = np.zeros((n_samples, num_layers))
        e_per_layer = np.zeros((n_samples, num_layers))
        
        # Process in chunks
        with h5py.File(data_path, 'r') as f:
            events_dataset = f['events']
            
            for start_idx in tqdm(range(0, n_samples, CHUNK_SIZE), desc=f"Processing {size_name}"):
                end_idx = min(start_idx + CHUNK_SIZE, n_samples)
                chunk_size = end_idx - start_idx
                
                # Load chunk
                events_chunk = events_dataset[start_idx:end_idx]
                
                # Unnormalize chunk
                events_chunk[:, 0, :] = (events_chunk[:, 0, :] + 1) * (Xmax - Xmin) / 2 + Xmin
                events_chunk[:, 1, :] = (events_chunk[:, 1, :] + 1) * (Ymax - Ymin) / 2 + Ymin
                events_chunk[:, 2, :] = (events_chunk[:, 2, :] + 1) * (Zmax - Zmin) / 2 + Zmin
                events_chunk[:, -1, :] /= 1000  # preprocessing to match caloclouds
                
                # Calculate metrics for this chunk
                for i in range(num_layers):
                    layer_mask = (events_chunk[:, 1, :] > i) & (events_chunk[:, 1, :] < i+1)
                    clusters_per_layer[start_idx:end_idx, i] = layer_mask.sum(axis=1)
                    e_per_layer[start_idx:end_idx, i] = (events_chunk[:, -1, :] * layer_mask).sum(axis=1)
        
        # Save the numpy arrays
        np.save(clusters_per_layer_path, clusters_per_layer)
        np.save(e_per_layer_path, e_per_layer)
        
        print(f"Final shapes - clusters_per_layer: {clusters_per_layer.shape}, e_per_layer: {e_per_layer.shape}")
        
        # Visualize the results for this dataset
        visualize_metrics(size_name)

def visualize_metrics(size_name):
    """Create visualizations of the layer metrics"""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    output_dir = Path(per_layer_base) / size_name
    e_per_layer_path = output_dir / "e_per_layer.npy"
    clusters_per_layer_path = output_dir / "clusters_per_layer.npy"
    
    print('e_per_layer_path:', e_per_layer_path)
    print('clusters_per_layer_path:', clusters_per_layer_path)
    
    # Load data
    e_per_layer = np.load(e_per_layer_path)
    clusters_per_layer = np.load(clusters_per_layer_path)
    
    # Find maximum values and corresponding layers
    max_energy_overall = np.max(e_per_layer)
    max_clusters_overall = np.max(clusters_per_layer)
    
    # Find the maximum value in each layer
    max_energy_per_layer = np.max(e_per_layer, axis=0)
    max_clusters_per_layer = np.max(clusters_per_layer, axis=0)
    
    # Find the layer with the highest max value
    max_energy_layer = np.argmax(max_energy_per_layer) + 1  # +1 to make it 1-indexed
    max_clusters_layer = np.argmax(max_clusters_per_layer) + 1  # +1 to make it one-indexed
    
    # Print the results
    print("\n=== Maximum Values Analysis ===")
    print(f"Energy per layer - Global maximum: {max_energy_overall:.4f}")
    print(f"Clusters per layer - Global maximum: {max_clusters_overall:.0f}")
    print(f"Energy per layer - Maximum occurs in layer {max_energy_layer} with value {max_energy_per_layer[max_energy_layer-1]:.4f}")
    print(f"Clusters per layer - Maximum occurs in layer {max_clusters_layer} with value {max_clusters_per_layer[max_clusters_layer-1]:.0f}")
    
    # Calculate mean values for each layer
    mean_energy_per_layer = np.mean(e_per_layer, axis=0)
    mean_clusters_per_layer = np.mean(clusters_per_layer, axis=0)
    
    print("\n=== Layer-by-Layer Statistics ===")
    print("Layer\tMax Energy\tMean Energy\tMax Clusters\tMean Clusters")
    for i in range(45):
        print(f"{i+1}\t{max_energy_per_layer[i]:.4f}\t{mean_energy_per_layer[i]:.4f}\t{max_clusters_per_layer[i]:.0f}\t{mean_clusters_per_layer[i]:.2f}")
    
    # Create summary plot
    fig = plt.figure(figsize=(15, 10))
    
    # Energy plot
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(range(1, 46), [float(x) for x in max_energy_per_layer], 'r-', label='Max Energy')
    ax1.plot(range(1, 46), [float(x) for x in mean_energy_per_layer], 'b--', label='Mean Energy')
    ax1.set_title('Energy per Layer')
    ax1.set_xlabel('Layer Number')
    ax1.set_ylabel('Energy')
    ax1.grid(True)
    ax1.legend()
    
    # Clusters plot
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(range(1, 46), [int(x) for x in max_clusters_per_layer], 'r-', label='Max Clusters')
    ax2.plot(range(1, 46), [float(x) for x in mean_clusters_per_layer], 'b--', label='Mean Clusters')
    ax2.set_title('Clusters per Layer')
    ax2.set_xlabel('Layer Number')
    ax2.set_ylabel('Number of Clusters')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/summary_{size_name}.png", dpi=300)
    plt.close()
    
    # Create detailed histograms for each layer
    fig = plt.figure(figsize=(30, 15))
    
    # Create main GridSpec
    gs = gridspec.GridSpec(1, 2, figure=fig)
    
    # Create nested GridSpecs for each half
    gs1 = gridspec.GridSpecFromSubplotSpec(9, 5, subplot_spec=gs[0])
    gs2 = gridspec.GridSpecFromSubplotSpec(9, 5, subplot_spec=gs[1])
    
    # Plot energy per layer
    for i in range(45):
        ax = fig.add_subplot(gs1[i])
        h1 = ax.hist(e_per_layer[:, i].flatten(), bins=100, color='darkred', alpha=0.4, label='Calochallenge')
        ax.set_yscale('log')
        ax.text(0.95, 0.95, str(i + 1), transform=ax.transAxes, fontsize=16, 
                verticalalignment='top', horizontalalignment='right')
        # Add text showing max value
        ax.text(0.95, 0.85, f"Max: {max_energy_per_layer[i]:.2f}", transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', horizontalalignment='right')
        if i == 0:
            ax.legend()
    
    # Add title for first set of plots
    fig.text(0.25, 1, 'Energy per layer (log scale)', fontsize=30, ha='center')
    
    # Plot clusters per layer
    for i in range(45):
        ax = fig.add_subplot(gs2[i])
        h1 = ax.hist(clusters_per_layer[:, i].flatten(), bins=100, color='darkred', alpha=0.4, 
                     label='Calochallenge')
        ax.set_yscale('log')
        ax.text(0.95, 0.95, str(i + 1), transform=ax.transAxes, fontsize=16, 
                verticalalignment='top', horizontalalignment='right')
        # Add text showing max value
        ax.text(0.95, 0.85, f"Max: {max_clusters_per_layer[i]:.0f}", transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', horizontalalignment='right')
        if i == 0:
            ax.legend()
    
    # Add title for second set of plots
    fig.text(0.75, 1, 'Clusters per layer (log scale)', fontsize=30, ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/detailed_{size_name}.png", dpi=300)
    plt.close()
    
    print(f"Visualization completed for {size_name}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process calorimeter datasets')
    parser.add_argument('--create', action='store_true', help='Create subdatasets')
    parser.add_argument('--calculate', action='store_true', help='Calculate metrics')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--size', type=str, help='Dataset size to visualize')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    
    args = parser.parse_args()
    
    if args.all or args.create:
        create_subdatasets()
        
    if args.all or args.calculate:
        calculate_per_layer_metrics()
        
    if args.all or args.visualize:
        if args.size:
            visualize_metrics(args.size)
        else:
            for size_name in size_names:
                visualize_metrics(size_name)
    
    if not (args.create or args.calculate or args.visualize or args.all):
        parser.print_help()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)