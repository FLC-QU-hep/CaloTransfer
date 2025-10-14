import h5py
import numpy as np
import matplotlib.pyplot as plt
import gc
import sys
# import pandas as pd
from tqdm import tqdm
import seaborn as sns
from sklearn.model_selection import train_test_split

import time


import os

def free_memory():
    # Run the garbage collector to free up memory
    gc.collect()
    memory_usage = sys.getsizeof(gc.get_objects())
    print(f"Memory usage after garbage collection: {memory_usage} bytes")

def read_hdf5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        dataset_names = list(f.keys())
        print(f"Dataset names: {dataset_names}")
        showers = f[dataset_names[1]][:]
        incident = f[dataset_names[0]][:]
    return {"showers": showers, "incident": incident}

# def read_hdf5_file2(file_path):
#     with h5py.File(file_path, 'r') as f:
#         keys = list(f.keys())
#         energy = f[keys[0]][:]
#         events = f[keys[1]][:]
#     return keys, energy, events

def read_hdf5_file2(filename):
    """Read HDF5 file with minimal compression handling for faster loading."""
    import h5py
    import numpy as np
    
    print(f"Attempting to read: {filename}")
    
    try:
        # Disable chunk cache to reduce memory issues
        with h5py.File(filename, 'r', rdcc_nbytes=0) as f:
            keys = list(f.keys())            
            # Try to read the energy data with error handling
            try:
                energy = f[keys[0]][:]
            except OSError as e:
                print(f"Error reading energy data: {str(e)}")
                # Create empty array with proper shape if possible
                try:
                    dset = f[keys[0]]
                    energy = np.zeros(dset.shape, dtype=dset.dtype)
                    print(f"Created placeholder energy array of shape {dset.shape}")
                except:
                    energy = np.array([])
                    print("Created empty energy array")
            
            # Try to read the shower data with error handling
            try:
                shower = f[keys[1]][:]
            except OSError as e:
                print(f"Error reading shower data: {str(e)}")
                # Create empty array with proper shape if possible
                try:
                    dset = f[keys[1]]
                    shower = np.zeros(dset.shape, dtype=dset.dtype)
                    print(f"Created placeholder shower array of shape {dset.shape}")
                except:
                    shower = np.array([])
                    print("Created empty shower array")
            
            return keys, energy, shower
            
    except Exception as e:
        print(f"Failed to open or process file: {filename}")
        print(f"Error: {str(e)}")
        # Return empty arrays as placeholders
        return [], np.array([]), np.array([])
    
def loading_files(path="/beegfs/desy/user/valentel/CaloTransfer/data/calo-challenge/original/", n_datasets=[1,2,3,4]):
    """Load the CaloChallenge datas from the given path and return them as a list of dictionaries."""
    data_paths = [os.path.join(path, f'dataset_3_{i}.hdf5') for i in n_datasets]
    print(f"Data path(s): {data_paths}")
    data = [read_hdf5_file(dp) for dp in tqdm(data_paths, desc='Loading datas', file=sys.stdout)]

    for idx, data_dict in enumerate(data):
        print(' - + ' * 20)
        print(f"Dataset {idx + 1}:")
        print(f"  Shape of showers: {data_dict['showers'].shape}")
        print(f"  Shape of incident: {data_dict['incident'].shape}")

    return data

def save_hdf5(data_dict, file_path):
    """
    Save the given dictionary to an HDF5 file at the specified path.

    Parameters:
    data_dict: dict
        Dictionary containing data to save.
    file_path: str
        Full path to the HDF5 file to create.
    """
    with h5py.File(file_path, 'w') as h5f:
        for key, value in data_dict.items():
            h5f.create_dataset(key, data=value)

    print(f"Data saved to: {file_path}")

def plot_energy_histogram(dataset, bins=100, log_scale=True, title=''):
    """
    Plots a histogram of the incident energies in the dataset.
    """
    plt.figure(figsize=(7, 4))
    
    # Plot the histogram
    plt.hist(dataset, bins=bins, alpha=0.75, color='blue', edgecolor='black')
    
    # Plot the mean line
    mean_energy = dataset.mean()
    plt.axvline(mean_energy, color='r', linestyle='dashed', linewidth=1, label=f'Mean energy: {mean_energy:.2f} GeV')
    
    # Optionally set a logarithmic scale for the y-axis
    if log_scale:
        plt.yscale('log')
    
    # Add labels and title
    plt.xlabel('Incident Energy (GeV)')
    plt.ylabel('Number of Events')
    plt.title('Histogram of Incident Energies' + title, fontsize=26)
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    
    plt.show()

def normalize_incident(data):
    """
    Divide the 'incident' energies by 1000 for each dictionary in the list.
    This would serve as a conversion from MeV to GeV.
    
    Parameters:
    data (list of dict): List of dictionaries containing data information.
    
    Returns:
    list of dict: The updated list with normalized 'incident' energies.
    """

    for idx, dataset in enumerate(tqdm(data, desc='Normalizing incident')):
        print(f"Dataset {idx +1}: Type {type(dataset)}")
        print(f"Dataset {idx +1} keys: {dataset.keys()}")

        if isinstance(dataset, dict) and 'incident' in dataset:
            print(f"Max incident energy before normalization: {dataset['incident'].max()}")
            
            if dataset['incident'].max() > 1000:
                dataset['incident'] = dataset['incident'] / 1000.0

            print(f"Max incident energy after normalization: {dataset['incident'].max()}")
        else:
            raise TypeError(f"Expected dictionary with 'incident' key, got {type(dataset)} with keys {dataset.keys()}")
        plot_energy_histogram(dataset['incident'], title=f' Dataset {idx + 1}')
        free_memory()
    
    return data


def filter_events(data, energy_range=(10, 90)):
    """
    Filters events based on energy criteria, and prints the details.

    Parameters:
    data (list of dict): List of dictionaries containing data information.
    energy_range (tuple): A tuple specifying the min and max range of energy.

    Returns:
    list of dict: The filtered data.
    """
    energy_min, energy_max = energy_range

    filtered_data_list = []  # List to store filtered datas

    for idx, dataset in enumerate(tqdm(data, desc='Filtering datas')):
        energy_t = dataset['incident']
        events_t = dataset['showers']

        # Finding indices for energy range
        indices = np.where((energy_t > energy_min) & (energy_t < energy_max))[0]

        print('\n', ' - + ' * 20, '\n')
        print(f"Dataset {idx + 1} - Energy Range: [ {energy_min} - {energy_max} ] GeV \n Number of events: {len(indices)} -  Percentage of total data: {len(indices) / len(energy_t):.2%}")
        print(f" \n Shape of showers: {events_t[indices].shape} - Shape of incident: {energy_t[indices].shape} \n")
        print(' - ' * 20 )

        free_memory()

        plot_energy_histogram(energy_t[indices], title=f' Dataset {idx + 1}')


        # Create filtered data dictionary
        filtered_dataset = {
            "showers": events_t[indices],
            "incident": energy_t[indices]
        }
        
        # Add filtered data to the list
        filtered_data_list.append(filtered_dataset)

    return filtered_data_list

def to_point_cloud(data, incidents = None):
    """
    Converts the input data to a point cloud format and returns it as a dictionary.

    Parameters:
    data: list of dict
        List of dictionaries, each containing 'showers' and 'incident' as keys.

    Returns:
    dict
        Dictionary with keys 'showers' and 'incident', containing the processed point cloud data and incident energies.
    """
    showers_xyz = []  # List to store the converted shower data in Cartesian coordinates
    energies_all = []  # List to store the energies of all events

    # Find the maximum number of non-zero points across all events in all datasets
    # Check if 'showers' is at least 2D, if not, reshape or raise an error
    max_npoint = 0
    for data_entry in tqdm(data, desc='Finding max npoints in all datasets'):
        showers = data_entry['showers']
        if showers.ndim != 2:
            raise ValueError("'showers' array is not 2D. Please check your data format.")

        max_npoint = max(max_npoint, max(np.count_nonzero(showers[i, :]) for i in range(showers.shape[0])))

    print(f"Max number of points: {max_npoint} \n\n")


    # Convert each dataset to point cloud format
    for idx, data_entry in enumerate(data):
        
        showers = data_entry['showers']
        # Try to get the incident energies
        # if not None:

        try:
            energies = data_entry['incident']
        except KeyError:
            energies = data_entry['incident_energies']  # Fallback key        
            
        energies_all.append(energies)
        print(f" \nDataset {idx + 1} over {len(data)} \nShape of showers: {showers.shape} - Shape of incident: {energies.shape} \n")

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
            
            # Append the converted shower data to the list
            showers_xyz.append([shower])
    
    # Free up memory if needed 
    # free_memory() 

    # Merge the individual lists of shower data and energies into unified arrays
    showers_xyz = np.vstack(showers_xyz)
    energies_all = np.vstack(energies_all)



    # Create a dictionary to store the processed point cloud data
    point_cloud = {
        "showers": showers_xyz,
        "incident": energies_all
    }
    print( '\n' ,' - + ' * 20)
    print( ' \n Datasets merged into a single point cloud format.')
    print(f"Final: shape of showers: {point_cloud['showers'].shape} - shape of incident energies: {point_cloud['incident'].shape} \n")

    return point_cloud


def plt_scatter(shower):
    # sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Scatter Plots of Shower Coordinates', fontsize=26)

    color = 'darkred'

    axes[0].scatter(shower[0, :], shower[1, :], s=10, alpha=0.6, edgecolor='w', linewidth=0.5, color=color)
    axes[0].set_xlabel('x', fontsize=22)
    axes[0].set_ylabel('y', fontsize=22)
    axes[0].set_title('x vs y', fontsize=24)

    axes[1].scatter(shower[0, :], shower[2, :], s=10, alpha=0.6, edgecolor='w', linewidth=0.5, color=color)
    axes[1].set_xlabel('x', fontsize=22)
    axes[1].set_ylabel('z', fontsize=22)
    axes[1].set_title('x vs z', fontsize=24)

    axes[2].scatter(shower[1, :], shower[2, :], s=10, alpha=0.6, edgecolor='w', linewidth=0.5, color=color)
    axes[2].set_xlabel('y', fontsize=22)
    axes[2].set_ylabel('z', fontsize=22)
    axes[2].set_title('y vs z', fontsize=24)

    
    [axes[i].grid(False) for i in range(3)]

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plt_scatter_2(shower, cylindrical=False, title='Scatter Plots in Different Planes'):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.suptitle(title, fontsize=30)

    if cylindrical:
        r = shower[0, :]
        phi = shower[1, :]
        z = shower[2, :]

        x = r * np.cos(phi)
        y = r * np.sin(phi)

    else:
        x = shower[0, :]
        y = shower[1, :]
        z = shower[2, :]

    #segments
    num_segments_x = 18
    num_segments_y = 45
    num_segments_z = 50

    # First Subplot
    axs[0].scatter(x, y, s=.4, color='r', alpha=1)
    axs[0].set_ylim(0, 45)
    axs[0].set_xlim(-18, 18)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('XY Plane')

     # Plot equidistant segments along x,y-axis
    x_values = np.linspace(-num_segments_x, num_segments_x, num_segments_x + 1)
    for x_val in x_values:
        axs[0].axvline(x=x_val, color='b', alpha=0.2)
    y_values = np.linspace(0, num_segments_y, num_segments_y + 1)  # Exclude first and last points
    for y_val in y_values:
        axs[0].plot([-num_segments_x, num_segments_x], [y_val, y_val], color='b', alpha=0.2)
    
    # Second Subplot
    axs[1].scatter(x, z, s=.4, color='r', alpha=1)
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('z')
    axs[1].set_ylim(-18, 18)
    axs[1].set_xlim(-18, 18)
    axs[1].set_title('zY Plane')
    
    # # Plot concentric circles
    for radius in range(1, num_segments_x+1):  # 18 concentric circles
        circle = plt.Circle((0, 0), radius, color='black', alpha=0.3, fill=False)
        axs[1].add_artist(circle)
    theta = np.linspace(0, 2*np.pi, num_segments_z +1)
    # print(theta)
    for i in range(num_segments_z):
        axs[1].plot([0, 18*np.cos(theta[i])], [0, 18*np.sin(theta[i])], color='b', alpha=0.3)

    # Third Subplot
    axs[2].scatter(y, z, s=.4, color='r', alpha=1)
    axs[2].set_ylim(-18, 18)
    axs[2].set_xlim(0, 45)
    axs[2].set_xlabel('y')
    axs[2].set_ylabel('z')
    axs[2].set_title('YZ Plane')
    
    for i in range(3):
        axs[i].grid(False)

    # Plot equidistant segments along y-axis
    y_values = np.linspace(-num_segments_x, num_segments_x, num_segments_x+1)  # Exclude first and last points
    for y_val in y_values:
        axs[2].plot([-num_segments_y, num_segments_y], [y_val, y_val], color='b', alpha=0.2)
    x_values = np.linspace(0, num_segments_y, num_segments_y+1)
    for x_val in x_values:
        axs[2].axvline(x=x_val, color='b', alpha=0.2)

    plt.tight_layout()
    plt.show()

def to_cylindrical(events):
    """
    Convert Cartesian coordinates to cylindrical coordinates and clip the values.
    """
    # Extract coordinates and energy values
    x = events[0, :]  # Assuming x coordinates are in the first row
    z = events[1, :]  # Assuming y coordinates are in the second row
    y = events[2, :]  # Assuming z coordinates are in the third row
    e = events[3, :]  # Assuming energy values are in the fourth row
    
    
    # Calculate cylindrical coordinates
    r = np.sqrt(x**2 + y**2)  # Radial distance from the origin
    phi = np.arctan2(y, x)    # Azimuthal angle from the x-axis in radians
    phi[phi < 0] += 2 * np.pi # Adjusting the range of phi to [0, 2 * pi]
    
    return np.stack([r, phi, z, e], axis=0)

def to_cartesian(events):
    rho = events[0, :]
    phi = events[1, :]
    z   = events[2, :]
    e   = events[3, :]
    
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    return np.stack([x, z, y, e], axis=0)

def simple_noise_smearing(events, noise=0.4):
    #simple smearing on xyz
    batch_size = events.shape[0]
    n_points = events.shape[2]
    events_xz = events.copy()  # make a copy to avoid modifying the original array
    
    mask = events_xz[:, :3] == 0
    noise = np.random.uniform(-noise, noise, size=[batch_size, 3, n_points])
    
    events_xz[:, :3] = events_xz[:, :3] + noise
    events_xz[:, :3][mask] = 0
    
    return events_xz

def noise_cylindrical(n_points=6000, batch_size=1000, noise = 0.5 ):  # Changed the default parameter order
    # np.random.seed(42) # TODO remove this
    
    r = np.random.uniform(-noise, noise, size=[batch_size, n_points])
    
    phi_range = 2 * np.pi / 50  
    phi = np.random.uniform(- phi_range * noise , phi_range * noise  , size=[batch_size, n_points])  

    z = np.random.uniform(-noise, noise, size=[batch_size, n_points])
    
    out = np.stack((r, phi, z), axis=1)
    
    return out

def data_smearing(events, noise=.5, simple_noise = False):

    # events = events.copy()  # make a copy to avoid modifying the original array

    if simple_noise:
        events = simple_noise_smearing(events, noise)

    else:
        cylindricalvisible = np.zeros(events.shape)

        for i in tqdm(range(events.shape[0]), desc='Converting to cylindrical'):
            cylinder = to_cylindrical(events[i])
            cylindricalvisible[i] = cylinder
        print('+ - '* 15, '\n MinMax cylindrical without noise') 
        for i in range(3):
            print('events in {}:'.format(i), cylindricalvisible[:, i].min(), cylindricalvisible[:, i].max())
        
        batch_size = cylindricalvisible.shape[0]
        n_points = cylindricalvisible.shape[2]

        free_memory()
        
        mask = cylindricalvisible[:, :3] == 0 
        print('+ - '* 15, '\n\n Adding noise ...')
        noise_arr = noise_cylindrical(n_points, batch_size, noise) 

        free_memory()

        cylindricalvisible[:, :3] = cylindricalvisible[:, :3] + noise_arr 
        cylindricalvisible[:, :3][mask] = 0
        
        free_memory()

        print('+ - '* 15, '\n MinMax cylindrical with noise') 
        for i in range(3):
            print('events in {}:'.format(i), cylindricalvisible[:, i].min(), cylindricalvisible[:, i].max())
       
        events = np.zeros(events.shape)
        for i in tqdm(range(cylindricalvisible.shape[0]), desc='Converting back to cartesian'):
            cartesian = to_cartesian(cylindricalvisible[i])
            events[i] = cartesian

    return events


def sort_and_process(showers, incident):
    """
    Sort and process the data of cylindrical_smear (showers) and point_cloud['incident'] 
    according to the number of non-zero points (npoints).

    Parameters:
    - showers (np.ndarray): 3D array of shape (N, 4, M) containing cylindrical coordinates and other information.
    - point_cloud (dict): Dictionary containing at least the 'incident' key with an array of values associated with the showers.

    Returns:
    - sorted_data (dict): Dictionary containing the sorted showers and corresponding incident energy values.
    - max_npoints (int): Maximum value of npoints from the sorted array.
    """
    # Calculate the number of non-zero points (npoints) for each element in showers
    npoints = [np.count_nonzero(showers[i, 3, :]) for i in range(showers.shape[0])]

    # Convert npoints to a numpy array for easier manipulation in subsequent operations
    npoints = np.array(npoints)

    # Get the indices that would sort the npoints array
    sorted_indices = np.argsort(npoints)

    # Reorder showers and point_cloud['incident'] arrays according to the sorted indices
    sorted_simple_smearing = showers[sorted_indices]
    sorted_incident_energy = incident[sorted_indices]

    # Create a dictionary to hold the sorted data
    sorted_data = {
        'showers': sorted_simple_smearing,
        'incident': sorted_incident_energy
    }

    # Obtain the sorted npoints array
    sorted_npoints = npoints[sorted_indices]

    # Get the maximum value of sorted_npoints
    max_npoints = sorted_npoints[-1]

    # Print the maximum value of npoints and the shape of the sorted data
    print('\n\nmax:')
    print(max_npoints)
    print(f"Shape of sorted data: {sorted_data['showers'].shape}")

    return sorted_data, max_npoints

def rescaling_e(data_dict, factor=0.033):
    """
    Rescale energy values in the 'showers' dataset if they exceed a specific threshold.
    
    data: dict containing 'showers' and 'incident'.
    factor: float factor to rescale the energy.
    
    Returns: rescaled dataset.

    """

    data = data_dict.copy()
    if isinstance(data, dict) and 'showers' in data:
        events = data['showers']
        incident = data['incident']
        
        # Printing max incident energy before normalization
        max_energy_before = data_dict['showers'][:, 3, :].max()
        print(f"Max incident energy before normalization: {max_energy_before}")

        visible_energy = data_dict['showers'][:, 3, :][data_dict['showers'][:, 3, :] > 0]
        plt_visible_e(visible_energy, log_scale=True, title=' Before Rescaling')
        # Rescale if max energy value exceeds the threshold (here assumed as 100)

        if max_energy_before > 400:
            events[:, 3, :] *= factor

        visible_energy = events[:, 3, :][events[:, 3, :] > 0]
        plt_visible_e(visible_energy, log_scale=True, title=' After Rescaling')

        # Printing max incident energy after normalization
        max_energy_after = events[:, 3, :].max()
        print(f"Max incident energy after normalization: {max_energy_after}")
        
        rescaled_data = {
            "showers": events,
            "incident": incident
        }

        return rescaled_data
    else:
        raise TypeError(f"Expected dictionary with 'incident' key, got {type(data)} with keys {data.keys()}")


def plt_visible_e(dataset, log_scale=True, title=''):
    """
    Plots a histogram of the incident energies in the dataset.
    """
    plt.figure(figsize=(7, 4))
    # TODO: add 
    # visible_energy = events[:, 3, :][events[:, 3, :] > 0]

    # Plot the histogram
    plt.hist(dataset, bins=np.logspace(np.log(1e-7), np.log(dataset.max()), 200, base=np.e), alpha=0.75, color='blue', edgecolor='black')
    
    # Plot the mean line
    mean_energy = dataset.mean()
    plt.axvline(mean_energy, color='r', linestyle='dashed', linewidth=1, label=f'Mean energy: {mean_energy:.2f} GeV')
    
    # Optionally set a logarithmic scale for the y-axis
    if log_scale:
        plt.yscale('log')
        plt.xscale('log')
    
    # Add labels and title
    plt.xlabel('Visible  Energy (MeV)')
    plt.ylabel('Number of Events')
    plt.title('Visible Energy' + title, fontsize=26)
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    plt.xlim(1e-4, dataset.max())
    
    plt.show()

def cylindrical_histogram(point_cloud, num_layers=45, num_radial_bins=18, num_angular_bins=50, 
                          z_limit=(-1, 1), r_limit=(0, 1)):
    # Convert Cartesian coordinates (x, y, z) to cylindrical coordinates (z, phi, r)
    # print('point_cloud:', point_cloud.shape) 
    
    x, z, y, e = point_cloud # y and z are sawapped to match calocloud convention

    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    
    # Normalize phi to [0, 2 * np.pi]
    phi = (phi + 2 * np.pi) % (2 * np.pi)
    # phi[phi < 0] += 2 * np.pi 
    
    # Create histogram edges
    r_edges = np.linspace(r_limit[0], r_limit[1], num_radial_bins + 1)
    phi_edges = np.linspace(0, 2 * np.pi, num_angular_bins + 1)
    z_edges = np.linspace(z_limit[0], z_limit[1], num_layers + 1)
    
    # Create the cylindrical histogram
    histogram, _ = np.histogramdd(
        np.stack((z, phi, r), axis=-1),
        bins=(z_edges, phi_edges, r_edges),
        weights=e
    )
    
    return histogram

# Function to normalize data to a specified range
def normalize_to_range(data, min_val, max_val, new_min=-1, new_max=1):
    return ((data - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min

# Function to normalize the 'showers' data within the 'val' dictionary
def normalize_showers(val_events, 
                      Xmin=-18, Xmax=18,
                      Ymin=0, Ymax=45,
                      Zmin=-18, Zmax=18):
    
    val_events = np.zeros(val_events.shape)  # Initialize an array to store the normalized data
  # Initialize an array to store the normalized data

    # Calculate min and max values for each of the first three features (dimension 1)
    # for i in tqdm(range(3), desc='Normalizing showers'):
    #     min_val = val[:, i, :].min()
    #     max_val = val[:, i, :].max()
    #     print(f"Min value of {i}-th feature: {min_val} - Max value of {i}-th feature: {max_val}")

    #     # Normalize each event individually
    #     for j in range(val.shape[0]):
    #         val_events[j, i, :] = normalize_to_range(val[j, i, :], min_val, max_val, -1, 1)
            # if i == 1:
            #     val_events[j, i, :] = normalize_to_range(val[j, i, :], min_val, max_val, 0, 1)
    

    val_events[:, 0, :] = (val_events[:, 0, :] - Xmin) / (Xmax - Xmin) * 2 - 1
    val_events[:, 1, :] = (val_events[:, 1, :] - Ymin) / (Ymax - Ymin) * 2 - 1
    val_events[:, 2, :] = (val_events[:, 2, :] - Zmin) / (Zmax - Zmin) * 2 - 1

    # Verify min and max values after normalization
    for i in range(3):
        print(f"Min value of {i}-th feature: {val_events[:, i, :].min()} - Max value of {i}-th feature: {val_events[:, i, :].max()}")
    
    print("Normalization completed. Shape of normalized data:", val_events.shape)

    return val_events

def split_and_save_hdf5(preprocessed_data, 
                        validation_size=10000, 
                        save = True,
                        train_file='train_data.hdf5', 
                        validation_file='val_data.hdf5',
                        path='/beegfs/desy/user/valentel/CaloTransfer/data/calo-challenge/preprocessing/'
                        ):
    energy = preprocessed_data['incident']
    events = preprocessed_data['showers']

    print(f"Shape of energy: {energy.shape} - Shape of events: {events.shape}\n")
    print('+ - ' * 20, '\n')
    
    total_samples = energy.shape[0]
    validation_proportion = validation_size / total_samples

    if validation_size > total_samples:
        raise ValueError("Validation size is larger than the total number of samples.")
    
    train_events, val_events, train_energy, val_energy = train_test_split(
        events, energy, test_size=validation_proportion, random_state=42, shuffle=True)
    
    # normalise between -1 and 1 xyz
    # Normalize the dataset
    
    print('+ - ' * 20, '\n')
    print(f' Shape of validation events: {val_events.shape}')
    Xmin, Xmax = -18, 18
    Ymin, Ymax = 0, 45
    Zmin, Zmax = -18, 18

    #normalise
    val_events[:, 0, :] = (val_events[:, 0, :] - Xmin) / (Xmax - Xmin) * 2 - 1
    val_events[:, 1, :] = (val_events[:, 1, :] - Ymin) / (Ymax - Ymin) * 2 - 1
    val_events[:, 2, :] = (val_events[:, 2, :] - Zmin) / (Zmax - Zmin) * 2 - 1
    print(f' Shape of validation events: {val_events.shape}')
    hist_val = np.zeros((val_events.shape[0], 45, 50, 18))

    # Fill histograms
    for i in range(val_events.shape[0]):
        # event =   # Extract individual event
        hist_val[i] = cylindrical_histogram(val_events[i])  # Process event data

    val_events = hist_val
    # print("Projection and reshape of the validation set: \n\n Original shape of validation events: ", val_events.shape)
    # val_events = normalize_showers(val_events)
    # val_events_reshaped = np.zeros((validation_size, 45 * 50 * 18))

    # for i in tqdm(range(validation_size), desc='Processing validation events: '):
        
    #     hist = cylindrical_histogram(val_events[i])
    #     hist_reshaped = hist.reshape(45 * 50 * 18)
    #     val_events_reshaped[i] = hist_reshaped
    

    # print("Reshaped validation events: ", val_events_reshaped.shape)
    # val_events = normalize_showers(val_events_reshaped)

    train_data_dict = {
        'energy': train_energy,
        'events': train_events
    }
    val_data_dict = {
        'incident_energies': val_energy,
        'showers': val_events
    }

    print('\n','+ - ' * 20)
    print(f"\n Shape of energy: {train_energy.shape}, \n Shape of events: {train_events.shape}\n Keys: {train_data_dict.keys()}\n")
    print(f"\n Shape of energy: {val_energy.shape}, \n Shape of events: {val_events.shape}, \n Keys: {val_data_dict.keys()}\n")
    print('\n','+ - ' * 20)
    # Full paths to the files
    train_file_path = path + train_file
    validation_file_path = path + validation_file
    if save:
        # Save the train data to HDF5
        print(f"\nSaving train and val dictionaries in external hdf5 files:\n")
        save_hdf5(train_data_dict, train_file_path )
        save_hdf5(val_data_dict, validation_file_path)
        print('\n','+ - ' * 20)
        print('\n\n Done!')
    

    return train_data_dict, val_data_dict
