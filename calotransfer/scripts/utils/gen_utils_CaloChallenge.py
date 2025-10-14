import numpy as np
import torch
from tqdm import tqdm
# from .metadata import Metadata
from configs import Configs

import h5py
import sys
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False

# from utils.plotting import MAP, offset, layer_bottom_pos, cell_thickness, Xmax, Xmin, Zmax, Zmin

config = Configs()


def get_cog(x, y, z, e):
    epsilon = 1e-10
    coords = [x, y, z]
    results = []
    
    e_sum = e.sum(axis=1) + epsilon
    
    for coord in coords:
        weighted_sum = np.sum((coord * e), axis=1)
        results.append(weighted_sum / e_sum)
    
    return tuple(results)

def cylindrical_histogram(point_cloud, num_layers=45, num_radial_bins=18, num_angular_bins=50, range_limit=(-1, 1)):
    # Convert Cartesian coordinates (x, y, z) to cylindrical coordinates (z, phi, r)
    
    x, z, y, e = point_cloud  #consistent with caloclouds axis
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    # Normalize phi to [0, 2 * pi]
    phi = (phi + 2 * np.pi) % (2 * np.pi)
    
    # Create histogram edges
    r_edges = np.linspace(0, 1, num_radial_bins + 1)
    phi_edges = np.linspace(0, 2.0 * np.pi, num_angular_bins + 1)
    z_edges = np.linspace(-1, 1, num_layers+1)
    
        
    # Create the cylindrical histogram
    histogram, _ = np.histogramdd(
        np.stack((z, phi, r), axis=-1),
        bins=(z_edges, phi_edges, r_edges),
        weights=e
    )
    
    return histogram

def get_scale_coeff(num_shower=100, model=None, flow=None, config=config):
    
    
    print('Getting scale factor...')
    fake_showers_xyz = gen_showers_batch_w_occupancy_cog(model, flow, e_min=config.min_energy, e_max=config.max_energy,
                                                        config=config, num=num_shower, point_cloud=True)
    print('Generated fake_showers_xyz shape point cloud is:', fake_showers_xyz['showers'].shape)
    fake_showers = gen_showers_batch_w_occupancy_cog(model, flow, e_min = config.min_energy, e_max = config.max_energy, 
                                    config = config, num=num_shower)
    print('Generated fake_showers shape grid is:', fake_showers['showers'].shape)
    
    occ_fake = np.count_nonzero(fake_showers['showers'].reshape(fake_showers['showers'].shape[0], 45*50*18), axis=-1)
    fs_pc = to_point_cloud(fake_showers['showers'].reshape(1, fake_showers['showers'].shape[0], 45*50*18))
    fs_pc = fs_pc.squeeze()

    num_hits_fakel = (fake_showers_xyz['showers'][:, -1, :] > 0).sum(axis=1)  # get num points in the point clouds

    # Ensure num_hits_fakel and occ_fake are 1D arrays
    num_hits_fakel = num_hits_fakel.flatten()
    print('num_hits_fakel', len(num_hits_fakel))
    occ_fake = occ_fake.flatten()
    print('occ_fake', len(occ_fake))

    coef_fake = np.polyfit(num_hits_fakel, occ_fake, 3)

    return coef_fake

def get_scale_factor(num_clusters, num_shower = 5000, model = None, flow= None,):

    coef_real = np.array([ -1.24855436e-13,  9.16723716e-10,  9.99997473e-01,  1.47402061e-03]) # polyfit to 10-90GeV training data

    if model is None:
        coef_fake = np.array([ 7.67824367e-11, -5.83052179e-06,  8.30170804e-01,  2.54593746e+01]) #generated
    else :
        coef_fake = get_scale_coeff(num_shower, model, flow, config, )
        print('coef_fake', coef_fake)

    poly_fn_real = np.poly1d(coef_real)
    poly_fn_fake = np.poly1d(coef_fake) 
    
    scale_factor = poly_fn_fake(poly_fn_real(num_clusters))/num_clusters

    return 1./scale_factor

def get_shower(model, num_points, energy, cond_N, config, bs=1):
    
    e = torch.ones((bs, 1), device=config.device) * energy
    n = torch.ones((bs, 1), device=config.device) * cond_N

    if config.norm_cond: # same as defined in dataset.py
        e = torch.log((e + 1e-5) /config.min_energy) / np.log(config.max_energy/config.min_energy)
        n = torch.log((n + 1)/config.min_points) / np.log(config.max_points/config.min_points)
    cond_feats = torch.cat([e, n], -1)
        
    with torch.no_grad():
        if config.kdiffusion:
            fake_shower = model.sample(cond_feats, num_points, config)
        else:
            fake_shower = model.sample(cond_feats, num_points, config.flexibility)
    
    return fake_shower

# batch inference 
def gen_showers_batch(model, shower_flow, e_min, e_max, config, num=2000, bs=32, point_cloud = False):
   
    output = {}
    
    leyer_pos = np.arange(-0.98, 1, 0.0444)
    
    low_log = np.log10(e_min)  # convert to log space
    high_log = np.log10(e_max)  # convert to log space
    uniform_samples = np.random.uniform(low_log, high_log, num)
    
    # apply exponential function (base 10)
    log_uniform_samples = np.power(10, uniform_samples)
    log_uniform_samples.sort()
    cond_E_not_norm = torch.tensor(log_uniform_samples).view(num, 1).to(config.device).float()
    
    output['energy'] = log_uniform_samples
    log_uniform_samples = ( np.log((log_uniform_samples+1e-5)/config.min_energy) / np.log((config.max_energy-1e-5)/config.min_energy) ).reshape(-1)
 
    cond_E = torch.tensor(log_uniform_samples).view(num, 1).to(config.device).float()
    # print("cond_E:", cond_E_not_norm)
    
    samples = shower_flow.condition(cond_E).sample(torch.Size([num, ])).to(config.device).cpu().numpy()
    # plt.scatter(cond_E.cpu().numpy().reshape(-1), samples[:, 1], s=1, alpha=1)
    # plt.show()
    
    # visible_energy, num_points, e_per_layer, clusters_per_layer
    energy_sum = samples[:, 1]
    energy_sum = invers_transform_energy(energy_sum/1000, config)
    energy_sum = energy_sum.reshape(num, 1)


    num_clusters = samples[:, 0]
    num_clusters = invers_transform_points(num_clusters, config=config)
    # print(num_clusters)
    
    e_per_layer = samples[:, -45:]
    e_per_layer[e_per_layer < 0] = 0
    e_per_layer = e_per_layer / e_per_layer.sum(axis=1).reshape(num, 1)
    e_per_layer[e_per_layer<0] = e_per_layer[e_per_layer<0] * (-1)
    
    clusters_per_layer = samples[:, 5:50]
    clusters_per_layer = clusters_per_layer / clusters_per_layer.sum(axis=1).reshape(num, 1)
    clusters_per_layer = clusters_per_layer * num_clusters.reshape(num, 1)
    
    energy_sum[energy_sum < 0] = energy_sum[energy_sum < 0] * (-1)
    clusters_per_layer[clusters_per_layer < 0] = clusters_per_layer[clusters_per_layer < 0] * (-1)
    clusters_per_layer = clusters_per_layer.astype(int)

    output['num_points'] = clusters_per_layer.sum(axis=1)

    fake_showers_list = []

    for evt_id in tqdm(range(0, num, bs)):
        if (num - evt_id) < bs:
            bs = num - evt_id

        hits_per_layer_all = clusters_per_layer[evt_id : evt_id+bs] # shape (bs, num_layers) 
        max_num_clusters = hits_per_layer_all.sum(axis=1).max().astype(int) 
        cond_N = torch.Tensor(hits_per_layer_all.sum(axis=1)).to(config.device).unsqueeze(-1)
        
        # print('max cluster', max_num_clusters)
        # print('cond_N', cond_N)
        fs = get_shower(model, max_num_clusters, cond_E_not_norm[evt_id : evt_id+bs].to(config.device), cond_N, bs=bs, config=config)

        fs = fs.cpu().numpy()   
        # print_num_clusters(fs)
        
        # plot.plt_scatter(np.swapaxes(fs[-1], 1,0))
        # plt.show()

        if np.isnan(fs).sum() != 0:
            print('nans in showers!')
            fs[np.isnan(fs)] = 0

        for i, hits_per_layer in enumerate(hits_per_layer_all):
            n_hits_to_concat = max_num_clusters - hits_per_layer.sum()

            z_flow = np.repeat(leyer_pos, hits_per_layer)
            z_flow = np.concatenate([z_flow, np.zeros(n_hits_to_concat)])

            mask = np.concatenate([ np.ones( hits_per_layer.sum() ), np.zeros( n_hits_to_concat ) ])

            fs[i, :, 1][mask == 0] = 10
            idx_dm = np.argsort(fs[i, :, 1])
            fs[i, :, 1][idx_dm] = z_flow

            fs[i, :, :][z_flow==0] = 0

        fs[:, :, -1][fs[:, :, -1]  < 0] = 0
        
        length = 20000 - fs.shape[1]
        fs = np.concatenate((fs, np.zeros((bs, length, 4))), axis=1)
        fs = np.moveaxis(fs, -1, 1) 

        if point_cloud:
            if evt_id == 0:
                print_num_clusters(fs)
                print(fs.shape)
        else:
            fs = [[cylindrical_histogram(fs)] for fs in fs]
            fs = np.vstack(fs)
            if evt_id == 0:
                print_num_clusters(fs.reshape(-1, 1, 45*50*18))

        fake_showers_list.append(fs)
        
    fake_showers = np.vstack(fake_showers_list)
    
    if point_cloud:
        output['showers'] = fake_showers
        return output
    
    # energy per layer calibration
    sum_fake_showers = fake_showers.sum(axis=(2, 3)).reshape(num, 45, 1, 1)
    sum_fake_showers[sum_fake_showers == 0] = 1  # Avoid division by zero

    fake_showers = fake_showers / sum_fake_showers
    fake_showers[np.isnan(fake_showers)] = 0
    fake_showers = fake_showers * (e_per_layer * energy_sum).reshape(num, 45, 1, 1)
    
    fake_showers /= 0.033 # 0.033 is scaling factor for the energy

    output['showers'] = fake_showers
    
    return output


def generate_sorted_log_uniform_energies(e_min, e_max, num, device='cuda'):
    """
    Generate sorted log-uniformly distributed energies exactly between e_min and e_max.
    
    Args:
        e_min: Minimum energy in GeV (exact lower bound)
        e_max: Maximum energy in GeV (exact upper bound)
        num: Number of samples to generate
        device: Device to place the tensor on
    
    Returns:
        cond_E: Tensor of normalized conditional energies for the model
        raw_energies: Original sampled energies in GeV before normalization
    """
    # Method 1: Create evenly spaced points in log space - guarantees exact bounds
    t = np.linspace(0, 1, num)  # Linear space from 0 to 1
    raw_energies = e_min * np.power(e_max/e_min, t)  # Transform to log space between e_min and e_max
    
    # Verify bounds are exact
    assert abs(raw_energies[0] - e_min) < 1e-10
    assert abs(raw_energies[-1] - e_max) < 1e-10
    
    # Normalize using log scale - this maps [e_min, e_max] to [0, 1] exactly
    normalized_energies = (np.log(raw_energies/e_min) / 
                           np.log(e_max/e_min)).reshape(-1)
    
    # Convert to tensor and reshape for conditioning
    cond_E = torch.tensor(normalized_energies).view(num, 1).to(device).float()
    
    return cond_E, raw_energies

# batch inference 
def gen_showers_batch_w_occupancy_cog(model, shower_flow, e_min, e_max, config, num=2000, bs=32, 
                                        point_cloud = False, scale_occupancy = False, cog_cal=False,):
    

    output = {}
    leyer_pos = np.arange(-0.98, 1, 0.0444)
    
    low_log = np.log10(e_min)  # convert to log space
    high_log = np.log10(e_max)  # convert to log space
    uniform_samples = np.random.uniform(low_log, high_log, num)
    
    # apply exponential function (base 10)
    log_uniform_samples = np.power(10, uniform_samples)
    log_uniform_samples.sort()
    cond_E_not_norm = torch.tensor(log_uniform_samples).view(num, 1).to(config.device).float()
    
    output['energy'] = log_uniform_samples
    log_uniform_samples = ( np.log((log_uniform_samples)/config.min_energy) / np.log((config.max_energy)/config.min_energy) ).reshape(-1)
    print ('min energy', config.min_energy, 'max energy', (config.max_energy))
    cond_E = torch.tensor(log_uniform_samples).view(num, 1).to(config.device).float()
    
    samples = shower_flow.condition(cond_E).sample(torch.Size([num, ])).to(config.device).cpu().numpy()

    # visible_energy, num_points, e_per_layer, clusters_per_layer
    clusters_per_layer = samples[:, 2:47] * config.sf_norm_points # postprocessing
    clusters_per_layer = np.abs(clusters_per_layer)  # Ensure non-negative values
    if config.val_ds_type == '10-90GeV':
        upper_bound = 294
    elif config.val_ds_type == '1-1000GeV':
        upper_bound = 800

    clusters_per_layer = np.clip(clusters_per_layer, 0, upper_bound)  # Enforce upper bound, max number of points in a layer
    clusters_per_layer = clusters_per_layer.astype(int)
    num_clusters = clusters_per_layer.sum(axis=1)

    if scale_occupancy:
        scale_factor = get_scale_factor(num_clusters, num_shower = 2500, model = model, flow= shower_flow)
        print('scale factor', scale_factor)
        print(' + - '*20, '\n'*2)
        print('BEFORE RESCALE, # clusters:', num_clusters)
        num_clusters =  (num_clusters * scale_factor).astype(int)
        print('AFTER RESCALE, # clusters', num_clusters)
        print(' + - '*20, '\n'*2)
    else:
        num_clusters = (num_clusters).astype(int)    

    output['num_points'] = num_clusters
    
    fake_showers_list = []
    if cog_cal:
        cog_x, _, cog_z = normalize_coordinates(samples, config=config)

    for evt_id in tqdm(range(0, num, bs)):
        if (num - evt_id) < bs:
            bs = num - evt_id

        hits_per_layer_all = clusters_per_layer[evt_id : evt_id+bs] # shape (bs, num_layers) 
        max_num_clusters = hits_per_layer_all.sum(axis=1).max().astype(int) 
        cond_N = torch.Tensor(hits_per_layer_all.sum(axis=1)).to(config.device).unsqueeze(-1)
        
        fs = get_shower(model, max_num_clusters, cond_E_not_norm[evt_id : evt_id+bs], cond_N, bs=bs, config=config)
    
        fs = fs.cpu().numpy()   
        
        # plot.plt_scatter(np.swapaxes(fs[-1], 1,0))        

        if np.isnan(fs).sum() != 0:
            print('nans in showers!')
            fs[np.isnan(fs)] = 0
        
        
        for i, hits_per_layer in enumerate(hits_per_layer_all):
    
            n_hits_to_concat = max_num_clusters - hits_per_layer.sum()

            z_flow = np.repeat(leyer_pos, hits_per_layer)
            z_flow = np.concatenate([z_flow, np.zeros(n_hits_to_concat)])

            mask = np.concatenate([ np.ones( hits_per_layer.sum() ), np.zeros( n_hits_to_concat ) ])

            fs[i, :, 1][mask == 0] = 10
            idx_dm = np.argsort(fs[i, :, 1])
            fs[i, :, 1][idx_dm] = z_flow

            fs[i, :, :][z_flow==0] = 0

        fs[:, :, -1][fs[:, :, -1]  < 0] = 0
        
        length = 25000 - fs.shape[1]

        if scale_occupancy:
            length = int(30000 * get_scale_factor(20000)) - fs.shape[1] 
        fs = np.concatenate((fs, np.zeros((bs, length, 4))), axis=1)
        fs = np.moveaxis(fs, -1, 1)
        
        if cog_cal:
            # CoG calibration on X and Z
            cog = get_cog(fs[:, 0], fs[:, 1], fs[:, 2], fs[:, 3])
            fs[:, 0] -= (cog[0] - cog_x[evt_id : evt_id+bs])[: , None]
            fs[:, 2] -= (cog[2] - cog_z[evt_id : evt_id+bs])[: , None]

        if point_cloud:
            if evt_id == 0:
                print_num_clusters(fs)
                print(fs.shape)
        else:
            fs = [[cylindrical_histogram(fs)] for fs in fs]
            fs = np.vstack(fs)
            if evt_id == 0:
                print_num_clusters(fs.reshape(-1, 1, 45*50*18))

        fake_showers_list.append(fs)
    
    fake_showers = np.vstack(fake_showers_list)

    if point_cloud:
        output['showers'] = fake_showers
        return output
    
    # energy per layer calibration
    # Ensure no division by zero or invalid values
    # we projected to cylindrical histogram, now we need to normalize it: divide by sum of energy in each layer

    # fake_showers = fake_showers / fake_showers.sum(axis=(2,3)).reshape(num, 45, 1, 1)
    fake_showers[np.isnan(fake_showers)] = 0
    # fake_showers = fake_showers * (e_per_layer*energy_sum).reshape(num, 45, 1, 1)

    # postprocessing    
    fake_showers /= 0.033 # 0.033 is scaling factor for the energy

    output['showers'] = fake_showers
    
    return output


def print_num_clusters(data):
    npoint2 = np.count_nonzero(data[:,-1, :], axis=-1)
    max_npoint =  npoint2.max()
    print('max num cluster', max_npoint)
    return max_npoint

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