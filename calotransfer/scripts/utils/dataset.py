import os
import random
from copy import copy
import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import h5py
from tqdm.auto import tqdm

class PointCloudDataset(Dataset):
    def __init__(self, file_path, chech_overfiting=False, bs=32, max_ds_seq_len=6000, quantized_pos=True):
        self.dataset = h5py.File(file_path, 'r')
        self.Ymin, self.Ymax = 0, 30
        self.Xmin, self.Xmax = -200, 200
        self.Zmin, self.Zmax = -160, 240
        self.chech_overfiting = chech_overfiting
        self.bs = bs
        self.max_ds_seq_len = max_ds_seq_len

        self.quantized_pos = quantized_pos
        self.offset = 5.0883331298828125 / 6 # size of x36 granular grid

    def get_n_points(self, data, axis=-1):
        n_points_arr = (data[...,axis] != 0.0).sum(1)
        return n_points_arr

    def __getitem__(self, idx):
        
        if idx > self.bs and idx < self.__len__() - self.bs:
            event = self.dataset['events'][idx-int(self.bs/2) : idx+int(self.bs/2)]
            energy = self.dataset['energy'][idx-int(self.bs/2) : idx+int(self.bs/2)]
        elif idx < self.bs:
            event = self.dataset['events'][idx : idx+self.bs]
            energy = self.dataset['energy'][idx : idx+self.bs]
        else:
            event = self.dataset['events'][idx-self.bs : idx]
            energy = self.dataset['energy'][idx-self.bs : idx]
            
            
        max_len = (event[:, -1] > 0).sum(axis=1).max()
        event = event[:, :, self.max_ds_seq_len-max_len:]

        
        event[:, 2, :] = event[:, 2, :]
        event[:, 3, :] = event[:, 3, :] * 1000 # energy scale

        if not self.quantized_pos:
            pos_offset_x = np.random.uniform(0, self.offset, 1)
            pos_offset_z = np.random.uniform(0, self.offset, 1)
            event[:, 0, :] = event[:, 0, :] + pos_offset_x
            event[:, 2, :] = event[:, 2, :] + pos_offset_z
        
        event[:, 0, :] = (event[:, 0, :] - self.Xmin) * 2 / (self.Xmax - self.Xmin) - 1 # x coordinate normalization
        event[:, 1, :] = (event[:, 1, :] - self.Ymin) * 2 / (self.Ymax - self.Ymin) - 1 # y coordinate normalization
        event[:, 2, :] = (event[:, 2, :] - self.Zmin) * 2 / (self.Zmax - self.Zmin) - 1 # z coordinate normalization
        

        event = event[:, [0, 1, 2, 3]]
        
        event = np.moveaxis(event, -1, -2)
        
        event[event[:, :, -1] == 0] = 0

        # nPoints
        points = self.get_n_points(event, axis=-1).reshape(-1,1)
        

        return {'event' : event,
                'energy' : energy,
                'points' : points}

    def __len__(self):
        return len(self.dataset['events'])
    
class PointCloudDatasetGH(Dataset):
    def __init__(self, file_path, chech_overfiting=False, bs=32, max_ds_seq_len=1700):
        self.dataset = h5py.File(file_path, 'r')
        self.Ymin, self.Ymax = 0, 30
        self.Xmin, self.Xmax = 0, 30
        self.Zmin, self.Zmax = 0, 30
        self.chech_overfiting = chech_overfiting
        self.bs = bs
        self.max_ds_seq_len = max_ds_seq_len

    def __getitem__(self, idx):
        
        if idx > self.bs and idx < self.__len__() - self.bs:
            event = self.dataset['showers'][idx-int(self.bs/2) : idx+int(self.bs/2)]
            energy = self.dataset['genE'][idx-int(self.bs/2) : idx+int(self.bs/2)]
        elif idx < self.bs:
            event = self.dataset['showers'][idx : idx+self.bs]
            energy = self.dataset['genE'][idx : idx+self.bs]
        else:
            event = self.dataset['showers'][idx-self.bs : idx]
            energy = self.dataset['genE'][idx-self.bs : idx]
            
            
        max_len = (event[:, :, -1] > 0).sum(axis=1).max()
        event = event[:, :max_len, :]
        
        
        event[:, :, 0] = (event[:, :, 0] - self.Xmin) * 2 / (self.Xmax - self.Xmin) - 1 # x coordinate normalization
        event[:, :, 1] = (event[:, :, 1] - self.Ymin) * 2 / (self.Ymax - self.Ymin) - 1 # y coordinate normalization
        event[:, :, 2] = (event[:, :, 2] - self.Zmin) * 2 / (self.Zmax - self.Zmin) - 1 # z coordinate normalization
        
        
        event[event[:, :, -1] == 0] = 0
        

        return {'event' : event,
                'energy' : energy}

    def __len__(self):
        return len(self.dataset['showers'])
        
class CaloChallangeDataset(Dataset):
    def __init__(self, file_path, cfg, bs=32, max_ds_seq_len=22000, dataset_size=None, seed=42):
        dataset = h5py.File(file_path, 'r')
        
        self.dataset = {
            'events' : dataset['events'],
            'energy' : dataset['energy']
        }
        
        self.bs = bs
        self.max_ds_seq_len = max_ds_seq_len
        self.cfg = cfg

        if dataset_size is not None:
            np.random.seed(seed)
            indices = np.random.choice(len(self.dataset['events']), dataset_size, replace=False)
            indices.sort()
            self.dataset['events'] = self.dataset['events'][indices]
            self.dataset['energy'] = self.dataset['energy'][indices]
        
    def get_n_points(self, data, axis=-1):
        n_points_arr = (data[...,axis] != 0.0).sum(1)
        return n_points_arr

    def __getitem__(self, idx):
        
        if idx > self.bs and idx < self.__len__() - self.bs:
            event = self.dataset['events'][idx-int(self.bs/2) : idx+int(self.bs/2)].copy()
            energy = self.dataset['energy'][idx-int(self.bs/2) : idx+int(self.bs/2)].copy()
        elif idx < self.bs:
            event = self.dataset['events'][idx : idx+self.bs].copy()
            energy = self.dataset['energy'][idx : idx+self.bs].copy()
        else:
            event = self.dataset['events'][idx-self.bs : idx].copy()
            energy = self.dataset['energy'][idx-self.bs : idx].copy()

        max_len = (event[:, -1, :] > 0).sum(axis=1).max()
        event = event[:, :, :max_len]
        
        
        # event[:, 0, :] = (event[:, 0, :] - self.Xmin) * 2 / (self.Xmax - self.Xmin) - 1 # x coordinate normalization
        # event[:, 1, :] = (event[:, 1, :] - self.Ymin) * 2 / (self.Ymax - self.Ymin) - 1 # y coordinate normalization
        # event[:, 2, :] = (event[:, 2, :] - self.Zmin) * 2 / (self.Zmax - self.Zmin) - 1 # z coordinate normalization
        

        event = event[:, [0, 1, 2, 3]]
        
        event = np.moveaxis(event, -1, -2)

        # event[:, -1, :] /= 1000 # deposition energy to match the energy scale of the other datasets
        
        event[event[:, :, -1] == 0] = 0

        # nPoints
        points = self.get_n_points(event, axis=-1).reshape(-1,1)


        if self.cfg.norm_cond:
            energy = np.log((energy + 1e-5)/self.cfg.min_energy) / np.log(self.cfg.max_energy/self.cfg.min_energy)
            points = np.log((points + 1)/self.cfg.min_points) / np.log(self.cfg.max_points/self.cfg.min_points)
        

        return {'event' : event,
                'energy' : energy,
                'points' : points}

    def __len__(self):
        return len(self.dataset['events'])
    


if __name__ == "__main__":
    train_dset = CaloChallangeDataset(
        # file_path='/beegfs/desy/user/valentel/CaloTransfer/data/calo-challenge/dataset_3_xyz_smearing_10-90GeV.hdf5',
        file_path='/beegfs/desy/user/valentel/CaloTransfer/data/calo-challenge/preprocessing/train_prep_10-90GeV.hdf5',
        bs=32,
    )