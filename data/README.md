# Dataset Documentation

This directory contains the data used for training and evaluation. **Data files are not included in this repository** due to size constraints.

## Directory Structure

```
data/
├── README.md           # This file
├── source/             # Source domain (ILD detector)
│   ├── .gitkeep
│   └── photons_ILD.hdf5  (download required)
├── target/             # Target domain (CaloChallenge)
│   ├── .gitkeep
│   └── electrons_calo_challenge.hdf5  (download required)
├── processed/          # Preprocessed data (generated)
│   └── .gitkeep
└── raw/                # Optional: raw simulation files
    └── .gitkeep
```

## Source Domain: ILD Detector

**Dataset:** Photon electromagnetic showers in the ILD detector

**Download:**
```bash
cd data/source/
wget https://zenodo.org/records/10044175
```

**DOI:** [10.5281/zenodo.10044175](https://zenodo.org/records/10044175)

**Characteristics:**
- **Particle type:** Photons (γ)
- **Geometry:** Regular electromagnetic calorimeter
- **Energy range:** 10-90 GeV (uniform distribution)
- **Number of showers:** ~524k
- **Points per shower:** ~6,000 (average)
- **Calorimeter layers:** 30


## Target Domain: CaloChallenge Dataset

**Dataset:** Electron electromagnetic showers from the Fast Calorimeter Simulation Challenge

**Download:**
Follow instructions at the official CaloChallenge website:
[https://calochallenge.github.io/homepage/](https://calochallenge.github.io/homepage/)

**Alternatively:**
```bash
cd data/target/
# Download Dataset 3
wget https://zenodo.org/records/6366324
```

**Characteristics:**
- **Particle type:** Electrons (e⁻)
- **Geometry:** Cylindrical electromagnetic calorimeter
- **Energy range:** 1-1000 GeV (log-normal distribution)
- **Number of showers:** 100k
- **Points per shower:** ~20,000 (max)
- **Calorimeter layers:** 45



## Geometric Mismatch Summary

Transfer learning must handle these domain shifts:

| Property | Source (ILD) | Target (CaloChallenge) |
|----------|--------------|------------------------|
| **Particle** | Photon | Electron |
| **Geometry** | Regular | Cylindrical |
| **Energy range** | 10-90 GeV | 1-1000 GeV |
| **Energy distribution** | Uniform | Log-normal |
| **Avg points/shower** | ~6,000 | ~20,000 |
| **Layers** | 30 | 45 |
| **Max points** | ~8,000 | ~25,000 |

## Data Preprocessing

After downloading both datasets, run preprocessing:

```bash
python scripts/preprocess_data.py \
    --source data/source/photons_ILD.hdf5 \
    --target data/target/electrons_calo_challenge.hdf5 \
    --output data/processed/
```

This will:
1. Normalize point cloud coordinates
2. Standardize energy scales
3. Create train/validation/test splits
4. Generate few-shot subsets (100, 500, 1000 samples)

**Output files:**
```
data/processed/
├── source_train.hdf5
├── source_val.hdf5
├── target_train_full.hdf5
├── target_train_100.hdf5   # Few-shot: 100 samples
├── target_train_500.hdf5   # Few-shot: 500 samples
├── target_val.hdf5
└── target_test.hdf5
```

## Data Format Specifications

Both datasets use the HDF5 format with the following structure:

### Point Cloud Representation
Each shower is represented as a point cloud with shape `(N_points, 4)`:
- **Column 0:** x-coordinate (mm)
- **Column 1:** y-coordinate (mm)
- **Column 2:** z-coordinate (mm) / layer index
- **Column 3:** deposited energy (GeV)

### Variable-Length Showers
Showers have variable numbers of hits. We use:
- **Zero-padding:** Pad to maximum length with zeros
- **Mask:** Include mask array to identify valid points

```python
# Example loading code
import h5py
import numpy as np

with h5py.File('data/source/photons_ILD.hdf5', 'r') as f:
    showers = f['showers'][:]          # (N, max_points, 4)
    energies = f['incident_energies'][:]  # (N,)
    masks = f['masks'][:]              # (N, max_points) - 1 for valid, 0 for padding
    
# Get first shower with valid points only
shower_0 = showers[0][masks[0].astype(bool)]
print(f"Shower 0: {len(shower_0)} hits, incident energy: {energies[0]:.2f} GeV")
```

## Citations

If you use these datasets, please cite:

**ILD Source Domain:**
```bibtex
@dataset{ild_photons_2024,
  author = {...},
  title = {ILD Photon Electromagnetic Shower Dataset},
  year = {2024},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.10044175}
}
```

**CaloChallenge Target Domain:**
```bibtex
@article{calochallenge2023,
  title={The Fast Calorimeter Simulation Challenge},
  author={...},
  journal={...},
  year={2023}
}
```

## Data Availability Statement

All datasets used in this work are publicly available and can be downloaded from the sources listed above. No proprietary or restricted data is used.

## Contact

For questions about the data or preprocessing, contact:
- Lorenzo Valente: lorenzo.valente@uni-hamburg.de