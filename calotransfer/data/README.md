# Dataset Documentation

This directory contains the data used for training and evaluation. **Data files are not included in this repository** due to size constraints.

## Directory Structure

```
calotransfer/data/
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
cd calotransfer/data/source/
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
cd calotransfer/data/target/
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
| **Layers** | 30 | 45 |
| **Max points** | ~6k | ~20k |

## Data Preprocessing

The preprocessing on CaloChallenge dataset is done via the notebook [preprocessing.ipynb](../scripts/preprocessing/preprocessing.ipynb).


This will:
1. Normalize point cloud coordinates
2. Standardize energy scales
3. Create train/validation splits
4. Cylindrical smearing to facilitate the transfer

**Output files:**
```
calotransfer/data/processed/
├── source_train.hdf5
├── source_val.hdf5
├── target_train_full.hdf5
└── target_val.hdf5
```
