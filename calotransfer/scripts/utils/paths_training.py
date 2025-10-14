# Description: This file contains the paths to the training data and the trained models.
from pathlib import Path
val_ds_type = '1-1000GeV'  # Change this to '1-1000GeV' if needed

# Base paths
BASE_PATH_SF = '/data/dust/user/valentel/maxwell.merged/MyCaloTransfer/CaloTransfer/logs/dust_link/logs'
BASE_PATH_PW = BASE_PATH_SF  # Original BASE_PATH for pointwise
BASE_PATH_SHOWERS = '/data/dust/user/valentel/beegfs.migration/dust/evaluate/outputsout'
GEANT4_BASE_PATH = '/data/dust/user/valentel/maxwell.merged/MyCaloTransfer/CaloTransfer/data/calo-challenge/preprocessing/reduced_datasets/'

GEANT4_PATH = GEANT4_BASE_PATH + '1-1000GeV/evaluation/10k_val_dset4_prep_1-1000GeV_cylindrical.hdf5'


# Dataset mappings
DATASET_DIR_MAP = {

    # === 1-1000 GeV ===#
    '100k_1-1000': '100k_1-1000GeV',
    '50k_1-1000': '50k_1-1000GeV',
    '10k_1-1000': '10k_1-1000GeV',
    '5k_1-1000': '5k_1-1000GeV',
    '1k_1-1000': '1k_1-1000GeV',
    '500_1-1000': '500_1-1000GeV',
    '100_1-1000': '100_1-1000GeV',
}

DISPLAY_NAME_MAP = {

    # === 1-1000 GeV ===#

    # Fixed keys to match PW_ENTRIES
    '100k_1-1000': 'D = 1 x 10^5',
    '50k_1-1000': 'D = 5 x 10^4',
    '10k_1-1000': 'D = 1 x 10^4',
    '5k_1-1000': 'D = 5 x 10^3',
    '1k_1-1000': 'D = 1 x 10^3',
    '500_1-1000': 'D = 5 x 10^2',
    '100_1-1000': 'D = 1 x 10^2',
    
    # Added alternative keys with underscores to handle both formats
    '100k_1_1000': 'D = 1 x 10^5',
    '50k_1_1000': 'D = 5 x 10^4',
    '10k_1_1000': 'D = 1 x 10^4',
    '5k_1_1000': 'D = 5 x 10^3',
    '1k_1_1000': 'D = 1 x 10^3',
    '500_1_1000': 'D = 5 x 10^2',
    '100_1_1000': 'D = 1 x 10^2',
}

# Shower Flow Configurations
SF_ENTRIES = {
    
    #=== 1-1000 GeV ===#
    
    'finetune_1-1000': [
        ('100k_1-1000', '2025_05_06__18_00_43', '1000'),
        ('50k_1-1000', '2025_05_06__18_08_37', '525'),
        ('10k_1-1000', '2025_05_06__18_09_28', '600'),
        ('5k_1-1000',  '2025_05_06__18_09_28', '1000'),

        ('1k_1-1000',  '2025_05_06__18_09_03', '150'),
        ('500_1-1000', '2025_05_07__14_17_56', '375'),  
        ('100_1-1000', '2025_05_07__14_18_28', '900'),
    ],

    'vanilla_1-1000': [
        ('100k_1-1000', '2025_05_06__18_00_45', '600'),
        ('50k_1-1000', '2025_05_06__18_08_15', '525'),
        ('10k_1-1000', '2025_05_06__18_08_22', '875'),
        ('5k_1-1000',  '2025_05_06__18_08_35', '1000'),

        ('1k_1-1000',  '2025_05_06__18_08_43', '1000'),
        ('500_1-1000', '2025_05_07__14_18_23', '975'),
        ('100_1-1000', '2025_05_07__14_19_04', '1000'),
    ],
}

# Pointwise Configurations
PW_ENTRIES = {
    'finetune_full_v1_1_1000': { # seed=42
        '100k_1-1000':'2025_05_28__15_44_22',
        '50k_1-1000': '2025_05_28__15_44_28',
        '10k_1-1000': '2025_05_28__15_45_31',   
        '5k_1-1000':  '2025_05_28__15_45_20',
        '1k_1-1000':  '2025_05_28__15_45_57',
        '500_1-1000': '2025_05_28__15_47_08',
        '100_1-1000': '2025_05_28__16_00_25',
    },
}

# Function to build Shower Flow paths
def build_sf_paths(base_path, entries, dataset_map):
    paths = {}
    for tt, entries_list in entries.items():
        paths[tt] = {}
        for ds, date, step in entries_list:
            # First, get the correct directory part from the dataset map
            dir_part = dataset_map.get(ds, ds)  # Fall back to ds if not in map
            
            # Determine the correct training type folder name
            if tt == 'vanilla_1-1000':
                tt_in_path = 'vanilla'
            elif tt == 'finetune_1-1000':
                tt_in_path = 'finetune'
            elif tt == 'vanilla_10-90':
                tt_in_path = 'vanilla'
            elif tt == 'finetune_10-90':
                tt_in_path = 'finetune'
            else:
                tt_in_path = tt
                
            # Construct the full path
            path = f"{base_path}/Shower_flow_weights/{dir_part}/{tt_in_path}/ShowerFlow_{date}/ShowerFlow_{step}.pth"
            paths[tt][ds] = path
    return paths

# Function to build Pointwise paths
def build_pw_paths(base_path, entries):
    paths = {}
    for tt, ds_map in entries.items():
        # If tt starts with 'finetune', set it to 'finetune'
        if tt.startswith('finetune'):
            tt_modified = 'finetune'
        elif tt.startswith('vanilla'):
            tt_modified = 'vanilla'
        elif tt.startswith('lora'):
            tt_modified = 'finetune/lora'
        else:
            tt_modified = tt
        
        paths[tt] = {}
        for ds, date in ds_map.items():
            # Use the modified tt in the path
            path = f"{base_path}/MyCaloTransfer_diffusionweights/{tt_modified}/CaloChallange_CD{date}/ckpt_0.000000_{{training_step}}.pt"
            paths[tt][ds] = path
    return paths


def build_showers_paths(base_path, pw_entries, geant4_path, display_name_map, check_steps=None):
    """ Build paths for generated showers based on pointwise entries and GEANT4 path. """
    # Initialize paths dictionary
    paths = {}
    for tt in pw_entries:
        paths[tt] = {'GEANT4': geant4_path}
        for ds in pw_entries[tt]:
            if ds == 'GEANT4': continue
            date = pw_entries[tt].get(ds, '')
            if not date: continue
            
            # Fix: Normalize the dataset key to use hyphens consistently
            normalized_ds = ds.replace('_', '-')
            
            # Try to get display name, first with original key, then with normalized key
            if ds in display_name_map:
                display_name = display_name_map[ds]
            elif normalized_ds in display_name_map:
                display_name = display_name_map[normalized_ds]
            else:
                print(f"Warning: No display name found for dataset '{ds}' or '{normalized_ds}'")
                continue  # Skip this dataset if no display name is found

            if tt.startswith('finetune'):
                shower_path = f"{base_path}/{tt}/{ds}/CaloChallange_CD{date}_ckpt_0.000000_{{training_step}}.pt_pretrained"
            elif tt.startswith('vanilla'):
                shower_path = f"{base_path}/{tt}/{ds}/CaloChallange_CD{date}_ckpt_0.000000_{{training_step}}.pt"
            elif tt.startswith('lora'):
                shower_path = f"{base_path}/{tt}/{ds}/CaloChallange_CD{date}_ckpt_0.000000_{{training_step}}.pt_pretrained_lora"

            # Dynamic check if requested
            if check_steps:
                path_exists = False
                for step in check_steps:
                    test_path = shower_path.format(training_step=step)
                    if Path(test_path).exists():
                        path_exists = True
                        break
                
                if path_exists:
                    paths[tt][display_name] = shower_path
                else:
                    print(f"Warning: No checkpoint found for {tt}/{ds} at steps {check_steps}")
                    paths[tt][display_name] = None
            else:
                # No check, just assign
                paths[tt][display_name] = shower_path

    return paths


# Build paths
sf_ckpt_paths = build_sf_paths(BASE_PATH_SF, SF_ENTRIES, DATASET_DIR_MAP)
pointwise_ckpt_paths = build_pw_paths(BASE_PATH_PW, PW_ENTRIES)
showers_ckpt_paths = build_showers_paths(BASE_PATH_SHOWERS, PW_ENTRIES, GEANT4_PATH, DISPLAY_NAME_MAP)