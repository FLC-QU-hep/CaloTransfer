import argparse
import h5py
import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt

# Fix the import path issues
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# Now import with absolute paths relative to current directory
from models.vae_flow import *
from models.shower_flow import compile_HybridTanH_model, compile_HybridTanH_model_CaloC
from configs import Configs
import utils.gen_utils_CaloChallenge as gen_utils
import models
import models.epicVAE_nflows_kDiffusion as mdls
import models.allCond_epicVAE_nflow_PointDiff as mdls2
import utils.paths_trainings_cleaned as training_paths
import utils.finetune as ft

plt.rcParams['text.usetex'] = False

import k_diffusion as K


cfg = Configs()

#  define the energy range for the datase
e_min = 1.0  #  for 10-90 GeV range
e_max = 1000.0  
num_shower = 10_000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_size', type=str, required=True)
    parser.add_argument('--use_pretrained', type=str, required=True)
    parser.add_argument('--training_step', type=int, required=True)
    parser.add_argument('--use_ema', type=lambda x: x.lower() == 'true', required=True)
    parser.add_argument('--lora_rank', type=int, default=16, help='Rank for LoRA layers')
    parser.add_argument('--lora_alpha', type=int, default=16, help='Alpha for LoRA layers')
    args = parser.parse_args()

    cfg.dataset_size = args.dataset_size
    cfg.use_pretrained = False if args.use_pretrained.startswith('vanilla') else True
    training_step = args.training_step
    use_ema = args.use_ema
    cfg.lora_rank = args.lora_rank
    use_lora = args.use_pretrained.startswith('lora')
    cfg.use_lora = use_lora
    cfg.lora_alpha = args.lora_alpha

    print('\n\n','+'*50, '\n\n')
    print(f"dataset_size: {cfg.dataset_size}")
    print(f"e_min: {e_min}, e_max: {e_max}")

    print(f"use_pretrained: {args.use_pretrained}")
    print(f"training_step: {training_step}")
    print(f"lora_rank: {cfg.lora_rank}")
    print(f"use_ema: {use_ema}")
        
    print('\n\n','+'*50, '\n\n')

    training_strategy = args.use_pretrained 

    flow, distribution = compile_HybridTanH_model_CaloC(num_blocks=2, 
                                            num_inputs=92, ### adding 30 e layers 
                                            num_cond_inputs=1, 
                                            device=cfg.device)  # num_cond_inputs
    BASE_PATH = '/data/dust/user/valentel/beegfs.migration/dust/logs'

    # Determine the shower flow training strategy based on the dataset_size
    # Check if the dataset_size contains "1-1000" to determine if it's in the 1-1000 GeV range
    if "1-1000" in cfg.dataset_size:
        # For datasets in the 1-1000 range
        if cfg.use_pretrained:
            sf_training_strategy = 'finetune_1-1000'
        else:
            sf_training_strategy = 'vanilla_1-1000'
    else:
        print('dataset size: ', cfg.dataset_size)
        # For datasets in the 10-90 range
        if cfg.use_pretrained:
            sf_training_strategy = 'finetune_10-90'
        else:
            sf_training_strategy = 'vanilla_10-90'

    print('sf_training_strategy: ', sf_training_strategy)
    
    checkpoint_path_flow = training_paths.sf_ckpt_paths[sf_training_strategy][cfg.dataset_size]
    print('sf_training_strategy entries:', training_paths.SF_ENTRIES[sf_training_strategy])
    print('dataset in entries?', cfg.dataset_size in [entry[0] for entry in training_paths.SF_ENTRIES[sf_training_strategy]])
    print('dataset mapping:', training_paths.DATASET_DIR_MAP.get(cfg.dataset_size, "NOT FOUND"))
    
    
    def debug_path_construction(dataset_size, sf_training_strategy):
        print("\n--- DEBUG PATH CONSTRUCTION ---")
        print(f"Looking for dataset '{dataset_size}' in strategy '{sf_training_strategy}'")
        
        # Check if dataset is in SF_ENTRIES
        dataset_entries = [entry[0] for entry in training_paths.SF_ENTRIES.get(sf_training_strategy, [])]
        print(f"Available datasets in this strategy: {dataset_entries}")
        print(f"Dataset found in entries? {dataset_size in dataset_entries}")
        
        # Check dataset mapping
        mapped_dir = training_paths.DATASET_DIR_MAP.get(dataset_size, "NOT FOUND")
        print(f"Dataset directory mapping: '{dataset_size}' -> '{mapped_dir}'")
        
        # If there's a match, show the full path construction
        for entry in training_paths.SF_ENTRIES.get(sf_training_strategy, []):
            ds, date, step = entry
            if ds == dataset_size:
                dir_part = training_paths.DATASET_DIR_MAP.get(ds, ds)
                tt_in_path = 'vanilla' if sf_training_strategy == 'vanilla_1-1000' else 'finetune' if sf_training_strategy == 'finetune_1-1000' else sf_training_strategy
                path = f"{BASE_PATH}/Shower_flow_weights/{dir_part}/{tt_in_path}/ShowerFlow_{date}/ShowerFlow_{step}.pth"
                print(f"Full constructed path: {path}")
                print(f"Path exists? {os.path.exists(path)}")
        print("--- END DEBUG ---\n")

    # Add this right after determining sf_training_strategy in generation.py
    debug_path_construction(cfg.dataset_size, sf_training_strategy)
    
    
    if not os.path.exists(checkpoint_path_flow):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path_flow}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path_flow, weights_only=True)

    flow.load_state_dict(checkpoint['model'])
    flow.eval().to(cfg.device)

    print('checkpoint path for the flow model: ', checkpoint_path_flow)
    print('\n\n','+'*20,'flow model loaded', '+'*20)
    print('+'*50, '\n\n')
    
    # Get checkpoint path for pointwise diffusion model
    if "1-1000" in cfg.dataset_size:
        # Format the path directly for 1-1000 GeV datasets
        # Determine the appropriate directory (vanilla or finetune)
        if training_strategy.startswith('vanilla'):
            diff_dir = 'vanilla'
        elif training_strategy.startswith('finetune'):
            diff_dir = 'finetune'
        elif training_strategy.startswith('lora'):
            diff_dir = 'finetune/lora'
        else:
            diff_dir = training_strategy
            
        # Get the date from PW_ENTRIES
        date = training_paths.PW_ENTRIES[training_strategy][cfg.dataset_size]
        ckpt_path = f"{training_paths.BASE_PATH_PW}/MyCaloTransfer_diffusionweights/{diff_dir}/CaloChallange_CD{date}/ckpt_0.000000_{training_step}.pt"
    else:
        # Use the path generation from training_paths for 10-90 GeV ranges
        ckpt_path = training_paths.pointwise_ckpt_paths[training_strategy][cfg.dataset_size].format(training_step=training_step)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f'checkpoint path for the diffusion model {ckpt_path} \n\n')

    cfg.logdir = ckpt_path
    out_dir = f'/data/dust/user/valentel/beegfs.migration/dust/evaluate/outputsout/{training_strategy}/{cfg.dataset_size}/' + ckpt_path.split('/')[-2] + '_' + ckpt_path.split('/')[-1]
    # Add 'pretrained' to the output directory if cfg.use_pretrained is True
    if cfg.use_pretrained:
        out_dir += '_pretrained'
    if use_ema:
        out_dir += ''
    else:
        out_dir += '_no_ema'
    if use_lora:
        out_dir += '_lora'
    print(f'output directory: {out_dir}')
    # Create the output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
    caloclouds = 'edm'   # 'ddpm, 'edm', 'cm'
    if caloclouds == 'ddpm':
        
        cfg.sched_mode = 'quardatic'
        # cfg.sched_mode = 'linear'
        # cfg.sched_mode = 'sigmoid'
        model = mdls2.AllCond_epicVAE_nFlow_PointDiff(cfg).to(cfg.device)
        
        checkpoint = torch.load(cfg.logdir, map_location=torch.device('cpu'))
        # checkpoint = torch.load('/beegfs/desy/user/akorol/logs/point-cloud/CaloChallange2023_06_15__17_08_22/ckpt_0.000000_1537000.pt', map_location=torch.device('cpu'))

        model.load_state_dict(checkpoint['state_dict'])
        
    elif caloclouds == 'edm':
        cfg.kdiffusion = True   # EDM vs DDPM diffusion
        cfg.size = 1
        cfg.num_steps = 32 # 32
        cfg.sampler = 'heun'   # default 'heun'
        cfg.s_churn =  0.0     # , default 0.0  (if s_churn more than num_steps, it will be clamped to max value)
        cfg.s_noise = 1.0    # default 1.0   # noise added when s_churn > 0
        cfg.sigma_max = 80.0 #  5.3152e+00  # default 80.0
        cfg.sigma_min = 0.002   # default 0.002
        cfg.rho = 7. # default 7.0
        # # # baseline with lat_dim = 0, max_iter 10M, lr=1e-4 fixed, dropout_rate=0.0, ema_power=2/3 (long training)            USING THIS TRAINING
        cfg.dropout_rate = 0.0
        cfg.latent_dim = 0
        sys.modules['configs'] = __import__('CaloTransfer.configs', fromlist=[''])
        checkpoint = torch.load(cfg.logdir, map_location=torch.device(cfg.device), weights_only=False)    # max 5200000
        #initialize the model
        model = mdls.epicVAE_nFlow_kDiffusion(cfg).to(cfg.device)
        
        if use_lora:
            model = ft.load_lora_model_for_generation(
                                            pretrained_path=cfg.diffusion_pretrained_model_path,
                                            lora_checkpoint_path=ckpt_path, 
                                            cfg     =cfg,) 
                        
            # Generate multiple showers with different energies for debugging
            energies = [90, 500, 1000]
            point_count = [5000, 10_000, 20_000]  # Number of points for each energy
            for i in range(len(energies)):
                print(f"\n\n{'='*50}")
                print(f"Generating shower with energy {energies[i]} GeV")
                print(f"{'='*50}")
                
                # Generate shower
                generated = ft.generate_shower(model, cfg, energy=energies[i], num_points=point_count[i])
                
                # Plot 3D
                ft.plot_shower_3d(
                    generated, 
                    title=f'Generated Shower (Energy: {energies[i]} GeV)',
                    save_path=f'{out_dir}/shower_3d_energies_{energies[i]}.png'
                )
                
                # Plot projections
                ft.plot_shower_projections(
                    generated,
                    title=f'Generated Shower Projections (Energy: {energies[i]} GeV)',
                    save_path=f'{out_dir}/shower_projections_energies_{energies[i]}.png'
                )
                
                # Print some statistics
                if generated.shape[1] != 4:
                    generated = np.moveaxis(generated, 1, -1)
                
                total_energy = np.sum(generated[:, 3])
                print(f"Total energy in shower: {total_energy:.2f}")
                print(f"Number of hits: {generated.shape[0]}")
                print(f"Average hit energy: {np.mean(generated[:, 3]):.4f}")
        
            print("=== LoRA model ready for generation ===")
        else:
            # Original non-LoRA loading logic
            if use_ema:
                model.load_state_dict(checkpoint['others']['model_ema'])
            else:
                model.load_state_dict(checkpoint['state_dict'])

    elif caloclouds == 'cm':
        cfg.kdiffusion = True   # EDM vs DDPM diffusion
        cfg.num_steps = 1
        cfg.sigma_max = 80.0 #  5.3152e+00  # default 80.0
        # long baseline with lat_dim = 0, max_iter 1M, lr=1e-4 fixed, num_steps=18, bs=256, simga_max=80, epoch=2M, EMA
        cfg.dropout_rate = 0.0
        cfg.latent_dim = 256
        checkpoint = torch.load(cfg.logdir, map_location=torch.device(cfg.device))    # max 5200000
        model = mdls.epicVAE_nFlow_kDiffusion(cfg, distillation = True).to(cfg.device)
        model.load_state_dict(checkpoint['others']['model_ema'])
        
    model.eval()

    # samples= np.load('/gpfs/dust/maxwell/user/valentel/MyCaloTransfer/CaloTransfer/utils/per_layer/0_500GeV_samples.npy')
    
    print('model loaded')
    print('generating...\n', ' + - '*20, '\n')
    
    print(f"Using energy range: {e_min}-{e_max} GeV")
    
    output = gen_utils.gen_showers_batch_w_occupancy_cog(model, shower_flow=distribution, 
                                                        e_min=e_min, e_max=e_max,
                                                        config=cfg, num=num_shower, bs=32, 
                                                        scale_occupancy=True, cog_cal=False)
    fake_showers, cond_E = output['showers'].astype(np.float32), output['energy'].astype(np.float32)
    print("=== generated showers ===")
    print(f"Fake showers shape: {fake_showers.shape}")
    print(f"Fake showers mean: {np.mean(fake_showers)}")
    print(f"Conditional energies: {cond_E}")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dataset_file = h5py.File(f"{out_dir}/showers.hdf5", 'w')
    dataset_file.create_dataset('incident_energies',
                    data=cond_E,
                    compression='gzip')
    dataset_file.create_dataset('showers',
                    data=fake_showers.reshape(len(fake_showers), -1),
                    compression='gzip')
    dataset_file.close()
    print('=== file saved in ', out_dir, ' ===')
    print(' + - '*20, '\n')
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################


    sys.path.append('/data/dust/user/valentel/maxwell.merged/MyCaloTransfer/CaloTransfer/homepage/code')
    import argparse
    import evaluate_2 as evaluate
    import HighLevelFeatures as HLF

    # change directory to the one with the evaluation script
    os.chdir('/data/dust/user/valentel/maxwell.merged/MyCaloTransfer/CaloTransfer/homepage/code')

    INPUT_FILE = f"{out_dir}/showers.hdf5" # REPLACE THIS WITH YOUR GENERATED EVENTS
    
    # Use the appropriate reference file based on dataset range
    if "1-1000" in cfg.dataset_size:
        REFERENCE_FILE = '/data/dust/user/valentel/maxwell.merged/MyCaloTransfer/CaloTransfer/data/calo-challenge/preprocessing/reduced_datasets/1-1000GeV/evaluation/10k_val_dset4_prep_1-1000GeV.hdf5'
    else:
        REFERENCE_FILE = '/data/dust/user/valentel/maxwell.merged/MyCaloTransfer/CaloTransfer/data/calo-challenge/preprocessing/reduced_datasets/10-90GeV/evaluation/10k_val_dset4_prep_10-90GeV.hdf5'
    
    MODE = 'all' # not really needed here because the nb is interactive
    DATASET = '3'
    OUTPUT_DIR = f"{out_dir}/evaluation/"
    SOURCE_DIR = f"{out_dir}/source/"

    parser_replacement = {
        'input_file': INPUT_FILE, 'reference_file': REFERENCE_FILE, 'mode': MODE, 'dataset': DATASET, 
        'output_dir': OUTPUT_DIR, 'source_dir': SOURCE_DIR, }
    args = argparse.Namespace(**parser_replacement)
    args.min_energy = {'1-photons': 10, '1-pions': 10,
                        '2': 0.5e-3/0.033, '3': 0.5e-3/0.033}[args.dataset]
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################

    # reading in source file
    source_file = h5py.File(args.input_file, 'r')

    # checking if it has correct shape
    evaluate.check_file(source_file, args)

    # preparing output directory
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)  

    # preparing source directory
    if not os.path.isdir(args.source_dir):
        os.makedirs(args.source_dir)

    # extracting showers and energies from source file
    shower, energy = evaluate.extract_shower_and_energy(source_file, which='input')

    # creating helper class for high-level features
    particle = {'1-photons': 'photon', '1-pions': 'pion',
                '2': 'electron', '3': 'electron'}[args.dataset]
    hlf = HLF.HighLevelFeatures(particle, filename='binning_dataset_{}.xml'.format(args.dataset.replace('-', '_')))


    # reading in reference
    if os.path.splitext(args.reference_file)[1] == '.hdf5':
        print("using .hdf5 reference")
        reference_file = h5py.File(args.reference_file, 'r')
        evaluate.check_file(reference_file, args, which='reference')
        reference_hlf = HLF.HighLevelFeatures(particle, filename='binning_dataset_{}.xml'.format(
            args.dataset.replace('-', '_')))
        reference_shower, reference_energy = evaluate.extract_shower_and_energy(reference_file, which='reference')
        reference_hlf.Einc = reference_energy
        evaluate.save_reference(reference_hlf, args.reference_file + '.pkl')

    elif os.path.splitext(args.reference_file)[1] == '.pkl':
        print("using .pkl file for reference")
        reference_hlf = evaluate.load_reference(args.reference_file)
    else:
        raise ValueError("reference_file must be .hdf5 or .pkl!")
    
    # evaluation mode 'avg': average of given showers
    print("Plotting average shower...")
    _ = hlf.DrawAverageShower(shower, filename=os.path.join(args.output_dir, 
                                                            'average_shower_dataset_{}.png'.format(args.dataset)),
                                    title="Shower average")
    if hasattr(reference_hlf, 'avg_shower'):
        pass
    else:
        reference_hlf.avg_shower = reference_shower.mean(axis=0, keepdims=True)
        evaluate.save_reference(reference_hlf, args.reference_file + '.pkl')
    _ = hlf.DrawAverageShower(reference_hlf.avg_shower, 
                            filename=os.path.join(args.output_dir, 'reference_average_shower_dataset_{}.png'.format(
                                            args.dataset)),
                            title="Shower average reference dataset")
    print("Plotting average shower: DONE.\n")


    # evaluation mode 'avg-E': average showers at different energy ranges
    print("Plotting average showers for different energies ...")
    if '1' in args.dataset:
        target_energies = 2**np.linspace(8, 23, 16)
        plot_title = ['shower average at E = {} MeV'.format(int(en)) for en in target_energies]
    else:
        target_energies = 10**np.linspace(3, 6, 4)
        plot_title = []
        for i in range(3, 7):
            plot_title.append('shower average for E in [{}, {}] MeV'.format(10**i, 10**(i+1)))
    for i in range(len(target_energies)-1):
        filename = 'average_shower_dataset_{}_E_{}.png'.format(args.dataset,
                                                                    target_energies[i])
        which_showers = ((energy >= target_energies[i]) & (energy < target_energies[i+1])).squeeze()
        _ = hlf.DrawAverageShower(shower[which_showers],
                                filename=os.path.join(args.output_dir, filename),
                                title=plot_title[i])
        if hasattr(reference_hlf, 'avg_shower_E'):
            pass
        else:
            reference_hlf.avg_shower_E = {}
        if target_energies[i] in reference_hlf.avg_shower_E:
            pass
        else:
            which_showers = ((reference_hlf.Einc >= target_energies[i]) & (reference_hlf.Einc < target_energies[i+1])).squeeze()
            reference_hlf.avg_shower_E[target_energies[i]] = reference_shower[which_showers].mean(axis=0, keepdims=True)
            evaluate.save_reference(reference_hlf, args.reference_file + '.pkl')

            _ = hlf.DrawAverageShower(reference_hlf.avg_shower_E[target_energies[i]],
                                    filename=os.path.join(args.output_dir, 'reference_'+filename),
                                    title='reference '+plot_title[i])

    print("Plotting average shower for different energies: DONE.\n")

    args.x_scale = 'log'

    # evaluation mode 'hist': plotting histograms of high-level features and printing/saving the sepration power
    # (equivalent to running hist-p for plotting and hist-chi for the separation power)
    print("Calculating high-level features for histograms ...")
    hlf.CalculateFeatures(shower)
    hlf.Einc = energy

    print("Calculating high-level features for histograms: DONE.\n")
    if reference_hlf.E_tot is None:
        reference_hlf.CalculateFeatures(reference_shower)
        evaluate.save_reference(reference_hlf, args.reference_file + '.pkl')
    print("Calculating high-level features for histograms: DONE.\n")

    with open(os.path.join(args.output_dir, 'histogram_chi2_{}.txt'.format(args.dataset)), 'w') as f:
        f.write('List of chi2 of the plotted histograms, see eq. 15 of 2009.03796 for its definition.\n')

    with open(os.path.join(args.output_dir, 'histogram_kl_{}.txt'.format(args.dataset)), 'w') as f:
        f.write('List of kld of the plotted histograms.\n')
        
    print("Plotting histograms ...")
    evaluate.plot_histograms(hlf, reference_hlf, args)
    print("Plotting histograms: DONE. \n")

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
#baseline
# with open(f'{out_dir}/histogram_chi2_3.txt', 'r') as f: 
#     scorese_baseline = f.readlines()
#     scorese_baseline = [float(x) for x in scorese_baseline if x[0] == '0']

# #current
# with open(f'{out_dir}/histogram_chi2_3.txt', 'r') as f: 
#     scorese_current = f.readlines()
#     scorese_current = [float(x) for x in scorese_current if x[0] == '0']


# #plotting
# plt.figure(figsize=(10, 8))
# plt.plot(scorese_baseline[:46], label='baseline')
# plt.plot(scorese_current[:46], label='current')
# plt.xlabel('feature')
# plt.ylabel('chi2')
# plt.title('chi2 of the energies')
# plt.legend()
# plt.savefig(f'{out_dir}/chi2_energy.png')

# plt.figure(figsize=(10, 8))
# plt.plot(scorese_baseline[46:], label='baseline')
# plt.plot(scorese_current[46:], label='current')
# plt.xlabel('feature')
# plt.ylabel('chi2')
# plt.title('chi2 of the others')
# plt.legend()
# plt.savefig(f'{out_dir}/chi2_others.png')