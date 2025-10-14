class Configs():
    
    def __init__(self):
            
    # Experiment Name
        self.name = 'CaloChallange_CD'  # options: [TEST_, kCaloClouds_, CaloClouds_, CD_]
        self.comet_project = 'mycalotransfer'   # options: ['k-CaloClouds', 'calo-consistency']
        self.Acomment = 'long baseline with lat_dim = 0, max_iter 1M, lr=1e-4 fixed, num_steps=18, bs=256, simga_max=80, epoch=2M, EMA'  # log_iter 100
        self.log_comet = True
            

    # Model arguments
        self.model_name = 'epicVAE_nFlow_kDiffusion'             # choices=['flow', 'AllCond_epicVAE_nFlow_PointDiff', 'epicVAE_nFlow_kDiffusion]
        self.latent_dim = 0     # caloclouds default: 256
        self.beta_1 = 1e-4
        self.beta_T = 0.02
        self.sched_mode = 'quardatic'  # options: ['linear', 'quardatic', 'sigmoid]
        self.flexibility = 0.0
        self.truncate_std = 2.0
        self.latent_flow_depth = 14
        self.latent_flow_hidden_dim = 256
        self.num_samples = 4
        self.features = 4
        self.sample_num_points = 2048
        self.kl_weight = 1e-3   # default: 0.001 = 1e-3
        self.residual = False            # choices=[True, False]   # !! for CaloClouds was True, but for EDM False might be better (?)
        
        self.cond_features = 2       # number of conditioning features (i.e. energy+points=2)
        self.norm_cond = True    # normalize conditioniong to [-1,1]
        self.kld_min = 1.0       # default: 0.0

        # EPiC arguments
        self.use_epic = False
        self.epic_layers = 5
        self.hidden_dim = 128        # default: 128
        self.sum_scale = 1e-3
        self.weight_norm = True

        # for n_flows model
        self.flow_model = 'PiecewiseRationalQuadraticCouplingTransform'
        self.flow_transforms = 10
        self.flow_layers = 2
        self.flow_hidden_dims = 128
        self.tails = 'linear'
        self.tail_bound = 10
        
        self.kdiffusion = True 

    # Data
        self.dataset = 'calo-challange' # choices=['x36_grid', 'clustered', 'getting_high', 'calo-challange']
        self.dataset_key = '100k_1-1000' #options=['all', '10k', '1k'] +'pretraining_cc2' + 'all_10-90'
        self.val_ds_type = '1-1000GeV' # 1-1000GeV, 10-90GeV, pretraining_cc2
        self.use_pretrained = True
        self.dataset_size = 1e4
        from utils.paths_configs import dataset_paths
        self.dataset_path = dataset_paths.get(self.dataset_key)
        # Throw an error if the dataset_key is not found
        if self.dataset_path is None:
            raise ValueError(f"Unknown dataset_key: {self.dataset_key}")

        self.quantized_pos = False

    # Dataloader
        self.workers = 4
        self.train_bs = 64 #      # k-diffusion: 128 / CD: 256
        self.pin_memory = False         # choices=[True, False]
        self.shuffle = True             # choices=[True, False]
        
        if self.val_ds_type == '10-90GeV':
            self.max_points = 5149
            self.min_points = 4
            self.max_energy = 89.9990994908
            self.min_energy = 10.0010047095
        elif self.val_ds_type == '1-1000GeV': #to be adapted for the dataset
            self.max_points = 19111
            self.min_points = 17
            self.max_energy = 999.7959524087546
            self.min_energy =  1.0006331434870916

    # Optimizer and scheduler
        self.optimizer = 'RAdam'         # choices=['Adam', 'AdamW','RAdam']
        self.lr = 5e-5 #if self.use_pretrained else 2e-4     ## calochallenge -  [5-1]e-4     # Caloclouds default: 2e-3, consistency model paper: approx. 1e-5
        self.end_lr = 1e-5 #if self.use_pretrained else 2e-4  ## domain adaptation thing - [5-1]e-5
        self.weight_decay = 0
        self.max_grad_norm = 10
        self.sched_start_epoch = 100 * 1e3
        self.sched_end_epoch = 300 * 1e3
        self.max_iters = 1.1e6

    # Others
        self.device = 'cuda'
        self.logdir = '/data/dust/user/valentel/beegfs.migration/dust/logs/MyCaloTransfer_diffusionweights'
        self.seed = 42
        self.val_freq =  10_000  #  1e3          # saving intervall for checkpoints

        self.test_freq = 30 * 1e3   
        self.test_size = 400
        self.tag = None
        self.log_iter = 100   # log every n iterations, default: 100

    # EMA scheduler
        self.ema_type = 'inverse'
        self.ema_power = 0.6667   # depends on the number of iterations, 2/3=0.6667 good for 1e6 iterations, 3/4=0.75 good for less
        self.ema_max_value = 0.9999
        
    # EDM diffusion parameters for training
        self.model = {
            # "sigma_data" : [0.08, 0.35, 0.08, 0.5],    ## default parameters for EDM pape = 0.5, might need to adjust for our dataset (meaning the std of our data) / or a seperate sigma for each feature?
            "sigma_data" : 0.5,
            # "has_variance" : False,
            # "loss_config" : "karras",
            "sigma_sample_density" : {
                "type": "lognormal",
                "mean": -1.2,
                "std": 1.2
                }
            }
        self.dropout_mode = 'all'     # options: 'all',  'mid'  location of the droput layers
        self.dropout_rate = 0.0       # EDM: approx. 0.1, Caloclouds default: 0.0
        self.diffusion_loss = 'l2'    # l2 or l1

    # EDM diffusion parameters for sampling    / also used in CM distillation
        self.num_steps = 18      # EDM paper: 18
        self.sampler = 'heun'
        self.sigma_min = 0.002  # EDM paper: 0.002, k-diffusion config: 0.01
        self.sigma_max = 80.0
        self.rho = 7.0    # exponent in EDM boundaries
        self.s_churn = 0.0
        self.s_noise = 1.0

    # Transfer Learning parameters
        self.diffusion_pretrained_model_path = '/pretrained/Pretrained_CaloClouds-PointWiseNet/ckpt_0.000000_2000000.pt'
        self.time_embedded = True # use time embedded diffusion
        self.train_pwise_layers = [ 0, 1, 2, 3, 4, 5  ] # [3, 4, 5 ] # the layer with adaptive training top3
        self.lora_rank = 8
        self.lora = False
        self.bitfit = False
        self.lora_alpha = 8

        # Shower Flow parameters
        if self.val_ds_type == '1-1000GeV':
            self.sf_norm_energy = 2.65 # 2.65 for 1-1000 GeV , 0.265 for 10-90 GeV
            self.sf_norm_points = 800 #800 for 1-1000 GeV, 400 for 10-90 GeV
        elif self.val_ds_type == '10-90GeV':
            self.sf_norm_energy = 0.265
            self.sf_norm_points = 400
        self.sf_use_pretrained = True
        self.sf_pretrained_model_path = '/pretrained/Pretrained_CaloClouds-ShowerFlow/ShowerFlow_3580.pth' # 2 blocks, 92 entries

        self.sf_weight_decay = 0.0
        self.sf_max_grad_norm = 10
        

    # Consistency Distillation parameters
        self.model_path = 'kCaloClouds_2023_06_29__23_08_31/ckpt_0.000000_2000000.pt'
        self.use_ema_trainer = True
        self.start_ema = 0.95
        self.cm_random_init = False    # kinda like consistency training, but still with a teacher score function

    def __repr__(self) -> str:
            strings = [f'-------- {self.__class__.__name__} ---------']
 
            for key, value in self.__dict__.items():
                if callable(value):
                    continue

                strings.append(f'\t- {key}: {value}')

            strings.append('-' * 80)
            return '\n'.join(strings)