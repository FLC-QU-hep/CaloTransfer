#!/bin/bash

#SBATCH --time=1-00:00:00    # Set to 5 days
#SBATCH --nodes 1
#SBATCH --output ./train-score/finals/final_evaluation-%j.out      # terminal output
#SBATCH --error ./train-score/finals/final_evaluation-%j.err
#SBATCH --export=None
#SBATCH --partition maxgpu 
#SBATCH --job-name final_evaluation

# Activate conda environment
module load maxwell mamba
. mamba-init
conda activate calo-transfer

cd /data/dust/user/valentel/maxwell.merged/MyCaloTransfer/CaloTransfer/
# Initialize the training_steps array

training_steps=(
    10_000
    50_000
    100_000

    150_000
    200_000
    250_000

    500_000
    750_000
    1_000_000 
)
#max 2 strategies ata time
training_strategies=(  
        # 'vanilla' # baseline
        # 'vanilla_v1'
        # 'vanilla_v2'
        # 'vanilla_v3'

        # vanilla_v4
        # 'vanilla_v5'
        # 'vanilla_v6'

        # vanilla_v4_w5k_10-90 
        # vanilla_v5_w5k_10-90
        # vanilla_v6_w5k_10-90

        # 'finetune_full_v1'
        # 'finetune_full_v2'
        # 'finetune_full_v3'

        # 'finetune'
        # 'finetune_3layers_v2'
        # 'finetune_3layers_v3'
        # 'finetune_3layers_v4'
        # 'finetune_3layers_v5'

        # 'finetune_3layers_v6'
        # 'finetune_3layers_v7'
        # 'finetune_3layers_v8'
        # 'finetune_full_v1_w5k_10-90'
        # 'finetune_full_v2_w5k_10-90'
        # 'finetune_full_v3_w5k_10-90'

        # 'lora_full_v1'
        # 'lora_full_v2'
        # 'lora_full_v3'

        # === 1-1000GeV ===
        # 'vanilla_full_v1_1_1000'
        # 'vanilla_full_v2_1_1000'
        # 'vanilla_full_v3_1_1000'
        # 'vanilla_full_v4_1_1000'
        # 'vanilla_full_v5_1_1000'

        # 'finetune_full_v1_1_1000'
        # 'finetune_full_v2_1_1000'
        # 'finetune_full_v3_1_1000'
        # 'finetune_full_v4_1_1000'
        # 'finetune_full_v5_1_1000'

        # 'finetune_head_v1_1_1000'

        # 'finetune_bitfit_v1_1_1000'
        # 'finetune_bitfit_v2_1_1000'
        # 'finetune_bitfit_v3_1_1000'
        # 'finetune_bitfit_v4_1_1000'
        # 'finetune_bitfit_v5_1_1000'

        # 'finetune_top3_v1_1_1000'
        # 'finetune_top3_v2_1_1000'
        # 'finetune_top3_v3_1_1000'
        # 'finetune_top3_v4_1_1000'
        # 'finetune_top3_v5_1_1000'

        # 'lora_r1_v1_1_1000'  # seed = 41, r = 1, alpha = 1
        # 'lora_r2_v1_1_1000'  
        # 'lora_r4_v1_1_1000'  # seed = 42, r = 2, alpha = 2
        # 'lora_r8_v1_1_1000'
        # 'lora_r8_v2_1_1000'  # seed = 43, r = 4, alpha = 4
        # 'lora_r8_v3_1_1000'  # seed = 44, r = 8, alpha = 8
        # 'lora_r16_v1_1_1000'
        # 'lora_r32_v1_1_1000'  # seed = 41, r = 32, alpha = 32
        # 'lora_r32a48_v1_1_1000'  # seed = 41, r = 48, alpha = 48
        # 'lora_r48_v1_1_1000'  # seed = 41, r = 32, alpha = 32
        # 'lora_r64_v1_1_1000'  # seed = 41, r = 32, alpha = 32
        
        # 'lora_r106_v1_1_1000'  # seed = 44, r = 8, alpha = 8
        # 'lora_r106_v2_1_1000'  # seed = 42, r = 106, alpha = 106
        # 'lora_r106_v3_1_1000'  # seed = 43, r = 106, alpha = 106
        # 'lora_r106_v4_1_1000'  # seed = 43, r = 106, alpha = 106
        'lora_r106_v5_1_1000'  # seed = 43, r = 106, alpha = 106
        
        # 'lora_r204_v1_1_1000'  # seed = 44, r = 106, alpha = 106
    )   

use_ema=(
    "True"
    # "False"

)

# Loop through all training steps
for training_step in "${training_steps[@]}"; do
    # Join the training_strategies array with commas
    strategies_string=$(IFS=,; echo "[${training_strategies[*]}]")
    echo "Starting evaluation for training_step=${training_step}, training strategies=${strategies_string}, use_ema=${use_ema}"
    python evaluation_models.py \
    --training_step "$training_step" \
    --training_strategies "$strategies_string" \
    --use_ema "${use_ema}"
done
exit