#!/bin/bash

#SBATCH --time=1-00:00:00    # Set the maximum runtime to 5 days
#SBATCH --nodes 1            # Request 1 node
#SBATCH --output /CaloTransfer/calotransfer/scripts/train-score/generation/generation-%j.out  # Standard output file
#SBATCH --error /CaloTransfer/calotransfer/scripts/train-score/generation/generation-%j.err   # Standard error file
#SBATCH --export=None
#SBATCH --constraint="GPU"
#SBATCH --partition maxgpu   # Specify the partition to use
#SBATCH --job-name ShowerGeneration  # Set the job name

# Activate conda environment
module load maxwell mamba
. mamba-init
conda activate calo-transfer

cd CaloTransfer/calotransfer/scripts/evaluation/  # Change to the project directory

dataset_size=( 
       
        
        ##### 1-1000GeV #####
        '100k_1-1000'
        '50k_1-1000'
        '10k_1-1000'
        '5k_1-1000'
        '1k_1-1000'
        '500_1-1000'
        '100_1-1000'
    )

use_pretrained=(
                
                ##### 1-1000GeV #####
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
                # 'finetune_top3_v1_1_1000'
                # 'finetune_top3_v2_1_1000'  
                # 'finetune_top3_v3_1_1000'
                # 'finetune_top3_v4_1_1000'
                # 'finetune_top3_v5_1_1000'

                # 'finetune_bitfit_v1_1_1000'
                # 'finetune_bitfit_v2_1_1000'
                # 'finetune_bitfit_v3_1_1000'
                'finetune_bitfit_v4_1_1000'
                # 'finetune_bitfit_v5_1_1000'

                # 'lora_r1_v1_1_1000'  # seed = 41, r = 1, alpha = 1
                # 'lora_r2_v1_1_1000'  # seed = 42, r = 2, alpha = 2
                # 'lora_r4_v1_1_1000'  # seed =
                # 'lora_r8_v1_1_1000'
                # 'lora_r8_v2_1_1000'
                # 'lora_r8_v3_1_1000'  # seed = 44, r = 8, alpha = 8

                # 'lora_r16_v1_1_1000'  # seed = 42, r = 16, alpha = 16
                # 'lora_r32_v1_1_1000'  # seed = 43, r = 32, alpha = 32
                # 'lora_r32a48_v1_1_1000'  # seed = 43, r = 48, alpha = 48
                # 'lora_r48_v1_1_1000'  # seed = 43, r = 48, alpha = 48
                
                # 'lora_r64_v1_1_1000'  
                # 'lora_r106_v1_1_1000'  # seed = 44, r = 106, alpha = 106
                # 'lora_r106_v2_1_1000'  # seed = 42, r = 106, alpha = 106
                # 'lora_r106_v3_1_1000'  # seed = 42, r = 106, alpha = 106
                # 'lora_r106_v4_1_1000'  # seed = 42, r = 106, alpha = 106
                # 'lora_r106_v5_1_1000'  # seed = 42, r = 106, alpha = 106


                # 'lora_r204_v1_1_1000'  # seed = 45, r = 204, alpha = 204
                )

LORA_RANK=106 # Fixed LoRA rank for all runs
LORA_ALPHA=106 # Fixed LoRA alpha for all runs

# Initialize the training_steps array with 10k
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

use_ema=(
    "True"
)

echo "===================================================================="
echo "Starting generation runs..."
echo "===================================================================="
# Print dataset configuration
echo "Dataset Configuration:"
for ds in "${dataset_size[@]}"; do
    echo "  - Size: $ds"
done
echo ""
# Print training parameters
echo "Generation Parameters:"
echo "  - Use Pretrained: ${use_pretrained[@]}"
echo "  - Use EMA: ${use_ema[@]}"
echo "  - Generation Steps: ${training_steps[@]}"
echo ""

# Loop through all combinations
start_time=$(date +%s)  # Record the start time
for use_pretrain in "${use_pretrained[@]}"; do
    for ema in "${use_ema[@]}"; do
        for training_step in "${training_steps[@]}"; do
            for ds in "${dataset_size[@]}"; do
                # Extract the dataset size from the key-value pair
                echo "Starting training for dataset_size=${ds}, use_pretrained=${use_pretrain}, training_step=${training_step}, use_ema=${ema}"
                python -m CaloTransfer.generation \
                    --dataset_size "$ds" \
                    --use_pretrained "$use_pretrain" \
                    --training_step "$training_step" \
                    --use_ema "$ema" \
                    --lora_rank "$LORA_RANK" \
                    --lora_alpha "$LORA_ALPHA"
                echo "Completed training for dataset_size=${ds}, use_pretrained=${use_pretrain}, training_step=${training_step}, use_ema=${ema}"
            done
        done
    done
done

end_time=$(date +%s)  # Record the end time
elapsed_time=$((end_time - start_time))  # Calculate the elapsed time
echo "Total elapsed time: $elapsed_time seconds"

exit
