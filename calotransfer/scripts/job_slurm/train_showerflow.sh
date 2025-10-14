#!/bin/bash

#SBATCH --time=5-00:00:00    # Set to 5 days
#SBATCH --nodes 1
#SBATCH --output /output/train-score/train-showerflow/training-%j.out      # terminal output
#SBATCH --error /output/train-score/train-showerflow/training-%j.err

#SBATCH --export=None
#SBATCH --partition maxgpu
#SBATCH --constraint="GPUx1&V100|GPUx1&A100"
#SBATCH --job-name showerflow-train

#SBATCH --array=0-6 # 0-6 %7 parallel jobs

# Add staggered delay between different dataset jobs
sleep $((SLURM_ARRAY_TASK_ID * 5))

# Print job information header
echo "===================================================================="
echo "Starting PointWise Training Job"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Start Time: $(date)"
echo "===================================================================="

# Activate conda environment
module load maxwell mamba
. mamba-init
conda activate calo-transfer

cd /data/dust/user/valentel/maxwell.merged/CaloTransfer/calotransfer/scripts/

# Dataset configurations
declare -a dataset_sizes=(
    
    # "pretrained_cc2"

    "100k_1-1000"
    "50k_1-1000"
    "10k_1-1000"
    "5k_1-1000" 
    "1k_1-1000"   
    "500_1-1000"   
    "100_1-1000"
)

IFS=':' read -r ds_key ds_value <<< "${dataset_sizes[$SLURM_ARRAY_TASK_ID]}"

# Print dataset configuration
echo "Dataset Configuration:"
echo "  - Key: $ds_key"
echo ""

# Rest of your parameters
NUM_ITERATIONS=2
sf_use_pretrained=("False") # True or False

# Print training parameters
echo "Training Parameters:"
echo "  - Iterations: $NUM_ITERATIONS"
echo "  - Use Pretrained: ${sf_use_pretrained[@]}"
echo ""

# Seed generation
seeds=()
for i in $(seq 0 $NUM_ITERATIONS); do
    seeds+=($((45 + i)))
done

# Print seeds
echo "Generated Seeds: ${seeds[@]}"
echo "===================================================================="
echo "Starting training runs..."
echo ""

# Main training loop
for i in $(seq 1 $NUM_ITERATIONS); do
    for sf_use_pretrain in "${sf_use_pretrained[@]}"; do
        echo "----------------------------------------------------"
        echo "Starting Iteration $i/$NUM_ITERATIONS"
        echo "  - Pretrained: $sf_use_pretrain"
        echo "  - Seed: ${seeds[$i-1]}"
        echo "----------------------------------------------------"
        
        # Pass environment variables instead of creating the trainer with args
        export DATASET_KEY="$ds_key"
        export SF_USE_PRETRAIN="$sf_use_pretrain"
        export SEED="${seeds[$i-1]}"
        
        python showerflow_train.py \
            --dataset_key "$ds_key" \
            --sf_use_pretrain "$sf_use_pretrain" \
            --seed "${seeds[$i-1]}"
            
        echo "----------------------------------------------------"
        echo "Completed Iteration $i/$NUM_ITERATIONS"
        echo "----------------------------------------------------"
        echo ""
    done
done

echo "===================================================================="
echo "All training iterations completed successfully!"
echo "End Time: $(date)"
echo "===================================================================="

exit