#!/bin/bash

#SBATCH --time=5-00:00:00    # Set to 5 days
#SBATCH --nodes 1
#SBATCH --output /output/train-score/eval_sf/eval-%j.out      # terminal output
#SBATCH --error /output/train-score/eval_sf/eval-%j.err

#SBATCH --export=None

#SBATCH --partition maxgpu
#SBATCH --constraint="GPU"
#SBATCH --job-name SFeval-epochs

# GPUx1&A100
# constraint "GPU"
# Activate conda environment
module load maxwell mamba
. mamba-init
conda activate calo-transfer

cd CaloTransfer/calotransfer/scripts/evaluation/

python showerflow_evaluation_epochs.py

exit