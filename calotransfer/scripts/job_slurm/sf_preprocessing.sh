#!/bin/bash
# No SLURM directives here - will use your default environment settings

# Change to the project directory - FIX: Use correct path
cd CaloTransfer/calotransfer/scripts/preprocessing/

# Create log directory if it doesn't exist
mkdir -p logs

# Define timestamp for log files
timestamp=$(date +"%Y%m%d_%H%M%S")

# Function to display usage instructions
function show_usage {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --create         Create subdatasets from original dataset"
    echo "  --calculate      Calculate per-layer metrics for all datasets"
    echo "  --visualize      Visualize all datasets"
    echo "  --size SIZE      Visualize specific dataset size (e.g., '50k')"  # FIX: Remove _1-1000
    echo "  --all            Run all steps (create, calculate, visualize)"
    echo "  --help           Show this help message"
    echo
    echo "Example: $0 --all"
    echo "Example: $0 --calculate --visualize --size 10k"  # FIX: Remove _1-1000
}

# Check if Python script exists
if [ ! -f "showerflow_train.py" ]; then
    echo "Error: showerflow_train.py not found in $(pwd)!"
    exit 1
fi

# Parse command line arguments
ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --create)
            ARGS="$ARGS --create"
            shift
            ;;
        --calculate)
            ARGS="$ARGS --calculate"
            shift
            ;;
        --visualize)
            ARGS="$ARGS --visualize"
            shift
            ;;
        --size)
            ARGS="$ARGS --size $2"
            shift 2
            ;;
        --all)
            ARGS="$ARGS --all"
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# If no arguments provided, show usage
if [ -z "$ARGS" ]; then
    show_usage
    exit 1
fi

# Create results directory if it doesn't exist
mkdir -p results

# Print environment information
echo "===== Environment Information ====="
echo "Host: $(hostname)"
echo "Current directory: $(pwd)"
echo "Python version: $(python --version 2>&1)"
echo "Available modules:"
module list
echo "=================================="

# Install required packages to user directory
echo "Installing required packages..."
pip install --user h5py numpy matplotlib tqdm

# Run the Python script with the provided arguments
echo "Starting calorimeter dataset processing at $(date)"
echo "Command: python showerflow_train.py $ARGS"

# Run with memory monitoring
echo "Available memory before starting:"
free -h

# FIX: Simplified execution without tee/PIPESTATUS complexity
python showerflow_train.py $ARGS > logs/calo_processing_${timestamp}.log 2>&1
RESULT=$?

# Check if the script executed successfully
if [ $RESULT -eq 0 ]; then
    echo "Processing completed successfully at $(date)"
    echo "Log saved to logs/calo_processing_${timestamp}.log"
    echo "Results saved to the results directory"
    
    # Show last few lines of successful log
    echo "Last 10 lines of output:"
    tail -n 10 logs/calo_processing_${timestamp}.log
else
    echo "Error occurred during processing (exit code: $RESULT). Check the log file for details."
    echo "Log saved to logs/calo_processing_${timestamp}.log"
    
    # Print the last few lines of the log file to help diagnose the issue
    echo "Last 20 lines of the log file:"
    tail -n 20 logs/calo_processing_${timestamp}.log
    
    exit 1
fi