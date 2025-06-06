    #!/bin/bash

# Compute number of parallel jobs as nproc - 5
NUM_JOBS=$(( $(nproc) - 5 ))

# Create a timestamped working directory
WORKDIR="./TPUFIseg128_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WORKDIR"
cd "$WORKDIR" || exit 1

for i in $(seq 0 $((NUM_JOBS - 1))); do
  (
    echo "Starting thread $i"

    # Clone the repo into a unique subdirectory
    git clone https://github.com/ManiSadati/TPU-FI.git TPUFIseg128_$i

    cd TPUFIseg128_$i || exit 1

    # Activate Conda environment
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate TFlite

    # Run the script
    python fi_segmentation.py --model 128 --iterations 15

    echo "Thread $i finished"
  ) &
done

# Wait for all jobs to finish
wait

echo "All jobs completed."
