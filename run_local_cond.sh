# Unconditional
# conda activate zeodiff

# Setup time
t1=$(date +%s)

cd zeodiff
python run.py with train=True \
        self_condition=True \
        target_prop='CON' \
        train_dataset='../data/train_augmented/' \
        test_dataset='../data/test/' \
        n_gpu=0 devices=[0] batch_size=16


# Print elapsed time (goes to job_output.txt)
t2=$(date +%s)
elapsed_seconds=$((t2 - t1))
days=$((elapsed_seconds / 86400))
hours=$(( (elapsed_seconds % 86400) / 3600 ))
minutes=$(( (elapsed_seconds % 3600) / 60 ))
seconds=$((elapsed_seconds % 60))
echo "elapsed ${days} days, ${hours} hours, ${minutes} minutes, ${seconds} seconds"

