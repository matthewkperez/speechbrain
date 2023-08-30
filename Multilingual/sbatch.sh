#SBATCH --nodes=1 # We want two nodes (servers)
#SBATCH --ntasks-per-node=1 # we will run once the next srun per node
#SBATCH --gres=gpu:1 # we want 4 GPUs per node
#SBATCH --job-name=SBisSOcool
#SBATCH --cpus-per-task=10 # the only task will request 10 cores
#SBATCH --time=20:00:00 # Everything will run for 20H.
#SBATCH --partition=spgpu # partition
#SBATCH --account=engin1 # partition

# # We jump into the submission dir
# cd ${SLURM_SUBMIT_DIR}

# And we call the srun that will run --ntasks-per-node times (once here) per node
srun srun_script.sh