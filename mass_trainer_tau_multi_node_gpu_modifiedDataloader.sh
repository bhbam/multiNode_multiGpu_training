#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=16
#SBATCH --image=docker:ereinha/ngc-23.04-v0:arrow
#SBATCH --time=00:30:00
#SBATCH -J multi_test
#SBATCH --output=slurm-%j.out
#SBATCH --mail-user=bbbam@crimson.ua.eduuuuu
#SBATCH --mail-type=ALL
#SBATCH --account=m4392
#SBATCH --mem-per-gpu=55G

# nvidia-smi

# export FI_MR_CACHE_MONITOR=userfaultfd
# export HDF5_USE_FILE_LOCKING=FALSE
# export NCCL_NET_GDR_LEVEL=PHB
# export MASTER_ADDR=$(hostname)
# export SLURM_CPU_BIND="cores"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SCRATCH=/pscratch/sd/b/bbbam

srun --unbuffered --export=ALL shifter python3 mass_trainer_tau_multi_node_gpu_modifiedDataloader.py --epochs=50 --base_lr=.00004 --epsilon=0.0000001 --batch_size=3072 --warmup=5 --checkpoint_folder=mass_regression_test
