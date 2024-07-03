#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --image=docker:ereinha/ngc-23.04-v0:arrow
#SBATCH --time=24:00:00
#SBATCH -J multi_3
#SBATCH --output=slurm_3node-%j.out
#SBATCH --mail-user=bbbam@crimson.ua.eduuuuu
#SBATCH --mail-type=ALL
#SBATCH --account=m4392


# nvidia-smi

# export FI_MR_CACHE_MONITOR=userfaultfd
# export HDF5_USE_FILE_LOCKING=FALSE
# export NCCL_NET_GDR_LEVEL=PHB
# export MASTER_ADDR=$(hostname)
# export SLURM_CPU_BIND="cores"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SCRATCH=/pscratch/sd/b/bbbam

srun --unbuffered --export=ALL shifter python3 mass_trainer_tau_multi_node_gpu_h5data.py --WandB=True --epochs=300 --base_lr=.00004 --epsilon=0.0000001 --batch_size=1024 --warmup=15 --num_worker=8 --checkpoint_folder=resnet50_1node --n_train=-1 --n_valid=-1
