#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --image=docker:ereinha/ngc-23.04-v0:arrow
#SBATCH --time=00:05:00
#SBATCH --mail-user=bbbam@crimson.ua.eduuuuu
#SBATCH --mail-type=ALL
#SBATCH --account=m4392
#SBATCH -J resnet18_node_1_test
#SBATCH --output=slurm_resnet18_node_1_test_%J.out




# nvidia-smi

# export FI_MR_CACHE_MONITOR=userfaultfd
# export HDF5_USE_FILE_LOCKING=FALSE
# export NCCL_NET_GDR_LEVEL=PHB
# export MASTER_ADDR=$(hostname)
export SLURM_CPU_BIND=None
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SCRATCH=/pscratch/sd/b/bbbam
# TIMESTAMP=$(date +%Y_%m_%d_%H_%M_%S)
# export TIMESTAMP
srun --unbuffered --export=ALL shifter python3 mass_trainer_tau_multi_node_gpu_h5data.py --WandB --run_test --epochs=2 --base_lr=.00004 --epsilon=0.01 --batch_size=1024 --warmup=1 --num_worker=8 --checkpoint_folder=resnet18_test --n_train=10000 --n_valid=5000 --n_test=5000
