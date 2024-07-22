#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --image=docker:ereinha/ngc-23.04-v0:arrow
#SBATCH --time=00:10:00
#SBATCH --mail-user=bbbam@crimson.ua.eduuuuu
#SBATCH --mail-type=ALL
#SBATCH --account=m4392
#SBATCH -J Testing_model
#SBATCH --output=slurm_Testing_model_%J.out




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
srun --unbuffered --export=ALL shifter python3 mass_inference_tau_multi_node_gpu_h5data.py --mass=12 --best_epoch=103 --final_model_dir=/pscratch/sd/b/bbbam/resnet34_modified_final/13_channels_massregressor_multi_node_2024_07_12_06_59_GPUS_16 --batch_size=1024  --num_worker=8 --checkpoint_folder=resnet34_modified_final --n_test=-1 
