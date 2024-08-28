#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --image=docker:bhimbam/ngc-23.04-v0:pyarrow_timm
#SBATCH --time=24:00:00
#SBATCH --mail-user=bbbam@crimson.ua.eduuuuu
#SBATCH --mail-type=ALL
#SBATCH --account=m4392
#SBATCH -J EffNet_E1
#SBATCH --output=slurm_EffNet_E1_4_%J.out
#SBATCH -C gpu&hbm80g

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
srun --unbuffered --export=ALL shifter python3 mass_trainer_tau_multi_node_gpu_h5data.py --resblocks=1 --WandB --run_test --model_name=EffNet_E1 --epochs=300 --base_lr=0.0001 --epsilon=0.01 --batch_size=1024 --warmup=30 --num_worker=8 --checkpoint_folder=EffNet_E1 --n_train=-1 --n_valid=-1 --n_test=-1