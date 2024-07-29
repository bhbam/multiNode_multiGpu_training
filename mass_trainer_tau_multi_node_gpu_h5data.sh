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
#SBATCH -J ResNet_B6
#SBATCH --output=slurm_ResNet_B6_4_%J.out


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
srun --unbuffered --export=ALL shifter python3 mass_trainer_tau_multi_node_gpu_h5data.py --WandB --run_test --model_name=ResNet_B6 --epochs=250 --base_lr=.00004 --epsilon=0.01 --batch_size=1024 --warmup=25 --num_worker=8 --checkpoint_folder=ResNets_old_no_ieta_iphi --n_train=-1 --n_valid=-1 --n_test=-1
