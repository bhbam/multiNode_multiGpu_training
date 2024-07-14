#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --image=docker:ereinha/ngc-23.04-v0:arrow
#SBATCH --time=24:00:00
#SBATCH -J resnet34_modified_with_test_data
#SBATCH --output=slurm_resnet34_modified_with_test_data-%j.out
#SBATCH --mail-user=bbbam@crimson.ua.eduuuuu
#SBATCH --mail-type=ALL
#SBATCH --account=m4392
# ##SBATCH --mem=256



# nvidia-smi

# export FI_MR_CACHE_MONITOR=userfaultfd
# export HDF5_USE_FILE_LOCKING=FALSE
# export NCCL_NET_GDR_LEVEL=PHB
# export MASTER_ADDR=$(hostname)
export SLURM_CPU_BIND=None
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SCRATCH=/pscratch/sd/b/bbbam

srun --unbuffered --export=ALL shifter python3 mass_trainer_tau_multi_node_gpu_h5data.py --WandB=True --run_test=True --epochs=300 --base_lr=.00004 --epsilon=0.0000001 --batch_size=800 --warmup=30 --num_worker=8 --checkpoint_folder=resnet34_modified_with_test_data --n_train=-1 --n_valid=-1 --n_test=-1
