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
#SBATCH -J ResNet_mN1p2To22
#SBATCH --output=slurm_ResNet_mN1p2To22%J.out


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
# srun --unbuffered --export=ALL shifter python3 mass_trainer_tau_multi_node_gpu_with_ieta_iphi_min_max_scaling_h5data.py --m0_scale=14 --resblocks=3  --WandB --run_test --data_path=/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_m3p6To14_pt30T0300_unbiased_combined_h5  --model_name=ResNet_min_max_m3p6To14 --epochs=300 --base_lr=0.0001 --epsilon=0.01 --batch_size=1024 --warmup=30 --num_worker=8 --checkpoint_folder=ResNet_min_max_m3p6To14 --n_train=-1 --n_valid=-1 --n_test=-1
srun --unbuffered --export=ALL shifter python3 mass_trainer_tau_multi_node_gpu_with_ieta_iphi_min_max_scaling_h5data.py --m0_scale=14 --resblocks=3 --WandB --run_test --data_path=/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_m3p6To18_pt30T0300_original_unbiased_combined_v2_h5 --model_name=ResNet_min_max_m3p6To18_v2 --epochs=300 --base_lr=0.0001 --epsilon=0.01 --batch_size=1024 --warmup=30 --num_worker=8 --checkpoint_folder=ResNet_min_max_m3p6To18_v2 --n_train=-1 --n_valid=-1 --n_test=-1
# srun --unbuffered --export=ALL shifter python3 mass_trainer_tau_multi_node_gpu_with_ieta_iphi_min_max_scaling_h5data.py --m0_scale=14 --resblocks=3 --WandB --run_test --data_path=/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_m1p2To18_pt30T0300_original_combined_unbiased_h5  --model_name=ResNet_min_max_m1p2To18 --epochs=300 --base_lr=0.0001 --epsilon=0.01 --batch_size=1024 --warmup=30 --num_worker=8 --checkpoint_folder=ResNet_min_max_m1p2To18 --n_train=-1 --n_valid=-1 --n_test=-1
# srun --unbuffered --export=ALL shifter python3 mass_trainer_tau_multi_node_gpu_with_ieta_iphi_min_max_scaling_h5data.py --m0_scale=14 --resblocks=3 --WandB --run_test --data_path=/pscratch/sd/b/bbbam/IMG_aToTauTau_mNeg1p2T018_unbiased_original_combined_with_stretching_lower_unphy_bins_h5  --model_name=ResNet_min_max_mN1p2To18 --epochs=300 --base_lr=0.0001 --epsilon=0.01 --batch_size=1024 --warmup=30 --num_worker=8 --checkpoint_folder=ResNet_min_max_mN1p2To18 --n_train=-1 --n_valid=-1 --n_test=-1
# srun --unbuffered --export=ALL shifter python3 mass_trainer_tau_multi_node_gpu_with_ieta_iphi_min_max_scaling_h5data.py --m0_scale=14 --resblocks=3 --WandB --run_test --data_path=/pscratch/sd/b/bbbam/IMG_aToTauTau_mNeg1p2T018_unbiased_original_combined_with_stretching_lower_unphy_and_upper_mass_bins_h5  --model_name=ResNet_min_max_mN1p2To22 --epochs=300 --base_lr=0.0001 --epsilon=0.01 --batch_size=1024 --warmup=30 --num_worker=8 --checkpoint_folder=ResNet_min_max_mN1p2To22 --n_train=-1 --n_valid=-1 --n_test=-1