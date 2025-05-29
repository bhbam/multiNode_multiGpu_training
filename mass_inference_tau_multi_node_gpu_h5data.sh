#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --image=docker:bhimbam/ngc-23.04-v0:pyarrow_timm
#SBATCH --time=00:15:00
#SBATCH --mail-user=bbbam@crimson.ua.eduuuuu
#SBATCH --mail-type=ALL
#SBATCH --account=m4392
#SBATCH -J Testing_model
#SBATCH --output=slurm_Testing_ResNet_no_ieta_iphi_%J.out




# nvidia-smi


export SLURM_CPU_BIND=None
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SCRATCH=/pscratch/sd/b/bbbam


srun --unbuffered --export=ALL shifter python3 mass_inference_tau_multi_node_gpu_h5data.py --resblocks=3 --mass=3p7 --best_epoch=300 --final_model_dir=/pscratch/sd/b/bbbam/ResNet_B3_no_iphi_ieta_Nodes_4.0/ResNet_B3_no_iphi_ieta_13_channel_massregressor_2024_12_12_10:39:53_GPUS_16/Models --test_data_path=/pscratch/sd/b/bbbam/run_3_with_trigger_signals_normalized_combined --batch_size=32  --num_worker=8 --checkpoint_folder=ResNet_B3_no_iphi_ieta_Nodes_4.0_inference_from_Ruchi_data --n_test=-1 &
srun --unbuffered --export=ALL shifter python3 mass_inference_tau_multi_node_gpu_h5data.py --resblocks=3 --mass=14  --best_epoch=300 --final_model_dir=/pscratch/sd/b/bbbam/ResNet_B3_no_iphi_ieta_Nodes_4.0/ResNet_B3_no_iphi_ieta_13_channel_massregressor_2024_12_12_10:39:53_GPUS_16/Models --test_data_path=/pscratch/sd/b/bbbam/run_3_with_trigger_signals_normalized_combined --batch_size=32  --num_worker=8 --checkpoint_folder=ResNet_B3_no_iphi_ieta_Nodes_4.0_inference_from_Ruchi_data --n_test=-1 &
srun --unbuffered --export=ALL shifter python3 mass_inference_tau_multi_node_gpu_h5data.py --resblocks=3 --mass=4   --best_epoch=300 --final_model_dir=/pscratch/sd/b/bbbam/ResNet_B3_no_iphi_ieta_Nodes_4.0/ResNet_B3_no_iphi_ieta_13_channel_massregressor_2024_12_12_10:39:53_GPUS_16/Models --test_data_path=/pscratch/sd/b/bbbam/run_3_with_trigger_signals_normalized_combined --batch_size=32  --num_worker=8 --checkpoint_folder=ResNet_B3_no_iphi_ieta_Nodes_4.0_inference_from_Ruchi_data --n_test=-1 &
srun --unbuffered --export=ALL shifter python3 mass_inference_tau_multi_node_gpu_h5data.py --resblocks=3 --mass=6   --best_epoch=300 --final_model_dir=/pscratch/sd/b/bbbam/ResNet_B3_no_iphi_ieta_Nodes_4.0/ResNet_B3_no_iphi_ieta_13_channel_massregressor_2024_12_12_10:39:53_GPUS_16/Models --test_data_path=/pscratch/sd/b/bbbam/run_3_with_trigger_signals_normalized_combined --batch_size=32  --num_worker=8 --checkpoint_folder=ResNet_B3_no_iphi_ieta_Nodes_4.0_inference_from_Ruchi_data --n_test=-1 &
wait
