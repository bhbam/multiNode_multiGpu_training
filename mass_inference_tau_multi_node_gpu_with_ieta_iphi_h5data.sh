#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --image=docker:bhimbam/ngc-23.04-v0:pyarrow_timm
#SBATCH --time=00:10:00
#SBATCH --mail-user=bbbam@crimson.ua.eduuuuu
#SBATCH --mail-type=ALL
#SBATCH --account=m4392
#SBATCH -J Testing_model
#SBATCH --output=slurm_Testing_ResNet_B3_%J.out




# nvidia-smi


export SLURM_CPU_BIND=None
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SCRATCH=/pscratch/sd/b/bbbam


srun --unbuffered --export=ALL shifter python3 mass_inference_tau_multi_node_gpu_with_ieta_iphi_h5data.py --resblocks=3 --mass=3p7 --best_epoch=47 --final_model_dir=/pscratch/sd/b/bbbam/ResNet_B3_with_ieta_iphi_including_negative_mass_Nodes_4.0/ResNet_B3_with_ieta_iphi_including_negative_mass_13_channel_massregressor_2025_01_16_13:20:08_GPUS_16/Models  --test_data_path=/pscratch/sd/b/bbbam/run_3_with_trigger_signals_normalized_combined --batch_size=32  --num_worker=8 --checkpoint_folder=ResNet_B3_with_ieta_iphi_including_negative_mass_Nodes_4.0 --n_test=-1 &
srun --unbuffered --export=ALL shifter python3 mass_inference_tau_multi_node_gpu_with_ieta_iphi_h5data.py --resblocks=3 --mass=14  --best_epoch=47 --final_model_dir=/pscratch/sd/b/bbbam/ResNet_B3_with_ieta_iphi_including_negative_mass_Nodes_4.0/ResNet_B3_with_ieta_iphi_including_negative_mass_13_channel_massregressor_2025_01_16_13:20:08_GPUS_16/Models  --test_data_path=/pscratch/sd/b/bbbam/run_3_with_trigger_signals_normalized_combined --batch_size=32  --num_worker=8 --checkpoint_folder=ResNet_B3_with_ieta_iphi_including_negative_mass_Nodes_4.0 --n_test=-1 &
srun --unbuffered --export=ALL shifter python3 mass_inference_tau_multi_node_gpu_with_ieta_iphi_h5data.py --resblocks=3 --mass=4   --best_epoch=47 --final_model_dir=/pscratch/sd/b/bbbam/ResNet_B3_with_ieta_iphi_including_negative_mass_Nodes_4.0/ResNet_B3_with_ieta_iphi_including_negative_mass_13_channel_massregressor_2025_01_16_13:20:08_GPUS_16/Models  --test_data_path=/pscratch/sd/b/bbbam/run_3_with_trigger_signals_normalized_combined --batch_size=32  --num_worker=8 --checkpoint_folder=ResNet_B3_with_ieta_iphi_including_negative_mass_Nodes_4.0 --n_test=-1 &
srun --unbuffered --export=ALL shifter python3 mass_inference_tau_multi_node_gpu_with_ieta_iphi_h5data.py --resblocks=3 --mass=6   --best_epoch=47 --final_model_dir=/pscratch/sd/b/bbbam/ResNet_B3_with_ieta_iphi_including_negative_mass_Nodes_4.0/ResNet_B3_with_ieta_iphi_including_negative_mass_13_channel_massregressor_2025_01_16_13:20:08_GPUS_16/Models  --test_data_path=/pscratch/sd/b/bbbam/run_3_with_trigger_signals_normalized_combined --batch_size=32  --num_worker=8 --checkpoint_folder=ResNet_B3_with_ieta_iphi_including_negative_mass_Nodes_4.0 --n_test=-1 &
wait
