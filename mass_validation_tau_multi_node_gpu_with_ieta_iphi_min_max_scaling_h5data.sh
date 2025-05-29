#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --image=docker:bhimbam/ngc-23.04-v0:pyarrow_timm
#SBATCH --time=00:30:00
#SBATCH --mail-user=bbbam@crimson.ua.eduuuuu
#SBATCH --mail-type=ALL
#SBATCH --account=m4392
#SBATCH -J Testing_model
#SBATCH --output=slurm_Testing_ResNet_%J.out




# nvidia-smi


export SLURM_CPU_BIND=None
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SCRATCH=/pscratch/sd/b/bbbam


srun --unbuffered --export=ALL shifter python3 mass_validation_tau_multi_node_gpu_with_ieta_iphi_min_max_scaling_h5data.py --resblocks=3 --mass=3p7  --best_epoch=200 --final_model_dir=/global/cfs/cdirs/m4392/bbbam/jupyter_notebook/ResNet_min_max_scaling_PL/ResNet_min_max_m3p6To14_Nodes_4.0/ResNet_min_max_m3p6To14_13_channel_massregressor_2025_05_11_10:29:04_GPUS_16/Models      --batch_size=320  --num_worker=8 --checkpoint_folder=ResNet_min_max_scaling_signal_test_PL/ResNet_min_max_m3p6To14    --n_test=-1 &
srun --unbuffered --export=ALL shifter python3 mass_validation_tau_multi_node_gpu_with_ieta_iphi_min_max_scaling_h5data.py --resblocks=3 --mass=3p7  --best_epoch=200 --final_model_dir=/global/cfs/cdirs/m4392/bbbam/jupyter_notebook/ResNet_min_max_scaling_PL/ResNet_min_max_m3p6To18_Nodes_4.0/ResNet_min_max_m3p6To18_v2_13_channel_massregressor_2025_05_18_00:03:56_GPUS_16/Models   --batch_size=320  --num_worker=8 --checkpoint_folder=ResNet_min_max_scaling_signal_test_PL/ResNet_min_max_m3p6To18    --n_test=-1 &
srun --unbuffered --export=ALL shifter python3 mass_validation_tau_multi_node_gpu_with_ieta_iphi_min_max_scaling_h5data.py --resblocks=3 --mass=3p7  --best_epoch=200 --final_model_dir=/global/cfs/cdirs/m4392/bbbam/jupyter_notebook/ResNet_min_max_scaling_PL/ResNet_min_max_m1p2To18_Nodes_4.0/ResNet_min_max_m1p2To18_13_channel_massregressor_2025_05_12_09:33:12_GPUS_16/Models      --batch_size=320  --num_worker=8 --checkpoint_folder=ResNet_min_max_scaling_signal_test_PL/ResNet_min_max_m1p2To18    --n_test=-1 &
srun --unbuffered --export=ALL shifter python3 mass_validation_tau_multi_node_gpu_with_ieta_iphi_min_max_scaling_h5data.py --resblocks=3 --mass=3p7  --best_epoch=200 --final_model_dir=/global/cfs/cdirs/m4392/bbbam/jupyter_notebook/ResNet_min_max_scaling_PL/ResNet_min_max_mN1p2To18_Nodes_4.0/ResNet_min_max_mN1p2To18_13_channel_massregressor_2025_05_12_09:33:12_GPUS_16/Models    --batch_size=320  --num_worker=8 --checkpoint_folder=ResNet_min_max_scaling_signal_test_PL/ResNet_min_max_mN1p2p6To18 --n_test=-1 &
srun --unbuffered --export=ALL shifter python3 mass_validation_tau_multi_node_gpu_with_ieta_iphi_min_max_scaling_h5data.py --resblocks=3 --mass=3p7  --best_epoch=200 --final_model_dir=/global/cfs/cdirs/m4392/bbbam/jupyter_notebook/ResNet_min_max_scaling_PL/ResNet_min_max_mN1p2To22_Nodes_4.0/ResNet_min_max_mN1p2To22_13_channel_massregressor_2025_05_14_02:15:53_GPUS_16/Models    --batch_size=320  --num_worker=8 --checkpoint_folder=ResNet_min_max_scaling_signal_test_PL/ResNet_min_max_mN1p2To22   --n_test=-1 &
srun --unbuffered --export=ALL shifter python3 mass_validation_tau_multi_node_gpu_with_ieta_iphi_min_max_scaling_h5data.py --resblocks=3 --mass=14   --best_epoch=200 --final_model_dir=/global/cfs/cdirs/m4392/bbbam/jupyter_notebook/ResNet_min_max_scaling_PL/ResNet_min_max_m3p6To14_Nodes_4.0/ResNet_min_max_m3p6To14_13_channel_massregressor_2025_05_11_10:29:04_GPUS_16/Models      --batch_size=320  --num_worker=8 --checkpoint_folder=ResNet_min_max_scaling_signal_test_PL/ResNet_min_max_m3p6To14    --n_test=-1 &
srun --unbuffered --export=ALL shifter python3 mass_validation_tau_multi_node_gpu_with_ieta_iphi_min_max_scaling_h5data.py --resblocks=3 --mass=14   --best_epoch=200 --final_model_dir=/global/cfs/cdirs/m4392/bbbam/jupyter_notebook/ResNet_min_max_scaling_PL/ResNet_min_max_m3p6To18_Nodes_4.0/ResNet_min_max_m3p6To18_v2_13_channel_massregressor_2025_05_18_00:03:56_GPUS_16/Models   --batch_size=320  --num_worker=8 --checkpoint_folder=ResNet_min_max_scaling_signal_test_PL/ResNet_min_max_m3p6To18    --n_test=-1 &
srun --unbuffered --export=ALL shifter python3 mass_validation_tau_multi_node_gpu_with_ieta_iphi_min_max_scaling_h5data.py --resblocks=3 --mass=14   --best_epoch=200 --final_model_dir=/global/cfs/cdirs/m4392/bbbam/jupyter_notebook/ResNet_min_max_scaling_PL/ResNet_min_max_m1p2To18_Nodes_4.0/ResNet_min_max_m1p2To18_13_channel_massregressor_2025_05_12_09:33:12_GPUS_16/Models      --batch_size=320  --num_worker=8 --checkpoint_folder=ResNet_min_max_scaling_signal_test_PL/ResNet_min_max_m1p2To18    --n_test=-1 &
srun --unbuffered --export=ALL shifter python3 mass_validation_tau_multi_node_gpu_with_ieta_iphi_min_max_scaling_h5data.py --resblocks=3 --mass=14   --best_epoch=200 --final_model_dir=/global/cfs/cdirs/m4392/bbbam/jupyter_notebook/ResNet_min_max_scaling_PL/ResNet_min_max_mN1p2To18_Nodes_4.0/ResNet_min_max_mN1p2To18_13_channel_massregressor_2025_05_12_09:33:12_GPUS_16/Models    --batch_size=320  --num_worker=8 --checkpoint_folder=ResNet_min_max_scaling_signal_test_PL/ResNet_min_max_mN1p2p6To18 --n_test=-1 &
srun --unbuffered --export=ALL shifter python3 mass_validation_tau_multi_node_gpu_with_ieta_iphi_min_max_scaling_h5data.py --resblocks=3 --mass=14   --best_epoch=200 --final_model_dir=/global/cfs/cdirs/m4392/bbbam/jupyter_notebook/ResNet_min_max_scaling_PL/ResNet_min_max_mN1p2To22_Nodes_4.0/ResNet_min_max_mN1p2To22_13_channel_massregressor_2025_05_14_02:15:53_GPUS_16/Models    --batch_size=320  --num_worker=8 --checkpoint_folder=ResNet_min_max_scaling_signal_test_PL/ResNet_min_max_mN1p2To22   --n_test=-1 &
wait
