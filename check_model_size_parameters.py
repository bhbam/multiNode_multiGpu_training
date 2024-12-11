import numpy as np
import os, glob, random, time, sys, pickle, glob, h5py
import argparse
import pyarrow.parquet as pq
from torch_resnet_concat import *
import torch
from torch import distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# from regression_Models import *
from torch_resnet_concat import *

in_channels=13

model1 = ResNet(in_channels,3,[8,16,32,64])
num_params1 = sum(p.numel() for p in model1.parameters() if p.requires_grad)
print(f'Total number of parameters--> ResNet : {num_params1}')

model2 = ResNet_BN(in_channels,3,[8,16,32,64])
num_params2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
print(f'Total number of parameters--> ResNet_BN : {num_params2}')


model3 = ResNet_no_ieta_iphi(in_channels, 3, [8,16,32,64])
num_params3 = sum(p.numel() for p in model3.parameters() if p.requires_grad)
print(f'Total number of parameters--> ResNet_no_ieta_iphi : {num_params3}')






# # Load trained Model
# trained_model_path = glob.glob('/pscratch/sd/b/bbbam/ResNet_B3_ieta_iphi_Nodes_4.0/ResNet_B3_ieta_iphi_13_channel_massregressor_2024_11_27_07:09:50_GPUS_16/Models/*300*.pt')[0]
# old_checkpoint = torch.load(trained_model_path)
# model1.load_state_dict(old_checkpoint['model_state_dict'])
# # Set the model to evaluation mode (optional for inference)
# model1.eval()
# num_params = sum(p.numel() for p in tmodel1.parameters() if p.requires_grad)
# print(f'Total trainable parameters in the trained model: {num_params}')