import glob
import os
import shutil
import random
import json
import numpy as np
import h5py
import math
import argparse
from tqdm import tqdm

file ='/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_tauDR0p4_m1p2To17p2_dataset_2_unbaised_v2_normalised_combined.hd5'

data = h5py.File(f'{file}', 'r')
num_images = len(data['all_jet'])
print("Totao number of events :", num_images)
num_images = 100
batch_size = 20

for start_idx in tqdm(range(0, num_images, batch_size)):
    end_idx = min(start_idx + batch_size, num_images)
    images_batch = data["all_jet"][start_idx:end_idx, :, :, :]
    am_batch = data["am"][start_idx:end_idx, :]
    ieta_batch = data["ieta"][start_idx:end_idx, :]
    iphi_batch = data["iphi"][start_idx:end_idx, :]
    m0_batch = data["m0"][start_idx:end_idx, :]
    print("am:  ",am_batch)
    break
data.close()
