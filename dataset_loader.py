import torch, h5py, random
from torch.utils.data import *
import pyarrow.parquet as pq
import math
'''mass transformation function: converted to network unit'''

def transform_y(y, m0_scale):
    return y/m0_scale

def inv_transform_y(y, m0_scale):
    return y*m0_scale

def transform_norm_y(y, mean, std):
    return (y - mean) / std

def inv_transform_norm_y(y, mean, std):
    return y * std + mean

''' data loder defination without ieta and iphi'''


class H5Dataset(Dataset):
    def __init__(self, file_path, indices):
        self.file_path = file_path
        self.indices = indices
        self.file = h5py.File(file_path, 'r')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        data = self.file['all_jet'][index]
        am = self.file['am'][index]
        return data, am

# with lazy loadings
class H5Dataset_(Dataset):
    def __init__(self, file_path, indices ):
        self.indices = indices
        self.file_path = file_path

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        with h5py.File(self.file_path, 'r') as file:
            data = file['all_jet'][idx]
            am = file['am'][idx]
            return data, am





## Efficient h5 data loading
class ChunkedSampler(Sampler):
    def __init__(self, data_source, chunk_size=3200, shuffle=False):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.num_chunks = len(data_source) // chunk_size
        self.indices = sorted(data_source)
        self.shuffle = shuffle

    def shuffle_indices(self):
        chunk_indices = [self.indices[i * self.chunk_size:(i + 1) * self.chunk_size] for i in range(self.num_chunks)]
        random.shuffle(chunk_indices)
        self.indices = [idx for chunk in chunk_indices for idx in chunk]

    def __iter__(self):
        if self.shuffle:
            self.shuffle_indices()
        return iter(self.indices)

    def __len__(self):
        return len(self.data_source)

class ChunkedDistributedSampler(Sampler):
    def __init__(self, data_source, chunk_size=3200, shuffle=False, num_replicas=None, rank=None):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.num_chunks = len(data_source) // chunk_size
        self.indices = sorted(data_source)
        self.shuffle = shuffle
        self.num_replicas = num_replicas if num_replicas is not None else torch.distributed.get_world_size()
        self.rank = rank if rank is not None else torch.distributed.get_rank()
        self.num_samples = int(math.ceil(len(self.indices) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def shuffle_indices(self):
        chunk_indices = [self.indices[i * self.chunk_size:(i + 1) * self.chunk_size] for i in range(self.num_chunks)]
        random.shuffle(chunk_indices)
        self.indices = [idx for chunk in chunk_indices for idx in chunk]

    def __iter__(self):
        if self.shuffle:
            self.shuffle_indices()

        # Ensure that all replicas have the same number of samples
        indices = self.indices + self.indices[:(self.total_size - len(self.indices))]
        assert len(indices) == self.total_size

        # Subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]

        return iter(indices)

    def __len__(self):
        return self.num_samples

class RegressionDataset(Dataset):
    def __init__(self, h5_path, transforms=None, preload_size=3200):
        self.h5_path = h5_path
        self.transforms = transforms
        self.preload_size = preload_size
        self.h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)
        self.data = self.h5_file['all_jet']
        self.labels = self.h5_file['am']
        self.ieta = self.h5_file['ieta']
        self.iphi = self.h5_file['iphi']
        self.dataset_size = self.data.shape[0]

        self.chunk_size = self.data.chunks

        self.preloaded_data = None
        self.preloaded_labels = None
        self.preload_start = -1

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        preload_start = (idx // self.preload_size) * self.preload_size
        if preload_start != self.preload_start:
            self.preload_start = preload_start
            preload_end = min(preload_start + self.preload_size, self.dataset_size)
            self.preloaded_data = self.data[preload_start:preload_end]
            self.preloaded_labels = self.labels[preload_start:preload_end]
            self.preloaded_ieta = self.ieta[preload_start:preload_end]
            self.preloaded_iphi = self.iphi[preload_start:preload_end]

        local_idx = idx - self.preload_start
        data = self.preloaded_data[local_idx]
        labels = self.preloaded_labels[local_idx]
        ieta = self.preloaded_ieta[local_idx]
        iphi = self.preloaded_iphi[local_idx]
        if self.transforms:
            data = self.transforms(data)
        return torch.from_numpy(data), torch.from_numpy(labels),torch.from_numpy(iphi),torch.from_numpy(ieta)

    def __del__(self):
        self.h5_file.close()
