import numpy as np
import os
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, data_file):
        self.data = np.load(data_file)
        
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        sound = self.data[idx, :, :, :]
        return torch.from_numpy(sound)

    def get_mustd(self):
        dim = self.data.shape[3]
        mean = np.mean(self.data.reshape(-1,dim), axis=0).astype(np.float32)
        std = np.std(self.data.reshape(-1,dim), axis=0).astype(np.float32)
        return mean, std
        

class TestDataset(Dataset):
    def __init__(self, data_file):
        self.data = np.load(data_file)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        sound = self.data[idx, :, :, :]
        return torch.from_numpy(sound)


if __name__ == '__main__':
    path = "D:/neural-processes/sound_dataset/"
    dirs = os.listdir( path )

    data={}
    for file in dirs:
        file_path = os.path.join(path, file)
        data[file] = np.load(file_path)
    data_all = np.array((list(data.values()))).astype(np.float32)

    