import torch
from torch.utils import data
import pandas as pd
from ast import literal_eval

class AudioVisionDataset(data.Dataset):

    def __init__(self, filename):
        featuresdf = pd.read_csv(filename)
        self.data = featuresdf['feature'].apply(literal_eval)
        self.target = featuresdf['siren_present']
        self.n_samples = self.data.shape[0]
    
    def __len__(self):   # Length of the dataset.
        return self.n_samples
    
    def __getitem__(self, index):   # Function that returns one point and one label.
        return torch.Tensor(self.data[index]), torch.Tensor(self.target[index])