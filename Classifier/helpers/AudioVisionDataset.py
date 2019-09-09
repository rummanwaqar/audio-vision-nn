import torch
from torch.utils import data
import pandas as pd
from ast import literal_eval
import numpy as np

class AudioVisionDataset(data.Dataset):

    dtype = torch.FloatTensor
    
    def __init__(self, filename, dtype):
        featuresdf = pd.read_csv(filename)
        self.data = featuresdf['feature']
        self.target = featuresdf['class_id']
        self.n_samples = self.data.shape[0]
        self.dtype = dtype
        
    def __len__(self):   # Length of the dataset.
        return self.n_samples
    
    def __getitem__(self, index):   # Function that returns one point and one label.
        temp = np.asarray(literal_eval(self.data[index]))
        feature = torch.from_numpy(temp).long().type(self.dtype)
        # print(self.target[index])
        val = [self.target[index].item()]
#         print(val)
        target = torch.LongTensor(val).type(torch.cuda.LongTensor)
        return feature, target