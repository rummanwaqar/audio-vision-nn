import numpy as np
import torch.nn as nn
import torch.nn.functional as F



# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
# from keras.optimizers import Adam
# from keras.utils import np_utils
from sklearn import metrics 

class AudioVisionModel(nn.Module):
    num_rows = 14
    num_columns = 174
    num_channels = 1

     def __init__(self):
        super(Model, self).__init__()
        self.seq1 = nn.Sequential()
        self.conv1 = nn.Conv2d(1, 16, 2)
        self.conv2 = nn.Conv2d(16, 32, 2)
        self.conv3 = nn.Conv2d(32, 64, 2)
        self.conv4 = nn.Conv2d(64, 128, 2)

        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.2)
        self.drop4 = nn.Dropout(p=0.2)

        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.maxpool4 = nn.MaxPool2d(2)

        self.avgPool = nn.AvgPool2d(2)
        self.dense = nn.Softmax(2)

    def forward(self, input):
        x = self.drop1(self.maxpool1(F.relu(self.conv1(input)))
        x = self.drop2(self.maxpool2(F.relu(self.conv2(input)))
        x = self.drop3(self.maxpool3(F.relu(self.conv3(input)))
        x = self.drop4(self.maxpool4(F.relu(self.conv4(input)))
        return (self.dense(self.avgPool(x)))