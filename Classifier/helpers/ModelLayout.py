import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class AudioVisionModel(nn.Module):
    num_rows = 14
    num_columns = 174
    num_channels = 1

    def __init__(self):
        super(AudioVisionModel, self).__init__()
        self.seq1 = nn.Sequential()
        self.conv1 = nn.Conv2d(1, 14, 2)
        self.conv2 = nn.Conv2d(14, 28, 2)
        self.conv3 = nn.Conv2d(28, 56, 2)
        self.conv4 = nn.Conv2d(56, 112, 2)

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

    def forward(self, x):

        out1 = self.drop1(self.maxpool1(F.relu(self.conv1(x))))
        out2 = self.drop2(self.maxpool2(F.relu(self.conv2(out1))))
        # out3 = self.drop3(self.maxpool3(F.relu(self.conv3(out2))))
        # out4 = self.drop4(self.maxpool4(F.relu(self.conv4(out3))))
        return (self.dense(self.avgPool(out2)))