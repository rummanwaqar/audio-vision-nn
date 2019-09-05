import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class AudioVisionModel(nn.Module):
    num_rows = 14
    num_columns = 174
    num_channels = 1

    def __init__(self):
        super(AudioVisionModel, self).__init__()

        self.conv1 = nn.Conv1d(1, 14, 1)
        self.conv2 = nn.Conv1d(14, 28, 1)

        self.drop1 = nn.Dropout(p=0.2)

        self.maxpool1 = nn.MaxPool1d(2)
        self.maxpool2 = nn.MaxPool1d(2)
        self.maxpool3 = nn.MaxPool1d(2)

        self.dense1 = nn.Linear(28*3, 64)
        self.dense2 = nn.Linear(64,2)
        # self.dense3 = nn.Softmax(2)

    def forward(self, x):
        print(x.size())

        out1 = self.maxpool1(F.relu(self.conv1(x)))
        out2 = self.maxpool2(F.relu(self.conv2(out1)))

        out2 = out2.view(-1, 28*3 )
        # out3 = self.maxpool3(F.relu(self.conv3(out2)))
        out4 = self.dense2(self.drop1(F.relu(self.dense1(out2))))
        return out4