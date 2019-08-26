
import torchaudio
import numpy as np
from torchaudio import transforms

class MFCCExtractor():

    def extract_features(self, file_name):
    
        audio, sample_rate = torchaudio.load(file_name) 
        mfccs = transforms.MFCC(sample_rate=sample_rate, n_mfcc=40)(audio)
        mfccsscaled = np.mean(mfccs.detach().numpy(), axis=0)    
        return mfccsscaled
