
import torchaudio
import numpy as np
import librosa
from librosa import feature

class MFCCExtractor():

    def __init__(self):
        print("No. of features: 14")

    def extract_features(self, file_name):
    
        audio, sample_rate = librosa.load(file_name) 
        mfccs = feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return mfccs.T