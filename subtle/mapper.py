import numpy as np
from subtle.function import morlet_cwt

class Mapper:
    def __init__(self, fs):
        self.fs=fs

    def get_spectrogram(self, X, omega=5, n_channels=50):
        assert isinstance(X, np.array), 'X should be numpy array'
        assert X.ndim==2, 'dimension of X should be 2'
        n_frames, n_features = X.shape

        cwts = []
        for i in range(num_features):
            x = X[:, i]
            cwt = morlet_cwt(x, fs=self.fs, omega=omega, n_channels=n_channels).T # [n_frames, n_channels]
            cwts.append(cwt)
        cwts = np.hstack(cwts) # [n_frames, n_channels * n_features]
        return cwts