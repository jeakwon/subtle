import numpy as np
from sklearn.decomposition import PCA
from openTSNE import TSNE
from umap import UMAP
from typing import List

from subtle.module import morlet_cwt
from subtle.module import Phenograph

class Data:
    def __init__(self, X):
        self.X = X

class Mapper:
    def __init__(self, fs, embedding_method='umap', n_pca=100, n_train_frames=120000):
        self.fs=fs
        self.trained=False
        self.n_train_frames = n_train_frames

        self.pca = PCA(n_pca)
        self.embedding = UMAP()

    def train(self, Xs):
        dataset = [Data(X) for X in Xs]

        for data in dataset:
            data.spectrogram = self.get_spectrogram(X)

        S = np.concatenate([data.spectrogram for data in dataset])
        S = np.random.permutation(S)[:self.n_train_frames]
        PC = self.pca.fit_transform(S)
        Z = self.embedding.fit_transform(PC)
        y = self.subcluster.fit_predict(Z)

        for X in Xs:
            data = Data()
            data.X = X
            data.spectrogram = self.get_spectrogram(X)
                
            S = np.concatenate([data.S for data in dataset])
        S = np.random.permutation(S)[:self.maxN]
            data.pc = self.pca.fit_transform(data.spectrogram)

            outputs.append(data)

    def get_spectrogram(self, X, omega=5, n_channels=50):
        assert isinstance(X, np.ndarray), 'X should be numpy array'
        assert X.ndim==2, 'dimension of X should be 2'
        
        n_frames, n_features = X.shape
        
        cwts = []
        for i in range(n_features):
            x = X[:, i]
            cwt = morlet_cwt(x, fs=self.fs, omega=omega, n_channels=n_channels).T # [n_frames, n_channels]
            cwts.append(cwt)
        cwts = np.hstack(cwts) # [n_frames, n_channels * n_features]
        return cwts