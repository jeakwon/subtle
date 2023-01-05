import numpy as np
from scipy import signal

def morlet_cwt(x, fs, omega, n_channels):
    f_nyquist = fs/2
    freq = np.linspace(f_nyquist/10, f_nyquist, n_channels)
    widths = omega*fs/(2*freq*np.pi)
    return np.abs(signal.cwt(x, signal.morlet2, widths, w=omega))

import phenograph
from sklearn.neighbors import KNeighborsClassifier

class Phenograph:
    def __init__(
        self, 
        clustering_algo= "leiden",
        k=30,
        directed=False,
        prune=False,
        min_cluster_size=10,
        jaccard=True,
        primary_metric="euclidean",
        **kwargs,
    ):
        self.clustering_algo=clustering_algo
        self.k=k
        self.directed=directed
        self.prune=prune
        self.min_cluster_size=min_cluster_size
        self.jaccard=jaccard
        self.primary_metric=primary_metric
        self.kwargs=kwargs
    
    def fit(self, X, y=None):
        return self._fit(X)
    
    def predict(self, X):
        return self._predict(X)

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)
        
    def _fit(self, X):
        communities, graph, Q = phenograph.cluster(
            X,
            clustering_algo=self.clustering_algo,
            k=self.k,
            directed=self.directed,
            prune=self.prune,
            min_cluster_size=self.min_cluster_size,
            jaccard=self.jaccard,
            primary_metric=self.primary_metric,
            **self.kwargs
            )
        self.neigh = KNeighborsClassifier(n_neighbors=1).fit(X, communities)
        self.labels = np.unique(communities)
        return self
    
    def _predict(self, X):
        return self.neigh.predict(X)