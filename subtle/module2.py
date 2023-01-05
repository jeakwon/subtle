import numpy as np
from scipy import signal
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.cluster import contingency_matrix
import phenograph

def morlet_cwt(x, fs, omega, n_channels):
    f_nyquist = fs/2
    freq = np.linspace(f_nyquist/10, f_nyquist, n_channels)
    widths = omega*fs/(2*freq*np.pi)
    return np.abs(signal.cwt(x, signal.morlet2, widths, w=omega))

class Data:
    def __init__(self, X):
        self.X = X

    def __repr__(self):
        return 'SUBTLE Data'

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



# matlab -> python
# https://github.com/bermanlabemory/behavioral-evolution

def deterministicInformationBottleneck(pXY, k, f0=None, beta=1, tol=1e-6, maxIter=1000):
    if isinstance(pXY, list):
        a = np.unique(pXY[0])
        b = np.unique(pXY[1])
        pXY = np.histogram2d(pXY[0], pXY[1], bins=(a,b))[0]
    pXY = pXY / np.sum(pXY)
    pX = np.sum(pXY, axis=0)
    pY_X = pXY / pX
    
    s = pXY.shape
    N = s[0]
    M = s[1]
    
    if f0 is None:
        f = np.random.randint(k, size=N)
    else:
        f = f0
    
    pT = np.zeros(k)
    pY_T = np.zeros((k,M))
    
    for i in range(k):
        pT[i] = np.sum(pX[f==i])
    idx = pT > 0
    HT = -np.sum(pT[idx]*np.log2(pT[idx]))
    
    for i in range(k):
        if pT[i] > 0:
            pY_T[i,:] = np.sum(pXY[f==i,:], axis=0) / pT[i]
        else:
            pY_T[i,:] = 0
    pYT = pY_T * pT[:, np.newaxis]
    pY = np.sum(pYT)
    temp = pYT * np.log2(pYT / (pT[:, None]*pY))
    IYT = np.sum(temp[~np.isnan(temp) & ~np.isinf(temp)])
    
    n = 1
    while True:
        previousJ = HT - beta*IYT
        
        DKLs = findListKLDivergences(pY_X.T, pY_T)
        fMat = np.subtract(np.log2(pT), beta*DKLs[0])
        
        f = np.argmax(fMat, axis=1)
        
        for i in range(k):
            pT[i] = np.sum(pX[f==i])
        idx = pT > 0
        HT = -np.sum(pT[idx]*np.log2(pT[idx]))
        
        for i in range(k):
            if pT[i] > 0:
                pY_T[i,:] = np.sum(pXY[f==i,:], axis=0) / pT[i]
            else:
                pY_T[i,:] = 0
        pYT = pY_T*pT[:, np.newaxis]
        pY = np.sum(pYT, axis=0)
        temp = pYT * np.log2(pYT / (pT[:, None]*pY))
        IYT = np.sum(temp[~np.isnan(temp) & ~np.isinf(temp)])

        J = HT - beta*IYT

        if abs(J-previousJ) < tol or n >= maxIter:
            break
        else:
            n = n + 1

    vals = np.unique(f)
    if len(vals) < k:
        for i in range(len(vals)):
            f[f == vals[i]] = i
        pY_T = pY_T[vals,:]
        pT = pT[vals]

    return f, IYT, HT, pY_T, pT

def findListKLDivergences(data, data2):
    logData = np.log2(data)
    logData[np.isnan(logData) | np.isinf(logData)] = 0
    
    entropies = -np.sum(np.multiply(data, logData), axis=1)
    logData2 = np.log2(data2)
    logData2[np.isnan(logData2) | np.isinf(logData2)] = 0
    
    D = - np.dot(data, logData2.T)
    
    D = D - entropies[:, None]
    
    return D, entropies

def findParetoFront(X):
    d = len(X[0,:])
    N = len(X[:,0])
    
    idx = np.zeros(N, dtype=bool)
    temp = np.zeros((N,d), dtype=bool)
    for i in range(N):
        temp[:] = False
        for j in range(d):
            temp[:,j] = X[i,j] < X[:,j]
        
        if np.max(np.sum(temp, axis=1)) < d:
            idx[i] = True
    
    return idx

def run_DIB(X, Y, N=10000, minClusters=2, maxClusters=30, minLogBeta=-1, maxLogBeta=4, readout=100):
    
    betas = np.zeros(N)
    numClusters = np.zeros(N, dtype=int)
    clusterings = [None]*N
    IYTs = np.zeros(N)
    HTs = np.zeros(N)
    
    a = np.unique(X)
    b = np.unique(Y)
    pXY = np.histogram2d(X, Y, bins=(len(a),len(b)))[0]
    
    for i in range(N):
        betas[i] = 10**(minLogBeta + (maxLogBeta-minLogBeta)*np.random.rand())
        k = minClusters + np.random.randint(maxClusters - minClusters)
        clusterings[i],IYTs[i],HTs[i],_,_ = deterministicInformationBottleneck(pXY,k,None,betas[i],1e-6,1000)
        numClusters[i] = len(np.unique(clusterings[i]))
        if i%readout == 0:
            print ('Calculating for Iteration #%6i out of %6i' % (i,N))
    
    idx = findParetoFront(np.vstack((-HTs, IYTs)).T)
    clusterings = [clusterings[i] for i in np.where(idx)[0]]
    IYTs = IYTs[idx]
    HTs = HTs[idx]
    numClusters = numClusters[idx]
    
    sortIdx = np.argsort(IYTs)
    clusterings = [clusterings[i] for i in sortIdx]
    IYTs = IYTs[sortIdx]
    HTs = HTs[sortIdx]
    numClusters = numClusters[sortIdx]
    
    idx = np.hstack((0,np.where(np.diff(IYTs) > 1e-10)[0]+1))
    clusterings = [clusterings[i] for i in idx]
    IYTs = IYTs[idx]
    HTs = HTs[idx]
    numClusters = numClusters[idx]
    
    clusterValues = np.unique(numClusters)
    clusterChoices = np.zeros(len(numClusters), dtype=bool)
    for i in range(len(clusterValues)):
        idx = np.where(numClusters == clusterValues[i])[0][-1]
        clusterChoices[idx] = True
    supclusters = []
    for c in clusterings[clusterChoices]:
        if len(np.unique(c)) == 1:
            c_prev = c
        else:
            c = assign_cluster(c, c_prev)
        supcluster = {x:c[x] for x in np.unique(X)}
        supclusters.append(supcluster)
    return supclusters