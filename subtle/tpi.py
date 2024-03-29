import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans

def transition_matrix(transitions, states):
    state2index = {k:i for i, k in enumerate(states)}
    
    transitions = list(map(lambda x: state2index[x], transitions))
    n = len(states)
    
    M = np.zeros(shape=(n, n))

    for (i,j) in zip(transitions, transitions[1:]):
        M[i, j] += 1

    return pd.DataFrame(data=M, columns=states, index=states)

def temporal_proximity_index(transition_matrix, centroids):
    d = pairwise_distances(centroids, centroids)+1e-12
    d_inv = 1/d
    np.fill_diagonal(d_inv, -np.inf)
    np.fill_diagonal(transition_matrix, 0)

    w = softmax(d_inv, axis=1)
    p = normalize(transition_matrix, norm='l1', axis=1)
    np.fill_diagonal(p, 0)
    np.fill_diagonal(w, 0)    

    return (w*p).sum()

def temporal_connectivity(embeddings, ks=[2,4,8,16,32,64,128,256], seed=None):
    tpis = []
    for k in ks:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init='auto')
        labels = kmeans.fit_predict(embeddings)
        tm = transition_matrix(labels, range(k))
        transition_probability = normalize(tm, norm='l1', axis=1)
        centroids = kmeans.cluster_centers_
        tpi = temporal_proximity_index(transition_probability, centroids)
        tpis.append(tpi)

    return tpis
