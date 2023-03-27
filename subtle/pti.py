

def proximal_transition_index(transition_matrix, centroids):
    d = pairwise_distances(centroids, centroids)+1e-12
    d_inv = 1/d
    np.fill_diagonal(d_inv, -np.inf)
    np.fill_diagonal(transition_matrix, 0)

    w = softmax(d_inv, axis=1)
    p = normalize(transition_matrix, norm='l1', axis=1)
    np.fill_diagonal(p, 0)
    np.fill_diagonal(w, 0)    

    return (w*p).sum()

def temporal_connectivity(embeddings, ks=[2,4,8,16,32,64,128, 256], seed=None):
    ptis = []
    for k in ks:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=seed)
        labels = kmeans.fit_predict(embeddings)
        transition_matrix = kn.utils.transition_matrix(labels, range(k))
        transition_probability = normalize(transition_matrix, norm='l1', axis=1)
        centroids = kmeans.cluster_centers_
        pti = proximal_transition_index(transition_probability, centroids)
        ptis.append(pti)

    return ptis