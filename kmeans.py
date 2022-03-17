import torch
from sklearn.cluster import KMeans as KMeans_sk
import numpy

class KMeans(torch.nn.Module):
    def __init__(self, n_clusters, random_state=None):
        super().__init__()
        self.n_clusters = n_clusters
        self.kmeans = KMeans_sk(n_clusters=self.n_clusters, random_state=random_state)

    def fit(self, X):
        if torch.is_tensor(X):
            X = X.numpy()
        self.kmeans.fit(X)
        self.centroids = torch.from_numpy(self.kmeans.cluster_centers_).double()

    def forward(self, x):
        N = x.shape[0]
        distances = torch.cdist(x, self.centroids)**2
        top_val = torch.topk(distances, k=2, dim=1, largest=False).values
        fx = torch.diff(top_val).squeeze()
        return fx

    def decision(self, X):
        distances = torch.cdist(X, self.centroids)**2
        return distances.argmin(-1)
