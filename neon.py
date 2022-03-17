import torch
import numpy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# soft minpooling layer
def smin(X, s, dim=-1):
    return -(1/s)*torch.logsumexp(-s*X, dim=dim) + (1/s)*numpy.log(X.shape[dim])

# soft maxpooling layer
def smax(X, s, dim=-1):
    return (1/s)*torch.logsumexp(s*X, dim=dim) - (1/s)*numpy.log(X.shape[dim])

class NeuralizedKMeans(torch.nn.Module):
    def __init__(self, kmeans):
        super().__init__()
        self.n_clusters = kmeans.n_clusters
        self.kmeans = kmeans
        K, D = kmeans.centroids.shape
        self.W = torch.empty(K, K-1, D, dtype=torch.double)
        self.b = torch.empty(K, K-1, dtype=torch.double)
        for c in range(K):
            for kk in range(K-1):
                k = kk if kk < c else kk + 1
                self.W[c, kk] = 2*(kmeans.centroids[c] - kmeans.centroids[k])
                self.b[c, kk] = (torch.norm(kmeans.centroids[k])**2 -
                            torch.norm(kmeans.centroids[c])**2)

    def h(self, X):
        z = torch.einsum('ckd,nd->nck', self.W, X) + self.b
        return z

    def forward(self, X, c=None):
        h = self.h(X)
        out = h.min(-1).values
        if c is None:
            return out.max(-1).values
        else:
            return out[:,c]

def inc(z, eps=1e-9):
    return z + eps*(2*(z >= 0) - 1)

def beta_heuristic(model, X):
    fc = model(X)
    return 1/fc.mean()

def neon(model, X, beta):
    R = torch.zeros_like(X)
    if not torch.is_tensor(beta):
        beta = torch.tensor(beta)
    for i in range(X.shape[0]):
        x = X[[i]]
        ### forward
        h = model.h(x)
        out = h.min(-1).values
        c = out.argmax()
        ### backward
        pk = torch.nn.functional.softmin(beta*h[:,c], dim=-1)
        Rk = out[:,c] * pk
        knc = [k for k in range(model.n_clusters) if k!=c]
        Z = model.W[c]*(x - .5*(model.kmeans.centroids[[c]] + model.kmeans.centroids[knc]))
        Z = Z / inc(Z.sum(-1, keepdims=True))
        R[i] = (Z * Rk.view(-1,1)).sum(0)
    return R
