import numpy as np
import torch as tr
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix

cmap = cm.get_cmap('tab20', 20)

def plot_explanation(x, R, feature_names=None, vlim=None):
    if feature_names is None:
        feature_names = list(range(len(x)))
    else:
        feature_names = [fn.replace('_',' ') for fn in feature_names]
    plt.figure(figsize=(4,2))
    plt.subplot(121)
    plt.title('data point')
    plt.gca().set_axisbelow(True), plt.grid(linestyle='dashed')
    negative = x.clamp(max=0)
    positive = x.clamp(min=0)
    plt.barh(range(len(feature_names)), negative, color='c')
    plt.barh(range(len(feature_names)), positive, color='m')
    plt.yticks(range(len(feature_names)), feature_names)
    plt.gca().invert_yaxis()
    plt.xlim(0,1.1)

    plt.subplot(122)
    plt.title('feature relevance')
    plt.gca().set_axisbelow(True), plt.grid(linestyle='dashed')
    negative = R.clamp(max=0)
    positive = R.clamp(min=0)
    if vlim is None:
        vlim = max(abs(negative).max(), positive.max()) + .3
    plt.barh(range(len(feature_names)), negative, color='b')
    plt.barh(range(len(feature_names)), positive, color='r')
    plt.xlim(-vlim, vlim)
    plt.yticks(range(len(feature_names)),[]*len(feature_names))
    plt.gca().invert_yaxis()
