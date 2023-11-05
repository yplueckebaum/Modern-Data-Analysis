import matplotlib.pyplot
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy as sp
import numpy as np


def load_data():
    data = sp.io.loadmat('dataset1.mat')
    set1 = data['set1a']
    set2 = data['set1b']
    return set1, set2


if __name__ == '__main__':
    set1, set2 = load_data()
    # write your code here
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].scatter(set1.T[0], set1.T[1])
    axes[0, 0].set_title('Set 1 - Original')
    axes[1, 0].scatter(set2.T[0], set2.T[1])
    axes[1, 0].set_title('Set 2 - Original')
    whiten_data = PCA(whiten=True, n_components=2)  # both 2 dim?
    set1_w = whiten_data.fit_transform(set1)
    whiten_data = PCA(whiten=True, n_components=2)
    set2_w = whiten_data.fit_transform(set2)
    axes[0, 1].scatter(set1_w.T[0], set1_w.T[1])
    axes[0, 1].set_title('Set 1 - Whitened')
    axes[1, 1].scatter(set2_w.T[0], set2_w.T[1])
    axes[1, 1].set_title('Set 2 - Whitened')

    plt.tight_layout()
    fig.show()

    # What happens?
    # The data gets centered around the origin and the variance becomes the same in all directions (unit).
    # The information that gets lost is the correlation between the two dimensions, as well as the information
    # about the original variance of the data in the two dimensions and the original offset from the origin.

    # Print variance and mean
    print('Set 1 - Original')
    print('Mean: ', np.mean(set1, axis=0))
    print('Variance: ', np.var(set1, axis=0))
    print('Set 1 - Whitened')
    print('Mean: ', np.mean(set1_w, axis=0))
    print('Variance: ', np.var(set1_w, axis=0))
    print('Set 2 - Original')
    print('Mean: ', np.mean(set2, axis=0))
    print('Variance: ', np.var(set2, axis=0))
    print('Set 2 - Whitened')
    print('Mean: ', np.mean(set2_w, axis=0))
    print('Variance: ', np.var(set2_w, axis=0))
