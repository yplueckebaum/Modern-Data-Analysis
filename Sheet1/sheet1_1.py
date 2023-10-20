#import scikitlearn
import matplotlib.pyplot
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    data = sp.io.loadmat('dataset1.mat')
    set1 = data['set1a']
    set2 = data['set1b']
    return set1, set2


if __name__ == '__main__':
    set1,set2 = load_data()
    fig,axes = plt.subplots(2,2)
    axes[0,0].scatter(set1.T[0],set1.T[1])
    axes[1,0].scatter(set2.T[0],set2.T[1])
    whiten_data = PCA(whiten=True,n_components=2) # both 2 dim?
    set1_w = whiten_data.fit_transform(set1)
    set2_w = whiten_data.fit_transform(set2)
    axes[0,1].scatter(set1_w.T[0],set1_w.T[1])
    axes[1,1].scatter(set2_w.T[0],set2_w.T[1])


    fig.show()

    # write your code here
