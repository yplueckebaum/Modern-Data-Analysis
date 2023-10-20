#import scikitlearn
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
    # write your code here
