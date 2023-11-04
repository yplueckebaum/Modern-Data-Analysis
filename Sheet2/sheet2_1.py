#import scikitlearn
from sklearn.decomposition import NMF
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    data = sp.io.loadmat('olivettifaces.mat')
    set1 = data['faces']
    set1 = set1.transpose(1, 0)
 #   set1 = set1.reshape(400, 64,64)
    return set1




# X is shape (sequencelength, dimension)
# D is shape (sequencelength, number of components)
# A is shape (number of components, dimension)
def solveNMF(X, steps = 100, n_components = None):
    #TODO
    return

if __name__ == '__main__':
    set1 = load_data()
    solveNMF(set1, steps=10, n_components=10)
