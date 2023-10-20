#import scikitlearn
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import sksfa
import pickle


def load_data():
    with open('excersise2.pkl', 'rb') as f: savedict = pickle.load(f)
    mean = savedict['mean']
    components = savedict['components']
    video_pca = savedict['video_pca']
    return mean, components, video_pca


if __name__ == '__main__':
    mean, components, x = load_data()
    #write your code here
