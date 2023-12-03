import numpy
from sklearn.linear_model import OrthogonalMatchingPursuit
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orth

def load_data():
    data = sp.io.loadmat('image.mat')
    signal = data['y'][:, 0]
    phi = data["Phi"]
    return signal, phi

def reconstruct(D, x):
    omp = OrthogonalMatchingPursuit()
    omp.fit(D, x)
    a = omp.coef_
    print("|a|_0: ", np.count_nonzero(a))
    return a

def compressed_sensing(phi, y):
    D = np.identity(1024)
    D3 = phi.T @ D
    # find coefficients using OMP
    a = reconstruct(D3, y)
    x_r = phi.T @ D @ a
    x_r2 = D @ a
    err = np.linalg.norm(y - x_r)
    print("error: ", err)
    return x_r, x_r2

def display_signal(x, title):
    shape = (32, 32)
    img = x.reshape(shape)
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()

if __name__ == '__main__':
    y,phi = load_data()
    # reconstruct signal via compressed sensing
    # y ~ phi.T @ D @ a
    x_r, x_r2 = compressed_sensing(phi, y)
    # x ~ D @ a
    #x_r = np.linalg.pinv(phi.T) @ x_r
    title = "Reconstructed signal using Compressed Sensing"
    display_signal(x_r2, title)
    # reconstruct signal via pseudo inverse
    phi_pseudo_inv = np.linalg.pinv(phi.T)
    x_p = phi_pseudo_inv @ y
    title = "Reconstructed signal using the pseudo inverse"
    display_signal(x_p, title)

