
from sklearn.linear_model import OrthogonalMatchingPursuit
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orth

def load_data():
    data = sp.io.loadmat('signal.mat')
    signal = data['x'][:, 0]
    D = data["D"]
    D2 = data["D2"]
    return signal, D, D2

def reconstruct(D, x):
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=3)
    omp.fit(D, x)
    a = omp.coef_
    x_r = D @ a
    return x_r

def compressed_sensing(D, x, error_list):
    np.random.seed(0)
    # initializing uniformly distributed random vectors
    phi_ext = np.random.uniform(-1, 1, (100, 100))
    # orthonormalizing
    phi_ext = orth(phi_ext)
    for m in range(1, 11):
        # reduced measurement basis
        col_idx = np.random.choice(100, m, replace=False)
        phi = phi_ext[:, col_idx]
        # representation of signal in measurement basis
        y = phi.T @ x
        D3 = phi.T @ D
        # find coefficients using OMP
        x_r = reconstruct(D3, y)
        # L2 norm for error
        err = np.linalg.norm(y - x_r)
        error_list.append(err)
    return error_list

def visualize(error_list):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), error_list, marker='o')
    plt.xlabel('Number of Measurements (m)')
    plt.ylabel('Reconstruction Error')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    x, D, D2 = load_data()
    # reconstruct signal via OMP
    x_r_D = reconstruct(D, x)
    x_r_D2 = reconstruct(D2, x)
    # calculate error via L1 norm
    error_D = np.sum(np.abs(x - x_r_D))
    error_D2 = np.sum(np.abs(x - x_r_D2))
    print(f"error for D: {error_D}, for D2: {error_D2}")
    # Compressed Sensing
    error_list = []
    error_list = compressed_sensing(D, x, error_list)
    visualize(error_list)
    # the error is minimal for m >= 7



