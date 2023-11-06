import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sklearn.decomposition


def load_data():
    data = sp.io.loadmat('olivettifaces.mat')
    set1 = data['faces']
    set1 = set1.transpose(1, 0)
    #   set1 = set1.reshape(400, 64,64)
    return set1


def get_image_from_data(data, index):
    return data[index].reshape(64, 64).T


def stitch_images(images, rows, cols, res):
    image = np.zeros((rows * res, cols * res))
    for i in range(rows):
        for j in range(cols):
            image[i * res:(i + 1) * res, j * res:(j + 1) * res] = images[i * cols + j]
    return image


# X is shape (sequencelength, dimension)
# D is shape (sequencelength, number of components)
# A is shape (number of components, dimension)
def solveNMF(X, steps=100, n_components=None):
    if n_components is None:
        n_components = X.shape[0]
    # initialize D and A
    D = np.random.rand(X.shape[0], n_components)
    A = np.random.rand(n_components, X.shape[1])
    errors = []
    # iterate
    for i in range(steps):
        # update A
        A = np.linalg.inv(D.T.dot(D)).dot(D.T).dot(X)
        A[A < 0] = 0
        # update D
        D = np.linalg.inv(A.dot(A.T)).dot(A).dot(X.T).T
        D[D < 0] = 0
        # print error (L1)
        errors.append(np.abs((X - D.dot(A))).mean())
        print(errors[-1])
    return D, A, errors


if __name__ == '__main__':
    set1 = load_data()

    plt.imshow(get_image_from_data(set1, 0), cmap='gray')
    plt.show()
    n_components = 10
    D, A, errors = solveNMF(set1, steps=10, n_components=n_components)
    # plot primitives
    plt.imshow(stitch_images([get_image_from_data(A, i) for i in range(n_components)], 2, 5, 64), cmap='gray')
    plt.show()
    # plot error
    plt.plot(errors)
    plt.show()
    # plot reconstruction
    index = 3
    plt.imshow(stitch_images([get_image_from_data(set1, index), get_image_from_data(D.dot(A), index)], 1, 2, 64), cmap='gray')
    plt.show()

    # b)
    nmf = sk.decomposition.NMF(alpha_W=1,l1_ratio=1,verbose=1,tol=0.05,n_components=n_components)
    solution_NMF = nmf.fit_transform(set1)

    plt.imshow(stitch_images([get_image_from_data(nmf.components_, i) for i in range(n_components)], 2, 5, 64), cmap='gray')
    plt.show()
    plt.imshow(stitch_images([get_image_from_data(set1, index),get_image_from_data(solution_NMF.dot(nmf.components_),index)],1,2,64),cmap='gray')
    plt.show()
    # discussion:
    # directionality of face is lost, both models imagine glasses and  a beard and sparsity loss leads to only three real components




