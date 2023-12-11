import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


# generate 10 random R2 vectors

def find_MEB(S):
    # Find the minimum enclosing ball of a set of points in R^n
    # S: set of points
    # Return: center, radius

    # init one alpha for each datapoint
    alpha = cp.Variable(S.shape[0])
    prob = cp.Problem(cp.Minimize(alpha * S * alpha.T - cp.sum(alpha[i]) * S[i].T * S[i]),
                      [alpha >= 0,
                       cp.sum(alpha) == 1])
    prob.solve()

    center = prob.value

    return center, radius


def find_MEB_approx(S, eps):
    # Find the minimum enclosing ball of a set of points in R^n
    # S: set of points
    # eps: precision
    # Return: center, radius

    return center, radius


np.random.seed(42)

test = np.random.rand(10, 2)
