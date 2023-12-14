import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

def find_MEB(S):
    # Find the minimum enclosing ball of a set of points in R^n
    # S: set of points
    # Return: center, radius

    # init one alpha for each datapoint
    alpha = cp.Variable(S.shape[1])

    # using quadratic program from cp
    # see doc : https://www.cvxpy.org/examples/basic/quadratic_program.html
    objective = cp.Minimize(cp.quad_form(alpha, (S.T @ S))
                            - cp.sum(cp.multiply(alpha, (S.T @ S).diagonal())))

    prob = cp.Problem(objective,
                      [alpha >= 0,
                       cp.sum(alpha) == 1])

    # the solution will be stored in alpha.value
    prob.solve()

    center = S @ alpha.value # or S.T ?

    radius = (alpha.value.T @ (S.T @ S) @ alpha.value
              - cp.sum([alpha.value[i] * (S[i] @ S[i]) for i in range(S.shape[1])]))
    radius = cp.sqrt(radius)

    return center, radius


def find_MEB_approx(S, eps):
    # Find the minimum enclosing ball of a set of points in R^n
    # S: set of points
    # eps: precision
    # Return: center, radius

    
    #return center, radius



# generate 10 random R2 vectors
np.random.seed(42)
test_set = np.random.rand(10, 2)

center, radius = find_MEB(test_set)
print(f"center: {center}, radius: {radius}")
#find_MEB_approx(test_set, 0.1)