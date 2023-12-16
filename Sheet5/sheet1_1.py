import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


def find_MEB(S):
    # Find the minimum enclosing ball of a set of points in R^n
    # S: set of points
    # Return: center, radius

    # init one alpha for each datapoint
    alpha = cp.Variable(S.shape[0])

    # using quadratic program from cp
    # see doc : https://www.cvxpy.org/examples/basic/quadratic_program.html
    objective = cp.Minimize(cp.quad_form(alpha, (S @ S.T), assume_PSD=True)
                            - cp.sum(cp.multiply(alpha, (S @ S.T).diagonal())))

    prob = cp.Problem(objective,
                      [alpha >= 0,
                       cp.sum(alpha) == 1])

    # the solution will be stored in alpha.value
    prob.solve()

    center = S.T @ alpha.value
    # Equivalent
    center2 = np.sum(np.array([alpha.value[i] * S[i] for i in range(len(S))]), axis=0)
    radius = cp.abs(cp.quad_form(alpha, (S @ S.T))
                    - cp.sum(cp.multiply(alpha, (S @ S.T).diagonal())))
    radius = cp.sqrt(radius)
    # Equivalent
    radius2 = np.sqrt(np.sum(np.array([alpha.value[i] * np.square(np.linalg.norm(center2 - S[i])) for i in range(len(S))])))
    assert np.allclose(center, center2)
    assert np.allclose(np.array(radius.value), np.array(radius2))
    return center, radius.value


def find_MEB_approx(S, eps, visualize=True):
    # Find the minimum enclosing ball of a set of points in R^n
    # S: set of points
    # eps: precision
    # Return: center, radius

    # choose random point
    s = S[np.random.randint(0, S.shape[0])]
    # choose furthest point from s
    # apparently square.. why?
    d = np.square(np.linalg.norm(S - s, axis=1))
    s2 = S[np.argmax(d)]
    assert s2 is not s
    s = [s, s2]
    while True:
        center, radius = find_MEB(np.array(s))
        if visualize:
            # plot the points and the circle
            fig, ax = plt.subplots()
            # quadratic aspect ratio
            ax.set_aspect(1)
            # plot rest of points
            ax.scatter(S[:, 0], S[:, 1], s=1, color='b')
            # plot current core set
            ax.scatter(np.array(s)[:, 0], np.array(s)[:, 1], s=1, color='r')
            # plot current circle
            circle = plt.Circle(center, radius, color='r', fill=False, lw=0.3)
            ax.add_artist(circle)
            # plot extended circle
            circle = plt.Circle(center, radius * (1 + eps), color='g', fill=False, lw=0.3)
            ax.add_artist(circle)
            # Zoom accordingly
            #ax.set_xlim(center[0] - radius * (1 + eps) * (1 + eps), center[0] + radius * (1 + eps) * (1 + eps))
            #ax.set_ylim(center[1] - radius * (1 + eps) * (1 + eps), center[1] + radius * (1 + eps) * (1 + eps))
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            plt.show()

        distances = np.linalg.norm(S - center, axis=1)
        if any(distances > radius * (1 + eps)):
            # add furthest point
            s.append(S[np.argmax(distances)])
        else:
            break

    return center, radius


# generate 10 random R2 vectors
np.random.seed(42)
test_set = np.random.randn(10, 2)

center, radius = find_MEB(test_set)
print(f"center: {center}, radius: {radius}")

# plot the points and the circle
fig, ax = plt.subplots()
# quadratic aspect ratio
ax.set_aspect(1)

ax.scatter(test_set[:, 0], test_set[:, 1], s=1, color='b')
circle = plt.Circle(center, radius, color='r', fill=False, lw=0.3)
ax.add_artist(circle)
# Zoom accordingly
ax.set_xlim(center[0] - radius * 1.2, center[0] + radius * 1.2)
ax.set_ylim(center[1] - radius * 1.2, center[1] + radius * 1.2)
plt.show()

test_set = np.random.randn(1000, 2)

center, radius = find_MEB_approx(test_set, 0.1)

# plot the points and the circle
fig, ax = plt.subplots()
# quadratic aspect ratio
ax.set_aspect(1)
ax.scatter(test_set[:, 0], test_set[:, 1], s=1, color='b')
circle = plt.Circle(center, radius, color='r', fill=False, lw=0.3)
ax.add_artist(circle)
# Zoom accordingly
#ax.set_xlim(center[0] - radius * 1.2, center[0] + radius * 1.2)
#ax.set_ylim(center[1] - radius * 1.2, center[1] + radius * 1.2)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
plt.show()
