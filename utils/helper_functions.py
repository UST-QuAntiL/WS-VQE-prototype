import numpy as np
import matplotlib.pyplot as plt

def generate_hermitian(n=2, a_range=(-5, 5), complex=True, sparsity=None):
    shape = (2**n, 2**n)

    # generate random matrix
    matrix = generate_matrix(shape, a_range, complex, sparsity)

    # make it hermitian
    for i in range(shape[0]):
        for j in range(shape[1]):
            if i == j:
                # values on diagonal must be real
                matrix[i,j] = np.real(matrix[i,j])
            if i > j:
                # values below diagonal must be conjugate of corresponding values above diagonal
                matrix[i,j] = np.conjugate(matrix[j,i])

    return matrix

def generate_matrix(shape=(2, 2), a_range=(-5, 5), complex=True, sparsity=None):
    # generate uniform random (complex) matrix

    matrix = np.random.uniform(*a_range, shape)
    if complex:
        matrix = matrix + 1.j * np.random.uniform(*a_range, shape)
    # apply sparsity
    if sparsity:
        for i in range(shape[0]):
            for j in range(shape[1]):
                matrix[i,j] = np.random.choice([0, matrix[i,j]], p=[sparsity, 1-sparsity])

    return np.matrix(matrix)


def power_method(matrix, num_iterations=10): # by power iteration
    # choose random initial vector
    bk = np.random.rand(matrix.shape[1]) + 1.j*np.random.rand(matrix.shape[1])
    bk = np.matrix(bk).T

    for i in range(num_iterations):
        bk1 = np.dot(matrix, bk)
        bk = bk1 / np.linalg.norm(bk1)

    return bk

def inverse_power_method(matrix, shift=0, num_iterations=10):
    inv_matrix = np.linalg.inv(matrix-shift*np.identity(matrix.shape[0]))
    return power_method(inv_matrix, num_iterations)

def get_gershgorins(matrix, draw=False):
    centers, radii = [], []
    if draw:
        fig, ax = plt.subplots()
    for i in range(matrix.shape[0]):
        centers.append(matrix[i,i])
        radii.append(np.sum([ np.abs(matrix[i,j]) if j is not i else 0 for j in range(matrix.shape[0]) ]))
        if draw:
            ax.add_patch(plt.Circle((np.real(centers[i]), np.imag(centers[i])), radii[i], fill=False))

    if draw:
        ax.set_xlim(min(np.subtract(np.real(centers), radii)), np.max(np.add(np.real(centers), radii)))
        ax.set_ylim(min(np.subtract(np.imag(centers), radii)), np.max(np.add(np.imag(centers), radii)))
        plt.show()
    return centers, radii

def get_gershgorin_extrema(matrix):
    centers, radii = get_gershgorins(matrix, draw=False)
    min_bound = min(np.subtract(np.real(centers), radii))
    max_bound = max(np.add(np.real(centers), radii))
    return min_bound, max_bound
