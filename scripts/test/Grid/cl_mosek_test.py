from mosek.fusion import *
import numpy as np
import os
import sys

from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix

from scripts.utils.log_utils import logger

def main(args):
    A = [[3.0, 1.0, 2.0, 0.0],
         [2.0, 1.0, 3.0, 1.0],
         [0.0, 2.0, 0.0, 3.0]]
    c = [3.0, 1.0, 5.0, 1.0]

    A = np.array(A)

    rows = []
    cols = []
    values = []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i,j] > 0:
                rows.append(i)
                cols.append(j)
                values.append(A[i,j])

    # Create a model with the name 'lo1'
    with Model("lo1") as M:

        # Create variable 'x' of length 4
        x = M.variable("x", 4, Domain.greaterThan(0.0))

        # Create constraints
        M.constraint(x.index(1), Domain.lessThan(10.0))
        # M.constraint("c1", Expr.dot(A[0], x), Domain.lessThan(25.0))
        # M.constraint("c2", Expr.dot(A[1], x), Domain.lessThan(25.0))
        # M.constraint("c3", Expr.dot(A[2], x), Domain.lessThan(25.0))
        B = Matrix.sparse(3, 4, rows, cols, values)
        C = None
        M.constraint("c1", Expr.mul(B, x), Domain.lessThan(25.0))

        # Set the objective function to (c^t * x)
        M.objective("obj", ObjectiveSense.Maximize, Expr.dot(c, x))

        # Solve the problem
        M.solve()

        # Get the solution values
        sol = x.level()
        print(sol)

def mosek_for_lap():
    embed_X = np.random.rand(4,2)
    num = 4
    N_sqrt = 2
    grid = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / N_sqrt, N_sqrt),
                                           np.linspace(0, 1 - 1.0 / N_sqrt, N_sqrt))) \
            .reshape(-1, 2)
    cost_matrix = cdist(grid, embed_X, "euclidean")
    sparse_cost_matrix = csr_matrix(cost_matrix)
    data = sparse_cost_matrix.data
    indptr = sparse_cost_matrix.indptr
    indices = sparse_cost_matrix.indices
    variable_indices = list(range(len(indices)))
    logger.debug("knn construction is finished, variables num: {}".format(len(data)))
    M = Model("lo1")
    # x = M.variable("x", len(data), Domain.greaterThan(0)); logger.info(">=0")
    x = M.variable("x", len(data), Domain.binary())
    logger.info("binary")
    A = []
    C = []
    hole_num, radish_num = sparse_cost_matrix.shape
    for i in range(hole_num):
        idx = variable_indices[indptr[i]:indptr[i + 1]]
        res = [[i, j, 1] for j in idx]
        # A = A + res
        A.append(res)

    logger.info("constructing A, A length: {}"
                .format(len(A)))

    for i in range(radish_num):
        idx = [variable_indices[indices[indptr[j]:indptr[j + 1]].tolist().index(i) + indptr[j]]
               for j in range(hole_num) if i in indices[indptr[j]:indptr[j + 1]]]
        res = [[i, j, 1] for j in idx]
        # C = C + res
        C.append(res)
    logger.info("constructing C, C length: {}".format(len(C)))
    A = [x for j in A for x in j]
    C = [x for j in C for x in j]
    A = list(zip(*A))
    C = list(zip(*C))
    logger.info("finished python's convert sparse matrix")
    A = Matrix.sparse(hole_num, len(data), list(A[0]), list(A[1]), list(A[2]))
    C = Matrix.sparse(radish_num, len(data), list(C[0]), list(C[1]), list(C[2]))
    logger.info("finished mosek's convert sparse matrix")

    logger.info("adding constraints")
    M.constraint(Expr.mul(A, x), Domain.lessThan(1))
    M.constraint(Expr.mul(C, x), Domain.equalsTo(1))

    M.objective("obj", ObjectiveSense.Minimize, Expr.dot(data, x))
    M.setLogHandler(sys.stdout)
    logger.info("begin solving")
    M.solve()
    logger.info(x.level())

    from lapjv import lapjv
    rowes, coles, _ = lapjv(cost_matrix)
    print(rowes)

if __name__ == '__main__':
    # main("a")
    mosek_for_lap()