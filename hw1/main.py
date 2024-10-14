from math import sqrt


def parse(X, A, B):
    f = open("input.txt", 'r')
    lin = f.readline()
    nr_lin = 0
    while lin:
        coef = -1
        sig = 0
        read_vars = 0
        for c in lin:
            if c == '-':
                sig = 1
                continue
            if c == '+':
                sig = 0
                continue
            if '0' < c < '9':
                if coef == -1:
                    coef = 0
                coef = coef * 10 + int(c)
                continue
            if c in " =":
                continue
            if coef == -1:
                coef = 1
            if sig == 1:
                coef *= (-1)
            if c == '\n':
                B[nr_lin] = coef
                coef = -1
                sig = 0
                continue
            A[nr_lin][read_vars] = coef
            if nr_lin == 0:
                X[read_vars] = c
            read_vars += 1
            coef = -1
            sig = 0
        lin = f.readline()
        nr_lin += 1
    if sig == 1:
        coef *= (-1)
    B[nr_lin - 1] = coef


def determinant(X):
    if len(X) == 2:
        return X[0][0] * X[1][1] - X[0][1] * X[1][0]
    if len(X) != 3 and len(X[0]) != 3:
        print("The matrix is not 3x3 or 2x2.")
        return
    return X[0][0] * (X[1][1] * X[2][2] - X[1][2] * X[2][1]) \
        - X[0][1] * (X[1][0] * X[2][2] - X[1][2] * X[2][0]) \
        + X[0][2] * (X[1][0] * X[2][1] - X[1][1] * X[2][0])

def trace(X):
    if len(X) != len(X[0]):
        print("Non-square matrix does not have a trace.\n")
        return
    res = 0
    for i in range(len(X)):
        res += X[i][i]
    return res

def vector_norm(X):
    sq_sum = 0
    for i in range(len(X)):
        sq_sum += X[i] * X[i]
    return sqrt(sq_sum)

def transpose(X):
    X_trans = [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]
    return X_trans

def mul_matrix_vector(M, V):
    if len(M[0]) != len(V):
        print("Matrix and vector don't have matching shapes.")
        return
    res = [sum([M[i][j] * V[j] for j in range(len(V))]) for i in range(len(M))]
    return res

def crammer(poz, A, B):
    A_var = [[ A[i][j] if j != poz else B[i] for j in range(len(A[0]))] for i in range(len(A))]
    return determinant(A_var) / determinant(A)

def solve_crammer(A, B):
    return [crammer(i, A, B) for i in range(len(B))]

def minorant(i, j, A):
    return [[A[t + (t >= i)][k + (k >= j)] for k in range(len(A[0]) - 1)] for t in range(len(A) - 1)]

def solve_inverse(A, B):
    cofactor = [[(-1) ** (i + j) * determinant(minorant(i, j, A)) for j in range(len(A[0]))] for i in range(len(A))]
    inv_det_A = 1 / determinant(A)
    adj_A = transpose(cofactor)
    A_inverse = [[ inv_det_A * adj_A[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    return mul_matrix_vector(A_inverse, B)

rows = 3
cols = 3
X = ['x' for _ in range(rows)]
A = [[0] * cols for _ in range(rows)]
B = [0 for _ in range(rows)]

parse(X, A, B)
print(solve_crammer(A, B))
print(solve_inverse(A, B))
