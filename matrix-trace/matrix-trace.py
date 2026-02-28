import numpy as np

def matrix_trace(A):
    total = 0
    for i in range(len(A)):
        total += A[i][i]   # A[i][i] = diagonal element
    return total

