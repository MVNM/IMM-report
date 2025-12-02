import numpy as np
import numpy.linalg.linalg as npla
import cv2

F = np.array([[-9.91901e-09, 2.5815e-07, -2.80463e-05],
              [-9.35804e-08, 8.28675e-09, -0.000774303],
              [-1.43722e-05, 0.00068278, 0.0241982]])

KL = np.array([[730.469, 0, 378],
               [0, 730.469, 504],
               [0, 0, 1]])

KR = np.array([[730.469, 0, 378],
               [0, 730.469, 504],
               [0, 0, 1]])

E = KL.T @ F @ KR

U, S, Vt = npla.svd(E)

# S should be of the form (l, l, 0) but the S that is retrieved is not exactly of that form in practice.
s = (S[0] + S[1]) / 2.0
S = np.array([s, s, 0.0])

W = np.array([[0, -1,  0],
              [1,  0,  0],
              [0,  0,  1]])

t1 = U @ W @ np.diag(S) @ U.T
t2 = U @ W.T @ np.diag(S) @ U.T

t1 = np.array([t1[2][1], t1[0][2], t1[1][0]])
t2 = np.array([t2[2][1], t2[0][2], t2[1][0]])

R1 = U @ npla.inv(W) @ Vt
R2 = U @ npla.inv(W.T) @ Vt

candidates = [
    (R1, t1),
    (R1, t2),
    (R2, t1),
    (R2, t2)
]

print(R1)
print(R2)
