from math import acos, sin
import numpy as np

R = np.array([[0.504, -0.523, 0.6866],
              [0.526, 0.816, 0.236],
              [-0.684, 0.242, 0.687]])

theta = acos((R.trace()-1)/2)

v = 1/(2*sin(theta)) * np.array([[R[2][1] - R[1][2]],
                                 [R[0][2] - R[2][0]],
                                 [R[1][0] - R[0][1]]])

print(f"theta={theta},\nv={v}")