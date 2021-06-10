import numpy as np


# 171842,172118
a_id = 2
b_id = 8

# Parameters
M = 0.5 + b_id  # kg
m = 0.1 + 0.1*a_id  # kg
L = 0.3  # m
I = 0.006  # kg*m2
b = 0.1  # N*s/m
g = 9.80665  # m/s2


# x = [y, y', th, th']


a_22 = -(I + m*L**2)*b / (I*(M+m) + M*m*L**2)  # y'' from y'
a_23 = -(m**2 * g * L**2) / (I*(M+m) + M*m*L**2)  # y'' from th
b_2 = (I + m*L**2) / (I*(M+m) + M*m*L**2)  # y'' from F

a_42 = -b*m*L / (I*(M+m) + M*m*L**2)  # th'' from y'
a_43 = g*m*L*(M+m) / (I*(M+m) + M*m*L**2)  # th'' from th
b_4 = -m*L / (I*(M+m) + M*m*L**2)  # th'' from F

# State model
A = np.array([[0,    1,    0, 0],
              [0, a_22, a_23, 0],
              [0,    0,    0, 1],
              [0, a_42, a_43, 0]])

B = np.array([[ 0],
              [b_2],
              [ 0],
              [b_4]])

C = np.eye(4)
D = np.zeros((4, 1))

print(A)
print(B)
print(C)
print(D)