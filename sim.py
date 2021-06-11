import numpy as np
import matplotlib.pyplot as plt
import control

np.set_printoptions(linewidth=300)


def sim_step(x, x_ref, A, B, C, K, dt=0.001):
    """
    Simulate one time step for given system, with given state-space model
    :param x: state at time t
    :param x_ref: reference state
    :param A: state transition matrix
    :param B: input matrix
    :param C: output matrix
    :param K: controller from state
    :param dt: time-step size
    :return: state at time t+dt, output at time t
    """
    x_diff = x_ref - x  # Difference in state

    u = K@x_diff  # Force
    x_next = x + A@x*dt + B*u*dt
    y = C @ x

    return x_next, y


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

# print(A)
# print(B)
# print(C)
# print(D)


x = np.array([[-10],
              [0],
              [0.05],
              [0]])

x_ref = np.array([[2],
                  [0],
                  [0],
                  [0]])
x_t = []
Pc = control.ctrb(A, B)
# print(Pc)
Pc_inv = np.linalg.inv(Pc)
# print(Pc_inv)
new_ch_poly = A @ A @ A @ A + 4 * A @ A @ A + 6 * A @ A + 4 * A + np.eye(4)
K = np.array([0, 0, 0, 1]) @ Pc_inv @ new_ch_poly

Nsteps = 1000
x0 = []
x1 = []
x2 = []
x3 = []
for t in range(Nsteps):
    x, _ = sim_step(x=x, x_ref=x_ref, A=A, B=B, C=C, K=K, dt=0.01)
    x0.append(x[0, 0])
    x1.append(x[1, 0])
    x2.append(x[2, 0])
    x3.append(x[3, 0])

print(x0)
plt.plot(x0)  # Polozenie
plt.plot(x1)  # Predkosc
plt.plot(x2)  # Kat wahadla
plt.plot(x3)  # Predkosc katowa wahadla
plt.show()
