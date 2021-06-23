import numpy as np
import matplotlib.pyplot as plt
import control
import scipy

plt.rcParams["figure.figsize"] = (8, 7)

# Hyperparameters
num_steps = 10000
dt = 0.01
plot_ss = False
plot_lqr = True


def sim_step(x, x_ref, A, B, C, K, d_t):
    """
    Simulate one time step for given system, with given state-space model
    :param x: state at time t
    :param x_ref: reference state
    :param A: state transition matrix
    :param B: input matrix
    :param C: output matrix
    :param K: controller from state
    :param d_t: time-step size
    :return: state at time t+dt, output at time t
    """
    x_diff = x_ref - x  # Difference in state

    u = K @ x_diff  # Force
    x_next = x + A @ x * dt + B * u * d_t
    y = C @ x

    return x_next, y


def get_output(x_0, x_ref, A, B, C, K, iterations, d_t):
    """
    Simulate outputs for given system, with given state-space model
    :param x_0: initial state
    :param x_ref: reference state
    :param A: state transition matrix
    :param B: input matrix
    :param C: output matrix
    :param K: controller from state
    :param iterations: number of steps
    :param d_t: time-step size
    :return: outputs for given step range
    """
    x = x_0
    y = [[] for _ in range(4)]
    for t in range(iterations):
        x, _ = sim_step(x, x_ref, A, B, C, K, d_t)
        y[0].append(x[0, 0])
        y[1].append(x[1, 0])
        y[2].append(x[2, 0])
        y[3].append(x[3, 0])
    return y


def plot_output(y):
    """
    Plot each output on separate subplot, from given array of four signals
    :param y: array of four signal values
    """
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].set_title('cart position [m]')
    axs[0, 0].plot(y[0])

    axs[0, 1].set_title('cart velocity [m/s]')
    axs[0, 1].plot(y[1])

    axs[1, 0].set_title("pendulum's angle [rad]")
    axs[1, 0].plot(y[2])

    axs[1, 1].set_title("pendulum's angular velocity [rad/s]")
    axs[1, 1].plot(y[3])
    plt.show()


# 171842,172118
a_id = 2
b_id = 8

# Parameters
M = 0.5 + b_id  # kg
m = 0.1 + 0.1 * a_id  # kg
L = 0.3  # m
I = 0.006  # kg*m2
b = 0.1  # N*s/m
g = 9.80665  # m/s2

# x = [y, y', th, th']
a_22 = -(I + m * L ** 2) * b / (I * (M + m) + M * m * L ** 2)  # y'' from y'
a_23 = -(m ** 2 * g * L ** 2) / (I * (M + m) + M * m * L ** 2)  # y'' from th
b_2 = (I + m * L ** 2) / (I * (M + m) + M * m * L ** 2)  # y'' from F

a_42 = b * m * L / (I * (M + m) + M * m * L ** 2)  # th'' from y'
a_43 = g * m * L * (M + m) / (I * (M + m) + M * m * L ** 2)  # th'' from th
b_4 = -m * L / (I * (M + m) + M * m * L ** 2)  # th'' from F

# State model
A = np.array([[0, 1, 0, 0],
              [0, a_22, a_23, 0],
              [0, 0, 0, 1],
              [0, a_42, a_43, 0]])

B = np.array([[0],
              [b_2],
              [0],
              [b_4]])

C = np.eye(4)
D = np.zeros((4, 1))

# Initial and desired state
x_init = np.array([[-10],
                   [0],
                   [0.05],
                   [0]])

x_des = np.array([[2],
                  [0],
                  [0],
                  [0]])

# State space feedback regulator
Pc = control.ctrb(A, B)
Pc_inv = np.linalg.inv(Pc)
des_ch_poly = np.linalg.matrix_power(A, 4) + 4 * np.linalg.matrix_power(A, 3) + \
              6 * np.linalg.matrix_power(A, 2) + 4 * A + np.eye(4)
K_ss = np.array([0, 0, 0, 1]) @ Pc_inv @ des_ch_poly

y_ss = get_output(x_init, x_des, A, B, C, K_ss, num_steps, dt)
if plot_ss:
    plot_output(y_ss)

# LQR regulator
Q = np.array([[1000, 0, 0, 0],
              [0, 1000, 0, 0],
              [0, 0, 1000, 0],
              [0, 0, 0, 1000]])
R = 0.01 * np.eye(1)
# Q = np.array([[0.5, 0, 0, 0],
#               [0, 0.1, 0, 0],
#               [0, 0, 0.1, 0],
#               [0, 0, 0, 0.1]])
# R = 100 * np.eye(1)
# Q = np.array([[1, 0, 0, 0],
#               [0, 1000, 0, 0],
#               [0, 0, 0.1, 0],
#               [0, 0, 0, 0.1]])
# R = 0.1 * np.eye(1)

P = scipy.linalg.solve_continuous_are(A, B, Q, R)
K_lqr = (np.linalg.inv(R) @ B.transpose() @ P)[0]
y_lqr = get_output(x_init, x_des, A, B, C, K_lqr, num_steps, dt)
if plot_lqr:
    plot_output(y_lqr)
