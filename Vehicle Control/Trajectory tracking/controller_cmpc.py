import casadi as ca
import numpy as np
import cvxpy as cp

def Setup_Derivative(param):

    def fun_jac_dt(x, u, param):
        L_f = param["L_f"]
        L_r = param["L_r"]
        h = param["h"]

        psi = x[2]
        v = x[3]
        delta = u[1]
        a = u[0]

        A = np.zeros((4, 4))
        B = np.zeros((4, 2))

        A[0, 0] = 1.0
        A[0, 2] = -h * v * np.sin(psi + np.arctan((L_r * np.arctan(delta)) / (L_f + L_r)))
        A[0, 3] = h * np.cos(psi + np.arctan((L_r * np.arctan(delta)) / (L_f + L_r)))

        A[1, 1] = 1.0
        A[1, 2] = h * v * np.cos(psi + np.arctan((L_r * np.arctan(delta)) / (L_f + L_r)))
        A[1, 3] = h * np.sin(psi + np.arctan((L_r * np.arctan(delta)) / (L_f + L_r)))

        A[2, 2] = 1.0
        A[2, 3] = (h * np.arctan(delta)) / (((L_r**2 * np.arctan(delta)**2) / (L_f + L_r)**2 + 1)**0.5 * (L_f + L_r))

        A[3, 3] = 1.0

        B[0, 1] = -(L_r * h * v * np.sin(psi + np.arctan((L_r * np.arctan(delta)) / (L_f + L_r)))) / ((delta**2 + 1) * ((L_r**2 * np.arctan(delta)**2) / (L_f + L_r)**2 + 1) * (L_f + L_r))
        B[1, 1] = (L_r * h * v * np.cos(psi + np.arctan((L_r * np.arctan(delta)) / (L_f + L_r)))) / ((delta**2 + 1) * ((L_r**2 * np.arctan(delta)**2) / (L_f + L_r)**2 + 1) * (L_f + L_r))
        B[2, 1] = (h * v) / ((delta**2 + 1) * ((L_r**2 * np.arctan(delta)**2) / (L_f + L_r)**2 + 1)**1.5 * (L_f + L_r))
        B[3, 0] = h

        return [A, B]

    return fun_jac_dt


def Controller_CMPC(x_bar, u_bar, x0, fun_jac_dt, param):
    dim_state = x_bar.shape[1]
    dim_ctrl = u_bar.shape[1]

    n_u = u_bar.size
    n_x = x_bar.size
    n_var = n_u + n_x

    n_eq = x_bar.shape[1] * u_bar.shape[0]  # dynamics
    n_ieq = u_bar.size  # input constraints

    # Define the parameters
    Q = np.eye(4) * 50
    R = np.eye(2) * 10
    L = np.eye(4) * 250

    # Define the cost function
    np.random.seed(1)
    P = np.zeros((n_var, n_var))
    for k in range(u_bar.shape[0]):
        P[k * dim_state:(k+1) * dim_state, k * dim_state:(k+1) * dim_state] = Q
        P[n_x + k * dim_ctrl:n_x + (k+1) * dim_ctrl, n_x + k * dim_ctrl:n_x + (k+1) * dim_ctrl] = R

    P[n_x - dim_state:n_x, n_x - dim_state:n_x] = L
    P = (P.T + P) / 2
    q = np.zeros((n_var, 1))

    # Define the constraints
    A = np.zeros((n_eq, n_var))
    b = np.zeros(n_eq)

    G = np.zeros((n_ieq, n_var))
    ub = np.zeros(n_ieq)
    lb = np.zeros(n_ieq)

    u_ub = np.array([4, 0.8])
    u_lb = np.array([-10, -0.8])

    for k in range(u_bar.shape[0]):
        AB = fun_jac_dt(x_bar[k, :], u_bar[k, :], param)
        A[k * dim_state:(k+1) * dim_state, k * dim_state:(k+1) * dim_state] = AB[0]
        A[k * dim_state:(k+1) * dim_state, (k+1) * dim_state:(k+2) * dim_state] = -np.eye(dim_state)
        A[k * dim_state:(k+1) * dim_state, n_x + k * dim_ctrl:n_x + (k+1) * dim_ctrl] = AB[1]

        G[k * dim_ctrl:(k+1) * dim_ctrl, n_x + k * dim_ctrl:n_x + (k+1) * dim_ctrl] = np.eye(dim_ctrl)
        ub[k * dim_ctrl:(k+1) * dim_ctrl] = u_ub - u_bar[k, :]
        lb[k * dim_ctrl:(k+1) * dim_ctrl] = u_lb - u_bar[k, :]

    # Define and solve the CVXPY problem
    x = cp.Variable(n_var)
    prob = cp.Problem(
        cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
        [G @ x <= ub, lb <= G @ x, A @ x == b, x[0:dim_state] == x0 - x_bar[0, :]]
    )
    prob.solve(verbose=False, max_iter=10000)

    return x.value[n_x:n_x + dim_ctrl] + u_bar[0, :]