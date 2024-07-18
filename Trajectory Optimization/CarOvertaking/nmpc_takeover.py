import casadi as ca
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import time

def nmpc_controller():
    # Declare simulation constants
    T = 4  # planning horizon
    N = 40  # Number of control intervals
    h = T / N

    # System dimensions
    Dim_state = 4
    Dim_ctrl = 2

    # Additional parameters
    x_init = ca.MX.sym('x_init', (Dim_state, 1))  # initial condition, state should be position relative to the leader car
    v_leader = ca.MX.sym('v_leader', (2, 1))  # leader car's velocity relative to ego car
    v_des = ca.MX.sym('v_des')
    delta_last = ca.MX.sym('delta_last')
    par = ca.vertcat(x_init, v_leader, v_des, delta_last)

    # Continuous dynamics model
    x_model = ca.MX.sym('xm', (Dim_state, 1))
    u_model = ca.MX.sym('um', (Dim_ctrl, 1))

    L_f = 1.0
    L_r = 1.0

    beta = ca.atan(L_r / (L_r + L_f) * ca.atan(u_model[1]))

    xdot = ca.vertcat(x_model[3] * ca.cos(x_model[2] + beta) - v_leader[0],
                      x_model[3] * ca.sin(x_model[2] + beta) - v_leader[1],
                      x_model[3] / L_r * ca.sin(beta),
                      u_model[0])

    # Discrete time dynamics model
    Fun_dynamics_dt = ca.Function('f_dt', [x_model, u_model, par], [xdot * h + x_model])

    # Declare model variables
    x = ca.MX.sym('x', (Dim_state, N + 1))
    u = ca.MX.sym('u', (Dim_ctrl, N))

    # Objective function terms
    L = 10 * x_model[1]**2 + x_model[2]**2 + 10 * (x_model[3] - v_des)**2
    P = L * 10
    L = L + 1 * u_model[0]**2 + 100 * u_model[1]**2

    Fun_cost_terminal = ca.Function('P', [x_model, par], [P])
    Fun_cost_running = ca.Function('Q', [x_model, u_model, par], [L])

    # State and control constraints
    state_ub = np.array([1e4, 1e4, 1e4, 1e4])
    state_lb = np.array([-1e4, -1e4, -1e4, -1e4])
    ctrl_ub = np.array([4, 0.4])
    ctrl_lb = np.array([-10, -0.4])

    # Upper and lower bounds for variables
    ub_x = np.matlib.repmat(state_ub, N + 1, 1)
    lb_x = np.matlib.repmat(state_lb, N + 1, 1)

    ub_u = np.matlib.repmat(ctrl_ub, N, 1)
    lb_u = np.matlib.repmat(ctrl_lb, N, 1)

    ub_var = np.concatenate((ub_u.reshape((Dim_ctrl * N, 1)), ub_x.reshape((Dim_state * (N + 1), 1))))
    lb_var = np.concatenate((lb_u.reshape((Dim_ctrl * N, 1)), lb_x.reshape((Dim_state * (N + 1), 1))))

    # Dynamics constraints: x[k+1] = x[k] + f(x[k], u[k]) * dt
    cons_dynamics = []
    ub_dynamics = np.zeros((N * Dim_state, 1))
    lb_dynamics = np.zeros((N * Dim_state, 1))
    for k in range(N):
        Fx = Fun_dynamics_dt(x[:, k], u[:, k], par)
        for j in range(Dim_state):
            cons_dynamics.append(x[j, k + 1] - Fx[j])

    # State constraints: G(x) <= 0
    cons_state = []
    for k in range(N):
        # Collision avoidance
        cons_state.append(1.0 - (x[1, k + 1]**2 / 4 + x[0, k + 1]**2 / 900))

        # Maximum lateral acceleration
        dx = (x[:, k + 1] - x[:, k]) / h
        ay = dx[2] * x[3, k]
        gmu = 0.5 * 0.6 * 9.81
        cons_state.append(ay - gmu)
        cons_state.append(-ay - gmu)

        # Lane keeping
        cons_state.append(-x[1, k + 1] - 1)
        cons_state.append(x[1, k + 1] - 3)

        # Steering rate
        if k >= 1:
            d_delta = u[1, k] - u[1, k - 1]
        else:
            d_delta = u[1, k] - delta_last
        cons_state.append(d_delta - 0.6 * h)
        cons_state.append(-d_delta - 0.6 * h)

    ub_state_cons = np.zeros((len(cons_state), 1))
    lb_state_cons = np.zeros((len(cons_state), 1)) - 1e5

    # Cost function
    J = Fun_cost_terminal(x[:, -1], par)
    for k in range(N):
        J = J + Fun_cost_running(x[:, k], u[:, k], par)

    # Initial condition as parameters
    cons_init = [x[:, 0] - x_init]
    ub_init_cons = np.zeros((Dim_state, 1))
    lb_init_cons = np.zeros((Dim_state, 1))

    # Define variables for NLP solver
    vars_NLP = ca.vertcat(u.reshape((Dim_ctrl * N, 1)), x.reshape((Dim_state * (N + 1), 1)))
    cons_NLP = cons_dynamics + cons_state + cons_init
    cons_NLP = ca.vertcat(*cons_NLP)
    lb_cons = np.concatenate((lb_dynamics, lb_state_cons, lb_init_cons))
    ub_cons = np.concatenate((ub_dynamics, ub_state_cons, ub_init_cons))

    # Create an NLP solver
    prob = {"x": vars_NLP, "p": par, "f": J, "g": cons_NLP}

    return prob, N, vars_NLP.shape[0], cons_NLP.shape[0], par.shape[0], lb_var, ub_var, lb_cons, ub_cons