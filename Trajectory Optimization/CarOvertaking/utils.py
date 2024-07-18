import casadi as ca
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import time

from nmpc_takeover import *

def eval_controller(par_init: np.ndarray):
    '''
    par_init: [x, y, yaw, v, v_x_leader, v_y_leader, v_x_desired]
        x: x distance between our car and leader car
        y: y distance between our car and leader car
        yaw: yaw angle of our car
        v: velocity of our car
        v_x_leader: x velocity of leader car
        v_y_leader: y velocity of leader car
        v_x_desired: desired takeover x velocity of our car
    '''
    
    # We define the default evaluation rate and other constants here
    h = 0.1
    N_sim = int(np.ceil(17/ h))
    Dim_state = 4
    Dim_ctrl  = 2
    
    # define some parameters
    x_init = ca.MX.sym('x_init', (Dim_state, 1)) # initial condition, the state should be position to the leader car
    v_leader = ca.MX.sym('v_leader',(2, 1))      # leader car's velocity
    v_des = ca.MX.sym('v_des')                   # desired speed of ego car
    delta_last = ca.MX.sym('delta_last')         # steering angle at last step
    par = ca.vertcat(x_init, v_leader, v_des, delta_last) # concatenate them
    
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

    # Discretized dynamics model
    Fun_dynmaics_dt = ca.Function('f_dt', [x_model, u_model, par], [xdot * h + x_model])
    
    # controller is constructed here:
    prob, N_mpc, n_x, n_g, n_p, lb_var, ub_var, lb_cons, ub_cons = nmpc_controller()
    # expected to provide
    # NLP problem, the problem size (n_x, n_g, n_p), horizon and bounds
    
    opts = {'ipopt.print_level': 3, 'print_time': 0} # , 'ipopt.sb': 'yes'}
    solver = ca.nlpsol('solver', 'ipopt', prob , opts)
    
    # our initial conditions, see if we want to change it ...
    state_0 = par_init[:4] #np.array([-80, 0.0, 0.15, 30.])
    d_last = 0
    # our initial conditions, see if we want to change it ...
    
    # logger of states
    xt = np.zeros((N_sim+1, Dim_state))
    ut = np.zeros((N_sim, Dim_ctrl))
    
    # place holder for warm start
    x0_nlp    = np.random.randn(n_x, 1) * 0.01 # np.zeros((n_x, 1))
    lamx0_nlp = np.random.randn(n_x, 1) * 0.01 # np.zeros((n_x, 1))
    lamg0_nlp = np.random.randn(n_g, 1) * 0.01 # np.zeros((n_g, 1))

    xt[0, :] = state_0
    
    # main loop of simulation
    for k in range(N_sim):
        state_0 = xt[k, :]
        
        # the leader car's velocity and desired velocity will not change in the planning horizon
        par_nlp = np.concatenate((state_0, par_init[4:], np.array([d_last]))) # state_0 + [v_x_leader, v_y_leader, v_x_desired, d_last]

        sol = solver(x0=x0_nlp, lam_x0=lamx0_nlp, lam_g0=lamg0_nlp,
                     lbx=lb_var, ubx=ub_var, lbg=lb_cons, ubg=ub_cons, p = par_nlp)
        
        x0_nlp = sol["x"].full()
        lamx0_nlp = sol["lam_x"].full()
        lamg0_nlp = sol["lam_g"].full()
                
        ut[k, :] = np.squeeze(sol['x'].full()[0:Dim_ctrl])
        d_last = ut[k, 1]
        
        ut[k, 0] = np.clip(ut[k, 0], -10, 4)
        ut[k, 1] = np.clip(ut[k, 1], -0.6, 0.6)
        
        xkp1 = Fun_dynmaics_dt(xt[k, :], ut[k, :], par_nlp)

        xt[k+1, :] = np.squeeze(xkp1.full())

    
    return xt, ut