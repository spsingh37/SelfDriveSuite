import casadi as ca
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import time

from utils import *

def GetCarModel(kappa_table, ts = np.linspace(0, 0.025, 10), ode = "cvodes"):
    # system dimensions
    Dim_state = 6 ## Ux Uy r | s r dpsi
    Dim_ctrl  = 2 ## Fx delta

    # Continuous time dynamics model
    xm = ca.MX.sym('xm', (Dim_state, 1))
    um = ca.MX.sym('um', (Dim_ctrl, 1))

    ##
    Fx = um[0]
    delta = um[1]

    Fxf, Fxr = chi_fr(Fx)

    ## resistance, need to deal with the discontinuity at 0
    Fd = param["Frr"] + param["Cd"] * xm[0]**2
    # Fd = Fd * ca.if_else(xm[0] >= 0, -1, 1)
    Fd = Fd * ca.tanh(- xm[0] * 10000)
    Fb = 0.0
    ## 
    
    af, ar = get_slip_angle(xm[0], xm[1], xm[2], delta, param)
    Fzf, Fzr = normal_load(Fx, param)
    Fxf, Fxr = chi_fr(Fx)

    #############################################################################
    ####### This cutoff is for simlation only, not for controller design ########
    Fxf = ca.if_else( ca.fabs(Fxf) >= param["mu_f"] * ca.fabs(Fzf), 
                     param["mu_f"] * ca.fabs(Fzf) * ca.sign(Fxf) ,
                       Fxf)
    
    Fxr = ca.if_else( ca.fabs(Fxr) >= param["mu_r"] * ca.fabs(Fzr), 
                     param["mu_r"] * ca.fabs(Fzr) * ca.sign(Fxr) ,
                       Fxr)
    #############################################################################

    Fyf = tire_model_sim(af, Fzf, Fxf, param["C_alpha_f"], param["mu_f"])
    Fyr = tire_model_sim(ar, Fzr, Fxr, param["C_alpha_r"], param["mu_r"])

    dUx  = (Fxf * ca.cos(delta) - Fyf * ca.sin(delta) + Fxr + Fd) / param["m"] + xm[2] * xm[1]
    dUy  = (Fyf * ca.cos(delta) + Fxf * ca.sin(delta) + Fyr + Fb) / param["m"] - xm[2] * xm[0]
    dr   = (param["L_f"] * (Fyf * ca.cos(delta) + Fxf * ca.sin(delta)) - param["L_r"] * Fyr) / param["Izz"] 
    
    dx   = ca.cos(xm[5]) * xm[0] - ca.sin(xm[5]) * xm[1]
    dy   = ca.sin(xm[5]) * xm[0] + ca.cos(xm[5]) * xm[1]
    dyaw = xm[2]
      
    xdot = ca.vertcat(dUx, dUy, dr, 
                       dx,  dy, dyaw)
    
    fx = ca.vertcat(Fxf, Fyf, Fzf, af, Fxr, Fyr, Fzr, ar, xdot)
    Fun_dynmaics_ct = ca.Function('f_ct', [xm, um], [fx])

    t = ca.MX.sym('t', (1, 1))

    #! We will simulate over 50 seconds, 1000 timesteps.
    dae={'x':xm, 'p':um, 't':t, 'ode':xdot}
    integrator = ca.integrator('integrator', ode, dae, {'grid':ts, 'output_t0':True})

    return integrator, Fun_dynmaics_ct


def SimVehicle(y0, controller, integrator, car_dynamics, N_sim, MPC = False, x0_nlp_in = None, lamx0_nlp_in = None, lamg0_nlp_in = None):
    ## y0: initial condition
    tire_force_log = np.zeros((14, 1))
    x_log = np.array(y0)
    u_log = np.array([[0, 0]]).T
    u = np.array([[300., 0]]).T

    if MPC:
        prob, N_mpc, n_x, n_g, n_p, lb_var, ub_var, lb_cons, ub_cons = controller(None)
        x0_nlp    = x0_nlp_in # np.zeros((n_x, 1))
        lamx0_nlp = lamx0_nlp_in # np.zeros((n_x, 1))
        lamg0_nlp = lamg0_nlp_in # np.zeros((n_g, 1))
        opts = {'ipopt.print_level': 3, 'print_time': 0, 'ipopt.max_iter': 100} # , 'ipopt.sb': 'yes'}
        # opts = {'ipopt.max_iter': 3000}
        solver = ca.nlpsol('solver', 'ipopt', prob , opts)

    start = time.time()
    for k in range(N_sim):
        uk_last = u + 0.0

        if MPC:
            init = y0
            sol = solver(x0=x0_nlp, lam_x0=lamx0_nlp, lam_g0=lamg0_nlp,
                        lbx=lb_var, ubx=ub_var, lbg=lb_cons, ubg=ub_cons, p=init)
            x0_nlp = sol["x"].full()
            lamx0_nlp = sol["lam_x"].full()
            lamg0_nlp = sol["lam_g"].full()

            u = np.reshape(sol['x'].full()[0:2], [2, 1])
        else:
            u = controller(y0, uk_last, k)
        
        # print(u.shape)
        ## we consider some dynamics of the car transmission system
        alpha = 0.7
        u = u * alpha + uk_last * (1 - alpha)

        ## Saturation of control efforts
        Fx_max = np.abs(param['Peng'] / (np.abs(y0[0]) + 0.001))

        u[0] =  np.clip(u[0], -Fx_max, Fx_max)  
        u[1] =  np.clip(u[1], -param['delta_max'], param['delta_max'])      
        ## 
        
        ## integrators
        sol = integrator(x0 = y0, p = u)
        y0 = sol["xf"][:, -1]
        ## 

        # print(x_log.shape)
        # print(sol["xf"][:, 1:].shape)
        x_log = np.concatenate((x_log, sol["xf"][:, 1:]), axis = 1)
        # x_log = np.concatenate((x_log, sol["xf"][:, 1:]))
        u_log = np.concatenate((u_log, u), axis = 1)

        dx = car_dynamics(sol["xf"][:, 1:], u)
        tire_force_log = np.concatenate((tire_force_log, np.reshape(dx, (14, -1) )), axis = 1)

        # dx = car_dynamics(y0, u)
        # tire_force_log = np.concatenate((tire_force_log, np.reshape(dx, (14, 1) )), axis = 1)

    end = time.time()
    print("Simulation completed. Spend %f secs"% (end - start))

    return x_log, u_log, tire_force_log
