import casadi as ca
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import time

## distribution of traction force
def chi_fr(Fx):

    ## front-drive
    # xf = 0.125 * ca.tanh(2 * (Fx + 0.5)) + 0.875
    # xr = 1. - xf

    ## rear-drive
    xr = 0.125 * ca.tanh(2 * (Fx + 0.5)) + 0.875
    xf = 1. - xr

    return xf * Fx, xr * Fx

def get_slip_angle(Ux, Uy, r, delta, param):
    L_r = param["L_r"]
    L_f = param["L_f"]
    
    a_f = ca.arctan2((Uy + L_f * r), Ux) - delta
    a_r = ca.arctan2((Uy - L_r * r), Ux)
    return a_f, a_r

def tire_model_sim(alpha, Fz, Fx, C_alpha, mu):

    Fy_max_sq = (mu * Fz)**2 - (Fx)**2
    Fy_max = ca.if_else( Fy_max_sq <=0, 0, ca.sqrt(Fy_max_sq))
    
    alpha_slip = ca.arctan(3 * Fy_max / C_alpha)
    Fy = ca.if_else(ca.fabs(alpha) <= alpha_slip, 
        - C_alpha * ca.tan(alpha) 
        + C_alpha**2 * ca.fabs(ca.tan(alpha)) * ca.tan(alpha) / (3 * Fy_max)
        - C_alpha**3 * ca.tan(alpha)**3 / (27 * Fy_max**2), 
        - Fy_max * ca.sign(alpha))
    
    return Fy

def tire_model_ctrl(alpha, Fz, Fx, C_alpha, mu):
    # for each tire
    xi = 0.85

    ## NaN
    # Fy_max = ca.sqrt((mu * Fz)**2 - (0.99 * Fx)**2)
    # Fy_max = ca.sqrt((mu * Fz)**2 - (0.99 * Fx)**2)

    ## if else
    # Fy_max_sq = (mu * Fz)**2 - (0.99 * Fx)**2
    # Fy_max = ca.if_else( Fy_max_sq <=0, 0, ca.sqrt(Fy_max_sq))

    F_offset = 2000
    ## hyperbola
    Fy_max_sq = (mu * Fz)**2 - (0.99 * Fx)**2
    Fy_max_sq = (ca.sqrt( Fy_max_sq**2 + F_offset) + Fy_max_sq) / 2
    Fy_max = ca.sqrt(Fy_max_sq)

    alpha_mod = ca.arctan(3 * Fy_max / C_alpha * xi)
    
    Fy = ca.if_else(ca.fabs(alpha) <= alpha_mod, - C_alpha * ca.tan(alpha) 
        + C_alpha**2 * ca.fabs(ca.tan(alpha)) * ca.tan(alpha) / (3 * Fy_max)
        - C_alpha**3 * ca.tan(alpha)**3 / (27 * Fy_max**2), 
        - C_alpha * (1 - 2 * xi + xi**2) * ca.tan(alpha)
        - Fy_max * (3 * xi**2 - 2 * xi**3) * ca.sign(alpha))
    return Fy

def normal_load(Fx, param):
    # for both tires
    L_r = param["L_r"]
    L_f = param["L_f"]
    m = param["m"]
    g = param["g"]
    hcg = param["hcg"]
    
    L = (L_r + L_f)
    F_zf = L_r / L * m * g - hcg / L * Fx
    F_zr = L_f / L * m * g + hcg / L * Fx
    
    return F_zf, F_zr

# param = { "m":1778., "Izz":3049.,"L_f":1.194, "L_r":1.436, "hcg":0.55, "Peng": 172. * 1000, "Frr": 218., "Cd": 0.4243,
#           "delta_max": 0.4712, "ddelta_max":0.3491, "C_alpha_f": 180. * 1000,"C_alpha_r": 300. * 1000,
#          "mu_f": 0.75,"mu_r": 0.8, "g":9.81, "mu_lim": 0.8} # from IEEE-TCST paper


param = { "m":1778., "Izz":3049.,"L_f":1.094-0.1, "L_r":1.536+0.1, "hcg":0.55, "Peng": 172. * 1000, "Frr": 218., "Cd": 0.4243,
          "delta_max": 0.4712, "ddelta_max":0.3491, "C_alpha_f": 180. * 1000,"C_alpha_r": 400. * 1000,
         "mu_f": 0.75,"mu_r": 0.6, "g":9.81, "mu_lim": 0.8}