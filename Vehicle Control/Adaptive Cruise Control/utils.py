import cvxpy as cp
import numpy as np
from scipy import integrate

def CarModel(t, x, Controller, param):
    
    if t <= param["switch_time"]:
        param["v0"] = param["v01"]
    if t > param["switch_time"]:
        param["v0"] = param["v02"]
    
    A, b, P, q = Controller(t, x, param)
    
    var = cp.Variable(2)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(var, P)+ q.T @ var),
                     [A @ var <= b])
    prob.solve()
    
    u = var.value[0]        
    u = np.clip(u, -param["Cdg"] * param["m"], param["Cag"] * param["m"])
    
    dx = np.array([param["v0"] - x[1], 
                   u / param["m"]])
    return dx

def sim_vehicle(Controller, param, y0):
    t0, t1 = 0, param["terminal_time"]                # start and end
    t = np.linspace(t0, t1, 200)  # the points of evaluation of solution
    # y0 = [250, 10]                   # initial value
    y = np.zeros((len(t), len(y0)))   # array for solution
    y[0, :] = y0
    r = integrate.ode( lambda t, x:CarModel(t, x, Controller, param) ).set_integrator("dopri5")  # choice of method
    r.set_initial_value(y0, t0)   # initial values
    for i in range(1, t.size):
       y[i, :] = r.integrate(t[i]) # get one more value, add it to the array
       if not r.successful():
           raise RuntimeError("Could not integrate")
    
    ### recover control input ###
    u = np.zeros((200, 1))
    for k in range(200):
        if t[k] <= param["switch_time"]:
            param["v0"] = param["v01"]
        if t[k] > param["switch_time"]:
            param["v0"] = param["v02"]
            
        A, b, P, q = Controller(t[k], y[k, :], param)
        var = cp.Variable(2)
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(var, P)+ q.T @ var),
                         [A @ var <= b])
        prob.solve()

        u[k] = var.value[0]
    ### recover control input ###

    v0 = t * 0
    v0[t <  param["switch_time"]] = param["v01"]
    v0[t >= param["switch_time"]] = param["v02"]
    Cdg = param["Cdg"]
    B   = y[:, 0] - 1.8 * y[:, 1] - 0.5 * (np.clip(y[:, 1] - v0, 0, np.inf))**2 / Cdg

    return t, B, y, u

