def Controller(t, x, param):
    import numpy as np
    vd = param["vd"]
    v0 = param["v0"]

    m = param["m"]
    Cag = param["Cag"]
    Cdg = param["Cdg"]
    
    lam = 1000
    alpha = 0.2
    w = 1000000
    h = ((x[1] - vd)**2)/2
    B = x[0] - ((v0 - x[1])**2)/(2*Cdg) - 1.8*x[1]
    A00 = (x[1] - vd)/m
    A10 = (1.8 + (x[1] - v0)/Cdg)/m
    b1 = v0 - x[1] + alpha*B
    
    P = np.zeros((2,2))
    P = np.array([[2, 0], [0, 2*w]])
    
    A = np.zeros([5, 2])
    A = np.array([[A00, -1], [A10, 0], [-1, 0], [1, 0], [0, -1]])


    b = np.zeros([5])
    b = np.array([[-lam*h],
                  [b1],
                  [Cdg*m],
                  [Cag*m],
                  [0]])
    b = b.reshape(5,)

    q = np.zeros([2, 1])
    
    return A, b, P, q