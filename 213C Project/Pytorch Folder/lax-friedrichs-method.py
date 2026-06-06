import numpy as np 
import pandas as pd

"""
Given some inputs this Lax-Friedrich Method will output a df that is a numerical solution of the Burger's equation
with a periodic boundary condition. Df will always have columns [x, t, time_step, u].
"""

def lf_method(M, x_grid, t_grid , g, filename):
    L = x_grid[-1]-x_grid[0]
    M = M 
    dx = L / M 
    x = np.arange(-L/2, L/2+dx , dx)
    u = g(x)

    ## CFL Condition
    dt = 0.8 * dx 
    lam = dt / dx

    ## Simulation Time
    T_max = t_grid[-1]
    N = int(np.ceil(T_max / dt))

    ## Burgers' flux function
    def f(x):
        return x**2 / 2

    def LF_flux(u,v):
        return f(u)-f(v)

    results = pd.DataFrame({
        'x': x,
        'u': u,
        'time_step': 0,
        't':0
    })

    ## main loop

    for n in range(1, N+1):
        u_new = u.copy() 

        for i in range(1,M):
            u_new[i] = 0.5*(u[i+1]+u[i-1]) - 0.5 * lam * (LF_flux(u[i+1],u[i-1])) 

        ## periodic boundary condition

        u_new[0] = 0.5 * (u[0]+u[M]) - 0.5 * lam * (LF_flux(u[1], u[M]))
        u_new[M] = 0.5 * (u[0]+u[M-1]) - 0.5 * lam * (LF_flux(u[0], u[M-1]))

        u = u_new 

        df2 = pd.DataFrame({
            "x":x,
            "u":u,
            "time_step": n,
            "t": n * dt 
        })

        results = pd.concat([results, df2], ignore_index=True)
    
    results.to_csv(f"Data/{filename}.csv", index = False)
    
def G(x):
    return np.exp(-16 * (x**2))

lf_method(M = 200, x_grid=[-5,5], t_grid=[0,2], g = G, filename="burgers_lf_tmax_2")