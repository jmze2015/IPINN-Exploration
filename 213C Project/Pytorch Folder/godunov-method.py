import numpy as np 
import pandas as pd
import plotly.express as px

def godunov_method(M, x_grid, t_grid , g, filename):
    L = x_grid[-1]-x_grid[0]
    M = M 
    dx = L / M 
    x = np.linspace(-L/2, L/2 , M+1)
    u = g(x)

    ## CFL Condition
    dt = 0.8 * dx / max(abs(u))
    lam = dt / dx

    ## Simulation Time
    T_max = t_grid[-1]
    N = int(np.ceil(T_max / dt))

    ## Burgers' flux function
    def f(x):
        return x**2 / 2

    def G_flux(u,v):
        if u <= v and u > 0:
            return f(u)
        elif u <= v and v < 0:
            return f(v)
        elif u <= 0 and 0 <= v:
            return 0 
        elif u > v and (u+v)/2 >= 0:
            return f(u)
        elif u > v and (u+v)/2 < 0:
            return f(v)
        else:
            print("Unhandled flux case")


    results = pd.DataFrame({
        'x': x,
        'u': u,
        'time_step': 0,
        't':0
    })

    ## Main loop

    for n in range(1, N+1):
        u_new = u.copy() 

        for i in range(1,M):
            u_new[i] = u[i] - lam * (G_flux(u[i], u[i+1]) - G_flux(u[i-1], u[i]))

        ## Dirichlet boundary condition

        u_new[0] = 0
        u_new[M] = 0

        u = u_new 

        df2 = pd.DataFrame({
            "x":x,
            "u":u,
            "time_step": n,
            "t": n * dt 
        })

        results = pd.concat([results, df2], ignore_index=True)
    
    results.to_csv(f"Data/{filename}.csv", index = False)

## The initial distribution from the literature

def G(x):
    return np.where(
        (-np.pi <= x) & (x <= np.pi),
        2*np.sin(3*x) + np.cos(2*x) - 1.0,
        0
    )


"""
Code below is to view initial distribution.
"""

# dx = 0.01
# x = np.arange(-10,10 + dx, dx)

# DF = pd.DataFrame({
#     "x" : x,
#     "y" : G(x)
# })

# fig = px.line(
#     DF,
#     x = "x",
#     y = "y",
#     title = "Initial Distribution"
# )

# fig.show()

godunov_method(M = 500, x_grid=[-10, 10], t_grid=[0,6], g = G, filename="M_500_literature_data")