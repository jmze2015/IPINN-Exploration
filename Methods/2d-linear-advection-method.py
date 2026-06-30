import numpy as np
import pandas as pd
import plotly.graph_objects as go

"""
In the following we define a method with square grid 10 x 10 and temporal boundaries [0, t_max].
We solve the 2D linear advection conservation law: u_t + a * u_x + b * u_y = 0.
The inputs are: M -- x/y grid subdivisions; T_max -- maximum time of simulation; V -- speed vector; 
g(x,y) -- initial distribution.
"""


def G(x,y):
        return np.exp(-0.2*((x**2)+(y**2)))

def F(x,y):
     return( 0.6 * np.exp(-0.2*(((x+3)**2)+((y-1)**2))) + 0.4* np.exp(-0.05*(((x-3)**2)+((y+2.5)**2))) )


def Multi_LA_method(m, T_max, V, g):
    M = m
    L = 10
    x = np.linspace(-L, L, M)
    y = np.linspace(-L, L, M)
    delta_x = x[1]-x[0]
    delta_y = y[1]-y[0]

    X, Y = np.meshgrid(x, y)

    ## wave speeds
    a = V[0]
    b = V[1]

    A = (abs(a)/delta_x) + (abs(b)/delta_y)
    delta_t = 0.8 * (1/A)
    t_max = T_max
    N = int(np.ceil(t_max / delta_t)) 

    solutions = []

    u = g(X, Y)

    for n in range(N):
        solutions.append(u.copy())

        u_new = np.zeros_like(u)

        for i in range(M):
            for j in range(M):
                im = (i - 1) % M
                ip = (i + 1) % M
                jm = (j - 1) % M
                jp = (j + 1) % M

                if a >= 0 and b >= 0:
                    # backward x, backward y
                    u_new[j, i] = (
                        u[j, i]
                        - (a*delta_t/delta_x)*(u[j, i] - u[j, im])
                        - (b*delta_t/delta_y)*(u[j, i] - u[jm, i])
                    )

                elif a < 0 and b >= 0:
                    # forward x, backward y
                    u_new[j, i] = (
                        u[j, i]
                        - (a*delta_t/delta_x)*(u[j, ip] - u[j, i])
                        - (b*delta_t/delta_y)*(u[j, i] - u[jm, i])
                    )

                elif a >= 0 and b < 0:
                    # backward x, forward y
                    u_new[j, i] = (
                        u[j, i]
                        - (a*delta_t/delta_x)*(u[j, i] - u[j, im])
                        - (b*delta_t/delta_y)*(u[jp, i] - u[j, i])
                    )

                else:
                    # a < 0 and b < 0
                    # forward x, forward y
                    u_new[j, i] = (
                        u[j, i]
                        - (a*delta_t/delta_x)*(u[j, ip] - u[j, i])
                        - (b*delta_t/delta_y)*(u[jp, i] - u[j, i])
                    )

        u = u_new.copy()


    frames = [
        go.Frame(
            data=[
                go.Surface(
                    x=X,
                    y=Y,
                    z=solutions[k],
                    cmin=0,
                    cmax=1
                )
            ],
            name=str(k)
        )
        for k in range(0, len(solutions), 5)
    ]

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=solutions[0],
                cmin=0,
                cmax=1
            )
        ],
        frames=frames
    )

    fig.update_layout(
        title="2D Linear Advection: Upwind Method: Speed = ("+str(V[0])+", "+str(V[1])+")",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="u",
            zaxis=dict(range=[0, 1])
        ),
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=80, redraw=True),
                                fromcurrent=True
                            )
                        ]
                    )
                ]
            )
        ]
    )

    fig.write_html(
    "multi_linear_advection.html",
    include_plotlyjs="cdn",
    full_html=True)

    fig.show()

Multi_LA_method(80, 80, [-1,1], G)


