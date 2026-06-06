# -*- coding: utf-8 -*-
"""
Created on Fri May 29 12:36:27 2026

@author: Sofia
"""
import numpy as np
import matplotlib.pyplot as plt

#define flux
def flux(u):
    return 0.5*u**2

#define Godunov flux
def godunov_flux(uL, uR):
    #shock
    if uL > uR:
        s = 0.5*(uL + uR)
        
        if s > 0:
            return flux(uL)
        else: 
            return flux(uR)
    
    #rarefaction
    else: 
        if uL >= 0:
            return flux(uL)
        elif uR <= 0: 
            return flux(uR)
        else: 
            return 0.0
#grid
N=400
x0=-10
x1=10

x= np.linspace(x0, x1, N)
h = x[1]-x[0]

#initial condition
u= np.where(
    (x>-np.pi) & (x<np.pi),
    2*np.sin(3*x)+ np.cos(2*x)-1.0,
    0 
)
    
CFL = .9
Tfinal = 6
t=0 

#loop
while t < Tfinal:
    umax = np.max(np.abs(u))
    
    if umax <1e-12:
        break
    
    #CFL condition
    k= CFL*h/umax
    
    if t + k > Tfinal: 
        k = Tfinal -t
    
    unew = u.copy()
    
    F = np.zeros(N+1)
    
    #update
    for i in range(1, N):
        F[i] = godunov_flux(u[i-1], u[i])
        
    for i in range(1, N-1):
        unew[i] = (
            u[i]
            - (k/h)*(F[i+1]-F[i])
        )
    #next time step
    u = unew
    t += k
    
#plot
plt.plot(x, u, linewidth=2)
plt.xlabel("x")
plt.ylabel("u")
plt.title("Godunov Method for Burgers Equation")
plt.grid(True)
plt.show()
        
np.save("x.npy", x)
np.save("u_godunov.npy", u)
    