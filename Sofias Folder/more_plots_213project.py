# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:45:20 2026

@author: Sofia
"""

import numpy as np
import matplotlib.pyplot as plt

# load grid
x = np.load("x.npy")

# load solutions
u_godunov = np.load("u_godunov.npy")
u_pinn = np.load("u_pinn.npy")
u_ipinn = np.load("u_ipinn.npy")

# plot
plt.figure(figsize=(8,5))

plt.plot(x, u_godunov, color="red", label="Godunov", linewidth=2)
plt.plot(x, u_pinn, label="PINN", linewidth=2.5, linestyle="--")

plt.xlabel("x")
plt.ylabel("u(x, t=6)")
plt.title("Burgers Equation: Godunov vs. PINN")
plt.legend()

plt.show()

plt.figure(figsize=(8,5))

plt.plot(x, u_godunov, color="red", label="Godunov", linewidth=2)
plt.plot(x, u_ipinn, color="black", label="IPINN",linewidth=2.5, linestyle="-.")

plt.xlabel("x")
plt.ylabel("u(x, t=6)")
plt.title("Burgers Equation: Godunov vs. IPINN")
plt.legend()

plt.show()
