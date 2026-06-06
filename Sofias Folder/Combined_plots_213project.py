# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 18:33:14 2026

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
plt.plot(x, u_ipinn, color="green", label="IPINN",linewidth=2.5, linestyle="-.")

plt.xlabel("x")
plt.ylabel("u(x, t=6)")
plt.title("Burgers Equation: Godunov vs. PINN vs. IPINN")
plt.legend()

plt.show()