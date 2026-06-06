# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:23:02 2026

@author: Sofia
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt


# Network
class IPINN(nn.Module):

    def __init__(self, layers):
        super().__init__()

        self.activation = nn.Tanh()

        net = []
        for i in range(len(layers)-1):
            net.append(nn.Linear(layers[i], layers[i+1]))

        self.layers = nn.ModuleList(net)

    def forward(self, x):

        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        return self.layers[-1](x)


# Initial condition
N0 = 1000

x0 = torch.linspace(-10, 10, N0).reshape(-1,1)
t0 = torch.zeros_like(x0)

x_np = x0.numpy().flatten()

u0_np = np.where(
    (x_np > -np.pi) & (x_np < np.pi),
    2*np.sin(3*x_np) + np.cos(2*x_np) - 1.0,
    0.0
)

u0 = torch.tensor(
    u0_np.reshape(-1,1),
    dtype=torch.float32
)


# Boundary conditions
Nb = 1000

t_b = 6*torch.rand(Nb,1)

x_left = -10*torch.ones(Nb,1)
x_right = 10*torch.ones(Nb,1)

x_b = torch.cat([x_left,x_right],dim=0)
t_b = torch.cat([t_b,t_b],dim=0)

u_b = torch.zeros_like(x_b)


# Combine Training data

x_data = torch.cat([x0,x_b],dim=0)
t_data = torch.cat([t0,t_b],dim=0)
u_data = torch.cat([u0,u_b],dim=0)


# Flux
def flux(u):
    return 0.5*u**2


# IPINN residual
def residual_pi(model, xa, xb, t):

    xa = xa.clone().detach().requires_grad_(True)
    xb = xb.clone().detach().requires_grad_(True)
    t  = t.clone().detach().requires_grad_(True)

    Na = model(torch.cat([xa,t],dim=1))
    Nb = model(torch.cat([xb,t],dim=1))
    N_dif = Nb-Na

    Ua = autograd.grad(
        Na,
        xa,
        grad_outputs=torch.ones_like(Na),
        create_graph=True
    )[0]

    Ub = autograd.grad(
        Nb,
        xb,
        grad_outputs=torch.ones_like(Nb),
        create_graph=True
    )[0]

    Fa = flux(Ua)
    Fb = flux(Ub)

    N_dif_t = autograd.grad(
        N_dif,
        t,
        grad_outputs=torch.ones_like(N_dif),
        create_graph=True
    )[0]


    R_pi = N_dif_t - (Fa - Fb)

    return R_pi


# Boundary loss
def boundary_loss(model, x_data, t_data, u_data):

    x_data = x_data.clone().detach().requires_grad_(True)

    N = model(torch.cat([x_data,t_data],dim=1))

    u_pred = autograd.grad(
        N,
        x_data,
        grad_outputs=torch.ones_like(N),
        create_graph=True
    )[0]
    
    R_b = u_pred-u_data

    return torch.mean(R_b**2)


# Total loss
def total_loss(
        model,
        xa,
        xb,
        t,
        x_data,
        t_data,
        u_data):

    R_pi = residual_pi(model, xa, xb, t)

    loss_pi = torch.mean(R_pi**2)

    loss_b = boundary_loss(
        model,
        x_data,
        t_data,
        u_data
    )

    return loss_pi + loss_b


# Random interval sampling
def sample_intervals(Nf):

    xa = -10 + 20*torch.rand(Nf,1)
    xb = -10 + 20*torch.rand(Nf,1)

    xa, xb = torch.min(xa,xb), torch.max(xa,xb)

    t = 6*torch.rand(Nf,1)

    return xa, xb, t


# Training loop
def train(
        model,
        optimizer,
        epochs,
        x_data,
        t_data,
        u_data):

    for epoch in range(epochs):

        xa, xb, t = sample_intervals(10000)

        optimizer.zero_grad()

        loss = total_loss(
            model,
            xa,
            xb,
            t,
            x_data,
            t_data,
            u_data,
        )

        loss.backward()

        optimizer.step()

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch} "
                f"Loss {loss.item():.6e}"
            )


# Train

layers = [2,100,100,100,100,1]

model = IPINN(layers)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

train(
    model,
    optimizer,
    epochs=3000,
    x_data=x_data,
    t_data=t_data,
    u_data=u_data
)


# Plot solution
model.eval()

x = np.linspace(-10,10,400)

x_tensor = torch.tensor(
    x[:,None],
    dtype=torch.float32,
    requires_grad=True
)

t_tensor = torch.full_like(
    x_tensor,
    6.0
)

N = model(torch.cat([x_tensor,t_tensor],dim=1))

u = autograd.grad(
    N,
    x_tensor,
    grad_outputs=torch.ones_like(N)
)[0]

u = u.detach().cpu().numpy().flatten()

plt.figure()
plt.plot(x,u)
plt.xlabel("x")
plt.ylabel("u(x,6)")
plt.title("IPINN Solution")
plt.grid()
plt.show()


# Check recovered initial condition
x0_g = x0.clone().detach().requires_grad_(True)

N0_pred = model(torch.cat([x0_g,t0],dim=1))

u0_pred = autograd.grad(
    N0_pred,
    x0_g,
    grad_outputs=torch.ones_like(N0_pred)
)[0]

u0_pred = u0_pred.detach().cpu().numpy().flatten()

plt.figure()
plt.plot(x_np,u0_np,label="Exact IC")
plt.plot(x_np,u0_pred,"--",label="Recovered IPINN IC")
plt.legend()
plt.show()

np.save("u_ipinn.npy", u)
