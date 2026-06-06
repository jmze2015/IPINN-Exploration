# -*- coding: utf-8 -*-
"""
Created on Fri May 29 21:04:01 2026

@author: Sofia
"""
#prevents crashes 
import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#machine learning/math packages
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt


# Neural Network
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()

        self.activation = nn.Tanh()

        #build layers
        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))

        self.layers = nn.ModuleList(layer_list)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        return x


#initial condition
N0=1000

x0=torch.linspace(-10, 10, N0).reshape(-1,1)
t0= torch.zeros_like(x0)

x_np = x0.numpy().flatten()

u0_np = np.where(
    (x_np > -np.pi) & (x_np < np.pi),
    2*np.sin(3*x_np) + np.cos(2*x_np) - 1.0,
    0
)

u0= torch.tensor(u0_np.reshape(-1,1), dtype=torch.float32)

#boundry conditions 
Nb= 1000

t_b = torch.rand(Nb,1)*6

x_left = -10*torch.ones(Nb,1)
x_right = 10*torch.ones(Nb,1)

x_b = torch.cat([x_left, x_right], dim=0)
t_b = torch.cat([t_b, t_b], dim=0)

#Dirichlet condition
u_b = torch.zeros_like(x_b)

#combine data
x_data = torch.cat([x0, x_b], dim=0)
t_data = torch.cat([t0, t_b], dim=0)
u_data = torch.cat([u0, u_b], dim=0)

x_data = x_data.detach()
t_data = t_data.detach()
u_data = u_data.detach()

#flux 
def flux(u):
    return 0.5*u**2


# Define Physics Informed Residual 
def Residual_PI(model, x, t):
    
    #enable differentiation
    x = x.clone().detach().requires_grad_(True)
    t= t.clone().detach().requires_grad_(True)

    u = model(torch.cat([x, t], dim=1))

    u_t = autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    F_x = u*u_x
    R_PI = u_t + F_x
    return R_PI


# Loss Functions
def bc_loss(model, x_data, t_data, u_data):
    N_theta = model(torch.cat([x_data, t_data], dim=1))
    R_b = N_theta - u_data
    return torch.mean(R_b**2)

def total_loss(model, x_f, t_f, x_data, t_data, u_data):
    R_pi = Residual_PI(model, x_f, t_f)
    loss_PI = torch.mean(R_pi**2)
    
    loss_b = bc_loss(model, x_data, t_data, u_data)

    return loss_PI + loss_b 


# Training Loop (minimize loss)
def train(model, optimizer, n_epochs,
          x_f, t_f, x_data, t_data, u_data):

    for epoch in range(n_epochs):
        
        model.train()

        optimizer.zero_grad()

        loss = total_loss(model, x_f, t_f, x_data, t_data, u_data)

        loss.backward()
        optimizer.step()
        

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")


#start training 
layers = [2, 100, 100, 100, 1]
model = PINN(layers)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

N_f = 10000
x_f = -10 + 20*torch.rand(N_f, 1)
t_f = 6*torch.rand(N_f, 1)

train(
      model,
      optimizer,
      n_epochs=3000,
      x_f=x_f,
      t_f=t_f,
      x_data=x_data,
      t_data=t_data,
      u_data=u_data
)

#plot 
# Put model in evaluation mode
model.eval()

# x values
x = np.linspace(-10, 10, 400)

# fixed time t = 6
t = 6*np.ones_like(x)

# convert to tensors
x_tensor = torch.tensor(x.reshape(-1,1), dtype=torch.float32)
t_tensor = torch.tensor(t.reshape(-1,1), dtype=torch.float32)

# evaluate network
with torch.no_grad():
    u_pred = model(torch.cat([x_tensor, t_tensor], dim=1))

    # convert back to numpy
    u_pred = u_pred.cpu().numpy().flatten()

# plot
plt.figure()
plt.plot(x, u_pred)
plt.xlabel("x")
plt.ylabel("u(x,6)")
plt.title("PINN Solution at t = 6")
plt.grid(True)
plt.show()


#estimate and plot initial conditions
with torch.no_grad():
    u_ic_pred = model(torch.cat([x0,t0],dim=1))
    u_ic_pred = u_ic_pred.cpu().numpy().flatten()
    
plt.plot(x_np, u0_np, label="Exact IC")
plt.plot(x_np, u_ic_pred, '--', label="Recovered PINN IC")
plt.legend()

np.save("u_pinn.npy", u_pred)

"""
#plot initial condition 
plt.figure()
plt.plot(x_np, u0_np)
plt.xlabel("x")
plt.ylabel("u(x,0)")
plt.title("Initial Condition")
plt.show()
"""