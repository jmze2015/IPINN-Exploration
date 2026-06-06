import sys
print(sys.executable)

import torch
print(torch.__version__)


import pandas as pd 
import numpy as np 
import plotly.express as px
import torch 
from torch import nn
from torch.utils.data import TensorDataset, DataLoader 


## loading linear advection data
all_data = pd.read_csv("Data/M_500_literature_data.csv")
training_data = all_data.sample(frac=0.10, random_state=123)

test_data = all_data.copy()

## inputs and outputs
X_train = training_data[["x", "t"]].to_numpy(dtype = np.float32)
y_train = training_data[["u"]].to_numpy(dtype = np.float32)

X_test = test_data[["x", "t"]].to_numpy(dtype = np.float32)
y_test = test_data[["u"]].to_numpy(dtype = np.float32)

## Now convert to PyTorch tentors
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)

X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)


class VanillaNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2,16),
            nn.Tanh(),
            nn.Linear(16,16),
            nn.Tanh(),
            nn.Linear(16,1)
        )
    def forward(self,x):
        return self.net(x)

model = VanillaNN()

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

epochs = 200

for epoch in range(epochs):
    model.train()

    total_loss = 0

    for X_batch, y_batch in train_loader:
        # Forward pass
        pred = model(X_batch)

        # Compute loss
        loss = loss_fn(pred, y_batch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 10 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")


model.eval()

with torch.no_grad():
    u_pred = model(X_test).numpy().flatten()


prediction_data = test_data.copy()
prediction_data["u_pred"] = u_pred 
prediction_data["abs_error"] = (prediction_data["u_pred"] - prediction_data["u"]).abs()
prediction_data["MSE_contr"] = prediction_data["abs_error"]**2

prediction_data.to_csv(
    "Results/literature-100epochs.csv", index = False
)

torch.save(
    model.state_dict(),
    "Models/literature-100epochs-vanilla-nn.pt"
)

