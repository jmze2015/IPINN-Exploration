import os
import pandas as pd 
import numpy as np 
import torch 
from torch import nn
from torch.utils.data import TensorDataset, DataLoader 



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


def PINN_method(input_filename, interior_frac, Epochs, epoch_every, output_filename, pde_weight=1.0):
    """
    Clean PINN method for Burgers' equation:

        u_t + (u^2/2)_x = 0.

    Data loss is computed only on:
        t = 0, x = xmin, x = xmax

    PDE loss is computed only on interior points:
        t > 0, xmin < x < xmax

    Assumes CSV has columns x, t, u.
    """

    os.makedirs("Results", exist_ok=True)
    os.makedirs("Models", exist_ok=True)

    all_data = pd.read_csv(input_filename)

    xmin = all_data["x"].min()
    xmax = all_data["x"].max()
    tmin = all_data["t"].min()

    # Boundary / initial data
    boundary_data = all_data[
        (all_data["t"] == tmin) |
        (all_data["x"] == xmin) |
        (all_data["x"] == xmax)
    ].copy()

    # Interior collocation data
    interior_data = all_data[
        (all_data["t"] > tmin) &
        (all_data["x"] > xmin) &
        (all_data["x"] < xmax)
    ].copy()

    interior_data = interior_data.sample(frac=interior_frac, random_state=123)

    # Boundary tensors: need x,t and known u
    X_boundary = boundary_data[["x", "t"]].to_numpy(dtype=np.float32)
    y_boundary = boundary_data[["u"]].to_numpy(dtype=np.float32)

    X_boundary = torch.tensor(X_boundary)
    y_boundary = torch.tensor(y_boundary)

    # Interior tensors: only need x,t
    X_interior = interior_data[["x", "t"]].to_numpy(dtype=np.float32)
    X_interior = torch.tensor(X_interior)

    # Full test data
    test_data = all_data.copy()
    X_test = test_data[["x", "t"]].to_numpy(dtype=np.float32)
    X_test = torch.tensor(X_test)

    boundary_dataset = TensorDataset(X_boundary, y_boundary)
    interior_dataset = TensorDataset(X_interior)

    boundary_loader = DataLoader(
        boundary_dataset,
        batch_size=128,
        shuffle=True
    )

    interior_loader = DataLoader(
        interior_dataset,
        batch_size=128,
        shuffle=True
    )

    model = VanillaNN()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def flux(u):
        return u**2 / 2

    for epoch in range(Epochs):
        model.train()

        total_loss = 0
        total_data_loss = 0
        total_pde_loss = 0

        num_batches = min(len(boundary_loader), len(interior_loader))

        for (X_b, y_b), (X_f,) in zip(boundary_loader, interior_loader):

            # -------------------------
            # Data loss on IC/BC points
            # -------------------------
            pred_b = model(X_b)
            data_loss = loss_fn(pred_b, y_b)

            # -------------------------
            # PDE loss on interior points
            # -------------------------
            X_f = X_f.clone().detach().requires_grad_(True)

            pred_f = model(X_f)
            f_pred = flux(pred_f)

            grad_u = torch.autograd.grad(
                pred_f,
                X_f,
                grad_outputs=torch.ones_like(pred_f),
                create_graph=True
            )[0]

            u_t = grad_u[:, 1:2]

            grad_f = torch.autograd.grad(
                f_pred,
                X_f,
                grad_outputs=torch.ones_like(f_pred),
                create_graph=True
            )[0]

            f_x = grad_f[:, 0:1]

            residual = u_t + f_x
            pde_loss = torch.mean(residual**2)

            # -------------------------
            # Total PINN loss
            # -------------------------
            loss = data_loss + pde_weight * pde_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_data_loss += data_loss.item()
            total_pde_loss += pde_loss.item()

        if epoch % epoch_every == 0:
            avg_loss = total_loss / num_batches
            avg_data_loss = total_data_loss / num_batches
            avg_pde_loss = total_pde_loss / num_batches

            print(
                f"Epoch {epoch}, "
                f"Loss: {avg_loss:.6f}, "
                f"Data Loss: {avg_data_loss:.6f}, "
                f"PDE Loss: {avg_pde_loss:.6f}"
            )

    model.eval()

    with torch.no_grad():
        u_pred = model(X_test).numpy().flatten()

    prediction_data = test_data.copy()
    prediction_data["u_pred"] = u_pred
    prediction_data["abs_error"] = (prediction_data["u_pred"] - prediction_data["u"]).abs()
    prediction_data["MSE_contr"] = prediction_data["abs_error"]**2

    prediction_data.to_csv(
        f"Results/{output_filename}.csv",
        index=False
    )

    torch.save(
        model.state_dict(),
        f"Models/{output_filename}-pinn.pt"
    )



## PINN_method(input_filename="Data/M_500_literature_data.csv", interior_frac=0.10, Epochs=5000, epoch_every= 100, output_filename="PINN_literature_5000epochs")




def PINN_regression(model_filename, test_filename, output_filename):
    ## load in the data we want to regress on
    new_data = pd.read_csv(f"Data/{test_filename}.csv")

    X_new = new_data[["x", "t"]].to_numpy(dtype=np.float32)
    X_new = torch.tensor(X_new)
    model = VanillaNN()
    model.load_state_dict(
        torch.load(f"Models/{model_filename}.pt", map_location="cpu")
    )

    model.eval()

    with torch.no_grad():
        u_pred = model(X_new).numpy().flatten()

    new_data["u_pred"] = u_pred
    new_data["abs_error"] = (new_data["u_pred"] - new_data["u"]).abs()
    new_data["MSE_contr"] = new_data["abs_error"]**2

    new_data.to_csv(f"Results/{output_filename}.csv", index=False)

PINN_regression(model_filename="PINN_literature_5000epochs-pinn", test_filename="burgers_lf_tmax_2", output_filename="burgers_on_lit_trained_PINN")