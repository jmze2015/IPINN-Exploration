import pandas as pd 
import numpy as np 
import torch 
from torch import nn
from torch.utils.data import TensorDataset, DataLoader 



"""
Vanilla NN Method
"""


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







def NN_method(input_filename, train_frac, Epochs, epoch_every, output_filename):
    """
    Loading analytical PDE solution or numerical solution data.
    Data is assumed to be a CSV with columns x, t, & u at a minimum.
    """
    
    all_data = pd.read_csv(input_filename)
    training_data = all_data.sample(frac= train_frac, random_state=123)

    test_data = all_data.copy()

    ## inputs and outputs
    X_train = training_data[["x", "t"]].to_numpy(dtype = np.float32)
    y_train = training_data[["u"]].to_numpy(dtype = np.float32)

    X_test = test_data[["x", "t"]].to_numpy(dtype = np.float32)
    y_test = test_data[["u"]].to_numpy(dtype = np.float32)

    ## Now convert to PyTorch tensors
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

    ## Defined outside of function
    model = VanillaNN()

    ## Typical MSE loss function
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    epochs = Epochs

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

        if epoch % epoch_every == 0:
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
        f"Results/{output_filename}.csv", index = False
    )

    torch.save(
        model.state_dict(),
        f"Models/{output_filename}-vanilla-nn.pt"
    )

"""
Example of how to run method below:

##NN_method("Data/lin-adv-2.csv", train_frac=0.10, Epochs= 200, epoch_every=10, output_filename="linear_advection_200_epochs")
"""

NN_method("Data/M_500_literature_data.csv", train_frac=0.10, Epochs= 1000, epoch_every=50, output_filename="NN_literature_1000_epochs")


def NN_regression(model_filename, test_filename, output_filename):
    ## load in the data we want to regress on
    new_data = pd.read_csv(f"Data/{test_filename}.csv")

    X_new = new_data[["x", "t"]].to_numpy(dtype=np.float32)
    X_new = torch.tensor(X_new)
    model = VanillaNN()
    model.load_state_dict(
        torch.load(f"Models/{model_filename}.pt")
    )

    model.eval()

    with torch.no_grad():
        u_pred = model(X_new).numpy().flatten()

    new_data["u_pred"] = u_pred
    new_data["abs_error"] = (new_data["u_pred"] - new_data["u"]).abs()
    new_data["MSE_contr"] = new_data["abs_error"]**2

    new_data.to_csv(f"Results/{output_filename}.csv", index=False)

    

## NN_regression(model_filename="M_500_literature-200epochs-vanilla-nn",  test_filename="burgers_lf_tmax_2", output_filename="burgers_on_lit_trained_NN")