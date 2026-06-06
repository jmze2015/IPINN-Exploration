import numpy as np
import pandas as pd
import plotly.express as px

rng = np.random.default_rng()

## parameters 
M = 200 
L = 5 
dx = L / M 
dt = 0.8 * dx 

## Simulation Time
T_max = 5 
N = int(np.ceil(T_max / dt))

x = np.arange(-L/2, L/2+dx , dx)
t = np.arange(0,T_max+dt, dt)
a = rng.uniform(-5, 5, 10)

l_a = len(a)
l_x = len(x)
l_t = len(t)

T = np.tile(np.repeat(t, l_x), l_a)
X = np.tile(np.tile(x, l_t),l_a)
A = np.repeat(a, l_x * l_t) 


def g(x):
    return np.exp(-16*(x**2))

validation_data = pd.DataFrame({
    "x" : X,
    "t" : T,
    "time_step" : np.round(T / dt).astype(int),
    "a" : A,
    "u" : g(X - A*T)
})

validation_data.to_csv("Data\lin_advection_validation_data.csv", index = False)

N_train = 1000000
df2 = pd.DataFrame({
    "x" : rng.uniform(-L/2, L/2+dx, N_train),
    "t" : rng.uniform(0, T_max + dt, N_train),
    "a" : rng.uniform(-5, 5, N_train)    
})


df2["u"] = g(df2["x"]- df2["a"]*df2["t"])
df2.to_csv("Data\lin-adv-2.csv", index = False)


## validation_data[["x", "t"]] : yields column(s)
## validation_data.iloc[9] : returns 10th row
## validation_data.iloc[9:20] : returns rows 10 through 20

## df.loc[5, "u"] or df.iloc[5,2] : for specific entries
## df[df["a"] == 2] : conditional picking




# ## code to view time evolution of num soln. ## a[i] must be int
# selected = validation_data[
#     validation_data["time_step"].isin([0,10, 20, 30]) & 
#     validation_data["a"].isin([2])
#     ].copy()


# fig = px.line(
# selected,
# x="x",
# y="u",
# title="Solution at Selected Time Steps"
# )

# fig.update_layout(
#     xaxis_range = [-2,10],
#     yaxis_range = [-0.1, 1.1]
# )

# fig.show()


 

