import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("pfr_2d.csv")

x_vals = np.sort(df["x"].unique())
y_vals = np.sort(df["y"].unique())

Nx = len(x_vals)
Ny = len(y_vals)

def to_2d(field):
    Z = np.zeros((Ny, Nx))
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            Z[j, i] = df[(df["x"]==x)&(df["y"]==y)][field].values[0]
    return Z

T2d = to_2d("T")

plt.contourf(x_vals, y_vals, T2d, levels=50, cmap="inferno")
plt.colorbar(label="Temperature [K]")
plt.xlabel("x [m]")
plt.ylabel("y index")
plt.title("Minimal 2D PFR Temperature")
plt.show()