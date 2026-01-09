import cantera as ct
import numpy as np
import csv

# Parameters
p = ct.one_atm
comp_air  = "O2:0.21, N2:0.79"
comp_fuel = "CH4:1, O2:2, N2:7.52"
T_air  = 300.0
T_fuel = 1500.0

A = 9.07e-4
vin = 30
Ny = 5

# time stepping
dt = 5e-4            # 初期 dt（安定条件で調整）
n_steps = 200000
length = 0.36

# physical diffusion coefficient (m^2/s) — 調整して試す
D_phys = 1e-4

# grid spacing in y (assume unit height normalized so dy = 1/Ny)
height = 0.01        # 実際の高さ [m]（適宜設定）
dy = height / Ny

# stability number
Dnum = D_phys * dt / (dy**2)
print(f"Dnum = {Dnum:.3e}")
if Dnum > 0.5:
    print(f"Warning: explicit diffusion unstable (Dnum={Dnum:.3e}). Reduce dt or D_phys.")
    # 自動で dt を下げる（安全策）
    dt = 0.4 * 0.5 * (dy**2) / D_phys
    Dnum = D_phys * dt / (dy**2)
    print(f"Adjusted dt -> {dt:.3e}, new Dnum={Dnum:.3e}")

# initialize gases and reactors
reactors = []
gas_list = []
for j in range(Ny):
    gas = ct.Solution("gri30.yaml")
    if j == 0:
        gas.TPX = T_fuel, p, comp_fuel
    else:
        gas.TPX = T_air, p, comp_air
    # seed tiny radicals in upper cells to help ignition sensitivity test
    if j > 0:
        # add tiny OH to upper cells (mass fraction)
        Y = gas.Y
        if "OH" in gas.species_names:
            idx = gas.species_index("OH")
            Y = Y.copy()
            Y[idx] += 1e-12
            Y /= Y.sum()
            gas.Y = Y
    gas_list.append(gas)

# choose reactor type (const-pressure recommended for ignition)
reactors = [ct.IdealGasConstPressureReactor(g) for g in gas_list]
sim = ct.ReactorNet(reactors)

# representative mdot and x update
rho0 = gas_list[0].density
mdot = rho0 * vin * A
x = 0.0
t = 0.0

# output
f = open("pfr_2d_diffusion_noneq.csv", "w", newline="")
writer = csv.writer(f)
writer.writerow(["x", "y", "t", "T"] + gas_list[0].species_names)

for step in range(n_steps):
    # advance by dt (incremental)
    sim.advance(sim.time + dt)
    t += dt

    # compute local densities and representative u (use average density or bottom cell)
    rho = np.array([r.thermo.density for r in reactors])
    # choose representative u per row (here use bottom cell density to compute u)
    u = mdot / (A * rho[0])
    x += u * dt
    if x > length:
        break

    # gather Y and enthalpy arrays
    Y = np.array([r.thermo.Y for r in reactors])   # shape (Ny, n_species)
    h = np.array([r.thermo.enthalpy_mass for r in reactors])

    # explicit diffusion in y for mass fractions and enthalpy
    Y_new = Y.copy()
    h_new = h.copy()

    for j in range(1, Ny-1):
        Y_new[j] = Y[j] + Dnum * (Y[j+1] - 2.0*Y[j] + Y[j-1])
        h_new[j] = h[j] + Dnum * (h[j+1] - 2.0*h[j] + h[j-1])

    # boundary treatment: allow top and bottom to exchange with neighbor (Neumann or Dirichlet)
    # here use one-sided diffusion for boundaries
    Y_new[0] = Y[0] + Dnum * (Y[1] - Y[0])
    Y_new[-1] = Y[-1] + Dnum * (Y[-2] - Y[-1])
    h_new[0] = h[0] + Dnum * (h[1] - h[0])
    h_new[-1] = h[-1] + Dnum * (h[-2] - h[-1])

    # clip and renormalize mass fractions to avoid negatives and ensure sum=1
    for j in range(Ny):
        Yj = np.clip(Y_new[j], 0.0, None)
        s = Yj.sum()
        if s <= 0:
            # fallback: keep previous Y
            Yj = Y[j].copy()
            s = Yj.sum()
        Y_new[j] = Yj / s

    # apply mixed states back to reactors (use HPY)
    for j in range(Ny):
        gas = ct.Solution("gri30.yaml")
        gas.HPY = h_new[j], p, Y_new[j]
        reactors[j].syncState()

    # output snapshot (optionally sample every N steps to reduce file size)
    for j in range(Ny):
        writer.writerow([x, j, t, reactors[j].T] + list(reactors[j].thermo.X))

f.close()
print("Finished. Steps:", step, "final x:", x)