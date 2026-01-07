import cantera as ct
import csv
import numpy as np

# ============================
# 1. Parameters
# ============================
p = ct.one_atm
Tin = 1500.0
comp = "CH4:1, O2:2, N2:7.52"

vin = 30.0
length = 0.36
A = 9.07e-4

Nx = 200      # x方向セル数（軽量化）
Ny = 5        # y方向ライン数（最小2D）
dx = length / Nx

# ============================
# 2. Gas template
# ============================
gas0 = ct.Solution("gri30.yaml")
gas0.TPX = Tin, p, comp
rho0 = gas0.density
mdot = rho0 * vin * A

# ============================
# 3. 2D PFR = Ny 本の 1D PFR
# ============================
results = []

for j in range(Ny):

    # --- 初期ガス ---
    gas = ct.Solution("gri30.yaml")
    gas.TPX = Tin, p, comp

    # --- 反応器 ---
    r = ct.IdealGasReactor(gas)
    r.volume = A * dx / Ny

    upstream = ct.Reservoir(gas)
    downstream = ct.Reservoir(gas)

    mfc = ct.MassFlowController(upstream, r, mdot=mdot/Ny)
    pc = ct.PressureController(r, downstream, primary=mfc, K=1e-5)

    sim = ct.ReactorNet([r])

    # --- 1D PFR を x 方向に進める ---
    for i in range(Nx):

        # upstream に現在状態をコピー
        gas.TDY = r.thermo.TDY
        upstream.syncState()

        # steady-state
        sim.reinitialize()
        sim.advance_to_steady_state()

        x = (i+1)*dx
        y = j

        results.append([x, y, r.T] + list(r.thermo.X))

# ============================
# 4. CSV 出力
# ============================
with open("pfr_2d.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y", "T"] + gas0.species_names)
    writer.writerows(results)

print("Finished minimal 2D PFR.")