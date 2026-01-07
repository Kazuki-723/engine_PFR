import cantera as ct
import csv

# ============================
# 1. Parameters
# ============================
p = ct.one_atm
Tin = 1500.0
comp = "CH4:1, O2:0.5, AR:0.5"

length = 0.36
A = 9.07e-4
mdot = 0.01

dt = 1e-7        # 小さめの dt
n_steps = 500000

# ============================
# 2. Gas & Reactor
# ============================
gas = ct.Solution("gri30.yaml")
gas.TPX = Tin, p, comp

# 反応器は閉じた系（流入なし）
r = ct.IdealGasConstPressureReactor(gas)

sim = ct.ReactorNet([r])

# ============================
# 3. Output
# ============================
f = open("pfr_noneq.csv", "w", newline="")
writer = csv.writer(f)
writer.writerow(["Distance (m)", "t (s)", "u (m/s)", "T (K)"] + gas.species_names)
# ============================
# 4. Marching
# ============================
x = 0.0
t = 0.0

for i in range(n_steps):

    rho = r.thermo.density
    u = mdot / (A * rho)

    dx = u * dt
    x += dx
    t += dt

    sim.advance(t)

    writer.writerow([x, t, u, r.T] + list(r.thermo.X))

    if x >= length:
        break

f.close()
print("Finished.")