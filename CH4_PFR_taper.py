import cantera as ct
import csv
import numpy as np

# ============================
# 1. Simulation parameters
# ============================
p = ct.one_atm
Tin = 1500.0
comp = 'CH4:1, O2:1, AR:0.5'

# ---- mdot を直接指定 ----
mdot = 0.01   # [kg/s]

length = 0.36
n_reactor = 1000
dx = length / n_reactor

# ---- 断面積を線形に変化させる ----
A_start = 9.07e-4      # 入口断面積
A_end   = 1.00e-2      # 出口断面積（例）
A_list = np.linspace(A_start, A_end, n_reactor)

# ============================
# 2. Gas & reactor setup
# ============================
gas = ct.Solution('gri30.yaml')
gas.TPX = Tin, p, comp

# 初期セルの体積
r = ct.IdealGasReactor(gas)
r.volume = A_list[0] * dx

upstream = ct.Reservoir(gas)
downstream = ct.Reservoir(gas)

mfc = ct.MassFlowController(upstream, r, mdot=mdot)
pc = ct.PressureController(r, downstream, primary=mfc, K=1e-5)

sim = ct.ReactorNet([r])

# ============================
# 3. Output
# ============================
outfile = open('pfr_area_variation.csv', 'w', newline='')
writer = csv.writer(outfile)
writer.writerow(['Distance (m)', 'Area (m2)', 'u (m/s)', 'res_time (s)',
                 'T (K)', 'P (Pa)'] + gas.species_names)

# ============================
# 4. Marching
# ============================
t_res = 0.0

for n in range(n_reactor):

    # ---- セル入口状態を upstream にコピー ----
    gas.TDY = r.thermo.TDY
    upstream.syncState()

    # ---- 定常まで解く ----
    sim.reinitialize()
    sim.advance_to_steady_state()

    # ---- 現在セルの断面積 ----
    A = A_list[n]

    # ---- 流速 ----
    rho = r.thermo.density
    u = mdot / (A * rho)

    # ---- 滞留時間 ----
    t_res += r.mass / mdot

    # ---- 出力 ----
    dist = (n + 1) * dx
    writer.writerow([dist, A, u, t_res, r.T, r.thermo.P] + list(r.thermo.X))

    # ---- 次セルの体積を更新 ----
    if n < n_reactor - 1:
        r.volume = A_list[n+1] * dx

outfile.close()
print("Finished. Saved to pfr_area_variation.csv")