import cantera as ct
import numpy as np
import csv

# ============================
# 1. Parameters
# ============================
p = ct.one_atm

comp_air  = "O2:0.21, N2:0.79"
comp_fuel = "CH4:1, O2:2, N2:7.52"

T_air  = 300.0
T_fuel = 1500.0

A   = 9.07e-4      # 断面積
vin = 0.1         # 代表流速 [m/s]
mdot = None        # 後で初期状態から決める

Ny = 5             # y セル数
dt = 1e-6          # 時間ステップ
n_steps = 50000
length = 0.36

Dmix = 0.8         # y方向の「拡散強さ」（0〜1程度で調整）

# ============================
# 2. Gas & Reactors
# ============================
reactors = []
gas_list = []

for j in range(Ny):
    gas = ct.Solution("gri30.yaml")
    if j == 0:
        gas.TPX = T_fuel, p, comp_fuel   # 下段ホットジェット
    else:
        gas.TPX = T_air, p, comp_air     # 他は空気
    gas_list.append(gas)

# 初期密度から mdot を決める（代表値）
rho0 = gas_list[0].density
mdot = rho0 * vin * A

for j in range(Ny):
    r = ct.IdealGasConstPressureReactor(gas_list[j])
    reactors.append(r)

sim = ct.ReactorNet(reactors)

# ============================
# 3. 出力
# ============================
f = open("pfr_2d_diffusion_noneq.csv", "w", newline="")
writer = csv.writer(f)
writer.writerow(["x", "y", "T"] + gas_list[0].species_names)

# ============================
# 4. Time marching
# ============================
t = 0.0
x = 0.0

for n in range(n_steps):
    t += dt

    # ---- まず全セルを非平衡で時間積分 ----
    sim.advance(t)

    # ---- 各セルの局所速度から代表 u を計算（ここでは下段を代表に）----
    rho = reactors[0].thermo.density
    u = mdot / (A * rho)
    x += u * dt
    if x > length:
        break

    # ---- y方向の拡散（explicit mixing）----
    Y = np.array([r.thermo.Y for r in reactors])
    h = np.array([r.thermo.enthalpy_mass for r in reactors])

    Y_new = Y.copy()
    h_new = h.copy()

    for j in range(1, Ny-1):
        Y_new[j] = Y[j] + Dmix * (Y[j+1] - 2*Y[j] + Y[j-1])
        h_new[j] = h[j] + Dmix * (h[j+1] - 2*h[j] + h[j-1])

    # 境界はそのまま（j=0 はジェット、j=Ny-1 は外側）
    # 必要なら壁条件などをここに入れる

    # ---- 拡散後の状態を反応器に反映 ----
    for j in range(1, Ny-1):
        gas = ct.Solution("gri30.yaml")
        gas.HPY = h_new[j], p, Y_new[j]
        reactors[j].syncState()

    # ---- 出力 ----
    for j in range(Ny):
        writer.writerow([x, j, reactors[j].T] + list(reactors[j].thermo.X))

f.close()
print("Finished.")