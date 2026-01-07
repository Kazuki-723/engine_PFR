import cantera as ct
import csv

# ============================
# 1. Parameters
# ============================
p = ct.one_atm
Tin = 1500.0
comp_main = "CH4:1, O2:2, N2:7.52"     # 一段目組成
comp_CH4 = "CH4:1"      # 追加燃料（純メタン）

length = 0.36
A = 9.07e-4
mdot_main = 0.01
mdot_CH4 = 0.005                        # 追加する CH₄ の流量
mdot_total = mdot_main                 # 初期は主流のみ

dt = 1e-7
n_steps = 500000

x_inj = 0.18                            # CH₄ 注入位置
injection_done = False

# ============================
# 2. Gas & Reactor
# ============================
gas = ct.Solution("gri30.yaml")
gas.TPX = Tin, p, comp_main

gas_CH4 = ct.Solution("gri30.yaml")
gas_CH4.TPX = Tin, p, comp_CH4

r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])

# ============================
# 3. Output
# ============================
f = open("pfr_two_stage_noneq.csv", "w", newline="")
writer = csv.writer(f)
writer.writerow(["Distance (m)", "t (s)", "u (m/s)", "T (K)"] + gas.species_names)

# ============================
# 4. Marching
# ============================
x = 0.0
t = 0.0

for i in range(n_steps):

    rho = r.thermo.density
    u = mdot_total / (A * rho)
    dx = u * dt
    x += dx
    t += dt

    # ---- CH₄ 注入イベント ----
    if (not injection_done) and (x >= x_inj):
        injection_done = True

        # 一段目出口状態
        gas_main = ct.Solution("gri30.yaml")
        gas_main.TDY = r.thermo.TDY

        # 混合：質量分率
        Y_mix = (mdot_main * gas_main.Y + mdot_CH4 * gas_CH4.Y) / (mdot_main + mdot_CH4)

        # 混合：比エンタルピ
        h_mix = (mdot_main * gas_main.enthalpy_mass +
                 mdot_CH4 * gas_CH4.enthalpy_mass) / (mdot_main + mdot_CH4)

        # 混合状態を反応器にセット
        gas.HPY = h_mix, p, Y_mix
        r.syncState()

        # 流量を更新
        mdot_total = mdot_main + mdot_CH4

    sim.advance(t)

    writer.writerow([x, t, u, r.T] + list(r.thermo.X))

    if x >= length:
        break

f.close()
print("Finished.")