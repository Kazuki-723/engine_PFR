import cantera as ct
import csv

# ============================
# 1. Simulation parameters
# ============================
p = ct.one_atm          # 圧力 [Pa]
Tin = 1500.0            # 入口温度 [K]
comp = 'CH4:1, O2:1, AR:0.5'  # 入口組成（モル比）

mdot = 0.01             # 質量流量を直接指定 [kg/s]

length = 0.36           # 反応器全長 [m]
area = 9.07e-4          # 断面積 [m^2]
n_reactor = 1000        # 分割セル数

# ============================
# 2. Gas & mass flow settings
# ============================
gas = ct.Solution('gri30.yaml')
gas.TPX = Tin, p, comp

dx = length / n_reactor

# ============================
# 3. Reactor setup
# ============================
r = ct.IdealGasReactor(gas)
r.volume = area * dx

upstream = ct.Reservoir(gas, name='upstream')
downstream = ct.Reservoir(gas, name='downstream')

# mfc作成
mfc = ct.MassFlowController(upstream, r, mdot=mdot)

pc = ct.PressureController(r, downstream, primary=mfc, K=1.0e-5)

sim = ct.ReactorNet([r])

# ============================
# 4. Output
# ============================
outfile = open('pfr_CH4_O2_Ar_mdot.csv', 'w', newline='')
writer = csv.writer(outfile)

header = ['Distance (m)', 'u (m/s)', 'res_time (s)',
          'T (K)', 'P (Pa)'] + gas.species_names
writer.writerow(header)

# ============================
# 5. PFR marching
# ============================
t_res = 0.0

for n in range(n_reactor):

    # セル入口状態を upstream にコピー
    gas.TDY = r.thermo.TDY
    upstream.syncState()

    # 定常まで解く
    sim.reinitialize()
    sim.advance_to_steady_state()

    # 距離
    dist = (n + 1) * dx

    # 流速 u = mdot / (A * ρ)
    rho = r.thermo.density
    u = mdot / (area * rho)

    # 滞留時間
    t_res += r.mass / mdot

    # 出力
    row = [dist, u, t_res, r.T, r.thermo.P] + list(r.thermo.X)
    writer.writerow(row)

outfile.close()

print("Finished. Results saved to 'pfr_CH4_O2_Ar_mdot.csv'")