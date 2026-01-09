import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
# ----------------------------
# ユーザー設定（ここを変更）
# ----------------------------
# Cantera 経由で平均分子量を得る場合（優先）
composition = "CH4:0.05, O2:0.21, N2:0.74"  # None にすると下の mean_M を使う

# 温度・圧力
temperature = 1500.0  # K
pressure = 101325.0   # Pa

# メッシュと時間刻み
dx = 0.01   # m (セル幅 x)
dy = 0.01   # m (セル幅 y)
dt = 5e-6   # s (微小時間)

# 大域流速（m/s）
bulk_flow = np.array([0.0, 0.0])  # (Ux, Uy)

# モンテカルロ設定
n_particles = 20000        # サンプル粒子数（統計精度に応じて増減）
plot_vectors = 300         # 描画する代表粒子数（上限）
random_seed = 12345

# ----------------------------
# 物理定数
# ----------------------------
kB = 1.380649e-23       # J/K
NA = 6.02214076e23      # 1/mol

# ----------------------------
# 平均分子量の決定（Cantera)
# ----------------------------
gas = ct.Solution("gri30.yaml")
gas.TPX = temperature, pressure, composition
# Cantera の mean_molecular_weight は kg/kmol（= kg per kmol）
mean_M_kg_per_kmol = gas.mean_molecular_weight
mean_M_kg_per_mol = mean_M_kg_per_kmol / 1000.0

# 粒子質量 [kg]
m_particle = mean_M_kg_per_mol / NA

print(f"mean_M = {mean_M_kg_per_mol:.6e} kg/mol, particle mass = {m_particle:.3e} kg")
print(f"T = {temperature} K, bulk_flow = {bulk_flow}, dt = {dt} s, dx,dy = {dx},{dy} m")
print(f"n_particles = {n_particles}")

# ----------------------------
# Maxwell-Boltzmann による速度成分サンプリング（2D）
# 2D 各成分は正規分布 N(0, sigma^2) に従う（独立）
# sigma^2 = k_B T / m
# ----------------------------
sigma2 = kB * temperature / m_particle
sigma = np.sqrt(sigma2)

# 理論値算出+dtの再決定
average_th = np.sqrt(8/np.pi * sigma2)
# dt = dx / average_th * 0.5
# print(f"overwrite dt = {dt} sec")

rng = np.random.default_rng(random_seed)
vx_thermal = rng.normal(loc=0.0, scale=sigma, size=n_particles)
vy_thermal = rng.normal(loc=0.0, scale=sigma, size=n_particles)

# 合成速度 = 熱速度 + 大域流速
vx_total = vx_thermal + bulk_flow[0]
vy_total = vy_thermal + bulk_flow[1]

# ----------------------------
# 初期位置: 原点中心のセル (i,j) の中心に置く
# セル (i,j) の領域は x in [-dx/2, dx/2], y in [-dy/2, dy/2]
# 微小時間 dt 後の位置を計算して、どのセルに入るか判定する
# 目的セルインデックスの取り扱い:
#   center: (0,0)
#   neighbors: (±1, ±1) の 4 つ
# ----------------------------
# 初期位置（セル中心）
x0 = 0.0
y0 = 0.0

# 移動
x_final = x0 + vx_total * dt
y_final = y0 + vy_total * dt

# セルインデックス（中心セルを (0,0) とする格子座標）
# index_x = floor((x - (-dx/2)) / dx) - offset 中心基準で計算
cell_x = np.floor((x_final + dx/2.0) / dx).astype(int)
cell_y = np.floor((y_final + dy/2.0) / dy).astype(int)

# カウント
counts = {
    "center": 0,
    "p1": 0,  # ( +1,  0 )
    "p2": 0,  # (  0, +1 )
    "p3": 0,  # ( -1,  0 )
    "p4": 0,  # (  0, -1 )
    "p5": 0,  # ( +1, +1 )
    "p6": 0,  # ( -1, +1 )
    "p7": 0,  # ( -1, -1 )
    "p8": 0,  # ( +1, -1 )    
    "others": 0
}

# 判定ループ（ベクトル化して高速化）
# mask_center: cell_x == 0 and cell_y == 0
mask_center = (cell_x == 0) & (cell_y == 0)
counts["center"] = np.count_nonzero(mask_center)

mask_p1 = (cell_x == 1) & (cell_y == 0)
counts["p1"] = np.count_nonzero(mask_p1)

mask_p2 = (cell_x == 0) & (cell_y == 1)
counts["p2"] = np.count_nonzero(mask_p2)

mask_p3 = (cell_x == -1) & (cell_y == 0)
counts["p3"] = np.count_nonzero(mask_p3)

mask_p4 = (cell_x == 0) & (cell_y == -1)
counts["p4"] = np.count_nonzero(mask_p4)

mask_p5 = (cell_x == 1) & (cell_y == 1)
counts["p5"] = np.count_nonzero(mask_p5)

mask_p6 = (cell_x == -1) & (cell_y == 1)
counts["p6"] = np.count_nonzero(mask_p6)

mask_p7 = (cell_x == -1) & (cell_y == -1)
counts["p7"] = np.count_nonzero(mask_p7)

mask_p8 = (cell_x == 1) & (cell_y == -1)
counts["p8"] = np.count_nonzero(mask_p8)

# others: それ以外（遠方へ行った粒子）
mask_others = ~(mask_center | mask_p1 | mask_p2 | mask_p3 | mask_p4 | mask_p5 | mask_p6 | mask_p7 | mask_p8)
counts["others"] = np.count_nonzero(mask_others)

# 割合（%）
percent = {k: 100.0 * v / n_particles for k, v in counts.items()}

print("\nParticle distribution after dt:")
for k in ["center", "p1", "p2", "p3", "p4", "others"]:
    print(f"  {k:7s}: {counts[k]:6d} particles  -> {percent[k]:6.3f} %")

# ----------------------------
# 可視化 1: 3x3 ヒートマップ（中心と 8 近傍を表示）
# ただし我々は 4 近傍のみ集計しているため、その他は 'others' に含める
# 配列配置（y 上が正）
#   [ ( -1, +1 ) , ( 0, +1 ) , ( +1, +1 ) ]
#   [ ( -1,  0 ) , ( 0,  0 ) , ( +1,  0 ) ]
#   [ ( -1, -1 ) , ( 0, -1 ) , ( +1, -1 ) ]
# ----------------------------
grid = np.zeros((3, 3))
# map counts to grid
# center at (1,1)  
grid = np.zeros((3, 3))
grid[0, 0] = percent["p6"]   # (-1,+1)
grid[0, 1] = percent["p2"]   # (0,+1) 
grid[0, 2] = percent["p5"]   # (+1,+1)
grid[1, 0] = percent["p3"]   # (-1,0)
grid[1, 1] = percent["center"]
grid[1, 2] = percent["p1"]   # (+1,0)
grid[2, 0] = percent["p7"]   # (-1,-1)
grid[2, 1] = percent["p4"]   # (0,-1)
grid[2, 2] = percent["p8"]   # (+1,-1)

# others を別表示（凡例）
others_pct = percent["others"]

fig = plt.figure(figsize=(10, 5))

# heatmap
ax1 = fig.add_subplot(1, 2, 1)
im = ax1.imshow(grid, origin='upper', cmap='viridis', vmin=0, vmax=np.max(grid) if np.max(grid)>0 else 1)
ax1.set_title("Percent to neighbor cells (center at middle) [%]")
ax1.set_xticks([0,1,2])
ax1.set_yticks([0,1,2])
ax1.set_xticklabels(['x=-1','x=0','x=+1'])
ax1.set_yticklabels(['y=+1','y=0','y=-1'])
for (i, j), val in np.ndenumerate(grid):
    ax1.text(j, i, f"{val:5.2f}%", ha='center', va='center', color='white' if val>0.5*np.max(grid) else 'black')
ax1.text(0.02, -0.12, f"others: {others_pct:.3f} %", transform=ax1.transAxes)

plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

# ----------------------------
# 可視化 2: 代表粒子ベクトル（原点から）
# ----------------------------
ax2 = fig.add_subplot(1, 2, 2)
# choose subset to plot
n_plot = min(plot_vectors, n_particles)
idx_plot = rng.choice(n_particles, size=n_plot, replace=False)
U = vx_total[idx_plot]
V = vy_total[idx_plot]

# scale arrows for display: choose scale so arrows visible
scale_display = 1.0  # no automatic scaling; adjust if needed
# plot arrows from origin
ax2.quiver(np.zeros(n_plot), np.zeros(n_plot), U, V, angles='xy', scale_units='xy', scale=scale_display, width=0.003, color='C1', alpha=0.7)
ax2.scatter(0, 0, color='k', s=20)
ax2.set_title(f"Representative particle vectors (n={n_plot})")
ax2.set_xlabel("vx [m/s]")
ax2.set_ylabel("vy [m/s]")
ax2.grid(True)
ax2.axis('equal')

# set axis limits based on percentiles to avoid extreme outliers dominating
lim = np.percentile(np.abs(np.concatenate([U, V])), 99) * 1.2
if lim == 0:
    lim = 1e-6
ax2.set_xlim(-lim, lim)
ax2.set_ylim(-lim, lim)

plt.tight_layout()
plt.show()

# ----------------------------
# 結果の要約出力
# ----------------------------
print("\nSummary (percent):")
for k in ["center", "p1", "p2", "p3", "p4", "others"]:
    print(f"  {k:7s}: {percent[k]:6.3f} %")

velosity = np.zeros_like(vx_total)
for i in range(n_particles):
    velosity[i] = np.sqrt(vx_total[i] ** 2 + vy_total[i] ** 2)
average_velosity = np.average(velosity)
print(f"average velosity ={average_velosity:6.3f} m/s")
print(f"theoritical average velosity(with no bulk) ={average_th:6.3f} m/s")
