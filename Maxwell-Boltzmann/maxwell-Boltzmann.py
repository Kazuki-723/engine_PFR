import numpy as np
import matplotlib.pyplot as plt
import MaxwellBoltzmann_func as MB
# ---------------------------------------------------------
# 100×100 メッシュで非反応の移流拡散を計算
# ---------------------------------------------------------
def simulate_advection_diffusion(
        nx=100, ny=100,
        dx=0.01, dt=5e-4,
        bulkflow=np.array([1000.0, 0.0]),
        temperature=1500.0,
        pressure=101325.0,
        composition_air="O2:0.21, N2:0.79",
        composition_inlet="CH4:0.05, O2:0.21, N2:0.74",
        steps=200
    ):
    """
    Maxwell-Boltzmann ベースの遷移確率を使って
    100×100 メッシュの非反応移流拡散を計算する。

    inlet 以外は空気組成で初期化する。
    """

    # --- 1. 遷移確率を inlet の組成で計算 ---
    P = MB.compute_mb_neighbor_distribution(
        dx=dx,
        dt=dt,
        bulkflow=bulkflow,
        temperature=temperature,
        pressure=pressure,
        composition=composition_inlet,
        n_particles=20000
    )

    # 近傍遷移確率
    w = {k: P[k] / 100.0 for k in P}

    # --- 2. メッシュ初期化（空気組成） ---
    C = np.zeros((ny, nx))  # スカラー場（濃度）

    # inlet（左端中央10セル）を 1.0 に固定
    mid = ny // 2
    C[mid-5:mid+5, 0] = 1.0

    inlet_mask = np.zeros_like(C, dtype=bool)
    inlet_mask[mid-5:mid+5, 0] = True

    # --- 3. 時間発展 ---
    for step in range(steps):

        C_new = np.zeros_like(C)

        for j in range(ny):
            for i in range(nx):

                val = C[j, i]

                C_new[j, i] += val * w["center"]

                if i+1 < nx: C_new[j, i+1] += val * w["p1"]
                if j+1 < ny: C_new[j+1, i] += val * w["p2"]
                if i-1 >= 0: C_new[j, i-1] += val * w["p3"]
                if j-1 >= 0: C_new[j-1, i] += val * w["p4"]

                if i+1 < nx and j+1 < ny: C_new[j+1, i+1] += val * w["p5"]
                if i-1 >= 0 and j+1 < ny: C_new[j+1, i-1] += val * w["p6"]
                if i-1 >= 0 and j-1 >= 0: C_new[j-1, i-1] += val * w["p7"]
                if i+1 < nx and j-1 >= 0: C_new[j-1, i+1] += val * w["p8"]

        # inlet を固定
        C_new[inlet_mask] = 1.0

        C = C_new

    return C


if __name__ == "__main__":

    # パラメータ設定
    nx = 100
    ny = 100
    dx = 0.01
    dt = 5e-6
    bulkflow = np.array([0.0, 0.0])
    temperature = 1500.0
    pressure = 101325.0
    composition_air = "O2:0.21, N2:0.79"
    composition_inlet = "CH4:0.05, O2:0.21, N2:0.74"

    # 200 step の移流拡散シミュレーション
    C = simulate_advection_diffusion(
        nx=nx,
        ny=ny,
        dx=dx,
        dt=dt,
        bulkflow=bulkflow,
        temperature=temperature,
        pressure=pressure,
        composition_air=composition_air,
        composition_inlet=composition_inlet,
        steps=200
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(C, origin='lower', cmap='inferno')
    plt.colorbar(label="Concentration")
    plt.title("Advection-Diffusion Concentration Field")
    plt.xlabel("x index")
    plt.ylabel("y index")
    plt.tight_layout()
    plt.show()
