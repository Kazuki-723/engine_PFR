import numpy as np
import cantera as ct

def compute_mb_neighbor_distribution(
        dx,
        dt,
        bulkflow,
        temperature,
        pressure,
        composition,
        n_particles=20000,
        random_seed=12345
    ):
    """
    Maxwell-Boltzmann 粒子を dt だけ移動させ、
    中心セル (0,0) から 8 近傍セルへの遷移確率を返す関数。

    Parameters
    ----------
    dx : float
        セル幅（dy も同じとする）
    dt : float
        微小時間ステップ
    bulkflow : array-like (2,)
        大域流速ベクトル [Ux, Uy]
    temperature : float
        温度 [K]
    pressure : float
        圧力 [Pa]
    composition : str
        Cantera 形式の組成文字列
    n_particles : int
        サンプリング粒子数
    random_seed : int
        乱数シード

    Returns
    -------
    percent : dict
        center, p1〜p8, others の分布確率（%）
    """

    # --- 物理定数 ---
    kB = 1.380649e-23
    NA = 6.02214076e23

    # --- Cantera で平均分子量を取得 ---
    gas = ct.Solution("gri30.yaml")
    gas.TPX = temperature, pressure, composition
    mean_M_kg_per_mol = gas.mean_molecular_weight / 1000.0  # kg/mol

    # 粒子質量
    m_particle = mean_M_kg_per_mol / NA

    # --- Maxwell-Boltzmann 速度分布 ---
    sigma2 = kB * temperature / m_particle
    sigma = np.sqrt(sigma2)

    rng = np.random.default_rng(random_seed)
    vx_thermal = rng.normal(0.0, sigma, n_particles)
    vy_thermal = rng.normal(0.0, sigma, n_particles)

    # 合成速度
    vx_total = vx_thermal + bulkflow[0]
    vy_total = vy_thermal + bulkflow[1]

    # --- 粒子移動 ---
    x_final = vx_total * dt
    y_final = vy_total * dt

    # --- セルインデックス判定 ---
    cell_x = np.floor((x_final + dx/2) / dx).astype(int)
    cell_y = np.floor((y_final + dx/2) / dx).astype(int)

    # --- カウント ---
    counts = {
        "center": 0,
        "p1": 0,  # (+1, 0)
        "p2": 0,  # (0, +1)
        "p3": 0,  # (-1, 0)
        "p4": 0,  # (0, -1)
        "p5": 0,  # (+1, +1)
        "p6": 0,  # (-1, +1)
        "p7": 0,  # (-1, -1)
        "p8": 0,  # (+1, -1)
        "others": 0
    }

    # ベクトル化判定
    counts["center"] = np.sum((cell_x == 0) & (cell_y == 0))
    counts["p1"]     = np.sum((cell_x == 1) & (cell_y == 0))
    counts["p2"]     = np.sum((cell_x == 0) & (cell_y == 1))
    counts["p3"]     = np.sum((cell_x == -1) & (cell_y == 0))
    counts["p4"]     = np.sum((cell_x == 0) & (cell_y == -1))
    counts["p5"]     = np.sum((cell_x == 1) & (cell_y == 1))
    counts["p6"]     = np.sum((cell_x == -1) & (cell_y == 1))
    counts["p7"]     = np.sum((cell_x == -1) & (cell_y == -1))
    counts["p8"]     = np.sum((cell_x == 1) & (cell_y == -1))

    mask_known = (
        (cell_x == 0) & (cell_y == 0) |
        (cell_x == 1) & (cell_y == 0) |
        (cell_x == 0) & (cell_y == 1) |
        (cell_x == -1) & (cell_y == 0) |
        (cell_x == 0) & (cell_y == -1) |
        (cell_x == 1) & (cell_y == 1) |
        (cell_x == -1) & (cell_y == 1) |
        (cell_x == -1) & (cell_y == -1) |
        (cell_x == 1) & (cell_y == -1)
    )

    counts["others"] = np.sum(~mask_known)

    # --- 割合（%） ---
    percent = {k: 100.0 * v / n_particles for k, v in counts.items()}

    return percent