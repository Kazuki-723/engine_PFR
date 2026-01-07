import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ============================
# 1. CSV 読み込み
# ============================
df = pd.read_csv("pfr_area_variation.csv")   # ← CSV 名を適宜変更

x = df["Distance (m)"]

# 存在する列を自動判定
columns = df.columns

# ============================
# 2. 温度プロファイル
# ============================
if "T (K)" in columns:
    plt.figure(figsize=(7,4))
    plt.plot(x, df["T (K)"], linewidth=2)
    plt.xlabel("Distance [m]")
    plt.ylabel("Temperature [K]")
    plt.title("Temperature Profile along PFR")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ============================
# 3. 流速プロファイル
# ============================
if "u (m/s)" in columns:
    plt.figure(figsize=(7,4))
    plt.plot(x, df["u (m/s)"], linewidth=2)
    plt.xlabel("Distance [m]")
    plt.ylabel("Velocity [m/s]")
    plt.title("Velocity Profile along PFR")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ============================
# 4. 圧力プロファイル
# ============================
if "P (Pa)" in columns:
    plt.figure(figsize=(7,4))
    plt.plot(x, df["P (Pa)"], linewidth=2)
    plt.xlabel("Distance [m]")
    plt.ylabel("Pressure [Pa]")
    plt.title("Pressure Profile along PFR")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ============================
# 5. 滞留時間プロファイル
# ============================
if "res_time (s)" in columns:
    plt.figure(figsize=(7,4))
    plt.plot(x, df["res_time (s)"], linewidth=2)
    plt.xlabel("Distance [m]")
    plt.ylabel("Residence Time [s]")
    plt.title("Residence Time along PFR")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ============================
# 6. 断面積プロファイル（任意）
# ============================
if "Area (m2)" in columns:
    plt.figure(figsize=(7,4))
    plt.plot(x, df["Area (m2)"], linewidth=2)
    plt.xlabel("Distance [m]")
    plt.ylabel("Area [m²]")
    plt.title("Cross-sectional Area Profile")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ============================
# 7. 主要化学種のモル分率
# ============================
species = ["CH4", "O2", "CO", "CO2", "H2O", "H2", "OH", "O", "AR", 
           "NO", "NO2", "N2O"]  # 必要に応じて追加

plt.figure(figsize=(10,6))

for sp in species:
    if sp in columns:
        plt.plot(x, df[sp], label=sp)

plt.xlabel("Distance [m]")
plt.ylabel("Mole Fraction")
plt.title("Species Profiles along PFR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()