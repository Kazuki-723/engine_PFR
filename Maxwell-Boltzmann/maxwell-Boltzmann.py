import numpy as np
import maxwellBoltzmann_func as mB

percent = mB.compute_mb_neighbor_distribution(
    dx=0.01,
    dt=5e-6,
    bulkflow=np.array([0.0, 0.0]),
    temperature=1500.0,
    pressure=101325.0,
    composition="CH4:0.05, O2:0.21, N2:0.74"
)

print(percent)