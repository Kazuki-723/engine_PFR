import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
from scipy.optimize import root


# ============================================================
# Geometry: Laval nozzle A(x)
# ============================================================
def laval_area_cosine(x, *, A_in, A_t, A_out, x_t, L):
    """
    Smooth converging-diverging nozzle using cosine blending.
    Returns A(x), dA/dx (dA/dx is not used in the isentropic solver, but kept for completeness).
    """
    x = float(x)
    if x <= x_t:
        c = np.cos(np.pi * x / x_t)
        s = np.sin(np.pi * x / x_t)
        A = A_t + (A_in - A_t) * 0.5 * (1.0 + c)
        dAdx = (A_in - A_t) * 0.5 * (-np.pi / x_t) * s
    else:
        xi = (x - x_t) / (L - x_t)
        c = np.cos(np.pi * xi)
        s = np.sin(np.pi * xi)
        A = A_t + (A_out - A_t) * 0.5 * (1.0 - c)
        dAdx = (A_out - A_t) * 0.5 * (np.pi / (L - x_t)) * s
    return A, dAdx


# ============================================================
# Thermo helpers
# ============================================================
def set_state_TP(gas, T, P, X0, chem_mode):
    """
    Set state at (T,P) with either frozen or equilibrium chemistry.
    - frozen: keep composition X0
    - equilibrium: equilibrate at (T,P)
    """
    gas.TPX = T, P, X0
    if chem_mode == "equilibrium":
        gas.equilibrate("TP")


def eval_thermo(gas, T, P, X0, chem_mode):
    """
    Returns (rho, a, s, h) for given (T,P) under chosen chemistry mode.
      rho: density [kg/m^3]
      a: sound speed [m/s]
      s: mass entropy [J/kg/K]
      h: mass enthalpy [J/kg]
    """
    set_state_TP(gas, T, P, X0, chem_mode)
    rho = gas.density
    a = gas.sound_speed
    s = gas.entropy_mass
    h = gas.enthalpy_mass
    return rho, a, s, h


# ============================================================
# 1) Find choked throat state and mdot* from stagnation state
# ============================================================
def choked_mdot_from_stagnation(gas0, A_t, chem_mode="frozen", Cd=1.0):
    """
    Given stagnation state gas0 (T0,P0,X0), compute choked mdot* for throat area A_t.

    We solve for throat (Tt,Pt) using:
      s(Tt,Pt) = s0
      h(Tt,Pt) + 0.5*a(Tt,Pt)^2 = h0   (since u=a at M=1)
    Then mdot* = Cd * rho_t * a_t * A_t
    """
    X0 = gas0.X  # composition reference

    # If equilibrium mode, also equilibrate stagnation state first (consistent definition of s0,h0)
    if chem_mode == "equilibrium":
        gas0.equilibrate("TP")

    s0 = gas0.entropy_mass
    h0 = gas0.enthalpy_mass
    T0 = gas0.T
    P0 = gas0.P

    def F(z):
        logT, logP = z
        T = np.exp(logT)
        P = np.exp(logP)
        rho, a, s, h = eval_thermo(gas0, T, P, X0, chem_mode)
        u = a  # choked
        return np.array([s - s0, (h + 0.5 * u * u) - h0], dtype=float)

    # Initial guess
    z0 = np.array([np.log(0.85 * T0), np.log(0.55 * P0)], dtype=float)

    sol = root(F, z0, method="hybr")
    if not sol.success:
        raise RuntimeError(f"[choke] root failed: {sol.message}")

    Tt = float(np.exp(sol.x[0]))
    Pt = float(np.exp(sol.x[1]))

    rho_t, a_t, s_t, h_t = eval_thermo(gas0, Tt, Pt, X0, chem_mode)
    mdot_star = Cd * rho_t * a_t * A_t

    info = {
        "Tt": Tt,
        "Pt": Pt,
        "rho_t": float(rho_t),
        "a_t": float(a_t),
        "s0": float(s0),
        "h0": float(h0),
        "s_t": float(s_t),
        "h_t": float(h_t),
        "Cd": float(Cd),
        "chem_mode": chem_mode,
    }
    return float(mdot_star), info


# ============================================================
# 2) Given mdot and A(x), solve isentropic nozzle state at each x
# ============================================================
def solve_isentropic_state_for_area(gas, A, mdot, X0, s0, h0, chem_mode, guess_TP):
    """
    Solve (T,P) for given area A and fixed mdot under:
      s(T,P)=s0
      h(T,P) + 0.5*u^2 = h0, where u = mdot / (rho(T,P)*A)

    Unknowns: (T,P) -> use log variables for positivity.
    """

    def G(z):
        logT, logP = z
        T = np.exp(logT)
        P = np.exp(logP)
        rho, a, s, h = eval_thermo(gas, T, P, X0, chem_mode)
        u = mdot / (rho * A)
        return np.array([s - s0, (h + 0.5 * u * u) - h0], dtype=float)

    logT0 = np.log(max(guess_TP[0], 1.0))
    logP0 = np.log(max(guess_TP[1], 1.0))
    z0 = np.array([logT0, logP0], dtype=float)

    sol = root(G, z0, method="hybr")
    if not sol.success:
        return None

    T = float(np.exp(sol.x[0]))
    P = float(np.exp(sol.x[1]))
    rho, a, s, h = eval_thermo(gas, T, P, X0, chem_mode)
    u = mdot / (rho * A)
    M = u / a
    return {
        "T": T,
        "P": P,
        "rho": float(rho),
        "a": float(a),
        "u": float(u),
        "M": float(M),
        "s": float(s),
        "h": float(h),
    }


def solve_nozzle_profile(
    gas0,
    *,
    A_in,
    A_t,
    A_out,
    L,
    x_t,
    n_points=300,
    chem_mode="frozen",
    Cd=1.0,
):
    """
    Full pipeline:
      - compute mdot* from stagnation state and throat area
      - solve isentropic states along x with continuation (subsonic -> throat -> supersonic)
    """
    gas = gas0
    X0 = gas.X

    # Ensure stagnation reference s0,h0 (consistent with mode)
    if chem_mode == "equilibrium":
        gas.equilibrate("TP")
    s0 = gas.entropy_mass
    h0 = gas.enthalpy_mass
    T0 = gas.T
    P0 = gas.P

    mdot_star, choke_info = choked_mdot_from_stagnation(
        gas, A_t, chem_mode=chem_mode, Cd=Cd
    )

    # Grid and area
    x = np.linspace(0.0, L, n_points)
    A = np.zeros_like(x)
    for i, xi in enumerate(x):
        A[i], _ = laval_area_cosine(xi, A_in=A_in, A_t=A_t, A_out=A_out, x_t=x_t, L=L)

    # Storage
    T = np.full_like(x, np.nan, dtype=float)
    P = np.full_like(x, np.nan, dtype=float)
    rho = np.full_like(x, np.nan, dtype=float)
    u = np.full_like(x, np.nan, dtype=float)
    a = np.full_like(x, np.nan, dtype=float)
    M = np.full_like(x, np.nan, dtype=float)

    # Find index nearest throat
    it = int(np.argmin(np.abs(x - x_t)))

    # --- Upstream (0 -> throat): subsonic continuation
    guess = (T0, P0)
    for i in range(0, it + 1):
        tries = [
            guess,
            (0.95 * guess[0], 0.90 * guess[1]),
            (1.02 * guess[0], 1.05 * guess[1]),
        ]
        sol_i = None
        for g in tries:
            sol_i = solve_isentropic_state_for_area(
                gas, A[i], mdot_star, X0, s0, h0, chem_mode, g
            )
            if (
                sol_i is not None and sol_i["M"] < 1.05
            ):  # upstream should be subsonic-ish
                break
        if sol_i is None:
            raise RuntimeError(
                f"[upstream] Failed at i={i}, x={x[i]:.6g}, A={A[i]:.6g}"
            )

        T[i], P[i], rho[i], u[i], a[i], M[i] = (
            sol_i["T"],
            sol_i["P"],
            sol_i["rho"],
            sol_i["u"],
            sol_i["a"],
            sol_i["M"],
        )
        guess = (T[i], P[i])

    # --- Downstream (throat -> L): aim for supersonic continuation
    # Start guess slightly biased to lower T,P (tends to higher velocity solution branch)
    guess = (0.95 * T[it], 0.80 * P[it])
    for i in range(it + 1, len(x)):
        tries = [
            guess,
            (0.90 * guess[0], 0.70 * guess[1]),
            (0.98 * T[i - 1], 0.85 * P[i - 1]),
        ]
        sol_i = None
        for g in tries:
            sol_i = solve_isentropic_state_for_area(
                gas, A[i], mdot_star, X0, s0, h0, chem_mode, g
            )
            if (
                sol_i is not None and sol_i["M"] > 0.95
            ):  # downstream should be >=~1 if fully expanded no-shock branch
                break
        if sol_i is None:
            # If it stubbornly converges to subsonic branch, we still return it, but warn via exception message
            # (You can relax this depending on use.)
            sol_i = solve_isentropic_state_for_area(
                gas, A[i], mdot_star, X0, s0, h0, chem_mode, (T[i - 1], P[i - 1])
            )
            if sol_i is None:
                raise RuntimeError(
                    f"[downstream] Failed at i={i}, x={x[i]:.6g}, A={A[i]:.6g}"
                )

        T[i], P[i], rho[i], u[i], a[i], M[i] = (
            sol_i["T"],
            sol_i["P"],
            sol_i["rho"],
            sol_i["u"],
            sol_i["a"],
            sol_i["M"],
        )
        guess = (T[i], P[i])

    return {
        "x": x,
        "A": A,
        "T": T,
        "P": P,
        "rho": rho,
        "u": u,
        "a": a,
        "M": M,
        "mdot": mdot_star,
        "choke_info": choke_info,
        "chem_mode": chem_mode,
        "stagnation": {"T0": T0, "P0": P0, "s0": float(s0), "h0": float(h0)},
        "geometry": {
            "A_in": A_in,
            "A_t": A_t,
            "A_out": A_out,
            "L": L,
            "x_t": x_t,
            "n_points": n_points,
        },
    }


# ============================================================
# Plotting
# ============================================================
def plot_results(res):
    x = res["x"]
    A = res["A"]
    M = res["M"]
    T = res["T"]
    P = res["P"]
    rho = res["rho"]

    mdot = res["mdot"]
    mode = res["chem_mode"]
    geom = res["geometry"]

    # 1) Geometry
    plt.figure()
    plt.plot(x, A)
    plt.xlabel("x [m]")
    plt.ylabel("Area A(x) [m^2]")
    plt.title("Nozzle Area Profile")
    plt.grid(True)

    # 2) Mach
    plt.figure()
    plt.plot(x, M)
    plt.axhline(1.0, linewidth=1.0)
    plt.xlabel("x [m]")
    plt.ylabel("Mach number [-]")
    plt.title(f"Mach Number (chem_mode={mode}, mdot={mdot:.6g} kg/s)")
    plt.grid(True)

    # 3) Temperature
    plt.figure()
    plt.plot(x, T)
    plt.xlabel("x [m]")
    plt.ylabel("T [K]")
    plt.title("Static Temperature")
    plt.grid(True)

    # 4) Pressure
    plt.figure()
    plt.plot(x, P)
    plt.xlabel("x [m]")
    plt.ylabel("P [Pa]")
    plt.title("Static Pressure")
    plt.grid(True)

    # 5) Density
    plt.figure()
    plt.plot(x, rho)
    plt.xlabel("x [m]")
    plt.ylabel(r"$\rho$ [kg/m$^3$]")
    plt.title("Density")
    plt.grid(True)

    # Show summary text in console too
    print("=== Summary ===")
    print(f"chem_mode      : {mode}")
    print(f"mdot*          : {mdot:.8g} kg/s")
    print(
        f"A_in, A_t, A_out: {geom['A_in']:.6g}, {geom['A_t']:.6g}, {geom['A_out']:.6g} m^2"
    )
    print(f"L, x_t         : {geom['L']:.6g}, {geom['x_t']:.6g} m")
    print(f"M(min,max)     : {np.nanmin(M):.6g}, {np.nanmax(M):.6g}")
    print("================")

    plt.show()


# ============================================================
# Main example
# ============================================================
def main():
    # ---------- User parameters ----------
    # Chemistry mode: "frozen" or "equilibrium"
    chem_mode = "equilibrium"  # <- change to "equilibrium" to allow local equilibrium
    Cd = 1.0  # discharge coefficient (ideal = 1)

    # Stagnation (reservoir) state
    T0 = 2000.0
    P0 = 20.0 * ct.one_atm

    # Mixture definition (example: methane-air)
    phi = 1.0
    fuel = "CH4"
    oxidizer = {"O2": 1.0, "N2": 3.76}

    # Geometry
    L = 0.30
    x_t = 0.12
    A_in = 3.0e-4
    A_t = 1.0e-4
    A_out = 5.0e-4

    n_points = 300
    # -----------------------------------

    gas0 = ct.Solution("gri30.yaml")
    gas0.set_equivalence_ratio(phi=phi, fuel=fuel, oxidizer=oxidizer)
    gas0.TP = T0, P0

    res = solve_nozzle_profile(
        gas0,
        A_in=A_in,
        A_t=A_t,
        A_out=A_out,
        L=L,
        x_t=x_t,
        n_points=n_points,
        chem_mode=chem_mode,
        Cd=Cd,
    )

    plot_results(res)


if __name__ == "__main__":
    main()
