import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn


# ============================================================
# Parameters
# ============================================================

R = 0.5
eps_r = 2.56
mu_r = 1.0
c0 = 299792458.0

f_min_GHz = 0.1
f_max_GHz = 3.0
df_GHz = 0.001

# True  -> dBsm
# False -> linear scale, m^2
PLOT_DB = True

m = np.sqrt(eps_r * mu_r)

freq_GHz = np.arange(f_min_GHz, f_max_GHz + 0.5 * df_GHz, df_GHz)
freq_Hz = freq_GHz * 1.0e9


# ============================================================
# Riccati-Bessel functions
# ============================================================

def psi(n, z):
    return z * spherical_jn(n, z)


def dpsi(n, z):
    return spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)


def xi(n, z):
    return z * (spherical_jn(n, z) + 1j * spherical_yn(n, z))


def dxi(n, z):
    h = spherical_jn(n, z) + 1j * spherical_yn(n, z)

    dh = (
        spherical_jn(n, z, derivative=True)
        + 1j * spherical_yn(n, z, derivative=True)
    )

    return h + z * dh


# ============================================================
# Mie coefficients for dielectric sphere
# ============================================================

def mie_coefficients_dielectric(x, m):
    nmax = int(np.ceil(x + 4.0 * x ** (1.0 / 3.0) + 2.0))
    n = np.arange(1, nmax + 1)

    px = psi(n, x)
    dpx = dpsi(n, x)

    xx = xi(n, x)
    dxx = dxi(n, x)

    mx = m * x

    pmx = psi(n, mx)
    dpmx = dpsi(n, mx)

    a_n = (m * pmx * dpx - px * dpmx) / (
        m * pmx * dxx - xx * dpmx
    )

    b_n = (pmx * dpx - m * px * dpmx) / (
        pmx * dxx - m * xx * dpmx
    )

    return n, a_n, b_n


# ============================================================
# Forward and backward Mie amplitudes
# ============================================================

def mie_forward_backward_amplitudes(k, R, m):
    x = k * R

    n, a_n, b_n = mie_coefficients_dielectric(x, m)

    # theta = 0 degrees:
    #
    # pi_n(1) = tau_n(1) = n(n+1)/2
    #
    # S(0) = 1/2 sum (2n+1)(a_n + b_n)
    S_forward = 0.5 * np.sum((2.0 * n + 1.0) * (a_n + b_n))

    # theta = 180 degrees:
    #
    # pi_n(-1)  = (-1)^(n+1) n(n+1)/2
    # tau_n(-1) = (-1)^n     n(n+1)/2
    #
    # S(pi) = 1/2 sum (2n+1)(-1)^n (b_n - a_n)
    signs = (-1.0) ** n

    S_backward = 0.5 * np.sum(
        (2.0 * n + 1.0) * signs * (b_n - a_n)
    )

    return S_forward, S_backward, len(n)


def to_dbsm(x):
    return 10.0 * np.log10(np.maximum(x, 1e-300))


# ============================================================
# Frequency sweep
# ============================================================

C_ext = np.zeros_like(freq_Hz, dtype=float)
RCS_0 = np.zeros_like(freq_Hz, dtype=float)
RCS_180 = np.zeros_like(freq_Hz, dtype=float)

ka_values = np.zeros_like(freq_Hz, dtype=float)
nmax_values = np.zeros_like(freq_Hz, dtype=int)

for i, f in enumerate(freq_Hz):
    k = 2.0 * np.pi * f / c0

    ka_values[i] = k * R

    S0, Spi, nmax = mie_forward_backward_amplitudes(k, R, m)

    nmax_values[i] = nmax

    # Extinction cross section from optical theorem:
    #
    # C_ext = 4 pi / k^2 * Re S(0)
    C_ext[i] = 4.0 * np.pi * np.real(S0) / (k * k)

    # Bistatic RCS-like value:
    #
    # sigma(theta) = 4 pi |S(theta)|^2 / k^2
    RCS_0[i] = 4.0 * np.pi * np.abs(S0) ** 2 / (k * k)
    RCS_180[i] = 4.0 * np.pi * np.abs(Spi) ** 2 / (k * k)


# ============================================================
# Plot scale
# ============================================================

if PLOT_DB:
    y_ext = to_dbsm(C_ext)
    y_rcs_0 = to_dbsm(RCS_0)
    y_rcs_180 = to_dbsm(RCS_180)

    ylabel_ext = r"$10\log_{10} C_{\mathrm{ext}}$, dBsm"
    ylabel_rcs = r"$10\log_{10} \sigma$, dBsm"
else:
    y_ext = C_ext
    y_rcs_0 = RCS_0
    y_rcs_180 = RCS_180

    ylabel_ext = r"$C_{\mathrm{ext}}$, m$^2$"
    ylabel_rcs = r"$\sigma$, m$^2$"


# ============================================================
# Plot 1: extinction cross section
# ============================================================

fig1, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(freq_GHz, y_ext)

ax1.set_xlabel("Frequency, GHz")
ax1.set_ylabel(ylabel_ext)
ax1.set_title(
    rf"Extinction cross section, Mie solution: "
    rf"$R={R}$ m, $\varepsilon_r={eps_r}$"
)
ax1.grid(True, alpha=0.35)

fig1.tight_layout()


# ============================================================
# Plot 2: RCS at theta = 0 degrees
# ============================================================

fig2, ax2 = plt.subplots(figsize=(10, 6))

ax2.plot(freq_GHz, y_rcs_0)

ax2.set_xlabel("Frequency, GHz")
ax2.set_ylabel(ylabel_rcs)
ax2.set_title(
    rf"Bistatic RCS at $\theta=0^\circ$, Mie solution: "
    rf"$R={R}$ m, $\varepsilon_r={eps_r}$"
)
ax2.grid(True, alpha=0.35)

fig2.tight_layout()


# ============================================================
# Plot 3: RCS at theta = 180 degrees
# ============================================================

fig3, ax3 = plt.subplots(figsize=(10, 6))

ax3.plot(freq_GHz, y_rcs_180)

ax3.set_xlabel("Frequency, GHz")
ax3.set_ylabel(ylabel_rcs)
ax3.set_title(
    rf"Bistatic RCS at $\theta=180^\circ$, Mie solution: "
    rf"$R={R}$ m, $\varepsilon_r={eps_r}$"
)
ax3.grid(True, alpha=0.35)

fig3.tight_layout()


# ============================================================
# Diagnostics
# ============================================================

print("Done.")
print(f"R = {R} m")
print(f"eps_r = {eps_r}")
print(f"frequency range = {f_min_GHz} .. {f_max_GHz} GHz")
print(f"frequency step = {df_GHz} GHz")
print(f"number of frequency points = {len(freq_GHz)}")
print(f"ka range = {ka_values[0]:.6f} .. {ka_values[-1]:.6f}")
print(f"nmax range = {nmax_values.min()} .. {nmax_values.max()}")

print()
print("Maxima:")
print(
    f"max C_ext = {C_ext.max():.6e} m^2 "
    f"at f = {freq_GHz[np.argmax(C_ext)]:.6f} GHz"
)
print(
    f"max RCS(0 deg) = {RCS_0.max():.6e} m^2 "
    f"at f = {freq_GHz[np.argmax(RCS_0)]:.6f} GHz"
)
print(
    f"max RCS(180 deg) = {RCS_180.max():.6e} m^2 "
    f"at f = {freq_GHz[np.argmax(RCS_180)]:.6f} GHz"
)

# Главное: этот вызов блокирует выполнение программы,
# поэтому окна matplotlib не закрываются сами.
plt.show()
