import numpy as np
import matplotlib.pyplot as plt

from scattnlay import fieldnlay, expancoeffs

def gold_drude(frequency):
    """
    Drude permittivity of gold.

    Parameters
    ----------
    frequency : float or np.ndarray
        Frequency in Hz.

    Returns
    -------
    complex or np.ndarray
        Relative permittivity epsilon(f).

    Time convention: exp(-i * omega * t)
    """

    h_eVs = 4.135667696e-15  # Planck constant, eV*s

    eps_inf = 5.0
    E_p = 8.9                # hbar * omega_p, eV
    Gamma = 0.0387           # hbar * gamma, eV

    E = h_eVs * frequency    # hbar*omega = h*f, eV

    return eps_inf - E_p**2 / (E * (E + 1j * Gamma))

def frequency_from_wavelength(wavelength):
    """
    Convert vacuum wavelength to frequency.

    Parameters
    ----------
    wavelength : float or np.ndarray
        Vacuum wavelength in meters.

    Returns
    -------
    float or np.ndarray
        Frequency in Hz.
    """

    C0 = 299_792_458.0

    return C0 / wavelength

# ============================================================
# USER PARAMETERS
# ============================================================

RADIUS = 10e-9

WAVELENGTH = 368.8e-9  # nm

FREQUENCY = frequency_from_wavelength(WAVELENGTH)

EPSILON = gold_drude(FREQUENCY)

HALF_SIZE = 4 * RADIUS                  # plot [-L, L]^2, m

GRID_SIZE = 1010                   # points per axis

OUTPUT_FILE = "mie_field.png"


RADIAL_MAX = 10 * RADIUS
RADIAL_POINTS = 30000
RADIAL_OUTPUT_FILE = "mie_field_radial.png"

# ============================================================
# CONSTANTS
# ============================================================

C0 = 299_792_458.0


# ============================================================
# MATERIAL PARAMETERS
# ============================================================

def refractive_index(eps):
    """
    m = sqrt(epsilon).

    For a passive material choose the branch with Im(m) >= 0.
    """

    m = np.sqrt(eps + 0j)

    if m.imag < 0:
        m = -m

    return m


# ============================================================
# FIELD CALCULATION
# ============================================================

def plot_mie_solution_coefficients(nmax=20):
    """
    Строит коэффициенты, с которыми векторные сферические
    гармоники реально входят в разложение электрического поля.

    Mie coefficients a_n, b_n, c_n, d_n считаются только scattnlay.
    """

    c0 = 299_792_458.0

    k0 = 2.0 * np.pi * FREQUENCY / c0

    x = np.array(
        [k0 * RADIUS],
        dtype=np.float64
    )

    m_value = np.sqrt(EPSILON + 0j)

    if m_value.imag < 0:
        m_value = -m_value

    m = np.array(
        [m_value],
        dtype=np.complex128
    )

    # --------------------------------------------------------
    # Mie coefficients from scattnlay
    # --------------------------------------------------------

    terms, a, b, c, d = expancoeffs(
        x,
        m,
        nmax=nmax
    )

    terms = int(terms)

    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)
    d = np.asarray(d)

    n = np.arange(1, terms + 1)

    # One homogeneous sphere:
    # 0  -> sphere interior
    # -1 -> exterior

    a_ext = a[-1, :terms]
    b_ext = b[-1, :terms]

    c_int = c[0, :terms]
    d_int = d[0, :terms]

    # --------------------------------------------------------
    # Plane-wave expansion coefficient
    #
    # E_n = i^n (2n+1)/(n(n+1))
    # --------------------------------------------------------

    En = (
        (1j ** n)
        * (2.0 * n + 1.0)
        / (n * (n + 1.0))
    )

    # --------------------------------------------------------
    # Actual coefficients in the E-field VSH expansion:
    #
    # Esc = sum En [ i a_n N^(3) - b_n M^(3) ]
    #
    # Ein = sum En [ c_n M^(1) - i d_n N^(1) ]
    # --------------------------------------------------------

    A =  1j * En * a_ext
    B =       -En * b_ext

    C =        En * c_int
    D = -1j * En * d_int

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.semilogy(
        n,
        np.abs(A),
        "o-",
        label=r"$|iE_n a_n|$"
    )

    plt.semilogy(
        n,
        np.abs(B),
        "o-",
        label=r"$|-E_n b_n|$"
    )

    plt.semilogy(
        n,
        np.abs(C),
        "o-",
        label=r"$|E_n c_n|$"
    )

    plt.semilogy(
        n,
        np.abs(D),
        "o-",
        label=r"$|-iE_n d_n|$"
    )

    plt.xlabel("Mode number n")
    plt.ylabel("Expansion coefficient magnitude")

    plt.title(
        "Coefficients of vector spherical harmonics\n"
        f"kR = {k0 * RADIUS:.4f}, "
        f"eps = {EPSILON.real:.3f} "
        f"{EPSILON.imag:+.3f}i"
    )

    plt.xticks(n)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return n, A, B, C, D
    
def plot_radial_field(k, m_sphere):
    """
    Plot |E|/E0 along the +z radial direction:

        x = 0
        y = 0
        z = r

    from r = 0 to RADIAL_MAX.
    """

    # Physical radial coordinate
    r = np.linspace(
        0.0,
        RADIAL_MAX,
        RADIAL_POINTS,
    )

    # scattnlay uses dimensionless coordinates k*r
    coord_x = np.zeros_like(r)
    coord_y = np.zeros_like(r)
    coord_z = k * r

    x = np.array(
        [k * RADIUS],
        dtype=np.float64,
    )

    m = np.array(
        [m_sphere],
        dtype=np.complex128,
    )

    # Exact Mie field
    terms, E, H = fieldnlay(
        x,
        m,
        coord_x,
        coord_y,
        coord_z,
    )

    # |E|
    Eabs = np.sqrt(
        np.abs(E[:, 0])**2
        + np.abs(E[:, 1])**2
        + np.abs(E[:, 2])**2
    )

    # Plot in nm
    r_nm = r * 1e9
    R_nm = RADIUS * 1e9

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        r_nm,
        Eabs,
        linewidth=2,
    )

    # Sphere boundary
    ax.axvline(
        R_nm,
        linestyle="--",
        linewidth=1.5,
        label="sphere boundary",
    )

    ax.set_xlabel("r, nm")
    ax.set_ylabel(r"$|\mathbf{E}|/E_0$")

    ax.set_title(
        r"Radial electric-field amplitude along $+z$"
        "\n"
        rf"$R={R_nm:g}$ nm, "
        rf"$f={FREQUENCY / 1e12:g}$ THz, "
        rf"$\varepsilon={EPSILON.real:.4g}"
        rf"{EPSILON.imag:+.4g}i$"
    )

    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        RADIAL_OUTPUT_FILE,
        dpi=220,
        bbox_inches="tight",
    )

    plt.show()

    return r, Eabs


def calculate_field():

    # --------------------------------------------------------
    # Vacuum wavenumber
    # --------------------------------------------------------

    k = 2.0 * np.pi * FREQUENCY / C0

    # --------------------------------------------------------
    # Relative refractive index of sphere
    #
    # surrounding medium = vacuum
    # --------------------------------------------------------

    m_sphere = refractive_index(EPSILON)

    # --------------------------------------------------------
    # scattnlay works with dimensionless quantities
    #
    # x = k R
    # --------------------------------------------------------

    x = np.array(
        [k * RADIUS],
        dtype=np.float64,
    )

    m = np.array(
        [m_sphere],
        dtype=np.complex128,
    )

    # --------------------------------------------------------
    # Physical grid in the plane x = 0
    #
    # horizontal coordinate: y
    # vertical coordinate:   z
    # --------------------------------------------------------

    y = np.linspace(
        -HALF_SIZE,
        HALF_SIZE,
        GRID_SIZE,
    )

    z = np.linspace(
        -HALF_SIZE,
        HALF_SIZE,
        GRID_SIZE,
    )

    Y, Z = np.meshgrid(y, z)

    # x = 0 plane
    X = np.zeros_like(Y)

    # --------------------------------------------------------
    # Flatten coordinates because fieldnlay expects
    # a list of spatial points.
    #
    # IMPORTANT:
    # fieldnlay coordinates are dimensionless:
    #
    #     coord = k * r
    # --------------------------------------------------------

    coord_x = (k * X).ravel()
    coord_y = (k * Y).ravel()
    coord_z = (k * Z).ravel()

    # --------------------------------------------------------
    # Exact Mie solution from scattnlay
    #
    # E.shape = (number_of_points, 3)
    # H.shape = (number_of_points, 3)
    # --------------------------------------------------------

    terms, E, H = fieldnlay(
        x,
        m,
        coord_x,
        coord_y,
        coord_z,
    )

    # --------------------------------------------------------
    # Electric-field magnitude
    #
    # |E| = sqrt(
    #       |Ex|^2
    #     + |Ey|^2
    #     + |Ez|^2
    # )
    #
    # Scattnlay uses incident field amplitude E0 = 1.
    # --------------------------------------------------------

    Eabs = np.sqrt(
        np.abs(E[:, 0])**2
        + np.abs(E[:, 1])**2
        + np.abs(E[:, 2])**2
    )

    # Return to 2D grid
    Eabs = Eabs.reshape(
        GRID_SIZE,
        GRID_SIZE,
    )

    return (
        y,
        z,
        Eabs,
        E,
        H,
        k,
        m_sphere,
        terms,
    )


# ============================================================
# PLOT
# ============================================================

def plot_field(y, z, Eabs):

    # Use cm on plot
    y_cm = y * 100.0
    z_cm = z * 100.0

    radius_cm = RADIUS * 100.0

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    image = ax.imshow(
        Eabs,
        origin="lower",
        extent=[
            y_cm[0],
            y_cm[-1],
            z_cm[0],
            z_cm[-1],
        ],
        aspect="equal",
        vmin=0.0,
        vmax=np.max(Eabs),
    )

    # Sphere boundary in the x=0 plane
    sphere = plt.Circle(
        (0.0, 0.0),
        radius_cm,
        fill=False,
        linewidth=2.0,
    )

    ax.add_patch(sphere)

    ax.set_xlabel("y, cm")
    ax.set_ylabel("z, cm")

    ax.set_title(
        rf"$|\mathbf{{E}}|/E_0$ in $x=0$"
        "\n"
        rf"$R={radius_cm:g}$ cm, "
        rf"$f={FREQUENCY / 1e9:g}$ GHz, "
        rf"$\varepsilon={EPSILON.real:.4g}"
        rf"{EPSILON.imag:+.4g}i$"
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        r"$|\mathbf{E}|/E_0$"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_FILE,
        dpi=220,
        bbox_inches="tight",
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

(
    y,
    z,
    Eabs,
    E,
    H,
    k,
    m_sphere,
    terms,
) = calculate_field()


# ============================================================
# DIAGNOSTICS
# ============================================================

Y, Z = np.meshgrid(y, z)

r = np.sqrt(
    Y**2 + Z**2
)

inside = r < RADIUS

center = GRID_SIZE // 2


print()
print("====================================")
print("PARAMETERS")
print("====================================")

print(f"R             = {RADIUS} m")
print(f"f             = {FREQUENCY / 1e9} GHz")
print(f"epsilon       = {EPSILON}")
print(f"sqrt(epsilon) = {m_sphere}")
print(f"kR            = {k * RADIUS}")
print(f"Mie terms     = {terms}")


print()
print("====================================")
print("FIELD")
print("====================================")

print(
    f"|E(0)| / E0   = {Eabs[center, center]}"
)

print(
    f"min inside    = {Eabs[inside].min()}"
)

print(
    f"mean inside   = {Eabs[inside].mean()}"
)

print(
    f"max inside    = {Eabs[inside].max()}"
)

print(
    f"global max    = {Eabs.max()}"
)


# ============================================================
# PLOT
# ============================================================

plot_field(
    y,
    z,
    Eabs,
)

# ============================================================
# RADIAL FIELD
# ============================================================

r_radial, Eabs_radial = plot_radial_field(
    k,
    m_sphere,
)

plot_mie_solution_coefficients(nmax=50)

print(
    f"Radial field saved to: {RADIAL_OUTPUT_FILE}"
)


print()
print(f"Saved to: {OUTPUT_FILE}")
