from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import shgo
from scattnlay import scattnlay


# ============================================================
# USER PARAMETERS
# ============================================================

# Relative permittivity of the sphere.
EPS_R = 2.56 + 0.0j

# Incident-wave frequency.
FREQUENCY_HZ = 0.6e9

# Search interval for fitted sphere radius.
RADIUS_BOUNDS_M = (0.49, 0.51)

# Folder containing:
#
#   sphere_sigma_hh_21.csv
#   sphere_sigma_hh_41.csv
#   ...
#
#   sphere_sigma_vv_21.csv
#   sphere_sigma_vv_41.csv
#   ...
#
DATA_DIR = Path(".")

OUTPUT_DIR = DATA_DIR / "convergence_plots"

ANGLE_COLUMN = "angle"
RSP_COLUMN = "rsp"

POLARIZATIONS = ("hh", "vv")

# Mie convention for the scattering plane used in these files:
#
#   HH -> S2
#   VV -> S1
#
MIE_COMPONENT = {
    "hh": "S2",
    "vv": "S1",
}

# Global radius optimization settings.
#
# SHGO is used instead of a local bounded minimizer because near
# a Mie resonance the error as a function of radius may have
# several local minima.
SHGO_N = 256
SHGO_ITERS = 3

# Plot settings.
FIGSIZE = (9.0, 6.0)
DPI = 180
FONT_SIZE = 14
LABEL_SIZE = 15
TITLE_SIZE = 16
LEGEND_SIZE = 13
TICK_SIZE = 12

plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
    }
)


# ============================================================
# Physical constants
# ============================================================

C0 = 299_792_458.0
LAMBDA0 = C0 / FREQUENCY_HZ
K0 = 2.0 * np.pi / LAMBDA0

REFRACTIVE_INDEX = np.sqrt(
    EPS_R + 0j
)

# Choose the passive branch when EPS_R is complex.
if REFRACTIVE_INDEX.imag < 0.0:
    REFRACTIVE_INDEX = -REFRACTIVE_INDEX


# ============================================================
# File discovery
# ============================================================

def find_numerical_files(
        data_dir: Path,
        polarization: str,
):
    """
    Find files

        sphere_sigma_<polarization>_<N>.csv

    Also accepts filenames produced by repeated downloads:

        sphere_sigma_hh_41(1).csv
        sphere_sigma_hh_61(3).csv

    Returns

        [(N1, path1), (N2, path2), ...]

    sorted by N.
    """
    patterns = [
        f"sphere_sigma_{polarization}_*.csv",
        f"sphere_simga_{polarization}_*.csv",
    ]

    regex = re.compile(
        rf"^sphere_(?:sigma|simga)_"
        rf"{re.escape(polarization)}_"
        rf"(\d+)(?:\(\d+\))?\.csv$"
    )

    files = []

    for pattern in patterns:
        for path in data_dir.glob(pattern):
            match = regex.match(
                path.name
            )

            if match is not None:
                number_of_nodes = int(
                    match.group(1)
                )

                files.append(
                    (
                        number_of_nodes,
                        path,
                    )
                )

    if not files:
        raise FileNotFoundError(
            f"No files found for polarization "
            f"'{polarization}' in {data_dir}"
        )

    # Do not silently choose between duplicates.
    by_n = {}

    for n, path in files:
        by_n.setdefault(
            n,
            [],
        ).append(path)

    duplicates = {
        n: paths
        for n, paths in by_n.items()
        if len(paths) > 1
    }

    if duplicates:
        text = "\n".join(
            f"N={n}: {[p.name for p in paths]}"
            for n, paths
            in sorted(
                duplicates.items()
            )
        )

        raise RuntimeError(
            "Several files correspond to the same "
            f"number_of_nodes for '{polarization}':\n"
            f"{text}"
        )

    return sorted(
        [
            (
                n,
                paths[0],
            )
            for n, paths in by_n.items()
        ],
        key=lambda item: item[0],
    )


# ============================================================
# Reading numerical RCS curves
# ============================================================

def read_curve(
        path: Path,
):
    df = pd.read_csv(
        path
    )

    required = {
        ANGLE_COLUMN,
        RSP_COLUMN,
    }

    if not required.issubset(
            df.columns
    ):
        raise ValueError(
            f"{path}: expected columns "
            f"{sorted(required)}, "
            f"got {list(df.columns)}"
        )

    angle_deg = df[
        ANGLE_COLUMN
    ].to_numpy(
        dtype=float
    )

    rsp_db = df[
        RSP_COLUMN
    ].to_numpy(
        dtype=float
    )

    order = np.argsort(
        angle_deg
    )

    return (
        angle_deg[order],
        rsp_db[order],
    )


# ============================================================
# Mie bistatic RCS
# ============================================================

def mie_rcs(
        radius_m,
        angle_deg,
        component,
):
    """
    Bistatic RCS

        sigma(theta)
        =
        4*pi/k0^2 * |S_1,2(theta)|^2.
    """
    if component not in {"S1", "S2"}:
        raise ValueError(
            f"Unknown Mie component: "
            f"{component}"
        )

    size_parameters = np.array(
        [K0 * radius_m],
        dtype=np.float64,
    )
    refractive_indices = np.array(
        [REFRACTIVE_INDEX],
        dtype=np.complex128,
    )
    theta = np.deg2rad(
        np.asarray(angle_deg, dtype=float)
    )

    (
        _, _, _, _, _, _, _, _, s1, s2,
    ) = scattnlay(
        size_parameters,
        refractive_indices,
        theta,
    )

    scattering_amplitude = (
        np.asarray(s1)
        if component == "S1"
        else np.asarray(s2)
    )

    sigma = (
            4.0
            * np.pi
            / K0**2
            * np.abs(
        scattering_amplitude
    )**2
    )

    return sigma


def mie_rcs_db(
        radius_m,
        angle_deg,
        component,
):
    sigma = mie_rcs(
        radius_m,
        angle_deg,
        component,
    )

    return (
            10.0
            * np.log10(
        np.maximum(
            sigma,
            1.0e-300,
        )
    )
    )


# ============================================================
# Radius fitting
# ============================================================

def fit_radius(
        angle_deg,
        numerical_rsp_db,
        component,
):
    """
    Fit radius using GLOBAL optimization of the L1 error in LINEAR RCS.

    Numerical data are converted from dB to linear RCS:

        sigma_num = 10^(RSP_dB / 10).

    The minimized functional is

        L1(R)
        =
        mean(
            abs(
                sigma_Mie(theta; R)
                - sigma_num(theta)
            )
        ).

    Using sum(abs(...)) instead of mean(abs(...)) gives
    exactly the same fitted radius.

    SHGO is used over the full interval RADIUS_BOUNDS_M.
    No vertical shift is fitted.
    """

    # Numerical RCS: dB -> linear scale.
    numerical_rsp_linear = 10.0 ** (
            numerical_rsp_db / 10.0
    )

    def objective_scalar(
            radius_m,
    ):
        mie_linear = mie_rcs(
            radius_m,
            angle_deg,
            component,
        )

        error_linear = (
                mie_linear
                - numerical_rsp_linear
        )

        # L1 objective in LINEAR RCS.
        #
        # mean(abs(error)) and sum(abs(error))
        # have exactly the same minimizer.
        return float(
            np.mean(
                np.abs(
                    error_linear
                )
            )
        )

    # scipy.optimize.shgo expects x to be a 1-D vector even for
    # a one-dimensional optimization problem.
    def objective_shgo(x):
        radius_m = float(
            np.asarray(x).reshape(-1)[0]
        )

        return objective_scalar(
            radius_m
        )

    result = shgo(
        objective_shgo,
        bounds=[
            RADIUS_BOUNDS_M,
        ],
        n=SHGO_N,
        iters=SHGO_ITERS,
        sampling_method="simplicial",
    )

    if not result.success:
        raise RuntimeError(
            "Global radius optimization failed: "
            f"{result.message}"
        )

    fitted_radius_m = float(
        np.asarray(result.x).reshape(-1)[0]
    )

    fitted_l1_linear = float(
        result.fun
    )

    # Optional diagnostic: SHGO also reports the local minima
    # discovered during the global search.
    local_minima = []

    if hasattr(result, "xl") and result.xl is not None:
        xl = np.asarray(result.xl)

        if xl.size > 0:
            xl = xl.reshape(-1, 1)

            funl = np.asarray(
                result.funl,
                dtype=float,
            ).reshape(-1)

            for x_local, f_local in zip(
                    xl,
                    funl,
            ):
                local_minima.append(
                    (
                        float(x_local[0]),
                        float(f_local),
                    )
                )

    if local_minima:
        print(
            "  SHGO local minima: "
            + ", ".join(
                f"R={r:.10f}, F={f:.6e}"
                for r, f in local_minima
            )
        )

    return (
        fitted_radius_m,
        fitted_l1_linear,
    )



# ============================================================
# Process all files
# ============================================================

def process_polarization(
        polarization,
):
    files = find_numerical_files(
        DATA_DIR,
        polarization,
    )

    component = MIE_COMPONENT[
        polarization
    ]

    rows = []

    for (
            number_of_nodes,
            path,
    ) in files:

        angle_deg, rsp_db = (
            read_curve(
                path
            )
        )

        radius_m, l1_linear = (
            fit_radius(
                angle_deg,
                rsp_db,
                component,
            )
        )

        rows.append(
            {
                "polarization":
                    polarization,
                "number_of_nodes":
                    number_of_nodes,
                "fitted_radius_m":
                    radius_m,
                "l1_linear":
                    l1_linear,
                "file":
                    path.name,
            }
        )

        print(
            f"{polarization.upper()}, "
            f"N={number_of_nodes:4d}, "
            f"R_fit={radius_m:.10f} m, "
            f"L1_linear={l1_linear:.6e} m^2"
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "number_of_nodes"
    )


# ============================================================
# Plot radius versus number_of_nodes
# ============================================================

def plot_radius_vs_number_of_nodes(
        result_df,
        output_dir,
):
    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    for polarization in POLARIZATIONS:
        subset = (
            result_df[
                result_df[
                    "polarization"
                ]
                == polarization
                ]
            .sort_values(
                "number_of_nodes"
            )
        )

        ax.plot(
            subset[
                "number_of_nodes"
            ],
            subset[
                "fitted_radius_m"
            ],
            "o-",
            linewidth=2.0,
            markersize=7,
            label=polarization.upper(),
        )

    ax.set_xlabel(
        "число узлов N"
    )

    ax.set_ylabel(
        "подобранный радиус R, м"
    )

    ax.set_title(
        "Эффективный радиус по фитированию ЭПР"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
            output_dir
            / "fitted_radius_vs_number_of_nodes.png"
    )

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_path



# ============================================================
# Finest-grid RCS comparison for HH and VV
# ============================================================

def plot_finest_grid_rcs_comparison(
        result_df,
        output_dir,
):
    """
    For each polarization, select the numerical file with the
    largest number_of_nodes and plot:

        1. Mie curve for R = 0.5 m;
        2. Mie curve for the fitted radius on this finest grid;
        3. numerical RCS curve.

    Radius fitting itself is done in LINEAR RCS inside fit_radius(),
    while this figure is displayed in dB.
    """

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16.0, 6.0),
        sharey=False,
    )

    for ax, polarization in zip(
            axes,
            POLARIZATIONS,
    ):
        subset = result_df[
            result_df["polarization"] == polarization
            ].copy()

        if subset.empty:
            ax.set_visible(False)
            continue

        # Select the finest grid for this polarization.
        finest_index = subset[
            "number_of_nodes"
        ].idxmax()

        finest_row = subset.loc[
            finest_index
        ]

        number_of_nodes = int(
            finest_row["number_of_nodes"]
        )

        fitted_radius_m = float(
            finest_row["fitted_radius_m"]
        )

        numerical_file = (
                DATA_DIR
                / str(finest_row["file"])
        )

        angle_deg, numerical_rsp_db = read_curve(
            numerical_file
        )

        component = MIE_COMPONENT[
            polarization
        ]

        # Mie curve for the nominal sphere.
        mie_nominal_db = mie_rcs_db(
            0.5,
            angle_deg,
            component,
        )

        # Mie curve for the radius obtained from the linear-RCS fit.
        mie_fitted_db = mie_rcs_db(
            fitted_radius_m,
            angle_deg,
            component,
        )

        ax.plot(
            angle_deg,
            mie_nominal_db,
            "--",
            linewidth=2.0,
            label=r"Mie, $R=0.5$ м",
        )

        ax.plot(
            angle_deg,
            mie_fitted_db,
            linewidth=2.1,
            label=(
                rf"Mie fit, "
                rf"$R={fitted_radius_m:.6f}$ м"
            ),
        )

        ax.plot(
            angle_deg,
            numerical_rsp_db,
            linewidth=2.1,
            label=(
                rf"расчёт, $N={number_of_nodes}$"
            ),
        )

        ax.set_title(
            f"{polarization.upper()}, "
            f"N = {number_of_nodes}"
        )

        ax.set_xlabel(
            "угол, град"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend(
            loc="best",
        )

    axes[0].set_ylabel(
        "ЭПР, дБ"
    )

    fig.suptitle(
        "ЭПР на самой мелкой сетке: "
        "расчёт и решение Ми",
        fontsize=TITLE_SIZE + 2,
        y=0.98,
    )

    fig.tight_layout(
        rect=[0.0, 0.0, 1.0, 0.94]
    )

    output_path = (
            output_dir
            / "finest_grid_rcs_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_path


# ============================================================
# Build a compact table with HH and VV side by side
# ============================================================

def build_radius_table(
        result_df,
):
    hh = (
        result_df[
            result_df[
                "polarization"
            ]
            == "hh"
            ][
            [
                "number_of_nodes",
                "fitted_radius_m",
                "l1_linear",
            ]
        ]
        .rename(
            columns={
                "fitted_radius_m":
                    "radius_hh_m",
                "l1_linear":
                    "l1_hh_linear",
            }
        )
    )

    vv = (
        result_df[
            result_df[
                "polarization"
            ]
            == "vv"
            ][
            [
                "number_of_nodes",
                "fitted_radius_m",
                "l1_linear",
            ]
        ]
        .rename(
            columns={
                "fitted_radius_m":
                    "radius_vv_m",
                "l1_linear":
                    "l1_vv_linear",
            }
        )
    )

    table = pd.merge(
        hh,
        vv,
        on="number_of_nodes",
        how="outer",
    )

    table = table.sort_values(
        "number_of_nodes"
    ).reset_index(
        drop=True
    )

    return table


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"EPS_R        = {EPS_R}"
    )

    print(
        f"frequency    = "
        f"{FREQUENCY_HZ / 1e9:.6f} GHz"
    )

    print(
        f"lambda0      = "
        f"{LAMBDA0:.9f} m"
    )

    print(
        f"radius bounds= "
        f"{RADIUS_BOUNDS_M}"
    )

    print()

    all_results = []

    for polarization in POLARIZATIONS:
        result = process_polarization(
            polarization
        )

        all_results.append(
            result
        )

    result_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    # Full table: one row per file.
    full_csv = (
            OUTPUT_DIR
            / "fitted_radii_full.csv"
    )

    result_df.to_csv(
        full_csv,
        index=False,
    )

    # Compact table: HH and VV radii in the same row.
    radius_table = build_radius_table(
        result_df
    )

    compact_csv = (
            OUTPUT_DIR
            / "fitted_radii_table.csv"
    )

    radius_table.to_csv(
        compact_csv,
        index=False,
    )

    figure_path = (
        plot_radius_vs_number_of_nodes(
            result_df,
            OUTPUT_DIR,
        )
    )

    finest_grid_figure_path = (
        plot_finest_grid_rcs_comparison(
            result_df,
            OUTPUT_DIR,
        )
    )

    print()
    print(
        "Fitted radii:"
    )

    print(
        radius_table.to_string(
            index=False
        )
    )

    print()
    print(
        f"Radius plot saved to: "
        f"{figure_path}"
    )

    print(
        f"Finest-grid RCS comparison saved to: "
        f"{finest_grid_figure_path}"
    )

    print(
        f"Table saved to: "
        f"{compact_csv}"
    )

    print(
        f"Full data saved to: "
        f"{full_csv}"
    )


if __name__ == "__main__":
    main()