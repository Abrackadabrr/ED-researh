from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# User settings
# ============================================================

DATA_DIR = Path(".")

POLARIZATIONS = ("hh", "vv")

ANGLE_COLUMN = "angle"
RSP_COLUMN = "rsp"

# False -> absolute errors.
# True  -> relative errors.
RELATIVE_ERRORS = False

INTERPOLATE_IF_NEEDED = True

# Example: FIT_FROM_N = 41
# None means use all available meshes in the fit.
FIT_FROM_N = None

OUTPUT_DIR = DATA_DIR / "convergence_plots"

# Combined output figure.
COMBINED_FIGURE_NAME = "rcs_grid_convergence_all.png"
RCS_CURVES_FIGURE_NAME = "rcs_curves_all.png"


# ============================================================
# Plot appearance
# ============================================================

FIGSIZE = (16, 11)
DPI = 180

FONT_SIZE = 14
AXIS_LABEL_SIZE = 15
TITLE_SIZE = 16
LEGEND_SIZE = 12
TICK_LABEL_SIZE = 12
SUPTITLE_SIZE = 19

LINE_WIDTH = 2.0
MARKER_SIZE = 7

plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
    }
)


# ============================================================
# File discovery
# ============================================================

def find_reference_file(data_dir: Path, polarization: str) -> Path:
    """Find mie_sphere_<polarization>.csv."""
    path = data_dir / f"mie_sigma_{polarization}.csv"

    if not path.is_file():
        raise FileNotFoundError(
            f"Mie reference file not found for polarization "
            f"'{polarization}': {path}"
        )

    return path


def find_numerical_files(data_dir: Path, polarization: str):
    """
    Find

        sphere_sigma_<polarization>_<number_of_nodes>.csv

    Names such as

        sphere_sigma_hh_41(3).csv

    are also accepted.
    """
    patterns = [
        f"sphere_sigma_{polarization}_*.csv",
        f"sphere_simga_{polarization}_*.csv",
    ]

    regex = re.compile(
        rf"^sphere_(?:sigma|simga)_{re.escape(polarization)}_"
        rf"(\d+)(?:\(\d+\))?\.csv$"
    )

    files = []

    for pattern in patterns:
        for path in data_dir.glob(pattern):
            match = regex.match(path.name)

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
            f"No numerical files found for "
            f"polarization '{polarization}'"
        )

    by_n = {}

    for number_of_nodes, path in files:
        by_n.setdefault(
            number_of_nodes,
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
            for n, paths in sorted(
                duplicates.items()
            )
        )

        raise RuntimeError(
            "Several numerical files correspond to the same "
            f"number_of_nodes for polarization '{polarization}':\n"
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
# Reading and interpolation
# ============================================================

def read_curve(path: Path):
    df = pd.read_csv(path)

    required = {
        ANGLE_COLUMN,
        RSP_COLUMN,
    }

    if not required.issubset(
            df.columns
    ):
        raise ValueError(
            f"{path}: expected columns {sorted(required)}, "
            f"got {list(df.columns)}"
        )

    angle = df[
        ANGLE_COLUMN
    ].to_numpy(
        dtype=float
    )

    rsp = df[
        RSP_COLUMN
    ].to_numpy(
        dtype=float
    )

    order = np.argsort(
        angle
    )

    return (
        angle[order],
        rsp[order],
    )


def put_on_reference_grid(
        angle_ref,
        angle_num,
        rsp_num,
):
    same_grid = (
            len(angle_ref)
            == len(angle_num)
            and np.allclose(
        angle_ref,
        angle_num,
        rtol=0.0,
        atol=1.0e-12,
    )
    )

    if same_grid:
        return rsp_num

    if not INTERPOLATE_IF_NEEDED:
        raise ValueError(
            "Angular grids are different and "
            "INTERPOLATE_IF_NEEDED=False"
        )

    if (
            angle_ref[0]
            < angle_num[0]
            or angle_ref[-1]
            > angle_num[-1]
    ):
        raise ValueError(
            "Numerical angular interval does not contain "
            "the complete reference interval"
        )

    return np.interp(
        angle_ref,
        angle_num,
        rsp_num,
    )


# ============================================================
# Error norms
# ============================================================

def c_norm(values):
    """
    C norm:

        ||f||_C = max |f|.
    """
    return np.max(
        np.abs(values)
    )


def l2_norm(values, angle_deg):
    """
    Continuous L2 norm with respect to scattering angle:

        ||f||_L2 =
            sqrt(
                integral |f(theta)|^2 d theta
            ).

    The integration variable is theta in radians.

    np.trapz is deliberately used instead of np.trapezoid
    for compatibility with older NumPy versions.
    """
    theta_rad = np.deg2rad(
        angle_deg
    )

    return np.sqrt(
        np.trapz(
            np.abs(values)**2,
            theta_rad,
            )
    )


def calculate_errors(
        reference_file: Path,
        numerical_files,
):
    angle_ref, rsp_ref = read_curve(
        reference_file
    )

    ref_c = c_norm(
        rsp_ref
    )

    ref_l2 = l2_norm(
        rsp_ref,
        angle_ref,
    )

    rows = []

    for (
            number_of_nodes,
            path,
    ) in numerical_files:

        angle_num, rsp_num = read_curve(
            path
        )

        rsp_num_on_ref = (
            put_on_reference_grid(
                angle_ref,
                angle_num,
                rsp_num,
            )
        )

        error = (
                rsp_num_on_ref
                - rsp_ref
        )

        error_c = c_norm(
            error
        )

        error_l2 = l2_norm(
            error,
            angle_ref,
        )

        if RELATIVE_ERRORS:
            error_c /= ref_c
            error_l2 /= ref_l2

        rows.append(
            {
                "number_of_nodes":
                    number_of_nodes,
                "C_error":
                    error_c,
                "L2_error":
                    error_l2,
                "file":
                    path.name,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "number_of_nodes"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Power-law fit
# ============================================================

def fit_power_law(
        number_of_nodes,
        errors,
):
    """
    Fit

        log(E_N)
        =
        slope * log(N)
        + intercept.

    If

        E_N ~ C N^{-p},

    then

        slope = -p,

    hence the convergence order is

        p = -slope.
    """
    number_of_nodes = np.asarray(
        number_of_nodes,
        dtype=float,
    )

    errors = np.asarray(
        errors,
        dtype=float,
    )

    mask = (
            np.isfinite(
                number_of_nodes
            )
            & np.isfinite(
        errors
    )
            & (
                    errors > 0.0
            )
    )

    if FIT_FROM_N is not None:
        mask &= (
                number_of_nodes
                >= FIT_FROM_N
        )

    n_fit = (
        number_of_nodes[
            mask
        ]
    )

    e_fit = (
        errors[
            mask
        ]
    )

    if len(n_fit) < 2:
        raise ValueError(
            "At least two points are required "
            "for a convergence fit"
        )

    slope, intercept = np.polyfit(
        np.log(
            n_fit
        ),
        np.log(
            e_fit
        ),
        deg=1,
    )

    order = -slope

    coefficient = np.exp(
        intercept
    )

    fitted_values = (
            coefficient
            * number_of_nodes**slope
    )

    return {
        "slope":
            slope,
        "order":
            order,
        "coefficient":
            coefficient,
        "fitted_values":
            fitted_values,
        "n_fit":
            n_fit,
        "e_fit":
            e_fit,
    }


# ============================================================
# Processing
# ============================================================

def process_polarization(
        data_dir: Path,
        polarization: str,
        output_dir: Path,
):
    reference = find_reference_file(
        data_dir,
        polarization,
    )

    numerical = find_numerical_files(
        data_dir,
        polarization,
    )

    result = calculate_errors(
        reference,
        numerical,
    )

    n_nodes = result[
        "number_of_nodes"
    ].to_numpy(
        dtype=float
    )

    fit_c = fit_power_law(
        n_nodes,
        result[
            "C_error"
        ].to_numpy(
            dtype=float
        ),
    )

    fit_l2 = fit_power_law(
        n_nodes,
        result[
            "L2_error"
        ].to_numpy(
            dtype=float
        ),
    )

    result = result.copy()

    result[
        "C_fit"
    ] = fit_c[
        "fitted_values"
    ]

    result[
        "L2_fit"
    ] = fit_l2[
        "fitted_values"
    ]

    csv_path = (
            output_dir
            / f"convergence_{polarization}.csv"
    )

    result.to_csv(
        csv_path,
        index=False,
    )

    print()
    print(
        f"Polarization: {polarization}"
    )

    print(
        f"Reference: {reference.name}"
    )

    print(
        result[
            [
                "number_of_nodes",
                "C_error",
                "L2_error",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        f"C norm:  slope = "
        f"{fit_c['slope']:.6f}, "
        f"order p = "
        f"{fit_c['order']:.6f}"
    )

    print(
        f"L2 norm: slope = "
        f"{fit_l2['slope']:.6f}, "
        f"order p = "
        f"{fit_l2['order']:.6f}"
    )

    return {
        "reference":
            reference,
        "data":
            result,
        "C_fit":
            fit_c,
        "L2_fit":
            fit_l2,
        "csv":
            csv_path,
    }


# ============================================================
# Combined 2 x 2 figure
# Columns = polarizations
# Rows    = norms
# ============================================================

def plot_subplot(
        ax,
        result,
        error_column,
        fit_key,
):
    data = result[
        "data"
    ]

    fit = result[
        fit_key
    ]

    n_nodes = data[
        "number_of_nodes"
    ].to_numpy(
        dtype=float
    )

    errors = data[
        error_column
    ].to_numpy(
        dtype=float
    )

    ax.loglog(
        n_nodes,
        errors,
        "o-",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label="численная ошибка",
    )

    ax.loglog(
        n_nodes,
        fit[
            "fitted_values"
        ],
        "--",
        linewidth=LINE_WIDTH,
        label=(
            rf"fit: $E_N \sim "
            rf"N^{{{fit['slope']:.3f}}}$"
            "\n"
            rf"порядок $p="
            rf"{fit['order']:.3f}$"
        ),
    )

    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )

    ax.legend(
        loc="best",
    )


def plot_combined(
        all_results,
        output_dir,
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=FIGSIZE,
        sharex=False,
    )

    # --------------------------------------------------------
    # Columns correspond to polarization:
    #   column 0 -> HH
    #   column 1 -> VV
    #
    # Rows correspond to norms:
    #   row 0 -> C
    #   row 1 -> L2
    # --------------------------------------------------------

    # Row 0: C norm
    plot_subplot(
        axes[0, 0],
        all_results["hh"],
        "C_error",
        "C_fit",
    )

    plot_subplot(
        axes[0, 1],
        all_results["vv"],
        "C_error",
        "C_fit",
    )

    # Row 1: L2 norm
    plot_subplot(
        axes[1, 0],
        all_results["hh"],
        "L2_error",
        "L2_fit",
    )

    plot_subplot(
        axes[1, 1],
        all_results["vv"],
        "L2_error",
        "L2_fit",
    )

    # --------------------------------------------------------
    # One column title per polarization.
    # It is placed above the entire column rather than repeated
    # in both subplots.
    # --------------------------------------------------------

    axes[0, 0].set_title(
        "HH",
        fontsize=TITLE_SIZE + 2,
        pad=14,
    )

    axes[0, 1].set_title(
        "VV",
        fontsize=TITLE_SIZE + 2,
        pad=14,
    )

    # --------------------------------------------------------
    # One row title per norm.
    # The axis labels themselves are common for the whole figure.
    # --------------------------------------------------------

    fig.text(
        0.075,
        0.69,
        r"$C$-норма",
        rotation=90,
        va="center",
        ha="center",
        fontsize=AXIS_LABEL_SIZE + 1,
    )

    fig.text(
        0.075,
        0.275,
        r"$L_2$-норма",
        rotation=90,
        va="center",
        ha="center",
        fontsize=AXIS_LABEL_SIZE + 1,
    )

    # No repeated axis labels inside individual panels.
    for ax in axes.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Common labels for the whole 2 x 2 figure.
    fig.supxlabel(
        r"число узлов $N$",
        fontsize=AXIS_LABEL_SIZE + 1,
        y=0.025,
    )

    fig.supylabel(
        "абс. ошибка в Дб",
        fontsize=AXIS_LABEL_SIZE + 1,
        x=0.018,
    )

    fig.suptitle(
        "Сеточная сходимость ЭПР",
        fontsize=SUPTITLE_SIZE,
        y=0.985,
    )

    # Leave enough space for common y-label and row labels.
    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.10,
        top=0.90,
        wspace=0.22,
        hspace=0.28,
    )

    output_path = (
            output_dir
            / COMBINED_FIGURE_NAME
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
# RCS curves: one additional figure for all input files
# ============================================================

def plot_all_rcs_curves(
        data_dir: Path,
        output_dir: Path,
):
    """
    Build one additional figure with two panels:

        left  -> HH
        right -> VV

    For each polarization:
      - reference Mie curve is labeled "Mie";
      - every numerical curve is labeled only by number_of_nodes.
    """
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6.5),
        sharey=False,
    )

    for ax, polarization in zip(
            axes,
            POLARIZATIONS,
    ):
        reference_file = find_reference_file(
            data_dir,
            polarization,
        )

        numerical_files = find_numerical_files(
            data_dir,
            polarization,
        )

        angle_ref, rsp_ref = read_curve(
            reference_file
        )

        # Reference curve.
        ax.plot(
            angle_ref,
            rsp_ref,
            linewidth=2.4,
            label="Mie",
        )

        # Numerical curves.
        for (
                number_of_nodes,
                path,
        ) in numerical_files:
            angle_num, rsp_num = read_curve(
                path
            )

            ax.plot(
                angle_num,
                rsp_num,
                linewidth=1.7,
                label=str(number_of_nodes),
            )

        ax.set_title(
            polarization.upper(),
            fontsize=TITLE_SIZE + 2,
            pad=12,
        )

        ax.set_xlabel("")

        ax.set_ylabel("")

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend(
            title=r"$N$",
            fontsize=LEGEND_SIZE,
            title_fontsize=LEGEND_SIZE,
            loc="best",
        )

    # Common axis labels for the complete figure.
    fig.supxlabel(
        "угол, град",
        fontsize=AXIS_LABEL_SIZE + 1,
        y=0.03,
    )

    fig.supylabel(
        "ЭПР, Дб",
        fontsize=AXIS_LABEL_SIZE + 1,
        x=0.025,
    )

    fig.suptitle(
        "ЭПР: решение Ми и сеточные расчёты",
        fontsize=SUPTITLE_SIZE,
        y=0.98,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.13,
        top=0.88,
        wspace=0.18,
    )

    output_path = (
            output_dir
            / RCS_CURVES_FIGURE_NAME
    )

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = {}

    for polarization in POLARIZATIONS:
        all_results[
            polarization
        ] = process_polarization(
            DATA_DIR,
            polarization,
            OUTPUT_DIR,
        )

    combined_path = plot_combined(
        all_results,
        OUTPUT_DIR,
    )

    rcs_curves_path = plot_all_rcs_curves(
        DATA_DIR,
        OUTPUT_DIR,
    )

    print()
    print(
        "Summary of convergence orders"
    )

    print(
        "--------------------------------"
    )

    for (
            polarization,
            result,
    ) in all_results.items():

        print(
            f"{polarization}: "
            f"p_C = "
            f"{result['C_fit']['order']:.6f}, "
            f"p_L2 = "
            f"{result['L2_fit']['order']:.6f}"
        )

    print()
    print(
        f"Combined convergence figure saved to: "
        f"{combined_path}"
    )

    print(
        f"RCS curves figure saved to: "
        f"{rcs_curves_path}"
    )


if __name__ == "__main__":
    main()
