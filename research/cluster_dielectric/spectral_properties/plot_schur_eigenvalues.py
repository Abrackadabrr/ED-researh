#!/usr/bin/env python3
"""Plot Schur eigenvalues from cluster_sphere_schur.cpp on the complex plane."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT = Path("sphere_schur_eigenvalues_Nx16.csv")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot eigenvalues from a Schur-decomposition CSV on the complex plane."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"input CSV file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output image; by default '<input stem>_complex_plane.png'",
    )
    parser.add_argument("--dpi", type=int, default=200, help="output resolution (default: 200)")
    parser.add_argument("--marker-size", type=float, default=10.0, help="scatter marker size (default: 10)")
    parser.add_argument("--title", help="custom plot title")
    return parser.parse_args()


def load_eigenvalues(path: Path) -> tuple[list[float], list[float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV file does not exist: {path}")

    real_parts: list[float] = []
    imaginary_parts: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or not {"real", "imag"}.issubset(reader.fieldnames):
            raise ValueError("Input CSV must contain 'real' and 'imag' columns")

        for line_number, row in enumerate(reader, start=2):
            try:
                real = float(row["real"])
                imaginary = float(row["imag"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid eigenvalue at CSV line {line_number}") from error

            if not math.isfinite(real) or not math.isfinite(imaginary):
                raise ValueError(f"Non-finite eigenvalue at CSV line {line_number}")

            real_parts.append(real)
            imaginary_parts.append(imaginary)

    if not real_parts:
        raise ValueError(f"Input CSV contains no eigenvalues: {path}")

    return real_parts, imaginary_parts


def plot_eigenvalues(
    real_parts: list[float],
    imaginary_parts: list[float],
    output_path: Path,
    dpi: int,
    marker_size: float,
    title: str | None,
) -> None:
    if dpi <= 0:
        raise ValueError("DPI must be positive")
    if marker_size <= 0:
        raise ValueError("Marker size must be positive")

    figure, axes = plt.subplots(figsize=(8, 7))
    axes.scatter(
        real_parts,
        imaginary_parts,
        s=marker_size,
        alpha=0.7,
        edgecolors="none",
        label=f"{len(real_parts)} eigenvalues",
    )
    axes.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes.set_xlabel(r"$\operatorname{Re}\lambda$")
    axes.set_ylabel(r"$\operatorname{Im}\lambda$")
    axes.set_title(title or "Eigenvalue distribution on the complex plane")
    axes.set_aspect("equal", adjustable="datalim")
    axes.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    axes.legend(loc="best")
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    output_path = arguments.output or arguments.input.with_name(
        f"{arguments.input.stem}_complex_plane.png"
    )
    real_parts, imaginary_parts = load_eigenvalues(arguments.input)
    plot_eigenvalues(
        real_parts,
        imaginary_parts,
        output_path,
        arguments.dpi,
        arguments.marker_size,
        arguments.title,
    )
    print(f"Saved plot with {len(real_parts)} eigenvalues to {output_path}")


if __name__ == "__main__":
    main()
