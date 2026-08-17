#!/usr/bin/env python3
"""Plot a planar section of a complex scalar or vector field from an EMW VTU snapshot.

Example:
    python3 plot_field_slice.py long_plate_dipole_scattering.vtu \
        --field total_field_magnitude --normal 0 1 0 --origin 3 0 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import pyvista as pv
except ImportError as error:
    raise SystemExit(
        "PyVista is required. Install it with: python3 -m pip install pyvista"
    ) from error


COMPONENT_INDEX = {"x": 0, "y": 1, "z": 2}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a planar section of a complex scalar or vector field from a VTU file."
    )
    parser.add_argument("vtk_file", type=Path, help="Input .vtu file produced by the C++ calculation.")
    parser.add_argument(
        "--field",
        default="solution",
        help="Complex scalar or vector field name without the _real/_imag suffix (default: solution).",
    )
    parser.add_argument(
        "--normal",
        nargs=3,
        type=float,
        metavar=("NX", "NY", "NZ"),
        default=(0.0, 1.0, 0.0),
        help="Section-plane normal (default: 0 1 0, the central xz section).",
    )
    parser.add_argument(
        "--origin",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Point on the section plane. The mesh center is used by default.",
    )
    parser.add_argument(
        "--component",
        choices=("norm", "x", "y", "z"),
        default="norm",
        help="Vector component to plot; scalar fields require norm (default: norm).",
    )
    parser.add_argument(
        "--quantity",
        choices=("magnitude", "real", "imaginary", "phase"),
        default="magnitude",
        help="Complex quantity to plot (default: magnitude).",
    )
    parser.add_argument("--cmap", default="viridis", help="PyVista/Matplotlib color map.")
    parser.add_argument(
        "--clim",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Optional fixed color limits.",
    )
    parser.add_argument("--show-edges", action="store_true", help="Draw section cell edges.")
    parser.add_argument("--show", action="store_true", help="Open an interactive window after rendering.")
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG path. By default it is created next to the input VTU file.",
    )
    return parser.parse_args()


def normalized(vector: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(result)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{name} must be a finite non-zero vector")
    return result / norm


def available_complex_fields(dataset: pv.DataSet) -> list[str]:
    real_fields = {
        name.removesuffix("_real")
        for name, values in dataset.cell_data.items()
        if name.endswith("_real")
        and (
            np.asarray(values).ndim == 1
            or (np.asarray(values).ndim == 2 and np.asarray(values).shape[1] == 3)
        )
    }
    imaginary_fields = {
        name.removesuffix("_imag")
        for name, values in dataset.cell_data.items()
        if name.endswith("_imag")
        and (
            np.asarray(values).ndim == 1
            or (np.asarray(values).ndim == 2 and np.asarray(values).shape[1] == 3)
        )
    }
    return sorted(real_fields & imaginary_fields)


def scalar_values(
    real: np.ndarray,
    imaginary: np.ndarray,
    field_name: str,
    component: str,
    quantity: str,
) -> tuple[np.ndarray, str]:
    if real.ndim == 1:
        if component != "norm":
            raise ValueError(f"Scalar field '{field_name}' requires --component norm")
        values = real + 1j * imaginary
        if quantity == "magnitude":
            return np.abs(values), f"|{field_name}|"
        if quantity == "real":
            return values.real, f"Re({field_name})"
        if quantity == "imaginary":
            return values.imag, f"Im({field_name})"
        return np.degrees(np.angle(values)), f"arg({field_name}), deg"

    if component == "norm":
        if quantity == "magnitude":
            return np.sqrt(np.sum(real * real + imaginary * imaginary, axis=1)), f"|{field_name}|"
        if quantity == "real":
            return np.linalg.norm(real, axis=1), f"|Re({field_name})|"
        if quantity == "imaginary":
            return np.linalg.norm(imaginary, axis=1), f"|Im({field_name})|"
        raise ValueError("phase requires --component x, y or z")

    component_index = COMPONENT_INDEX[component]
    values = real[:, component_index] + 1j * imaginary[:, component_index]
    if quantity == "magnitude":
        return np.abs(values), f"|{field_name}_{component}|"
    if quantity == "real":
        return values.real, f"Re({field_name}_{component})"
    if quantity == "imaginary":
        return values.imag, f"Im({field_name}_{component})"
    return np.degrees(np.angle(values)), f"arg({field_name}_{component}), deg"


def camera_up(normal: np.ndarray) -> np.ndarray:
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(reference, normal)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    up = reference - np.dot(reference, normal) * normal
    return up / np.linalg.norm(up)


def default_output_path(arguments: argparse.Namespace) -> Path:
    suffix = f"{arguments.field}_{arguments.quantity}_{arguments.component}_slice.png"
    return arguments.vtk_file.with_name(f"{arguments.vtk_file.stem}_{suffix}")


def main() -> None:
    arguments = parse_arguments()
    if not arguments.vtk_file.is_file():
        raise FileNotFoundError(f"VTU file does not exist: {arguments.vtk_file}")

    mesh = pv.read(arguments.vtk_file)
    fields = available_complex_fields(mesh)
    if arguments.field not in fields:
        available = ", ".join(fields) if fields else "none"
        raise ValueError(f"Field '{arguments.field}' is unavailable. Available fields: {available}")

    normal = normalized(arguments.normal, "normal")
    origin = np.asarray(arguments.origin if arguments.origin is not None else mesh.center, dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("origin must contain exactly three finite coordinates")

    section = mesh.slice(normal=normal, origin=origin)
    if section.n_cells == 0:
        raise ValueError(f"The plane does not intersect the mesh; mesh bounds are {mesh.bounds}")

    real_name = f"{arguments.field}_real"
    imaginary_name = f"{arguments.field}_imag"
    values, scalar_title = scalar_values(
        np.asarray(section.cell_data[real_name]),
        np.asarray(section.cell_data[imaginary_name]),
        arguments.field,
        arguments.component,
        arguments.quantity,
    )
    scalar_name = "field_slice_value"
    section.cell_data[scalar_name] = values

    output_path = arguments.output or default_output_path(arguments)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(off_screen=not arguments.show, window_size=(1600, 900))
    plotter.set_background("white")
    plotter.add_mesh(
        section,
        scalars=scalar_name,
        preference="cell",
        cmap=arguments.cmap,
        clim=arguments.clim,
        show_edges=arguments.show_edges,
        scalar_bar_args={"title": scalar_title},
    )
    plotter.add_axes()
    plotter.show_bounds(grid="front", location="outer", all_edges=True)
    plotter.add_title(
        f"{arguments.field}: {scalar_title}\n"
        f"origin={np.array2string(origin, precision=3)}, "
        f"normal={np.array2string(normal, precision=3)}"
    )
    plotter.view_vector(normal, camera_up(normal))
    plotter.enable_parallel_projection()

    if arguments.show:
        plotter.show(screenshot=str(output_path))
    else:
        plotter.show(screenshot=str(output_path), auto_close=True)

    print(f"Section contains {section.n_cells} cells")
    print(f"Image saved to {output_path}")


if __name__ == "__main__":
    main()
