#!/usr/bin/env python3
"""Spectrum predictor for the electromagnetic volume integral operator on a sphere.

The implementation follows section 4 of J. Rahola,
"On the Eigenvalues of the Volume Integral Operator of Electromagnetic Scattering",
SIAM J. Sci. Comput. 21(5), 2000.

For a physical relative permittivity eps (or, for backward compatibility,
a physical refractive index m) and sphere size x = k r, the script plots
points in the spectrum of I-K. Internally m^2 = eps:

  1. the branch-cut segment [1, m**2];
  2. isolated points produced by complex Mie resonances.

A fictitious resonant refractive index mu is mapped to the spectrum by

    lambda = 1 - (m**2 - 1) / (mu**2 - 1).

Only resonance points satisfying |lambda - 1| >= spectrum_tol are retained.
The complex roots are found by an automated two-dimensional scan followed by
local nonlinear refinement.  The scan mirrors the article's procedure of first
locating minima of the Mie denominators and then applying a root finder.
"""

from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import minimum_filter
from scipy.optimize import minimize
from scipy.special import spherical_jn, spherical_yn

CONFIG_FILENAME = "mie_spectrum_config.json"


@dataclass(frozen=True)
class Resonance:
    order: int
    family: str
    mu: complex
    spectral_point: complex
    relative_residual: float


@dataclass(frozen=True)
class SpectralConditionEstimate:
    """Condition-number lower estimate based on predicted spectral points."""

    minimum_modulus: float
    maximum_modulus: float
    spectral_ratio: float
    minimum_point: complex
    maximum_point: complex
    minimum_source: str
    maximum_source: str
    branch_minimum_modulus: float
    branch_maximum_modulus: float


@dataclass(frozen=True)
class EnclosingEllipse:
    """Ellipse containing the predicted spectrum in the complex plane."""

    center: complex
    major_semiaxis: float
    minor_semiaxis: float
    angle_radians: float
    shape_matrix: np.ndarray
    fit_method: str
    fit_iterations: int
    maximum_normalized_radius_squared: float
    point_count: int


@dataclass(frozen=True)
class GmresEllipseEstimate:
    """Chebyshev GMRES estimate based on an enclosing spectral ellipse."""

    valid: bool
    message: str
    relative_tolerance: float
    eigenvector_condition_number: float
    estimated_iterations: int | None
    residual_bound_at_estimate: float | None
    asymptotic_reduction_factor: float | None
    origin_ellipse_value: float
    focal_distance: float
    ellipse_parameter_rho: float | None
    transformed_origin: complex | None
    exterior_map_modulus: float | None
    ellipse: EnclosingEllipse


def spherical_hankel1(order: int, z: complex | np.ndarray) -> complex | np.ndarray:
    """Spherical Hankel function h_l^(1)(z)."""
    return spherical_jn(order, z) + 1j * spherical_yn(order, z)


def riccati_j_derivative(order: int, z: complex | np.ndarray) -> complex | np.ndarray:
    """Derivative [z j_l(z)]' with respect to z."""
    return spherical_jn(order, z) + z * spherical_jn(order, z, derivative=True)


def riccati_h1_derivative(order: int, z: complex | np.ndarray) -> complex | np.ndarray:
    """Derivative [z h_l^(1)(z)]' with respect to z."""
    h = spherical_hankel1(order, z)
    h_prime = spherical_jn(order, z, derivative=True) + 1j * spherical_yn(
        order, z, derivative=True
    )
    return h + z * h_prime



@lru_cache(maxsize=None)
def _exterior_values(order: int, x: float) -> tuple[complex, complex]:
    """Cache h_l^(1)(x) and [x h_l^(1)(x)]' for real x."""
    h_x = complex(spherical_hankel1(order, x))
    xi_prime_x = complex(riccati_h1_derivative(order, x))
    return h_x, xi_prime_x

def mie_denominator_terms(
    mu: complex | np.ndarray,
    order: int,
    x: float,
    family: str,
) -> tuple[complex | np.ndarray, complex | np.ndarray]:
    """Return the two terms of a Mie-coefficient denominator.

    family='c' corresponds to the denominator in formula (4.11):

        j_l(mu*x) [x h_l^(1)(x)]' - h_l^(1)(x) [mu*x*j_l(mu*x)]'.

    family='d' corresponds to the denominator in formula (4.12):

        mu^2 j_l(mu*x) [x h_l^(1)(x)]'
        - h_l^(1)(x) [mu*x*j_l(mu*x)]'.
    """
    if family not in {"c", "d"}:
        raise ValueError("family must be 'c' or 'd'")

    z = mu * x
    j = spherical_jn(order, z)
    j_prime = spherical_jn(order, z, derivative=True)
    psi_prime = j + z * j_prime

    h_x, xi_prime_x = _exterior_values(order, float(x))

    multiplier = 1.0 if family == "c" else mu**2
    first = multiplier * j * xi_prime_x
    second = h_x * psi_prime
    return first, second


def mie_denominator(
    mu: complex | np.ndarray,
    order: int,
    x: float,
    family: str,
) -> complex | np.ndarray:
    first, second = mie_denominator_terms(mu, order, x, family)
    return first - second


def relative_denominator_residual(mu: complex, order: int, x: float, family: str) -> float:
    first, second = mie_denominator_terms(mu, order, x, family)
    scale = abs(first) + abs(second) + np.finfo(float).tiny
    return float(abs(first - second) / scale)


def map_to_spectrum(mu: complex | np.ndarray, physical_m: complex) -> complex | np.ndarray:
    """Formula (4.13): fictitious index mu -> spectrum of I-K."""
    return 1.0 - (physical_m**2 - 1.0) / (mu**2 - 1.0)


def branch_cut_segment(physical_m: complex, count: int = 300) -> np.ndarray:
    """The branch-cut segment in sigma(I-K), from 1 to m^2.

    For a purely imaginary fictitious index mu=i*y,

        lambda(y) = 1 + (m^2-1)/(1+y^2),  y in [0, infinity),

    hence the image is exactly the straight segment joining m^2 and 1.
    """
    t = np.linspace(0.0, 1.0, count)
    return 1.0 + t * (physical_m**2 - 1.0)



def branch_cut_modulus_extrema(
    physical_m: complex,
) -> tuple[float, complex, float, complex]:
    """Return exact min/max of |lambda| on the segment [1, m^2].

    The minimum is the Euclidean distance from zero to the segment. The
    maximum of the convex function |lambda| on a segment is attained at an
    endpoint.
    """
    start = complex(1.0, 0.0)
    end = complex(physical_m**2)
    direction = end - start
    direction_norm_squared = abs(direction) ** 2

    if direction_norm_squared <= np.finfo(float).tiny:
        return abs(start), start, abs(start), start

    projection_parameter = -float(
        np.real(np.conjugate(direction) * start)
    ) / direction_norm_squared
    projection_parameter = min(1.0, max(0.0, projection_parameter))
    minimum_point = start + projection_parameter * direction

    if abs(start) >= abs(end):
        maximum_point = start
    else:
        maximum_point = end

    return (
        float(abs(minimum_point)),
        complex(minimum_point),
        float(abs(maximum_point)),
        complex(maximum_point),
    )


def estimate_spectral_condition_number(
    physical_m: complex,
    resonances: list[Resonance],
) -> SpectralConditionEstimate:
    """Estimate max|lambda|/min|lambda| over the predicted spectrum.

    For an actual nonsingular matrix A,

        kappa_2(A) >= max_j |lambda_j(A)| / min_j |lambda_j(A)|.

    Here the ratio is formed from the analytically predicted branch-cut
    segment and the retained Mie-resonance points. It is therefore a predicted
    spectral lower-bound estimate for the discretized matrix, not a rigorous
    bound unless these predicted points are known to be eigenvalues of that
    particular matrix.
    """
    (
        branch_minimum,
        branch_minimum_point,
        branch_maximum,
        branch_maximum_point,
    ) = branch_cut_modulus_extrema(physical_m)

    minimum_modulus = branch_minimum
    maximum_modulus = branch_maximum
    minimum_point = branch_minimum_point
    maximum_point = branch_maximum_point
    minimum_source = "branch_cut"
    maximum_source = "branch_cut"

    for item in resonances:
        modulus = float(abs(item.spectral_point))
        source = f"Mie resonance: l={item.order}, family={item.family}"

        if modulus < minimum_modulus:
            minimum_modulus = modulus
            minimum_point = item.spectral_point
            minimum_source = source

        if modulus > maximum_modulus:
            maximum_modulus = modulus
            maximum_point = item.spectral_point
            maximum_source = source

    if minimum_modulus <= np.finfo(float).tiny:
        spectral_ratio = float("inf")
    else:
        spectral_ratio = maximum_modulus / minimum_modulus

    return SpectralConditionEstimate(
        minimum_modulus=minimum_modulus,
        maximum_modulus=maximum_modulus,
        spectral_ratio=float(spectral_ratio),
        minimum_point=complex(minimum_point),
        maximum_point=complex(maximum_point),
        minimum_source=minimum_source,
        maximum_source=maximum_source,
        branch_minimum_modulus=branch_minimum,
        branch_maximum_modulus=branch_maximum,
    )



def predicted_spectrum_points(
    physical_m: complex,
    resonances: list[Resonance],
) -> np.ndarray:
    """Return planar points sufficient to enclose the predicted spectrum.

    Since an ellipse is convex, including the two endpoints 1 and m^2
    automatically includes the complete branch-cut segment [1, m^2].
    """
    values = [complex(1.0, 0.0), complex(physical_m**2)]
    values.extend(item.spectral_point for item in resonances)

    points = np.array([[value.real, value.imag] for value in values], dtype=float)
    return np.unique(points, axis=0)


def _pca_enclosing_ellipse(
    points: np.ndarray,
    minimum_axis_ratio: float,
    safety_factor: float,
) -> EnclosingEllipse:
    """Robust enclosing ellipse for collinear or nearly collinear points."""
    points = np.asarray(points, dtype=float)
    mean = np.mean(points, axis=0)
    centered = points - mean

    if len(points) >= 2 and np.linalg.norm(centered) > 0.0:
        _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
        rotation = right_vectors.T
    else:
        rotation = np.eye(2)

    local = points @ rotation
    minimum = np.min(local, axis=0)
    maximum = np.max(local, axis=0)
    midpoint = 0.5 * (minimum + maximum)
    half_width = 0.5 * (maximum - minimum)

    # Keep the first local axis as the major direction.
    if half_width[1] > half_width[0]:
        rotation = rotation[:, [1, 0]]
        local = points @ rotation
        minimum = np.min(local, axis=0)
        maximum = np.max(local, axis=0)
        midpoint = 0.5 * (minimum + maximum)
        half_width = 0.5 * (maximum - minimum)

    center_vector = rotation @ midpoint
    absolute_floor = max(
        1.0e-14,
        1.0e-12 * (1.0 + float(np.linalg.norm(center_vector))),
    )

    preliminary_major = max(float(half_width[0]), absolute_floor)
    preliminary_minor = max(
        float(half_width[1]),
        minimum_axis_ratio * preliminary_major,
        absolute_floor,
    )

    normalized_radius_squared = (
        ((local[:, 0] - midpoint[0]) / preliminary_major) ** 2
        + ((local[:, 1] - midpoint[1]) / preliminary_minor) ** 2
    )
    scale = max(
        1.0,
        float(np.sqrt(np.max(normalized_radius_squared))),
    ) * safety_factor

    major = preliminary_major * scale
    minor = preliminary_minor * scale
    shape_matrix = rotation @ np.diag([major**-2, minor**-2]) @ rotation.T

    shifted = points - center_vector
    maximum_value = float(
        np.max(np.einsum("ni,ij,nj->n", shifted, shape_matrix, shifted))
    )

    return EnclosingEllipse(
        center=complex(center_vector[0], center_vector[1]),
        major_semiaxis=major,
        minor_semiaxis=minor,
        angle_radians=float(np.arctan2(rotation[1, 0], rotation[0, 0])),
        shape_matrix=shape_matrix,
        fit_method="pca_fallback",
        fit_iterations=0,
        maximum_normalized_radius_squared=maximum_value,
        point_count=len(points),
    )


def minimum_volume_enclosing_ellipse(
    points: np.ndarray,
    algorithm_tolerance: float,
    maximum_iterations: int,
    safety_factor: float,
    minimum_axis_ratio: float,
    rank_tolerance: float,
) -> EnclosingEllipse:
    """Compute a minimum-volume enclosing ellipse using Khachiyan iteration.

    If the point set is collinear or numerically rank deficient, a thin
    PCA-aligned enclosing ellipse is used. The safety factor slightly enlarges
    the final ellipse so all input points are enclosed despite finite stopping
    tolerance.
    """
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("points must be a nonempty array of shape (N, 2)")

    centered = points - np.mean(points, axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)

    if (
        len(points) < 3
        or singular_values[0] <= np.finfo(float).tiny
        or singular_values[1] <= rank_tolerance * singular_values[0]
    ):
        return _pca_enclosing_ellipse(
            points,
            minimum_axis_ratio=minimum_axis_ratio,
            safety_factor=safety_factor,
        )

    point_count, dimension = points.shape
    lifted = np.vstack([points.T, np.ones(point_count)])
    weights = np.full(point_count, 1.0 / point_count)

    iterations = 0
    try:
        for iterations in range(1, maximum_iterations + 1):
            moment = (lifted * weights) @ lifted.T
            inverse_moment = np.linalg.inv(moment)
            leverage = np.sum(
                lifted * (inverse_moment @ lifted),
                axis=0,
            )

            selected = int(np.argmax(leverage))
            maximum_leverage = float(leverage[selected])

            if maximum_leverage <= dimension + 1.0 + algorithm_tolerance:
                break

            step = (
                maximum_leverage - dimension - 1.0
            ) / (
                (dimension + 1.0) * (maximum_leverage - 1.0)
            )

            if not 0.0 < step < 1.0:
                break

            weights *= 1.0 - step
            weights[selected] += step

        center_vector = points.T @ weights
        covariance = (
            points.T @ (weights[:, None] * points)
            - np.outer(center_vector, center_vector)
        )
        shape_matrix = np.linalg.inv(covariance) / dimension

        eigenvalues, eigenvectors = np.linalg.eigh(shape_matrix)
        if np.any(eigenvalues <= 0.0):
            raise np.linalg.LinAlgError("nonpositive ellipse shape eigenvalue")

        semiaxes = 1.0 / np.sqrt(eigenvalues)
        ordering = np.argsort(semiaxes)[::-1]
        semiaxes = semiaxes[ordering]
        eigenvectors = eigenvectors[:, ordering]

        shifted = points - center_vector
        normalized_values = np.einsum(
            "ni,ij,nj->n",
            shifted,
            shape_matrix,
            shifted,
        )

        # Guarantee enclosure after the finite Khachiyan iteration.
        scale = max(
            1.0,
            float(np.sqrt(np.max(normalized_values))),
        ) * safety_factor
        semiaxes *= scale
        shape_matrix /= scale**2

        shifted = points - center_vector
        maximum_value = float(
            np.max(
                np.einsum(
                    "ni,ij,nj->n",
                    shifted,
                    shape_matrix,
                    shifted,
                )
            )
        )

        return EnclosingEllipse(
            center=complex(center_vector[0], center_vector[1]),
            major_semiaxis=float(semiaxes[0]),
            minor_semiaxis=float(semiaxes[1]),
            angle_radians=float(
                np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
            ),
            shape_matrix=shape_matrix,
            fit_method="khachiyan_mvee",
            fit_iterations=iterations,
            maximum_normalized_radius_squared=maximum_value,
            point_count=point_count,
        )
    except np.linalg.LinAlgError:
        return _pca_enclosing_ellipse(
            points,
            minimum_axis_ratio=minimum_axis_ratio,
            safety_factor=safety_factor,
        )



def _ellipse_parameters_from_object(
    ellipse: EnclosingEllipse,
) -> np.ndarray:
    return np.array(
        [
            ellipse.center.real,
            ellipse.center.imag,
            np.log(ellipse.major_semiaxis),
            np.log(ellipse.minor_semiaxis),
            ellipse.angle_radians,
        ],
        dtype=float,
    )


def _ellipse_values_from_parameters(
    parameters: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Evaluate normalized squared ellipse radii for planar points."""
    center_x, center_y, log_axis_1, log_axis_2, angle = parameters
    axis_1 = float(np.exp(log_axis_1))
    axis_2 = float(np.exp(log_axis_2))

    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))

    delta_x = points[:, 0] - center_x
    delta_y = points[:, 1] - center_y

    local_1 = cosine * delta_x + sine * delta_y
    local_2 = -sine * delta_x + cosine * delta_y

    return (local_1 / axis_1) ** 2 + (local_2 / axis_2) ** 2


def _ellipse_from_parameters(
    parameters: np.ndarray,
    point_count: int,
    fit_method: str,
    fit_iterations: int,
    points: np.ndarray,
) -> EnclosingEllipse:
    center_x, center_y, log_axis_1, log_axis_2, angle = parameters
    axis_1 = float(np.exp(log_axis_1))
    axis_2 = float(np.exp(log_axis_2))

    # Store the semiaxes in descending order.
    if axis_2 > axis_1:
        axis_1, axis_2 = axis_2, axis_1
        angle += 0.5 * np.pi

    angle = float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    rotation = np.array(
        [[cosine, -sine], [sine, cosine]],
        dtype=float,
    )
    shape_matrix = (
        rotation
        @ np.diag([axis_1**-2, axis_2**-2])
        @ rotation.T
    )

    center = np.array([center_x, center_y], dtype=float)
    shifted = points - center
    normalized_values = np.einsum(
        "ni,ij,nj->n",
        shifted,
        shape_matrix,
        shifted,
    )

    return EnclosingEllipse(
        center=complex(center_x, center_y),
        major_semiaxis=axis_1,
        minor_semiaxis=axis_2,
        angle_radians=angle,
        shape_matrix=shape_matrix,
        fit_method=fit_method,
        fit_iterations=fit_iterations,
        maximum_normalized_radius_squared=float(
            np.max(normalized_values)
        ),
        point_count=point_count,
    )


def _closest_point_in_convex_hull(
    points: np.ndarray,
    tolerance: float,
    maximum_iterations: int,
) -> np.ndarray:
    """Find the minimum-norm point in conv(points)."""
    point_count = len(points)
    if point_count == 1:
        return points[0].copy()

    initial_weights = np.full(point_count, 1.0 / point_count)

    def objective(weights: np.ndarray) -> float:
        point = weights @ points
        return 0.5 * float(point @ point)

    def objective_jacobian(weights: np.ndarray) -> np.ndarray:
        point = weights @ points
        return points @ point

    result = minimize(
        objective,
        initial_weights,
        jac=objective_jacobian,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * point_count,
        constraints=[
            {
                "type": "eq",
                "fun": lambda weights: float(np.sum(weights) - 1.0),
                "jac": lambda weights: np.ones_like(weights),
            }
        ],
        options={
            "maxiter": maximum_iterations,
            "ftol": tolerance,
            "disp": False,
        },
    )

    weights = result.x
    weights = np.maximum(weights, 0.0)
    total = float(np.sum(weights))
    if total <= np.finfo(float).tiny:
        weights = initial_weights
    else:
        weights /= total

    return weights @ points


def _construct_origin_excluding_initial_ellipse(
    points: np.ndarray,
    origin_margin: float,
    minimum_axis_ratio: float,
    convex_hull_tolerance: float,
    convex_hull_maximum_iterations: int,
) -> EnclosingEllipse:
    """Construct a feasible ellipse from a separating direction.

    If zero is outside conv(points), the minimum-norm point q of the convex
    hull supplies a separating normal u=q/|q|. The ellipse is then constructed
    in (u,v) coordinates with its leftmost point strictly to the right of zero.
    """
    closest = _closest_point_in_convex_hull(
        points,
        tolerance=convex_hull_tolerance,
        maximum_iterations=convex_hull_maximum_iterations,
    )

    point_scale = max(
        1.0,
        float(np.max(np.linalg.norm(points, axis=1))),
    )
    closest_norm = float(np.linalg.norm(closest))

    if closest_norm <= convex_hull_tolerance * point_scale:
        raise ValueError(
            "The origin belongs to, or is numerically indistinguishable from, "
            "the convex hull of the predicted spectrum. No convex ellipse "
            "containing the spectrum can exclude the origin."
        )

    direction_1 = closest / closest_norm
    direction_2 = np.array(
        [-direction_1[1], direction_1[0]],
        dtype=float,
    )

    coordinate_1 = points @ direction_1
    coordinate_2 = points @ direction_2

    minimum_1 = float(np.min(coordinate_1))
    maximum_1 = float(np.max(coordinate_1))
    minimum_2 = float(np.min(coordinate_2))
    maximum_2 = float(np.max(coordinate_2))

    if minimum_1 <= 0.0:
        raise ValueError(
            "Failed to construct a strict separating direction for the origin."
        )

    span_1 = max(maximum_1 - minimum_1, 1.0e-6 * point_scale)
    center_1 = maximum_1 + max(1.0e-3 * point_scale, 0.05 * span_1)

    required_left_offset = center_1 * (
        1.0 - 1.0 / np.sqrt(1.0 + origin_margin)
    )
    left_offset = max(
        0.2 * minimum_1,
        1.1 * required_left_offset,
    )

    if left_offset >= 0.95 * minimum_1:
        left_offset = 0.95 * minimum_1

    axis_1 = center_1 - left_offset
    center_2 = 0.5 * (minimum_2 + maximum_2)

    horizontal = (coordinate_1 - center_1) / axis_1
    remaining = np.maximum(1.0 - horizontal**2, 1.0e-14)
    required_axis_2 = np.max(
        np.abs(coordinate_2 - center_2) / np.sqrt(remaining)
    )

    axis_floor = max(
        minimum_axis_ratio * axis_1,
        1.0e-10 * point_scale,
    )
    axis_2 = max(float(required_axis_2) * 1.01, axis_floor)

    center_vector = (
        center_1 * direction_1
        + center_2 * direction_2
    )
    angle = float(np.arctan2(direction_1[1], direction_1[0]))

    parameters = np.array(
        [
            center_vector[0],
            center_vector[1],
            np.log(axis_1),
            np.log(axis_2),
            angle,
        ],
        dtype=float,
    )

    ellipse = _ellipse_from_parameters(
        parameters=parameters,
        point_count=len(points),
        fit_method="separating_initial_ellipse",
        fit_iterations=0,
        points=points,
    )

    origin = np.zeros((1, 2), dtype=float)
    origin_value = float(
        _ellipse_values_from_parameters(parameters, origin)[0]
    )
    if (
        ellipse.maximum_normalized_radius_squared > 1.0 + 1.0e-8
        or origin_value < 1.0 + origin_margin - 1.0e-8
    ):
        raise ValueError(
            "Could not construct a feasible initial ellipse with the "
            "requested origin-exclusion margin."
        )

    return ellipse


def minimum_area_enclosing_ellipse_excluding_origin(
    points: np.ndarray,
    unconstrained_ellipse: EnclosingEllipse,
    origin_margin: float,
    point_margin: float,
    optimizer_tolerance: float,
    optimizer_maximum_iterations: int,
    optimizer_restarts: int,
    random_seed: int,
    minimum_axis_ratio: float,
    convex_hull_tolerance: float,
    convex_hull_maximum_iterations: int,
    feasibility_tolerance: float,
) -> EnclosingEllipse:
    r"""Minimize ellipse area while keeping zero strictly outside.

    The ellipse is parameterized by center c, two positive semiaxes a,b and a
    rotation angle theta. The nonlinear problem is

        minimize      log(a) + log(b)
        subject to    q(p_i) <= 1-point_margin,
                      q(0) >= 1+origin_margin,

    where q is the normalized squared ellipse radius. The problem is
    nonconvex; several deterministic/randomized starts are used.
    """
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(points) == 0:
        raise ValueError("At least one spectral point is required")

    origin = np.zeros((1, 2), dtype=float)
    unconstrained_parameters = _ellipse_parameters_from_object(
        unconstrained_ellipse
    )
    unconstrained_origin_value = float(
        _ellipse_values_from_parameters(
            unconstrained_parameters,
            origin,
        )[0]
    )

    # If the ordinary MVEE already satisfies the extra constraint, it remains
    # globally minimal for the constrained problem.
    if unconstrained_origin_value >= 1.0 + origin_margin:
        return EnclosingEllipse(
            center=unconstrained_ellipse.center,
            major_semiaxis=unconstrained_ellipse.major_semiaxis,
            minor_semiaxis=unconstrained_ellipse.minor_semiaxis,
            angle_radians=unconstrained_ellipse.angle_radians,
            shape_matrix=unconstrained_ellipse.shape_matrix,
            fit_method=unconstrained_ellipse.fit_method
            + "_already_excludes_origin",
            fit_iterations=unconstrained_ellipse.fit_iterations,
            maximum_normalized_radius_squared=(
                unconstrained_ellipse.maximum_normalized_radius_squared
            ),
            point_count=unconstrained_ellipse.point_count,
        )

    feasible_initial = _construct_origin_excluding_initial_ellipse(
        points=points,
        origin_margin=origin_margin,
        minimum_axis_ratio=minimum_axis_ratio,
        convex_hull_tolerance=convex_hull_tolerance,
        convex_hull_maximum_iterations=(
            convex_hull_maximum_iterations
        ),
    )
    feasible_parameters = _ellipse_parameters_from_object(
        feasible_initial
    )

    point_scale = max(
        1.0,
        float(np.max(np.linalg.norm(points, axis=1))),
    )
    minimum_axis = max(
        1.0e-8 * point_scale,
        minimum_axis_ratio * 1.0e-3 * point_scale,
    )
    maximum_axis = 1.0e4 * point_scale
    center_bound = 1.0e4 * point_scale

    bounds = [
        (-center_bound, center_bound),
        (-center_bound, center_bound),
        (np.log(minimum_axis), np.log(maximum_axis)),
        (np.log(minimum_axis), np.log(maximum_axis)),
        (-4.0 * np.pi, 4.0 * np.pi),
    ]

    def objective(parameters: np.ndarray) -> float:
        return float(parameters[2] + parameters[3])

    def objective_jacobian(parameters: np.ndarray) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0, 1.0, 0.0])

    def point_constraints(parameters: np.ndarray) -> np.ndarray:
        return (
            1.0
            - point_margin
            - _ellipse_values_from_parameters(parameters, points)
        )

    def origin_constraint(parameters: np.ndarray) -> float:
        return float(
            _ellipse_values_from_parameters(parameters, origin)[0]
            - 1.0
            - origin_margin
        )

    constraints = [
        {"type": "ineq", "fun": point_constraints},
        {"type": "ineq", "fun": origin_constraint},
    ]

    random_generator = np.random.default_rng(random_seed)
    starts = [feasible_parameters, unconstrained_parameters]

    for restart in range(max(0, optimizer_restarts - 2)):
        candidate = feasible_parameters.copy()
        perturbation_scale = 0.04 + 0.015 * (restart % 5)
        candidate[0:2] += random_generator.normal(
            scale=perturbation_scale * point_scale,
            size=2,
        )
        candidate[2:4] += random_generator.normal(
            scale=0.12,
            size=2,
        )
        candidate[4] += random_generator.normal(scale=0.25)
        starts.append(candidate)

    best_parameters = feasible_parameters
    best_objective = objective(feasible_parameters)
    best_iterations = 0
    successful_optimization = False
    best_message = "feasible separating construction"

    def is_feasible(parameters: np.ndarray) -> bool:
        point_values = _ellipse_values_from_parameters(
            parameters,
            points,
        )
        origin_value = float(
            _ellipse_values_from_parameters(
                parameters,
                origin,
            )[0]
        )
        return (
            float(np.max(point_values))
            <= 1.0 - point_margin + feasibility_tolerance
            and origin_value
            >= 1.0 + origin_margin - feasibility_tolerance
        )

    for start in starts:
        result = minimize(
            objective,
            start,
            jac=objective_jacobian,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": optimizer_maximum_iterations,
                "ftol": optimizer_tolerance,
                "disp": False,
            },
        )

        if not np.all(np.isfinite(result.x)):
            continue

        if is_feasible(result.x):
            current_objective = objective(result.x)
            if current_objective < best_objective:
                best_parameters = result.x.copy()
                best_objective = current_objective
                best_iterations = int(getattr(result, "nit", 0))
                successful_optimization = bool(result.success)
                best_message = str(result.message)

    fit_method = (
        "constrained_slsqp_origin_excluded"
        if successful_optimization
        else "constrained_feasible_origin_excluded"
    )

    ellipse = _ellipse_from_parameters(
        parameters=best_parameters,
        point_count=len(points),
        fit_method=fit_method,
        fit_iterations=best_iterations,
        points=points,
    )

    final_origin_value = float(
        _ellipse_values_from_parameters(
            best_parameters,
            origin,
        )[0]
    )

    if not is_feasible(best_parameters):
        raise RuntimeError(
            "The constrained ellipse optimizer did not return a feasible "
            "ellipse. Last status: " + best_message
        )

    # Include the achieved exclusion value in the method label for diagnostics.
    return EnclosingEllipse(
        center=ellipse.center,
        major_semiaxis=ellipse.major_semiaxis,
        minor_semiaxis=ellipse.minor_semiaxis,
        angle_radians=ellipse.angle_radians,
        shape_matrix=ellipse.shape_matrix,
        fit_method=(
            ellipse.fit_method
            + f"_origin_value_{final_origin_value:.6g}"
        ),
        fit_iterations=ellipse.fit_iterations,
        maximum_normalized_radius_squared=(
            ellipse.maximum_normalized_radius_squared
        ),
        point_count=ellipse.point_count,
    )

def enclosing_ellipse_curve(
    ellipse: EnclosingEllipse,
    point_count: int = 500,
) -> np.ndarray:
    """Sample the boundary of an enclosing ellipse as complex values."""
    parameter = np.linspace(0.0, 2.0 * np.pi, point_count)
    local = (
        ellipse.major_semiaxis * np.cos(parameter)
        + 1j * ellipse.minor_semiaxis * np.sin(parameter)
    )
    return ellipse.center + np.exp(1j * ellipse.angle_radians) * local


def _ellipse_residual_log_bound(
    iteration: int,
    rho: float,
    exterior_map_value: complex,
    eigenvector_condition_number: float,
) -> float:
    """Logarithm of the Chebyshev residual bound for one iteration."""
    log_rho = float(np.log(rho))
    log_exterior_modulus = float(np.log(abs(exterior_map_value)))

    # (rho^k + rho^-k) / |w^k + w^-k|, computed without overflow.
    numerator_correction = float(np.log1p(np.exp(-2.0 * iteration * log_rho)))
    inverse_power = np.exp(
        -2.0 * iteration * np.log(exterior_map_value)
    )
    denominator_correction = float(np.log(abs(1.0 + inverse_power)))

    return (
        float(np.log(eigenvector_condition_number))
        + iteration * (log_rho - log_exterior_modulus)
        + numerator_correction
        - denominator_correction
    )


def estimate_gmres_iterations_from_ellipse(
    physical_m: complex,
    resonances: list[Resonance],
    relative_tolerance: float,
    eigenvector_condition_number: float,
    maximum_gmres_iterations: int,
    ellipse_algorithm_tolerance: float,
    ellipse_maximum_iterations: int,
    ellipse_safety_factor: float,
    ellipse_minimum_axis_ratio: float,
    ellipse_rank_tolerance: float,
    ellipse_exclude_origin: bool,
    ellipse_origin_margin: float,
    ellipse_point_margin: float,
    ellipse_optimizer_tolerance: float,
    ellipse_optimizer_maximum_iterations: int,
    ellipse_optimizer_restarts: int,
    ellipse_random_seed: int,
    ellipse_convex_hull_tolerance: float,
    ellipse_convex_hull_maximum_iterations: int,
    ellipse_feasibility_tolerance: float,
) -> GmresEllipseEstimate:
    r"""Estimate unrestarted GMRES iterations using a spectral ellipse.

    Let the enclosing ellipse have center c, semiaxes a >= b, rotation theta,
    and focal distance f=sqrt(a^2-b^2). Define

        rho = (a+b)/f,
        zeta_0 = -c exp(-i theta)/f,
        w_0 = zeta_0 +/- sqrt(zeta_0^2-1),  |w_0| > 1.

    For a diagonalizable matrix A with eigenvector matrix V,

        ||r_k||/||r_0||
        <= kappa_2(V) (rho^k+rho^-k)/|w_0^k+w_0^-k|.

    The default configuration uses kappa_2(V)=1, i.e. an approximately normal
    matrix. A finite estimate requires the origin to lie strictly outside the
    enclosing ellipse.
    """
    points = predicted_spectrum_points(physical_m, resonances)
    unconstrained_ellipse = minimum_volume_enclosing_ellipse(
        points=points,
        algorithm_tolerance=ellipse_algorithm_tolerance,
        maximum_iterations=ellipse_maximum_iterations,
        safety_factor=ellipse_safety_factor,
        minimum_axis_ratio=ellipse_minimum_axis_ratio,
        rank_tolerance=ellipse_rank_tolerance,
    )

    if ellipse_exclude_origin:
        try:
            ellipse = minimum_area_enclosing_ellipse_excluding_origin(
                points=points,
                unconstrained_ellipse=unconstrained_ellipse,
                origin_margin=ellipse_origin_margin,
                point_margin=ellipse_point_margin,
                optimizer_tolerance=ellipse_optimizer_tolerance,
                optimizer_maximum_iterations=(
                    ellipse_optimizer_maximum_iterations
                ),
                optimizer_restarts=ellipse_optimizer_restarts,
                random_seed=ellipse_random_seed,
                minimum_axis_ratio=ellipse_minimum_axis_ratio,
                convex_hull_tolerance=(
                    ellipse_convex_hull_tolerance
                ),
                convex_hull_maximum_iterations=(
                    ellipse_convex_hull_maximum_iterations
                ),
                feasibility_tolerance=(
                    ellipse_feasibility_tolerance
                ),
            )
        except (ValueError, RuntimeError) as error:
            rotated_origin = (
                -unconstrained_ellipse.center
                * np.exp(-1j * unconstrained_ellipse.angle_radians)
            )
            origin_ellipse_value = float(
                (
                    rotated_origin.real
                    / unconstrained_ellipse.major_semiaxis
                )
                ** 2
                + (
                    rotated_origin.imag
                    / unconstrained_ellipse.minor_semiaxis
                )
                ** 2
            )
            return GmresEllipseEstimate(
                valid=False,
                message=str(error),
                relative_tolerance=relative_tolerance,
                eigenvector_condition_number=(
                    eigenvector_condition_number
                ),
                estimated_iterations=None,
                residual_bound_at_estimate=None,
                asymptotic_reduction_factor=None,
                origin_ellipse_value=origin_ellipse_value,
                focal_distance=0.0,
                ellipse_parameter_rho=None,
                transformed_origin=None,
                exterior_map_modulus=None,
                ellipse=unconstrained_ellipse,
            )
    else:
        ellipse = unconstrained_ellipse

    rotated_origin = (
        -ellipse.center * np.exp(-1j * ellipse.angle_radians)
    )
    origin_ellipse_value = float(
        (rotated_origin.real / ellipse.major_semiaxis) ** 2
        + (rotated_origin.imag / ellipse.minor_semiaxis) ** 2
    )

    if origin_ellipse_value <= 1.0 + 1.0e-12:
        return GmresEllipseEstimate(
            valid=False,
            message=(
                "The enclosing ellipse contains or touches the origin; "
                "the classical ellipse GMRES bound gives no convergence estimate."
            ),
            relative_tolerance=relative_tolerance,
            eigenvector_condition_number=eigenvector_condition_number,
            estimated_iterations=None,
            residual_bound_at_estimate=None,
            asymptotic_reduction_factor=None,
            origin_ellipse_value=origin_ellipse_value,
            focal_distance=0.0,
            ellipse_parameter_rho=None,
            transformed_origin=None,
            exterior_map_modulus=None,
            ellipse=ellipse,
        )

    major = ellipse.major_semiaxis
    minor = ellipse.minor_semiaxis
    circle_threshold = 1.0e-10 * major

    # Circle limit: p_k(z)=(1-z/c)^k gives (R/|c|)^k.
    if major - minor <= circle_threshold:
        reduction = major / abs(ellipse.center)
        if not 0.0 < reduction < 1.0:
            return GmresEllipseEstimate(
                valid=False,
                message=(
                    "The fitted ellipse is circular but does not exclude the origin."
                ),
                relative_tolerance=relative_tolerance,
                eigenvector_condition_number=eigenvector_condition_number,
                estimated_iterations=None,
                residual_bound_at_estimate=None,
                asymptotic_reduction_factor=None,
                origin_ellipse_value=origin_ellipse_value,
                focal_distance=0.0,
                ellipse_parameter_rho=None,
                transformed_origin=None,
                exterior_map_modulus=None,
                ellipse=ellipse,
            )

        target_log = float(np.log(relative_tolerance))
        log_prefactor = float(np.log(eigenvector_condition_number))
        log_reduction = float(np.log(reduction))

        estimated_iterations = None
        residual_bound = None
        for iteration in range(1, maximum_gmres_iterations + 1):
            current_log_bound = log_prefactor + iteration * log_reduction
            if current_log_bound <= target_log:
                estimated_iterations = iteration
                residual_bound = float(np.exp(current_log_bound))
                break

        return GmresEllipseEstimate(
            valid=estimated_iterations is not None,
            message=(
                "Circle-limit GMRES estimate computed."
                if estimated_iterations is not None
                else "The configured GMRES iteration limit was reached."
            ),
            relative_tolerance=relative_tolerance,
            eigenvector_condition_number=eigenvector_condition_number,
            estimated_iterations=estimated_iterations,
            residual_bound_at_estimate=residual_bound,
            asymptotic_reduction_factor=float(reduction),
            origin_ellipse_value=origin_ellipse_value,
            focal_distance=0.0,
            ellipse_parameter_rho=None,
            transformed_origin=None,
            exterior_map_modulus=None,
            ellipse=ellipse,
        )

    focal_distance = float(np.sqrt(major**2 - minor**2))
    rho = float((major + minor) / focal_distance)
    transformed_origin = complex(rotated_origin / focal_distance)

    square_root = np.sqrt(transformed_origin**2 - 1.0)
    candidates = (
        transformed_origin + square_root,
        transformed_origin - square_root,
    )
    exterior_map_value = max(candidates, key=abs)
    exterior_modulus = float(abs(exterior_map_value))

    if exterior_modulus <= rho * (1.0 + 1.0e-12):
        return GmresEllipseEstimate(
            valid=False,
            message=(
                "The conformal image of the origin is not outside the ellipse "
                "parameter rho; no finite Chebyshev estimate is available."
            ),
            relative_tolerance=relative_tolerance,
            eigenvector_condition_number=eigenvector_condition_number,
            estimated_iterations=None,
            residual_bound_at_estimate=None,
            asymptotic_reduction_factor=None,
            origin_ellipse_value=origin_ellipse_value,
            focal_distance=focal_distance,
            ellipse_parameter_rho=rho,
            transformed_origin=transformed_origin,
            exterior_map_modulus=exterior_modulus,
            ellipse=ellipse,
        )

    target_log = float(np.log(relative_tolerance))
    estimated_iterations = None
    residual_bound = None

    for iteration in range(1, maximum_gmres_iterations + 1):
        current_log_bound = _ellipse_residual_log_bound(
            iteration=iteration,
            rho=rho,
            exterior_map_value=exterior_map_value,
            eigenvector_condition_number=eigenvector_condition_number,
        )
        if current_log_bound <= target_log:
            estimated_iterations = iteration
            residual_bound = float(np.exp(current_log_bound))
            break

    asymptotic_factor = float(rho / exterior_modulus)

    return GmresEllipseEstimate(
        valid=estimated_iterations is not None,
        message=(
            "Chebyshev ellipse GMRES estimate computed."
            if estimated_iterations is not None
            else "The configured GMRES iteration limit was reached."
        ),
        relative_tolerance=relative_tolerance,
        eigenvector_condition_number=eigenvector_condition_number,
        estimated_iterations=estimated_iterations,
        residual_bound_at_estimate=residual_bound,
        asymptotic_reduction_factor=asymptotic_factor,
        origin_ellipse_value=origin_ellipse_value,
        focal_distance=focal_distance,
        ellipse_parameter_rho=rho,
        transformed_origin=transformed_origin,
        exterior_map_modulus=exterior_modulus,
        ellipse=ellipse,
    )


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _complex_payload(value: complex | None) -> dict | None:
    if value is None:
        return None
    return {"real": float(value.real), "imag": float(value.imag)}


def save_gmres_estimate_json(
    estimate: GmresEllipseEstimate,
    path: Path,
) -> None:
    """Save the ellipse and GMRES iteration estimate."""
    ellipse = estimate.ellipse
    payload = {
        "valid": estimate.valid,
        "message": estimate.message,
        "relative_residual_tolerance": estimate.relative_tolerance,
        "assumed_eigenvector_condition_number": (
            estimate.eigenvector_condition_number
        ),
        "estimated_unrestarted_gmres_iterations": estimate.estimated_iterations,
        "residual_bound_at_estimated_iteration": _finite_or_none(
            estimate.residual_bound_at_estimate
        ),
        "asymptotic_reduction_factor": _finite_or_none(
            estimate.asymptotic_reduction_factor
        ),
        "origin_ellipse_value": estimate.origin_ellipse_value,
        "ellipse": {
            "center": _complex_payload(ellipse.center),
            "major_semiaxis": ellipse.major_semiaxis,
            "minor_semiaxis": ellipse.minor_semiaxis,
            "angle_radians": ellipse.angle_radians,
            "angle_degrees": float(np.degrees(ellipse.angle_radians)),
            "focal_distance": estimate.focal_distance,
            "ellipse_parameter_rho": _finite_or_none(
                estimate.ellipse_parameter_rho
            ),
            "transformed_origin": _complex_payload(
                estimate.transformed_origin
            ),
            "exterior_map_modulus": _finite_or_none(
                estimate.exterior_map_modulus
            ),
            "fit_method": ellipse.fit_method,
            "fit_iterations": ellipse.fit_iterations,
            "maximum_normalized_radius_squared": (
                ellipse.maximum_normalized_radius_squared
            ),
            "point_count": ellipse.point_count,
        },
        "bound": {
            "formula": (
                "||r_k||/||r_0|| <= kappa_2(V) "
                "(rho^k+rho^(-k))/|w_0^k+w_0^(-k)|"
            ),
            "assumption": (
                "A is diagonalizable and close to normal; the default uses "
                "kappa_2(V)=1."
            ),
            "spectrum_model": (
                "The ellipse encloses the branch-cut segment [1,m^2] and all "
                "retained Mie-resonance points."
            ),
        },
    }

    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\\n")

def save_condition_estimate_json(
    estimate: SpectralConditionEstimate,
    path: Path,
) -> None:
    """Save the spectral condition-number estimate and its interpretation."""
    payload = {
        "predicted_minimum_eigenvalue_modulus": estimate.minimum_modulus,
        "predicted_maximum_eigenvalue_modulus": estimate.maximum_modulus,
        "predicted_spectral_condition_ratio": estimate.spectral_ratio,
        "minimum_point": {
            "real": estimate.minimum_point.real,
            "imag": estimate.minimum_point.imag,
            "source": estimate.minimum_source,
        },
        "maximum_point": {
            "real": estimate.maximum_point.real,
            "imag": estimate.maximum_point.imag,
            "source": estimate.maximum_source,
        },
        "branch_cut": {
            "minimum_modulus": estimate.branch_minimum_modulus,
            "maximum_modulus": estimate.branch_maximum_modulus,
        },
        "interpretation": {
            "formula": "kappa_2(A) >= max_j |lambda_j(A)| / min_j |lambda_j(A)|",
            "normal_matrix": (
                "For a normal matrix this ratio equals the 2-norm condition number."
            ),
            "nonnormal_matrix": (
                "For a nonnormal matrix it is only a lower bound when the values "
                "used are actual eigenvalues."
            ),
            "prediction_caveat": (
                "This file uses the analytically predicted spectrum; for a finite "
                "discretized matrix the result is a spectral estimate."
            ),
        },
    }

    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\\n")


def _grid_minimum_seeds(
    mu_grid: np.ndarray,
    denominator_values: np.ndarray,
    max_seeds: int,
) -> list[complex]:
    """Find candidate roots from local minima of log10|D_l(mu)|."""
    with np.errstate(all="ignore"):
        log_abs = np.log10(np.abs(denominator_values) + np.finfo(float).tiny)

    finite = np.isfinite(log_abs)
    if not np.any(finite):
        return []

    replacement = np.nanmax(log_abs[finite]) + 100.0
    log_abs = np.where(finite, log_abs, replacement)

    neighborhood_min = minimum_filter(log_abs, size=3, mode="nearest")
    mask = log_abs <= neighborhood_min + 1.0e-12

    typical = float(np.median(log_abs[finite]))
    depths = typical - log_abs
    mask &= depths >= 0.5
    flat_indices = np.flatnonzero(mask)
    if flat_indices.size == 0:
        return []

    if flat_indices.size > max_seeds:
        candidate_depths = depths.ravel()[flat_indices]
        keep = np.argpartition(candidate_depths, -max_seeds)[-max_seeds:]
        flat_indices = flat_indices[keep]

    order = np.argsort(depths.ravel()[flat_indices])[::-1]
    flat_indices = flat_indices[order]
    return [complex(value) for value in mu_grid.ravel()[flat_indices]]


def _axis_seeds(search_radius: float, x: float) -> list[complex]:
    """Generic seeds near the positive-real and negative-imaginary axes.

    We use the closed fourth quadrant as the canonical square-root branch:

        Re(mu) >= 0,   Im(mu) <= 0.

    In particular, the negative imaginary axis is part of the search domain.
    This is essential for plasmonic modes, whose quasistatic roots approach
    that axis as x = k r -> 0.
    """
    seeds: list[complex] = []

    real_step = max(
        np.pi / (2.0 * max(x, 1.0e-12)),
        search_radius / 80.0,
    )
    for re in np.arange(0.5 * real_step, search_radius, real_step):
        # Include the real axis and a small negative-imaginary offset.
        seeds.append(complex(re, 0.0))
        seeds.append(complex(re, -min(0.02, 0.02 * max(1.0, search_radius))))

    # Seeds directly on the negative imaginary axis plus nearby points in
    # the interior of the fourth quadrant.
    imag_limit = min(search_radius, 6.0)
    for y in np.arange(0.1, imag_limit + 0.05, 0.1):
        seeds.append(complex(0.0, -y))
        seeds.append(complex(min(0.01, 0.01 * search_radius), -y))

    return seeds


def _plasmon_quasistatic_seeds(
    order: int,
    search_radius: float,
) -> list[complex]:
    r"""Seeds targeted at surface-plasmon Mie resonances.

    In the quasistatic limit the electric multipole of order l has

        epsilon_l -> -(l+1)/l.

    On the outgoing resonance branch used here we choose

        mu_l = sqrt(epsilon_l) -> -i sqrt((l+1)/l),

    i.e. the negative imaginary axis.  Radiation moves the finite-x root
    slightly into the fourth quadrant, so several small positive-real
    offsets are used around the quasistatic value.
    """
    if order <= 0:
        return []

    y0 = float(np.sqrt((order + 1.0) / order))
    if y0 > search_radius * 1.02:
        return []

    # The root approaches Re(mu)=0 rapidly for small x.  The logarithmic set
    # of offsets prevents the search from missing a very narrow near-axis
    # minimum while still covering moderately shifted dynamic resonances.
    offsets = (0.0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2, 1.0e-1)
    seeds: list[complex] = []
    for re in offsets:
        if re <= search_radius:
            seeds.append(complex(re, -y0))
            # Small vertical perturbations help the secant iteration when the
            # exact dynamic root is displaced from the quasistatic estimate.
            seeds.append(complex(re, -0.95 * y0))
            seeds.append(complex(re, -1.05 * y0))
    return seeds


def _line_minimum_seeds(
    mu_values: np.ndarray,
    denominator_values: np.ndarray,
    max_seeds: int,
) -> list[complex]:
    """Find local minima of |D(mu)| along a one-dimensional search line."""
    mu_values = np.asarray(mu_values, dtype=complex).ravel()
    values = np.asarray(denominator_values, dtype=complex).ravel()

    with np.errstate(all="ignore"):
        log_abs = np.log10(np.abs(values) + np.finfo(float).tiny)

    finite = np.isfinite(log_abs)
    if not np.any(finite):
        return []

    replacement = np.nanmax(log_abs[finite]) + 100.0
    log_abs = np.where(finite, log_abs, replacement)

    # One-dimensional local minima, including endpoints only when they are
    # genuinely among the smallest values.
    local = minimum_filter(log_abs, size=3, mode="nearest")
    mask = log_abs <= local + 1.0e-12

    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []

    # Rank candidates by the absolute denominator value rather than by a
    # fixed depth threshold.  Near-axis plasmonic minima can be narrow and
    # otherwise disappear on a coarse global grid.
    ordering = np.argsort(log_abs[indices])
    indices = indices[ordering[:max_seeds]]
    return [complex(mu_values[index]) for index in indices]


def _near_imaginary_axis_seeds(
    order: int,
    x: float,
    family: str,
    search_radius: float,
    n_imag: int,
    max_seeds: int,
) -> list[complex]:
    """Dense targeted scan along/near the negative imaginary axis.

    This scan is the main robustness addition for plasmonic resonances.  The
    ordinary two-dimensional box may have a very coarse first Re(mu) cell when
    search_radius is large, while a plasmonic root may have Re(mu) << 1.
    """
    if search_radius <= 0.0:
        return []

    # Resolve the physically important interval |Im(mu)| = O(1) very densely,
    # while retaining a coarser tail up to the global search radius.
    near_limit = min(search_radius, 6.0)
    near_count = max(500, 8 * n_imag)
    y_near = np.linspace(1.0e-6, near_limit, near_count)

    if search_radius > near_limit:
        far_count = max(100, 2 * n_imag)
        y_far = np.linspace(near_limit, search_radius, far_count)[1:]
        y_values = np.concatenate([y_near, y_far])
    else:
        y_values = y_near

    # Include the boundary Re(mu)=0 exactly and several logarithmically small
    # offsets inside the fourth quadrant.
    real_offsets = [0.0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2]
    real_offsets = [re for re in real_offsets if re <= search_radius]

    seeds: list[complex] = []
    per_line = max(2, max_seeds // max(1, len(real_offsets)))

    for re in real_offsets:
        line = re - 1j * y_values
        with np.errstate(all="ignore"):
            values = mie_denominator(line, order, x, family)
        seeds.extend(
            _line_minimum_seeds(
                mu_values=line,
                denominator_values=values,
                max_seeds=per_line,
            )
        )

    return _deduplicate(seeds, tolerance=1.0e-4)


def _build_mu_grid(search_radius: float, n_real: int, n_imag: int) -> np.ndarray:
    """Build a fourth-quadrant grid including both coordinate axes.

    The real coordinate is intentionally nonuniform and strongly clustered
    near Re(mu)=0, because surface-plasmon roots approach the negative
    imaginary axis as kr -> 0.
    """
    if n_real < 2 or n_imag < 2:
        raise ValueError("grid_real and grid_imag must both be at least 2")

    near_real_limit = min(search_radius, 2.0)
    near_count = max(2, int(np.ceil(0.65 * n_real)))

    # Cubic map: many more nodes near zero, but the axis itself is included.
    s = np.linspace(0.0, 1.0, near_count)
    real_near = near_real_limit * s**3

    if search_radius > near_real_limit and near_count < n_real:
        real_far = np.linspace(
            near_real_limit,
            search_radius,
            n_real - near_count + 1,
        )[1:]
        real_axis = np.concatenate([real_near, real_far])
    else:
        real_axis = real_near

    # Im(mu)=0 is also part of the closed fourth quadrant.  The origin is
    # explicitly suppressed later so it cannot become a spurious minimum.
    imag_axis = np.linspace(-search_radius, 0.0, n_imag)

    return real_axis[np.newaxis, :] + 1j * imag_axis[:, np.newaxis]


def _refine_root(
    function: Callable[[complex], complex],
    seed: complex,
    root_tolerance: float = 1.0e-11,
    max_function_evaluations: int = 80,
) -> tuple[bool, complex]:
    """Refine one complex root with a lightweight complex secant iteration."""
    z0 = complex(seed)
    z1 = z0 + (1.0e-3 + 1.0e-3j) * (1.0 + abs(z0))

    try:
        f0 = complex(function(z0))
        f1 = complex(function(z1))
    except (OverflowError, ZeroDivisionError, ValueError, FloatingPointError):
        return False, complex(np.nan, np.nan)

    for _ in range(max_function_evaluations):
        if not (
            np.isfinite(z1.real)
            and np.isfinite(z1.imag)
            and np.isfinite(f1.real)
            and np.isfinite(f1.imag)
        ):
            return False, complex(np.nan, np.nan)

        denominator = f1 - f0
        if abs(denominator) <= np.finfo(float).eps * max(1.0, abs(f0), abs(f1)):
            return False, complex(np.nan, np.nan)

        z2 = z1 - f1 * (z1 - z0) / denominator
        if abs(z2 - z1) <= root_tolerance * (1.0 + abs(z2)):
            try:
                f2 = complex(function(z2))
            except (OverflowError, ZeroDivisionError, ValueError, FloatingPointError):
                return False, complex(np.nan, np.nan)
            success = (
                np.isfinite(z2.real)
                and np.isfinite(z2.imag)
                and np.isfinite(f2.real)
                and np.isfinite(f2.imag)
            )
            return bool(success), z2

        z0, f0 = z1, f1
        z1 = z2
        try:
            f1 = complex(function(z1))
        except (OverflowError, ZeroDivisionError, ValueError, FloatingPointError):
            return False, complex(np.nan, np.nan)

    return False, complex(np.nan, np.nan)


def _deduplicate(values: Iterable[complex], tolerance: float = 2.0e-6) -> list[complex]:
    unique: list[complex] = []
    for value in sorted(values, key=lambda z: (abs(z), z.real, z.imag)):
        if all(abs(value - previous) > tolerance * (1.0 + abs(value)) for previous in unique):
            unique.append(value)
    return unique


def find_resonances_for_order(
    order: int,
    x: float,
    physical_m: complex,
    spectrum_tol: float,
    search_radius: float,
    n_real: int,
    n_imag: int,
    max_seeds: int,
    residual_tol: float,
    root_tol: float,
    root_maxiter: int,
    mu_grid: np.ndarray | None = None,
) -> list[Resonance]:
    """Find both c_l and d_l resonances for one order.

    The expensive spherical Bessel values on the scan grid are computed once
    and reused for both Mie families.
    """
    resonances: list[Resonance] = []

    # The canonical branch for outgoing resonant eigen-permittivities is the
    # CLOSED fourth quadrant.  Axes are included deliberately.  A tolerance is
    # needed because a converged root on an axis may acquire a tiny numerical
    # excursion outside the quadrant.
    boundary_tol = max(1.0e-9, 50.0 * root_tol)

    if mu_grid is None:
        mu_grid = _build_mu_grid(search_radius, n_real, n_imag)

    z_grid = mu_grid * x
    with np.errstate(all="ignore"):
        j_grid = spherical_jn(order, z_grid)
        j_prime_grid = spherical_jn(order, z_grid, derivative=True)
        psi_prime_grid = j_grid + z_grid * j_prime_grid

    h_x, xi_prime_x = _exterior_values(order, float(x))
    common_second = h_x * psi_prime_grid
    denominator_grids = {
        "c": j_grid * xi_prime_x - common_second,
        "d": mu_grid**2 * j_grid * xi_prime_x - common_second,
    }

    # mu=0 is a trivial/degenerate zero for some orders because spherical
    # Bessel factors vanish there.  It is not a Mie resonance and must not be
    # allowed to dominate the minimum detector now that both axes are included.
    zero_mask = np.abs(mu_grid) < 1.0e-10
    for family in denominator_grids:
        denominator_grids[family] = np.where(
            zero_mask,
            complex(np.inf, 0.0),
            denominator_grids[family],
        )

    generic_axis_seeds = _axis_seeds(search_radius, x)
    plasmon_seeds = _plasmon_quasistatic_seeds(order, search_radius)

    for family in ("c", "d"):
        seeds = _grid_minimum_seeds(
            mu_grid=mu_grid,
            denominator_values=denominator_grids[family],
            max_seeds=max_seeds,
        )

        # Dedicated plasmonic search.  This is intentionally independent of
        # the coarse global grid.
        seeds.extend(
            _near_imaginary_axis_seeds(
                order=order,
                x=x,
                family=family,
                search_radius=search_radius,
                n_imag=n_imag,
                max_seeds=max_seeds,
            )
        )
        seeds.extend(plasmon_seeds)
        seeds.extend(generic_axis_seeds)
        seeds = _deduplicate(seeds, tolerance=1.0e-4)

        denominator = lambda mu, fam=family: mie_denominator(mu, order, x, fam)
        roots: list[complex] = []
        for seed in seeds:
            success, mu = _refine_root(
                denominator,
                seed,
                root_tolerance=root_tol,
                max_function_evaluations=root_maxiter,
            )
            if not success:
                continue
            if not np.isfinite(mu.real) or not np.isfinite(mu.imag):
                continue

            # Correct plasmonic search-domain test:
            #
            #     Re(mu) >= 0,   Im(mu) <= 0,
            #
            # including Re(mu)=0.  Previously the grid itself excluded the
            # imaginary axis, which is precisely where the quasistatic surface
            # plasmon roots accumulate.
            if mu.real < -boundary_tol or mu.imag > boundary_tol:
                continue

            # Snap only tiny numerical boundary violations back to the axes.
            if -boundary_tol <= mu.real < 0.0:
                mu = complex(0.0, mu.imag)
            if 0.0 < mu.imag <= boundary_tol:
                mu = complex(mu.real, 0.0)

            if abs(mu) < 1.0e-5:
                continue
            if mu.real > search_radius * 1.02 or -mu.imag > search_radius * 1.02:
                continue

            residual = relative_denominator_residual(mu, order, x, family)
            if residual > residual_tol:
                continue
            roots.append(mu)

        for mu in _deduplicate(roots):
            spectral_point = complex(map_to_spectrum(mu, physical_m))
            if abs(spectral_point - 1.0) < spectrum_tol:
                continue
            resonances.append(
                Resonance(
                    order=order,
                    family=family,
                    mu=mu,
                    spectral_point=spectral_point,
                    relative_residual=relative_denominator_residual(
                        mu, order, x, family
                    ),
                )
            )

    resonances.sort(
        key=lambda item: (item.order, item.family, abs(item.spectral_point - 1.0))
    )
    return resonances


def compute_spectrum(
    x: float,
    physical_m: complex,
    spectrum_tol: float = 1.0e-3,
    l_max: int | None = None,
    search_radius: float | None = None,
    n_real: int = 100,
    n_imag: int = 70,
    max_seeds: int = 20,
    residual_tol: float = 1.0e-8,
    root_tol: float = 1.0e-11,
    root_maxiter: int = 80,
    empty_orders_to_stop: int = 4,
    workers: int = 1,
    verbose: bool = True,
) -> list[Resonance]:
    if x <= 0.0:
        raise ValueError("kr must be positive")
    if spectrum_tol <= 0.0:
        raise ValueError("spectrum_tol must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")

    contrast = abs(physical_m**2 - 1.0)
    # If |lambda-1| >= tol, then |mu^2-1| <= |m^2-1|/tol and therefore
    # |mu| <= sqrt(1 + |m^2-1|/tol).  This gives a finite search box.
    automatic_search_radius = float(np.sqrt(1.0 + contrast / spectrum_tol))
    if search_radius is None:
        search_radius = automatic_search_radius
    elif search_radius <= 0.0:
        raise ValueError("search_radius must be positive")

    if l_max is None:
        # Resonances of order l require roughly |mu*x| >= l.  Once l is well
        # beyond x*search_radius, retained spectral points cannot occur.
        l_max = max(8, int(np.ceil(x * search_radius + 8.0)))

    all_resonances: list[Resonance] = []
    empty_streak = 0

    if verbose:
        print(f"kr = {x:g}")
        print(f"physical m = {physical_m}")
        print(f"spectral tolerance = {spectrum_tol:g}")
        print(f"automatic mu-plane radius bound = {automatic_search_radius:.6g}")
        print(f"used mu-plane search radius = {search_radius:.6g}")
        print(f"maximum Mie order = {l_max}")
        print(f"parallel workers = {workers}")

    mu_grid = _build_mu_grid(search_radius, n_real, n_imag)

    def process_result(order: int, current: list[Resonance]) -> bool:
        nonlocal empty_streak
        all_resonances.extend(current)

        if verbose:
            minimum_distance = min(
                (abs(item.spectral_point - 1.0) for item in current),
                default=float("nan"),
            )
            print(
                f"l={order:3d}: retained {len(current):3d} roots"
                + (
                    f", min |lambda-1|={minimum_distance:.3e}"
                    if current
                    else ""
                )
            )

        if current:
            empty_streak = 0
        else:
            empty_streak += 1

        return (
            empty_orders_to_stop > 0
            and empty_streak >= empty_orders_to_stop
            and order > x * search_radius
        )

    def solve_order(order: int) -> list[Resonance]:
        return find_resonances_for_order(
            order=order,
            x=x,
            physical_m=physical_m,
            spectrum_tol=spectrum_tol,
            search_radius=search_radius,
            n_real=n_real,
            n_imag=n_imag,
            max_seeds=max_seeds,
            residual_tol=residual_tol,
            root_tol=root_tol,
            root_maxiter=root_maxiter,
            mu_grid=mu_grid,
        )

    if workers == 1:
        for order in range(1, l_max + 1):
            if process_result(order, solve_order(order)):
                break
    else:
        # Evaluate orders in small batches so early stopping remains meaningful.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for first_order in range(1, l_max + 1, workers):
                batch = list(
                    range(first_order, min(first_order + workers, l_max + 1))
                )
                batch_results = list(executor.map(solve_order, batch))
                should_stop = False
                for order, current in zip(batch, batch_results):
                    if process_result(order, current):
                        should_stop = True
                        break
                if should_stop:
                    break

    # Remove duplicate spectral points generated by repeated numerical roots.
    unique: list[Resonance] = []
    for item in sorted(
        all_resonances,
        key=lambda r: (r.order, r.family, r.spectral_point.real, r.spectral_point.imag),
    ):
        if all(
            abs(item.spectral_point - previous.spectral_point)
            > 2.0e-6 * (1.0 + abs(item.spectral_point))
            or item.family != previous.family
            or item.order != previous.order
            for previous in unique
        ):
            unique.append(item)
    return unique


def save_resonances_csv(resonances: list[Resonance], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "l",
                "family",
                "mu_real",
                "mu_imag",
                "lambda_real",
                "lambda_imag",
                "abs_lambda",
                "abs_lambda_minus_1",
                "relative_residual",
            ]
        )
        for item in resonances:
            writer.writerow(
                [
                    item.order,
                    item.family,
                    f"{item.mu.real:.16e}",
                    f"{item.mu.imag:.16e}",
                    f"{item.spectral_point.real:.16e}",
                    f"{item.spectral_point.imag:.16e}",
                    f"{abs(item.spectral_point):.16e}",
                    f"{abs(item.spectral_point - 1.0):.16e}",
                    f"{item.relative_residual:.16e}",
                ]
            )


def plot_spectrum(
    x: float,
    physical_m: complex,
    resonances: list[Resonance],
    output_path: Path,
    show: bool,
    settings: dict,
    condition_estimate: SpectralConditionEstimate | None = None,
    gmres_estimate: GmresEllipseEstimate | None = None,
) -> None:
    """Plot the predicted spectrum using options from the JSON configuration."""
    segment = branch_cut_segment(
        physical_m,
        count=settings["branch_point_count"],
    )

    previous_font_size = plt.rcParams.get("font.size", 10.0)
    plt.rcParams["font.size"] = settings["font_size"]

    try:
        fig, ax = plt.subplots(
            figsize=(
                settings["figure_width"],
                settings["figure_height"],
            )
        )

        branch_kwargs = {
            "linewidth": settings["branch_line_width"],
            "linestyle": settings["branch_line_style"],
            "label": settings["branch_label"],
        }
        if settings["branch_color"] is not None:
            branch_kwargs["color"] = settings["branch_color"]

        ax.plot(
            segment.real,
            segment.imag,
            **branch_kwargs,
        )

        family_settings = {
            "c": {
                "marker": settings["c_marker"],
                "size": settings["c_marker_size"],
                "label": settings["c_label"],
                "color": settings["c_color"],
            },
            "d": {
                "marker": settings["d_marker"],
                "size": settings["d_marker_size"],
                "label": settings["d_label"],
                "color": settings["d_color"],
            },
        }

        for family in ("c", "d"):
            points = np.array(
                [
                    item.spectral_point
                    for item in resonances
                    if item.family == family
                ],
                dtype=complex,
            )
            if not points.size:
                continue

            family_config = family_settings[family]
            scatter_kwargs = {
                "marker": family_config["marker"],
                "s": family_config["size"],
                "label": family_config["label"],
            }
            if family_config["color"] is not None:
                scatter_kwargs["color"] = family_config["color"]

            ax.scatter(
                points.real,
                points.imag,
                **scatter_kwargs,
            )

        one_kwargs = {
            "marker": settings["one_marker"],
            "s": settings["one_marker_size"],
            "label": settings["one_label"],
        }
        if settings["one_color"] is not None:
            one_kwargs["color"] = settings["one_color"]
        ax.scatter([1.0], [0.0], **one_kwargs)

        if settings["show_gmres_ellipse"] and gmres_estimate is not None:
            ellipse_values = enclosing_ellipse_curve(
                gmres_estimate.ellipse,
                point_count=settings["gmres_ellipse_point_count"],
            )
            ellipse_kwargs = {
                "linewidth": settings["gmres_ellipse_line_width"],
                "linestyle": settings["gmres_ellipse_line_style"],
                "label": settings["gmres_ellipse_label"],
            }
            if settings["gmres_ellipse_color"] is not None:
                ellipse_kwargs["color"] = settings["gmres_ellipse_color"]
            ax.plot(
                ellipse_values.real,
                ellipse_values.imag,
                **ellipse_kwargs,
            )

        endpoint = physical_m**2
        endpoint_kwargs = {
            "marker": settings["endpoint_marker"],
            "s": settings["endpoint_marker_size"],
            "label": settings["endpoint_label"],
        }
        if settings["endpoint_color"] is not None:
            endpoint_kwargs["color"] = settings["endpoint_color"]
        ax.scatter(
            [endpoint.real],
            [endpoint.imag],
            **endpoint_kwargs,
        )

        if settings["show_zero_axes"]:
            ax.axhline(
                0.0,
                linewidth=settings["zero_axes_line_width"],
                color=settings["zero_axes_color"],
            )
            ax.axvline(
                0.0,
                linewidth=settings["zero_axes_line_width"],
                color=settings["zero_axes_color"],
            )

        ax.set_xlabel(settings["x_label"])
        ax.set_ylabel(settings["y_label"])

        title = settings["title"]
        if title is None:
            title = f"Predicted spectrum of I-K: kr={x:g}, m={physical_m}"
        ax.set_title(title)

        if settings["show_grid"]:
            ax.grid(
                True,
                alpha=settings["grid_alpha"],
                linestyle=settings["grid_line_style"],
                linewidth=settings["grid_line_width"],
            )

        if settings["show_legend"]:
            ax.legend(
                loc=settings["legend_location"],
                fontsize=settings["legend_font_size"],
            )

        if (
            settings["show_condition_annotation"]
            and condition_estimate is not None
        ):
            ratio = condition_estimate.spectral_ratio
            ratio_text = (
                r"$\infty$"
                if not np.isfinite(ratio)
                else f"{ratio:.6g}"
            )
            annotation = (
                r"$\min |\lambda|="
                + f"{condition_estimate.minimum_modulus:.6g}$"
                + "\n"
                + r"$\max |\lambda|="
                + f"{condition_estimate.maximum_modulus:.6g}$"
                + "\n"
                + r"$\kappa_{\mathrm{spectral}}="
                + ratio_text
                + "$"
            )
            ax.text(
                settings["condition_annotation_x"],
                settings["condition_annotation_y"],
                annotation,
                transform=ax.transAxes,
                horizontalalignment="left",
                verticalalignment="top",
                fontsize=settings["condition_annotation_font_size"],
                bbox={
                    "boxstyle": "round",
                    "facecolor": "white",
                    "alpha": settings["condition_annotation_alpha"],
                },
            )

        if (
            settings["show_gmres_annotation"]
            and gmres_estimate is not None
        ):
            if gmres_estimate.valid:
                gmres_text = (
                    r"$k_{\mathrm{GMRES}}="
                    + str(gmres_estimate.estimated_iterations)
                    + "$"
                    + "\n"
                    + r"$\varepsilon="
                    + f"{gmres_estimate.relative_tolerance:.1e}$"
                    + "\n"
                    + r"$q_{\mathrm{ell}}="
                    + f"{gmres_estimate.asymptotic_reduction_factor:.6g}$"
                )
            else:
                gmres_text = "GMRES ellipse estimate unavailable"

            ax.text(
                settings["gmres_annotation_x"],
                settings["gmres_annotation_y"],
                gmres_text,
                transform=ax.transAxes,
                horizontalalignment="left",
                verticalalignment=settings["gmres_annotation_vertical_alignment"],
                fontsize=settings["gmres_annotation_font_size"],
                bbox={
                    "boxstyle": "round",
                    "facecolor": "white",
                    "alpha": settings["gmres_annotation_alpha"],
                },
            )

        if settings["equal_aspect"]:
            ax.set_aspect("equal", adjustable="datalim")

        x_min = settings["x_min"]
        x_max = settings["x_max"]
        y_min = settings["y_min"]
        y_max = settings["y_max"]

        if x_min is not None or x_max is not None:
            current_x_min, current_x_max = ax.get_xlim()
            ax.set_xlim(
                current_x_min if x_min is None else x_min,
                current_x_max if x_max is None else x_max,
            )
        if y_min is not None or y_max is not None:
            current_y_min, current_y_max = ax.get_ylim()
            ax.set_ylim(
                current_y_min if y_min is None else y_min,
                current_y_max if y_max is None else y_max,
            )

        if settings["tight_layout"]:
            fig.tight_layout()

        fig.savefig(
            output_path,
            dpi=settings["dpi"],
            transparent=settings["transparent_background"],
            bbox_inches="tight" if settings["tight_bounding_box"] else None,
        )

        if show:
            plt.show()
        plt.close(fig)
    finally:
        plt.rcParams["font.size"] = previous_font_size



def _require_mapping(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"Configuration field '{name}' must be a JSON object")
    return value


def _require_number(mapping: dict, key: str, section: str) -> float:
    if key not in mapping:
        raise ValueError(f"Missing required configuration field '{section}.{key}'")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Configuration field '{section}.{key}' must be numeric")
    return float(value)


def _optional_number(
    mapping: dict,
    key: str,
    section: str,
    default: float | None,
) -> float | None:
    value = mapping.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Configuration field '{section}.{key}' must be numeric or null"
        )
    return float(value)


def _optional_integer(
    mapping: dict,
    key: str,
    section: str,
    default: int | None,
) -> int | None:
    value = mapping.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Configuration field '{section}.{key}' must be an integer or null"
        )
    return value


def _optional_boolean(
    mapping: dict,
    key: str,
    section: str,
    default: bool,
) -> bool:
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Configuration field '{section}.{key}' must be boolean")
    return value


def _optional_nullable_string(
    mapping: dict,
    key: str,
    section: str,
    default: str | None,
) -> str | None:
    value = mapping.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"Configuration field '{section}.{key}' must be a string or null"
        )
    return value


def _optional_string(
    mapping: dict,
    key: str,
    section: str,
    default: str,
) -> str:
    value = mapping.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Configuration field '{section}.{key}' must be a non-empty string"
        )
    return value


def load_config(config_path: Path) -> dict:
    """Read and validate mie_spectrum_config.json."""
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Place '{CONFIG_FILENAME}' in the same directory as the script."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {config_path}: line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    root = _require_mapping(raw, "root")
    problem = _require_mapping(root.get("problem"), "problem")
    search = _require_mapping(root.get("search", {}), "search")
    output = _require_mapping(root.get("output", {}), "output")
    analysis = _require_mapping(root.get("analysis", {}), "analysis")
    plot = _require_mapping(root.get("plot", {}), "plot")

    config = {
        "kr": _require_number(problem, "kr", "problem"),
        # Preferred interface: specify relative permittivity directly.
        "eps_real": _optional_number(problem, "eps_real", "problem", None),
        "eps_imag": _optional_number(problem, "eps_imag", "problem", 0.0),
        # Backward-compatible interface: specify refractive index m directly.
        "m_real": _optional_number(problem, "m_real", "problem", None),
        "m_imag": _optional_number(problem, "m_imag", "problem", 0.0),
        "spectrum_tol": _optional_number(
            search, "spectrum_tol", "search", 1.0e-3
        ),
        "l_max": _optional_integer(search, "l_max", "search", None),
        "search_radius": _optional_number(
            search, "search_radius", "search", None
        ),
        "grid_real": _optional_integer(search, "grid_real", "search", 100),
        "grid_imag": _optional_integer(search, "grid_imag", "search", 70),
        "max_seeds": _optional_integer(search, "max_seeds", "search", 20),
        "residual_tol": _optional_number(
            search, "residual_tol", "search", 1.0e-8
        ),
        "root_tol": _optional_number(search, "root_tol", "search", 1.0e-11),
        "root_maxiter": _optional_integer(
            search, "root_maxiter", "search", 80
        ),
        "empty_orders_to_stop": _optional_integer(
            search, "empty_orders_to_stop", "search", 4
        ),
        "parallel_workers": _optional_integer(
            search, "parallel_workers", "search", 1
        ),
        "plot_filename": _optional_string(
            output, "plot_filename", "output", "mie_spectrum.png"
        ),
        "csv_filename": _optional_string(
            output, "csv_filename", "output", "mie_resonances.csv"
        ),
        "show_plot": _optional_boolean(output, "show_plot", "output", False),
        "verbose": _optional_boolean(output, "verbose", "output", True),
        "condition_estimate_enabled": _optional_boolean(
            analysis, "condition_estimate_enabled", "analysis", True
        ),
        "condition_summary_filename": _optional_string(
            analysis,
            "condition_summary_filename",
            "analysis",
            "results/condition_estimate.json",
        ),
        "print_condition_estimate": _optional_boolean(
            analysis, "print_condition_estimate", "analysis", True
        ),
        "gmres_estimate_enabled": _optional_boolean(
            analysis, "gmres_estimate_enabled", "analysis", True
        ),
        "gmres_relative_tolerance": _optional_number(
            analysis, "gmres_relative_tolerance", "analysis", 1.0e-6
        ),
        "eigenvector_condition_number": _optional_number(
            analysis, "eigenvector_condition_number", "analysis", 1.0
        ),
        "gmres_maximum_iterations": _optional_integer(
            analysis, "gmres_maximum_iterations", "analysis", 100000
        ),
        "gmres_summary_filename": _optional_string(
            analysis,
            "gmres_summary_filename",
            "analysis",
            "results/gmres_estimate.json",
        ),
        "print_gmres_estimate": _optional_boolean(
            analysis, "print_gmres_estimate", "analysis", True
        ),
        "ellipse_algorithm_tolerance": _optional_number(
            analysis, "ellipse_algorithm_tolerance", "analysis", 1.0e-7
        ),
        "ellipse_maximum_iterations": _optional_integer(
            analysis, "ellipse_maximum_iterations", "analysis", 20000
        ),
        "ellipse_safety_factor": _optional_number(
            analysis, "ellipse_safety_factor", "analysis", 1.005
        ),
        "ellipse_minimum_axis_ratio": _optional_number(
            analysis, "ellipse_minimum_axis_ratio", "analysis", 1.0e-3
        ),
        "ellipse_rank_tolerance": _optional_number(
            analysis, "ellipse_rank_tolerance", "analysis", 1.0e-10
        ),
        "ellipse_exclude_origin": _optional_boolean(
            analysis, "ellipse_exclude_origin", "analysis", True
        ),
        "ellipse_origin_margin": _optional_number(
            analysis, "ellipse_origin_margin", "analysis", 1.0e-4
        ),
        "ellipse_point_margin": _optional_number(
            analysis, "ellipse_point_margin", "analysis", 1.0e-9
        ),
        "ellipse_optimizer_tolerance": _optional_number(
            analysis, "ellipse_optimizer_tolerance", "analysis", 1.0e-11
        ),
        "ellipse_optimizer_maximum_iterations": _optional_integer(
            analysis,
            "ellipse_optimizer_maximum_iterations",
            "analysis",
            4000,
        ),
        "ellipse_optimizer_restarts": _optional_integer(
            analysis, "ellipse_optimizer_restarts", "analysis", 16
        ),
        "ellipse_random_seed": _optional_integer(
            analysis, "ellipse_random_seed", "analysis", 12345
        ),
        "ellipse_convex_hull_tolerance": _optional_number(
            analysis, "ellipse_convex_hull_tolerance", "analysis", 1.0e-12
        ),
        "ellipse_convex_hull_maximum_iterations": _optional_integer(
            analysis,
            "ellipse_convex_hull_maximum_iterations",
            "analysis",
            5000,
        ),
        "ellipse_feasibility_tolerance": _optional_number(
            analysis, "ellipse_feasibility_tolerance", "analysis", 1.0e-8
        ),
        "plot_enabled": _optional_boolean(plot, "enabled", "plot", True),
        "figure_width": _optional_number(
            plot, "figure_width", "plot", 8.0
        ),
        "figure_height": _optional_number(
            plot, "figure_height", "plot", 6.5
        ),
        "dpi": _optional_integer(plot, "dpi", "plot", 200),
        "font_size": _optional_number(plot, "font_size", "plot", 11.0),
        "branch_point_count": _optional_integer(
            plot, "branch_point_count", "plot", 300
        ),
        "branch_line_width": _optional_number(
            plot, "branch_line_width", "plot", 2.0
        ),
        "branch_line_style": _optional_string(
            plot, "branch_line_style", "plot", "-"
        ),
        "branch_color": _optional_nullable_string(
            plot, "branch_color", "plot", None
        ),
        "branch_label": _optional_string(
            plot, "branch_label", "plot", r"branch cut: $[1,m^2]$"
        ),
        "c_marker": _optional_string(plot, "c_marker", "plot", "o"),
        "c_marker_size": _optional_number(
            plot, "c_marker_size", "plot", 42.0
        ),
        "c_color": _optional_nullable_string(
            plot, "c_color", "plot", None
        ),
        "c_label": _optional_string(
            plot, "c_label", "plot", r"zeros of $D_l^{(c)}$"
        ),
        "d_marker": _optional_string(plot, "d_marker", "plot", "s"),
        "d_marker_size": _optional_number(
            plot, "d_marker_size", "plot", 42.0
        ),
        "d_color": _optional_nullable_string(
            plot, "d_color", "plot", None
        ),
        "d_label": _optional_string(
            plot, "d_label", "plot", r"zeros of $D_l^{(d)}$"
        ),
        "one_marker": _optional_string(plot, "one_marker", "plot", "x"),
        "one_marker_size": _optional_number(
            plot, "one_marker_size", "plot", 70.0
        ),
        "one_color": _optional_nullable_string(
            plot, "one_color", "plot", None
        ),
        "one_label": _optional_string(
            plot, "one_label", "plot", r"$1$"
        ),
        "endpoint_marker": _optional_string(
            plot, "endpoint_marker", "plot", "o"
        ),
        "endpoint_marker_size": _optional_number(
            plot, "endpoint_marker_size", "plot", 80.0
        ),
        "endpoint_color": _optional_nullable_string(
            plot, "endpoint_color", "plot", None
        ),
        "endpoint_label": _optional_string(
            plot, "endpoint_label", "plot", r"$m^2$"
        ),
        "show_zero_axes": _optional_boolean(
            plot, "show_zero_axes", "plot", True
        ),
        "zero_axes_line_width": _optional_number(
            plot, "zero_axes_line_width", "plot", 0.6
        ),
        "zero_axes_color": _optional_string(
            plot, "zero_axes_color", "plot", "black"
        ),
        "show_grid": _optional_boolean(
            plot, "show_grid", "plot", True
        ),
        "grid_alpha": _optional_number(
            plot, "grid_alpha", "plot", 0.3
        ),
        "grid_line_style": _optional_string(
            plot, "grid_line_style", "plot", "-"
        ),
        "grid_line_width": _optional_number(
            plot, "grid_line_width", "plot", 0.8
        ),
        "show_legend": _optional_boolean(
            plot, "show_legend", "plot", True
        ),
        "show_condition_annotation": _optional_boolean(
            plot, "show_condition_annotation", "plot", True
        ),
        "show_gmres_ellipse": _optional_boolean(
            plot, "show_gmres_ellipse", "plot", True
        ),
        "gmres_ellipse_point_count": _optional_integer(
            plot, "gmres_ellipse_point_count", "plot", 500
        ),
        "gmres_ellipse_line_width": _optional_number(
            plot, "gmres_ellipse_line_width", "plot", 1.5
        ),
        "gmres_ellipse_line_style": _optional_string(
            plot, "gmres_ellipse_line_style", "plot", "--"
        ),
        "gmres_ellipse_color": _optional_nullable_string(
            plot, "gmres_ellipse_color", "plot", None
        ),
        "gmres_ellipse_label": _optional_string(
            plot, "gmres_ellipse_label", "plot", "enclosing GMRES ellipse"
        ),
        "show_gmres_annotation": _optional_boolean(
            plot, "show_gmres_annotation", "plot", True
        ),
        "gmres_annotation_x": _optional_number(
            plot, "gmres_annotation_x", "plot", 0.02
        ),
        "gmres_annotation_y": _optional_number(
            plot, "gmres_annotation_y", "plot", 0.02
        ),
        "gmres_annotation_vertical_alignment": _optional_string(
            plot,
            "gmres_annotation_vertical_alignment",
            "plot",
            "bottom",
        ),
        "gmres_annotation_font_size": _optional_number(
            plot, "gmres_annotation_font_size", "plot", 9.0
        ),
        "gmres_annotation_alpha": _optional_number(
            plot, "gmres_annotation_alpha", "plot", 0.8
        ),
        "condition_annotation_x": _optional_number(
            plot, "condition_annotation_x", "plot", 0.02
        ),
        "condition_annotation_y": _optional_number(
            plot, "condition_annotation_y", "plot", 0.98
        ),
        "condition_annotation_font_size": _optional_number(
            plot, "condition_annotation_font_size", "plot", 9.0
        ),
        "condition_annotation_alpha": _optional_number(
            plot, "condition_annotation_alpha", "plot", 0.8
        ),
        "legend_location": _optional_string(
            plot, "legend_location", "plot", "best"
        ),
        "legend_font_size": _optional_number(
            plot, "legend_font_size", "plot", 10.0
        ),
        "equal_aspect": _optional_boolean(
            plot, "equal_aspect", "plot", True
        ),
        "x_min": _optional_number(plot, "x_min", "plot", None),
        "x_max": _optional_number(plot, "x_max", "plot", None),
        "y_min": _optional_number(plot, "y_min", "plot", None),
        "y_max": _optional_number(plot, "y_max", "plot", None),
        "x_label": _optional_string(
            plot, "x_label", "plot", r"$\operatorname{Re}\lambda$"
        ),
        "y_label": _optional_string(
            plot, "y_label", "plot", r"$\operatorname{Im}\lambda$"
        ),
        "title": _optional_nullable_string(
            plot, "title", "plot", None
        ),
        "tight_layout": _optional_boolean(
            plot, "tight_layout", "plot", True
        ),
        "tight_bounding_box": _optional_boolean(
            plot, "tight_bounding_box", "plot", True
        ),
        "transparent_background": _optional_boolean(
            plot, "transparent_background", "plot", False
        ),
    }

    if config["kr"] <= 0.0:
        raise ValueError("Configuration field 'problem.kr' must be positive")

    has_eps = config["eps_real"] is not None
    has_m = config["m_real"] is not None
    if has_eps == has_m:
        raise ValueError(
            "Specify exactly one material representation: either "
            "problem.eps_real[/eps_imag] or problem.m_real[/m_imag]."
        )
    if config["spectrum_tol"] is None or config["spectrum_tol"] <= 0.0:
        raise ValueError(
            "Configuration field 'search.spectrum_tol' must be positive"
        )
    if (
        config["search_radius"] is not None
        and config["search_radius"] <= 0.0
    ):
        raise ValueError(
            "Configuration field 'search.search_radius' must be positive or null"
        )
    if config["l_max"] is not None and config["l_max"] <= 0:
        raise ValueError(
            "Configuration field 'search.l_max' must be positive or null"
        )

    for field in (
        "grid_real",
        "grid_imag",
        "max_seeds",
        "root_maxiter",
        "parallel_workers",
    ):
        if config[field] is None or config[field] <= 0:
            raise ValueError(
                f"Configuration field 'search.{field}' must be positive"
            )

    if config["grid_real"] < 2 or config["grid_imag"] < 2:
        raise ValueError(
            "Configuration fields 'search.grid_real' and 'search.grid_imag' "
            "must both be at least 2"
        )

    if (
        config["empty_orders_to_stop"] is None
        or config["empty_orders_to_stop"] < 0
    ):
        raise ValueError(
            "Configuration field 'search.empty_orders_to_stop' "
            "must be a nonnegative integer"
        )

    positive_analysis_fields = (
        "gmres_relative_tolerance",
        "eigenvector_condition_number",
        "gmres_maximum_iterations",
        "ellipse_algorithm_tolerance",
        "ellipse_maximum_iterations",
        "ellipse_safety_factor",
        "ellipse_minimum_axis_ratio",
        "ellipse_rank_tolerance",
        "ellipse_origin_margin",
        "ellipse_optimizer_tolerance",
        "ellipse_optimizer_maximum_iterations",
        "ellipse_optimizer_restarts",
        "ellipse_convex_hull_tolerance",
        "ellipse_convex_hull_maximum_iterations",
        "ellipse_feasibility_tolerance",
    )
    for field in positive_analysis_fields:
        if config[field] is None or config[field] <= 0:
            raise ValueError(
                f"Configuration field 'analysis.{field}' must be positive"
            )

    if config["gmres_relative_tolerance"] >= 1.0:
        raise ValueError(
            "Configuration field 'analysis.gmres_relative_tolerance' "
            "must be less than 1"
        )

    if config["ellipse_safety_factor"] < 1.0:
        raise ValueError(
            "Configuration field 'analysis.ellipse_safety_factor' "
            "must be at least 1"
        )

    if config["ellipse_minimum_axis_ratio"] > 1.0:
        raise ValueError(
            "Configuration field 'analysis.ellipse_minimum_axis_ratio' "
            "must not exceed 1"
        )

    if config["ellipse_point_margin"] is None:
        raise ValueError(
            "Configuration field 'analysis.ellipse_point_margin' is required"
        )
    if not 0.0 <= config["ellipse_point_margin"] < 1.0:
        raise ValueError(
            "Configuration field 'analysis.ellipse_point_margin' "
            "must lie in [0, 1)"
        )

    if config["ellipse_random_seed"] is None:
        raise ValueError(
            "Configuration field 'analysis.ellipse_random_seed' "
            "must be an integer"
        )

    positive_plot_fields = (
        "figure_width",
        "figure_height",
        "dpi",
        "font_size",
        "branch_point_count",
        "branch_line_width",
        "c_marker_size",
        "d_marker_size",
        "one_marker_size",
        "endpoint_marker_size",
        "zero_axes_line_width",
        "grid_line_width",
        "legend_font_size",
        "condition_annotation_font_size",
        "gmres_ellipse_point_count",
        "gmres_ellipse_line_width",
        "gmres_annotation_font_size",
    )
    for field in positive_plot_fields:
        if config[field] is None or config[field] <= 0:
            raise ValueError(
                f"Configuration field 'plot.{field}' must be positive"
            )

    if not 0.0 <= config["grid_alpha"] <= 1.0:
        raise ValueError(
            "Configuration field 'plot.grid_alpha' must lie in [0, 1]"
        )

    if not 0.0 <= config["condition_annotation_alpha"] <= 1.0:
        raise ValueError(
            "Configuration field 'plot.condition_annotation_alpha' "
            "must lie in [0, 1]"
        )

    if not 0.0 <= config["gmres_annotation_alpha"] <= 1.0:
        raise ValueError(
            "Configuration field 'plot.gmres_annotation_alpha' "
            "must lie in [0, 1]"
        )

    for field in (
        "condition_annotation_x",
        "condition_annotation_y",
        "gmres_annotation_x",
        "gmres_annotation_y",
    ):
        if not 0.0 <= config[field] <= 1.0:
            raise ValueError(
                f"Configuration field 'plot.{field}' must lie in [0, 1]"
            )

    if (
        config["x_min"] is not None
        and config["x_max"] is not None
        and config["x_min"] >= config["x_max"]
    ):
        raise ValueError("Configuration requires plot.x_min < plot.x_max")

    if (
        config["y_min"] is not None
        and config["y_max"] is not None
        and config["y_min"] >= config["y_max"]
    ):
        raise ValueError("Configuration requires plot.y_min < plot.y_max")

    return config


def main() -> None:
    script_directory = Path(__file__).resolve().parent
    config_path = script_directory / CONFIG_FILENAME
    config = load_config(config_path)

    if config["eps_real"] is not None:
        physical_eps = complex(config["eps_real"], config["eps_imag"])
        # Principal square root: Re(m) >= 0.  For passive eps with Im(eps)>0
        # this also gives Im(m)>0.  The resonance-root search in fictitious mu
        # is independent of this physical square-root choice.
        physical_m = complex(np.sqrt(physical_eps + 0j))
    else:
        physical_m = complex(config["m_real"], config["m_imag"])
        physical_eps = physical_m**2

    if config["verbose"]:
        print(f"physical eps = {physical_eps}")
        print(f"physical m   = {physical_m}")

    resonances = compute_spectrum(
        x=config["kr"],
        physical_m=physical_m,
        spectrum_tol=config["spectrum_tol"],
        l_max=config["l_max"],
        search_radius=config["search_radius"],
        n_real=config["grid_real"],
        n_imag=config["grid_imag"],
        max_seeds=config["max_seeds"],
        residual_tol=config["residual_tol"],
        root_tol=config["root_tol"],
        root_maxiter=config["root_maxiter"],
        empty_orders_to_stop=config["empty_orders_to_stop"],
        workers=config["parallel_workers"],
        verbose=config["verbose"],
    )

    plot_path = script_directory / config["plot_filename"]
    csv_path = script_directory / config["csv_filename"]
    condition_path = (
        script_directory / config["condition_summary_filename"]
    )
    gmres_path = script_directory / config["gmres_summary_filename"]
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    condition_path.parent.mkdir(parents=True, exist_ok=True)
    gmres_path.parent.mkdir(parents=True, exist_ok=True)

    save_resonances_csv(resonances, csv_path)

    condition_estimate = None
    if config["condition_estimate_enabled"]:
        condition_estimate = estimate_spectral_condition_number(
            physical_m=physical_m,
            resonances=resonances,
        )
        save_condition_estimate_json(condition_estimate, condition_path)

        if config["print_condition_estimate"]:
            print("\nPredicted spectral condition-number estimate:")
            print(
                "  min |lambda| = "
                f"{condition_estimate.minimum_modulus:.12g} "
                f"({condition_estimate.minimum_source})"
            )
            print(
                "  max |lambda| = "
                f"{condition_estimate.maximum_modulus:.12g} "
                f"({condition_estimate.maximum_source})"
            )
            print(
                "  max|lambda|/min|lambda| = "
                f"{condition_estimate.spectral_ratio:.12g}"
            )
            print(
                "  Interpretation: lower estimate for kappa_2(A); "
                "equality holds for a normal matrix."
            )

    gmres_estimate = None
    if config["gmres_estimate_enabled"]:
        gmres_estimate = estimate_gmres_iterations_from_ellipse(
            physical_m=physical_m,
            resonances=resonances,
            relative_tolerance=config["gmres_relative_tolerance"],
            eigenvector_condition_number=(
                config["eigenvector_condition_number"]
            ),
            maximum_gmres_iterations=config["gmres_maximum_iterations"],
            ellipse_algorithm_tolerance=(
                config["ellipse_algorithm_tolerance"]
            ),
            ellipse_maximum_iterations=(
                config["ellipse_maximum_iterations"]
            ),
            ellipse_safety_factor=config["ellipse_safety_factor"],
            ellipse_minimum_axis_ratio=(
                config["ellipse_minimum_axis_ratio"]
            ),
            ellipse_rank_tolerance=config["ellipse_rank_tolerance"],
            ellipse_exclude_origin=config["ellipse_exclude_origin"],
            ellipse_origin_margin=config["ellipse_origin_margin"],
            ellipse_point_margin=config["ellipse_point_margin"],
            ellipse_optimizer_tolerance=(
                config["ellipse_optimizer_tolerance"]
            ),
            ellipse_optimizer_maximum_iterations=(
                config["ellipse_optimizer_maximum_iterations"]
            ),
            ellipse_optimizer_restarts=(
                config["ellipse_optimizer_restarts"]
            ),
            ellipse_random_seed=config["ellipse_random_seed"],
            ellipse_convex_hull_tolerance=(
                config["ellipse_convex_hull_tolerance"]
            ),
            ellipse_convex_hull_maximum_iterations=(
                config["ellipse_convex_hull_maximum_iterations"]
            ),
            ellipse_feasibility_tolerance=(
                config["ellipse_feasibility_tolerance"]
            ),
        )
        save_gmres_estimate_json(gmres_estimate, gmres_path)

        if config["print_gmres_estimate"]:
            print("\nGMRES estimate from an enclosing spectral ellipse:")
            print(
                "  ellipse center = "
                f"{gmres_estimate.ellipse.center.real:.12g}"
                f"{gmres_estimate.ellipse.center.imag:+.12g}i"
            )
            print(
                "  semiaxes = "
                f"({gmres_estimate.ellipse.major_semiaxis:.12g}, "
                f"{gmres_estimate.ellipse.minor_semiaxis:.12g})"
            )
            print(
                "  origin ellipse value = "
                f"{gmres_estimate.origin_ellipse_value:.12g}"
            )
            if gmres_estimate.valid:
                print(
                    "  asymptotic reduction factor = "
                    f"{gmres_estimate.asymptotic_reduction_factor:.12g}"
                )
                print(
                    "  estimated unrestarted GMRES iterations for "
                    f"{gmres_estimate.relative_tolerance:.1e} = "
                    f"{gmres_estimate.estimated_iterations}"
                )
                print(
                    "  residual bound at this iteration = "
                    f"{gmres_estimate.residual_bound_at_estimate:.12g}"
                )
            else:
                print(f"  estimate unavailable: {gmres_estimate.message}")

    plot_settings = {
        key: config[key]
        for key in (
            "figure_width",
            "figure_height",
            "dpi",
            "font_size",
            "branch_point_count",
            "branch_line_width",
            "branch_line_style",
            "branch_color",
            "branch_label",
            "c_marker",
            "c_marker_size",
            "c_color",
            "c_label",
            "d_marker",
            "d_marker_size",
            "d_color",
            "d_label",
            "one_marker",
            "one_marker_size",
            "one_color",
            "one_label",
            "endpoint_marker",
            "endpoint_marker_size",
            "endpoint_color",
            "endpoint_label",
            "show_zero_axes",
            "zero_axes_line_width",
            "zero_axes_color",
            "show_grid",
            "grid_alpha",
            "grid_line_style",
            "grid_line_width",
            "show_legend",
            "show_condition_annotation",
            "show_gmres_ellipse",
            "gmres_ellipse_point_count",
            "gmres_ellipse_line_width",
            "gmres_ellipse_line_style",
            "gmres_ellipse_color",
            "gmres_ellipse_label",
            "show_gmres_annotation",
            "gmres_annotation_x",
            "gmres_annotation_y",
            "gmres_annotation_vertical_alignment",
            "gmres_annotation_font_size",
            "gmres_annotation_alpha",
            "condition_annotation_x",
            "condition_annotation_y",
            "condition_annotation_font_size",
            "condition_annotation_alpha",
            "legend_location",
            "legend_font_size",
            "equal_aspect",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "x_label",
            "y_label",
            "title",
            "tight_layout",
            "tight_bounding_box",
            "transparent_background",
        )
    }

    if config["plot_enabled"]:
        plot_spectrum(
            x=config["kr"],
            physical_m=physical_m,
            resonances=resonances,
            output_path=plot_path,
            show=config["show_plot"],
            settings=plot_settings,
            condition_estimate=condition_estimate,
            gmres_estimate=gmres_estimate,
        )

    print(f"\nConfiguration: {config_path}")
    print(f"Saved {len(resonances)} resonance points")
    if config["plot_enabled"]:
        print(f"Plot: {plot_path}")
    else:
        print("Plot: disabled in configuration")
    print(f"Table: {csv_path}")
    if config["condition_estimate_enabled"]:
        print(f"Condition estimate: {condition_path}")
    if config["gmres_estimate_enabled"]:
        print(f"GMRES estimate: {gmres_path}")


if __name__ == "__main__":
    main()
