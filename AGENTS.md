# ED-researh

## Purpose

This repository contains research calculations, reproducible numerical
experiments, prototypes, convergence studies, benchmarks, and examples built on
top of the EMW core library.

Treat this repository as a research workspace rather than the canonical home of
reusable EMW algorithms.

## Response language

Respond to the user primarily in Russian.

Mixed Russian-English technical prose is allowed and preferred when translation
could alter or obscure technical meaning. Do not translate mathematical
notation, standard CEM terminology, C++ identifiers, library/API names, or source
notation merely for linguistic consistency.

## Shared Codex skills

Shared electromagnetic skills are expected under:

    .agents/skills/

Use the relevant domain skills for SIE, VIE, VSIE, operator discretization, and
uniform-grid FFT-VIE work.

## Relationship to EMW

Prefer using existing public EMW APIs instead of duplicating core numerical
algorithms in this repository.

Before writing a new implementation:

1. inspect the current EMW API used by nearby experiments;
2. check whether the required functionality already exists in the core library;
3. write a local prototype only when the task is genuinely experimental;
4. if a prototype becomes generally reusable, recommend moving/refactoring the
   final implementation into `Electromagnetic-Waves-Scattering`.

Do not copy large pieces of core implementation into research code just to make
an experiment self-contained.

## Electromagnetic convention

Use the same canonical convention as EMW:

    exp(-i * omega * t)

with outgoing Green function

    G_k(x,y) = exp(+i * k * |x-y|) / (4 * pi * |x-y|).

When reproducing a paper or external formula that uses another convention,
record the source convention and translate it before comparing with EMW results.

## Research workflow

Every numerical experiment should make its assumptions and parameters explicit.
Record, when relevant:

- geometry and dimensions;
- material parameters;
- frequency, wavelength, and wave number;
- incident-field polarization and direction;
- mesh resolution;
- unknown and formulation (`E-VIE`, `J-VIE`, SIE, VSIE, etc.);
- basis/testing scheme;
- quadrature and singularity-treatment parameters;
- solver, restart, tolerance, and stopping criterion;
- preconditioner;
- number of threads/processes for performance experiments;
- random seed for randomized algorithms;
- output filenames and plotted quantities.

Do not silently alter experimental parameters to obtain a cleaner plot or a
more favorable benchmark.

## Reproducibility

Prefer experiment programs that can be rerun without manual editing of hidden
state.

When practical:

- keep key parameters near the top of the program or in a small explicit config;
- emit the effective parameters to stdout or metadata files;
- use deterministic seeds for randomized comparisons;
- save raw numerical data separately from plot rendering;
- keep plotting/analysis scripts reproducible.

If a result depends on a local data file, document its expected location and
format.

## Numerical comparisons

When comparing methods, ensure that they solve the same mathematical problem and
use compatible normalizations.

For accuracy studies, distinguish errors caused by:

1. geometric discretization;
2. basis/test discretization;
3. quadrature;
4. iterative-solver tolerance;
5. fast/approximate matrix representation;
6. post-processing.

Do not compare runtime numbers obtained at materially different accuracy without
stating that difference.

## Benchmarks

For timing experiments, report enough information to reproduce the result:

- matrix/grid/problem size;
- rank parameters if low-rank methods are used;
- FFT dimensions/padding if relevant;
- number of RHS vectors;
- thread/process count;
- solver tolerance;
- warmup/repetition policy;
- whether setup/precomputation is included in timing.

Keep algorithm setup time separate from repeated matvec/solve time when that
distinction matters.

## Validation hierarchy

If user explicitly define how to verify results of calculation, the use described path,
otherwise prefer, in order of strength for the task:

- analytical reference solution;
- independently implemented numerical reference;
- highly refined direct calculation;
- cross-check against a second formulation;
- internal consistency checks.

For scattering by spheres, use Mie theory when applicable (there is a submodule scattnlay, 
that implement various solutions of Mie-like problems). For structured VIE
implementations, compare dense, explicit Toeplitz, and FFT paths on small grids
before scaling up.

## Research conclusions

Keep measured evidence separate from general claims.

A result observed for one geometry, frequency range, mesh, or hardware platform
is evidence for that configuration, not a universal theorem.

When presenting conclusions, state the tested regime and relevant limitations.

## Promotion to core library

A research implementation is a candidate for EMW core when it has:

- a stable mathematical formulation;
- clear API boundaries;
- regression tests;
- documented normalization/sign conventions;
- numerical validation;
- evidence that the functionality is reusable beyond one experiment.

Until then, keep experimental code local to this repository. 

## Git handling

By the way, anywhen you **should not** commit something implicitly,
always ask permission from user for every modification of both local 
and remote repository.
