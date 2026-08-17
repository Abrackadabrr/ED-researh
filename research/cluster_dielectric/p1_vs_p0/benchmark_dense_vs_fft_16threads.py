import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fft
from threadpoolctl import threadpool_limits

N_THREADS = 16
N_TET = 1624
N_DENSE = 3 * N_TET
CELLS_PER_LAMBDA_M = 10
EPS_VALUES = np.arange(1.0, 100.0, 1.0)

RNG_SEED = 12345
DENSE_REPEATS = 20
FFT_REPEATS = 60
FFT_WARMUPS = 4

rng = np.random.default_rng(RNG_SEED)

A = (
    rng.standard_normal((N_DENSE, N_DENSE))
    + 1j*rng.standard_normal((N_DENSE, N_DENSE))
).astype(np.complex128)
x = (
    rng.standard_normal(N_DENSE)
    + 1j*rng.standard_normal(N_DENSE)
).astype(np.complex128)

with threadpool_limits(limits=N_THREADS):
    for _ in range(4):
        _ = A @ x
    dense_times = []
    for _ in range(DENSE_REPEATS):
        t0 = time.perf_counter()
        _ = A @ x
        dense_times.append(time.perf_counter() - t0)

dense_ms = 1e3*np.median(dense_times)

def make_fft_problem(n_phys):
    n_pad = fft.next_fast_len(2*n_phys - 1)
    shape = (n_pad, n_pad, n_pad)

    u = np.zeros((3, *shape), dtype=np.complex128)
    u[:, :n_phys, :n_phys, :n_phys] = (
        rng.standard_normal((3, n_phys, n_phys, n_phys))
        + 1j*rng.standard_normal((3, n_phys, n_phys, n_phys))
    )

    Gxx = (rng.standard_normal(shape)+1j*rng.standard_normal(shape)).astype(np.complex128)
    Gyy = (rng.standard_normal(shape)+1j*rng.standard_normal(shape)).astype(np.complex128)
    Gzz = (rng.standard_normal(shape)+1j*rng.standard_normal(shape)).astype(np.complex128)
    Gxy = (rng.standard_normal(shape)+1j*rng.standard_normal(shape)).astype(np.complex128)
    Gxz = (rng.standard_normal(shape)+1j*rng.standard_normal(shape)).astype(np.complex128)
    Gyz = (rng.standard_normal(shape)+1j*rng.standard_normal(shape)).astype(np.complex128)

    G = (Gxx, Gyy, Gzz, Gxy, Gxz, Gyz)
    vhat = np.empty((3, *shape), dtype=np.complex128)
    scratch = np.empty(shape, dtype=np.complex128)
    return n_pad, u, G, vhat, scratch

def fft_matvec(u, G, vhat, scratch):
    Gxx, Gyy, Gzz, Gxy, Gxz, Gyz = G

    uhat = fft.fftn(u, axes=(1,2,3), workers=N_THREADS)
    ux, uy, uz = uhat[0], uhat[1], uhat[2]

    np.multiply(Gxx, ux, out=vhat[0])
    np.multiply(Gxy, uy, out=scratch)
    np.add(vhat[0], scratch, out=vhat[0])
    np.multiply(Gxz, uz, out=scratch)
    np.add(vhat[0], scratch, out=vhat[0])

    np.multiply(Gxy, ux, out=vhat[1])
    np.multiply(Gyy, uy, out=scratch)
    np.add(vhat[1], scratch, out=vhat[1])
    np.multiply(Gyz, uz, out=scratch)
    np.add(vhat[1], scratch, out=vhat[1])

    np.multiply(Gxz, ux, out=vhat[2])
    np.multiply(Gyz, uy, out=scratch)
    np.add(vhat[2], scratch, out=vhat[2])
    np.multiply(Gzz, uz, out=scratch)
    np.add(vhat[2], scratch, out=vhat[2])

    return fft.ifftn(vhat, axes=(1,2,3), workers=N_THREADS)

rows = []
for eps_r in EPS_VALUES:
    n_phys = int(np.ceil(CELLS_PER_LAMBDA_M*np.sqrt(eps_r)/np.pi))
    n_pad, u, G, vhat, scratch = make_fft_problem(n_phys)

    for _ in range(FFT_WARMUPS):
        _ = fft_matvec(u, G, vhat, scratch)

    times = []
    for _ in range(FFT_REPEATS):
        t0 = time.perf_counter()
        _ = fft_matvec(u, G, vhat, scratch)
        times.append(time.perf_counter() - t0)

    fft_ms = 1e3*np.median(times)
    h = 2*np.pi/(CELLS_PER_LAMBDA_M*np.sqrt(eps_r))
    active_voxels = (4*np.pi/3)/h**3

    rows.append({
        "eps_r": eps_r,
        "n_phys": n_phys,
        "n_pad_fast": n_pad,
        "fft_points": n_pad**3,
        "active_voxels_est": active_voxels,
        "dense_ms_16t": dense_ms,
        "fft_ms_16t": fft_ms,
        "speedup_dense_over_fft": dense_ms/fft_ms,
    })

df = pd.DataFrame(rows)
df.to_csv("16thread_dense_vs_fft_sweep.csv", index=False)

plt.figure(figsize=(9.1,5.3))
plt.plot(df["eps_r"], df["speedup_dense_over_fft"], marker="o", markersize=3)
plt.axhline(1.0, linestyle="--", linewidth=1)
plt.yscale("log")
plt.xlim(min(EPS_VALUES), max(EPS_VALUES))
plt.xlabel(r"Relative permittivity $\varepsilon_r$")
plt.ylabel(r"Measured speedup $T_{\rm dense}/T_{\rm FFT}$")
plt.title(r"16-thread P0 matvec: 1624 tetrahedra vs 10 voxels/$\lambda_m$")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("16thread_dense_vs_fft_sweep.png", dpi=180)
plt.show()

print(f"Visible CPUs: {os.cpu_count()}")
print(f"Requested threads: {N_THREADS}")
print(f"Dense matvec: {dense_ms:.3f} ms")
print(df.to_string(index=False))
