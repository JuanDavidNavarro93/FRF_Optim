import equinox as eqx
import numpy as np
import jax.numpy as jnp
import pickle

# ── Patch CombinationKernel ────────────────────────────────────────────────
from gpjax.kernels.base import CombinationKernel
CombinationKernel.__abstractmethods__ = frozenset(
    m for m in CombinationKernel.__abstractmethods__ if m != '_reduce'
)
# Provide a no-op so any call to _reduce doesn't crash post-load
CombinationKernel._reduce = lambda self, x: x

# ── Unpickle ───────────────────────────────────────────────────────────────
MODEL_PATH   = "/work/ljk354/FRF_Optim/GP_Jax/800kHz_CUDA_Outputs_25/CUDA_gp_model.pkl"
MODEL_PREFIX = "/work/ljk354/FRF_Optim/GP_Jax/800kHz_CUDA_Outputs_25/CUDA_gp_model"

with open(MODEL_PATH, "rb") as f:
    loaded_model = pickle.load(f)

print("Pickle loaded! Keys:", list(loaded_model.keys()))

# ── Re-save portably ───────────────────────────────────────────────────────
eqx.tree_serialise_leaves(MODEL_PREFIX + "_mag_post.eqx",  loaded_model["mag_post"])
eqx.tree_serialise_leaves(MODEL_PREFIX + "_sign_post.eqx", loaded_model["sign_post"])

np.savez(MODEL_PREFIX + "_data.npz",
    mag_X          = np.array(loaded_model["mag_ds"].X),
    mag_y          = np.array(loaded_model["mag_ds"].y),
    sign_X         = np.array(loaded_model["sign_ds"].X),
    sign_y         = np.array(loaded_model["sign_ds"].y),
    log_abs_H_mean = np.array([loaded_model["meta"]["log_abs_H_mean"]]),
    log_abs_H_std  = np.array([loaded_model["meta"]["log_abs_H_std"]]),
    rho_values     = loaded_model["meta"]["rho_values"],
    freq_values    = loaded_model["freq_values"],
)
print("Re-saved in portable .eqx format.")

#!/usr/bin/env python3
"""
The University of Texas at San Antonio
Klesse School of Engineering and Integrated Design
Department of Mechanical, Aerospace and Industrial Engineering

Juan David Navarro PhD
David Restrepo PhD

TPMS Lattice Bandgap Optimization
===================================
Loads the trained two-channel GP metamodel and finds the relative density ρ*
that produces the deepest bandgap in a user-specified frequency range.

Optimization problem:
    ρ* = argmin_{ρ ∈ [ρ_min, ρ_max]}  mean_{f ∈ [f_lo, f_hi]}  log|Ĥ(ρ, f)|

Rationale for objective:
  - The magnitude GP is trained on log|H| → the objective is smooth and
    directly from the GP mean (no exponentiation needed)
  - Minimising mean(log|H|) in the band = minimising the geometric mean of
    |H| = deepest possible bandgap
  - beta > 0 adds the predictive std (upper confidence bound), making the
    result conservative: the bandgap is guaranteed even under GP uncertainty

Created: 05/27/2026
Modifications: None to date
"""

# ═════════════════════════════════════════════════════════════════════════════
# 1.  Imports
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("STEP 1 — Initiating Imports")

import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import config
from jax.scipy.special import ndtr
import gpjax as gpx
import equinox as eqx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize_scalar

config.update("jax_enable_x64", True)

print("Finished")
print("═" * 60)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  User settings  ← edit this block only
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("STEP 2 — Loading User Settings")

# Path prefix used when saving (no extension — three files are expected):
#   <MODEL_PREFIX>_mag_post.eqx
#   <MODEL_PREFIX>_sign_post.eqx
#   <MODEL_PREFIX>_data.npz
MODEL_PREFIX = "/work/ljk354/FRF_Optim/GP_Jax/800kHz_CUDA_Outputs_25/CUDA_gp_model"

# Target bandgap frequency range [kHz]
BAND_LOW  = 250.0   # lower edge of desired bandgap [kHz]
BAND_HIGH = 450.0   # upper edge of desired bandgap [kHz]

# Optimisation settings
N_GRID    = 500     # coarse grid points for landscape scan
BETA      = 0.0     # UCB weight on predictive std (0 = mean only; 1–2 = conservative)
                    # beta > 0 ensures the bandgap is present even accounting
                    # for GP uncertainty — useful when adding more Abaqus runs
                    # is expensive

# Visualisation: additional ρ values to plot for the FRF surface
N_SURFACE = 30      # number of ρ values for the surface comparison plot

print(f"  Target band   : [{BAND_LOW}, {BAND_HIGH}] kHz")
print(f"  UCB weight β  : {BETA}")
print("Finished")
print("═" * 60)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Load model
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("STEP 3 — Loading GP Model")

def _build_mag_posterior(n_datapoints: int) -> gpx.gps.ConjugatePosterior:
    """Rebuild the magnitude GP architecture (must match training script exactly)."""
    k_rho = gpx.kernels.Matern52(active_dims=[0],
                                  lengthscale=jnp.array(0.1),
                                  variance=jnp.array(1.0))
    k_f   = gpx.kernels.Matern52(active_dims=[1],
                                  lengthscale=jnp.array(50.0),
                                  variance=jnp.array(1.0))
    prior      = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(),
                           kernel=gpx.kernels.ProductKernel(kernels=[k_rho, k_f]))
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=n_datapoints)
    return prior * likelihood
# end _build_mag_posterior

def _build_sign_posterior(n_datapoints: int) -> gpx.gps.NonConjugatePosterior:
    """Rebuild the sign GP architecture (must match training script exactly)."""
    k_rho = gpx.kernels.Matern52(active_dims=[0],
                                  lengthscale=jnp.array(0.1),
                                  variance=jnp.array(1.0))
    k_f   = gpx.kernels.Matern52(active_dims=[1],
                                  lengthscale=jnp.array(50.0),
                                  variance=jnp.array(1.0))
    prior      = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(),
                           kernel=gpx.kernels.ProductKernel(kernels=[k_rho, k_f]))
    likelihood = gpx.likelihoods.Bernoulli(num_datapoints=n_datapoints)
    return prior * likelihood
# end _build_sign_posterior

def load_model(prefix: str) -> dict:
    """
    Load the two-channel GP metamodel saved with save_model_portable().

    Strategy
    --------
    eqx.tree_serialise_leaves saves only array leaves — no Python class
    references, no flax dependency.  To reload, we:
      1. Rebuild the GP posterior architecture from scratch (same structure
         as training, dummy hyperparameter values are fine — they get
         overwritten in step 2).
      2. Call eqx.tree_deserialise_leaves to pour the saved arrays back
         into the architecture.

    Parameters
    ----------
    prefix : path prefix used during save, e.g. "/work/.../tpms_gp_model"
             (no file extension — three files are expected)
    """
    # ── Load datasets and metadata ────────────────────────────────────────────
    data    = np.load(prefix + ".pkl")
    mag_ds  = gpx.Dataset(X=jnp.array(data["mag_X"]),
                          y=jnp.array(data["mag_y"]))
    sign_ds = gpx.Dataset(X=jnp.array(data["sign_X"]),
                          y=jnp.array(data["sign_y"]))

    meta = dict(
        log_abs_H_mean = float(data["log_abs_H_mean"][0]),
        log_abs_H_std  = float(data["log_abs_H_std"][0]),
        rho_values     = data["rho_values"],
        freq_values    = data["freq_values"],
    )
    freq_values = data["freq_values"]

    # ── Rebuild architectures, then load saved leaf arrays into them ──────────
    mag_post  = _build_mag_posterior(mag_ds.n)
    mag_post  = eqx.tree_deserialise_leaves(prefix + "_mag_post.eqx", mag_post)
    
    sign_post = _build_sign_posterior(sign_ds.n)
    sign_post = eqx.tree_deserialise_leaves(prefix + "_sign_post.eqx", sign_post)
    # ── Report ────────────────────────────────────────────────────────────────
    rho_v = meta["rho_values"]
    frqs  = freq_values
    print(f"  Prefix        : {prefix}")
    print(f"  Training ρ    : {np.round(rho_v, 4)}")
    print(f"  ρ range       : [{rho_v.min():.4f}, {rho_v.max():.4f}]")
    print(f"  Frequency range: [{frqs.min():.0f}, {frqs.max():.0f}] kHz  "
          f"({len(frqs)} points)")

    # Confirm learned hyperparameters loaded correctly
    mk0, mk1 = mag_post.prior.kernel.kernels
    sk0, sk1 = sign_post.prior.kernel.kernels
    # print(f"  Mag GP   — ℓ_ρ={mk0.lengthscale:.5f}  ℓ_f={mk1.lengthscale:.2f} kHz  "
    #       f"σ_noise={mag_post.likelihood.obs_stddev:.4e}")
    # print(f"  Sign GP  — ℓ_ρ={sk0.lengthscale:.5f}  ℓ_f={sk1.lengthscale:.2f} kHz")
    return dict(
        mag_post    = mag_post,
        sign_post   = sign_post,
        mag_ds      = mag_ds,
        sign_ds     = sign_ds,
        meta        = meta,
        freq_values = freq_values,
    )
# end load_model

model = load_model(MODEL_PREFIX)

rho_min   = model["meta"]["rho_values"].min()
rho_max   = model["meta"]["rho_values"].max()
freq_all  = model["freq_values"]

# Validate target band against model frequency range
assert BAND_LOW  >= freq_all.min(), \
    f"BAND_LOW={BAND_LOW} kHz is below model range ({freq_all.min()} kHz)"
assert BAND_HIGH <= freq_all.max(), \
    f"BAND_HIGH={BAND_HIGH} kHz exceeds model range ({freq_all.max()} kHz)"
assert BAND_LOW  <  BAND_HIGH, "BAND_LOW must be less than BAND_HIGH"

band_mask = (freq_all >= BAND_LOW) & (freq_all <= BAND_HIGH)
freq_band = freq_all[band_mask]
print(f"  Band contains  : {band_mask.sum()} frequency points")
print("Finished")
print("═" * 60)

# ═════════════════════════════════════════════════════════════════════════════
# 4.  Prediction utilities  (self-contained; no import from training script)
# ═════════════════════════════════════════════════════════════════════════════
def _build_X_test(rho_star: float, freq_values: np.ndarray) -> jnp.ndarray:
    """Stack (ρ*, f) input pairs for the GP."""
    n_f = len(freq_values)
    return jnp.stack([
        jnp.full((n_f,), float(rho_star)),
        jnp.array(freq_values, dtype=jnp.float64),
    ], axis=-1)
# end _build_X_test

def predict_magnitude_full(rho_star: float,
                            freq_values: np.ndarray,
                            model: dict) -> dict:
    """
    Predict magnitude channel at (ρ*, freq_values).

    Returns dict with:
        magnitude     : |H| in physical units
        magnitude_std : 1σ uncertainty in |H|  (delta method)
        mean_log      : GP mean of log|H|       (used in objective)
        std_log       : GP std  of log|H|       (used in UCB objective)
    """
    X_test = _build_X_test(rho_star, freq_values)

    latent_dist     = model["mag_post"].predict(X_test, train_data=model["mag_ds"])
    predictive_dist = model["mag_post"].likelihood(latent_dist)

    mean_std = np.array(predictive_dist.mean).ravel()
    std_std  = np.sqrt(np.array(predictive_dist.variance).ravel())

    mean_log = mean_std * model["meta"]["log_abs_H_std"] + model["meta"]["log_abs_H_mean"]
    std_log  = std_std  * model["meta"]["log_abs_H_std"]

    magnitude     = np.exp(mean_log)
    magnitude_std = magnitude * std_log

    return dict(magnitude=magnitude, magnitude_std=magnitude_std,
                mean_log=mean_log, std_log=std_log)
# end predict_magnitude_full


def predict_sign_full(rho_star: float,
                      freq_values: np.ndarray,
                      model: dict) -> dict:
    """
    Predict sign channel at (ρ*, freq_values).

    Returns dict with:
        prob_positive : P(H > 0) at each frequency
        pred_sign     : predicted sign ∈ {-1., +1.}
    """
    X_test = _build_X_test(rho_star, freq_values)

    latent_dist = model["sign_post"].predict(X_test, train_data=model["sign_ds"])

    mu    = jnp.array(latent_dist.mean).ravel()
    sigma = jnp.sqrt(jnp.array(latent_dist.variance).ravel())

    kappa         = 1.0 / jnp.sqrt(1.0 + jnp.pi * sigma**2 / 8.0)
    prob_positive = np.array(ndtr(kappa * mu))
    pred_sign     = np.where(prob_positive >= 0.5, 1.0, -1.0)

    return dict(prob_positive=prob_positive, pred_sign=pred_sign)
# end predict_sign_full

def predict_frf_full(rho_star: float,
                     freq_values: np.ndarray,
                     model: dict) -> dict:
    """
    Predict the full real-valued FRF at a new relative density ρ*.

    Returns
    -------
    dict with keys:
        H_pred        : (n_f,)  Reconstructed FRF (real-valued, signed)
        H_upper/lower : (n_f,)  ±1σ envelopes (sign applied)
        magnitude     : (n_f,)  |H|
        magnitude_std : (n_f,)  uncertainty in |H|
        mean_log      : (n_f,)  GP mean of log|H|
        std_log       : (n_f,)  GP std of log|H|
        prob_positive : (n_f,)  P(sign = +1)
        pred_sign     : (n_f,)  ∈ {-1., +1.}
    """
    mag  = predict_magnitude_full(rho_star, freq_values, model)
    sign = predict_sign_full(rho_star, freq_values, model)

    H_pred  = sign["pred_sign"] * mag["magnitude"]
    H_upper = sign["pred_sign"] * (mag["magnitude"] + mag["magnitude_std"])
    H_lower = sign["pred_sign"] * (mag["magnitude"] - mag["magnitude_std"])

    return dict(**mag, **sign,
                H_pred=H_pred, H_upper=H_upper, H_lower=H_lower)
# end predict_frf_full

# ═════════════════════════════════════════════════════════════════════════════
# 5.  Bandgap objective and metrics
# ═════════════════════════════════════════════════════════════════════════════
def bandgap_objective(rho_scalar: float,
                      model: dict,
                      freq_band: np.ndarray,
                      beta: float = 0.0) -> float:
    """
    Scalar objective to MINIMISE for the bandgap optimisation.

    obj(ρ) = mean_{f ∈ band}[ log|Ĥ(ρ,f)| + β · std(log|Ĥ(ρ,f)|) ]

    Interpretation
    --------------
    β = 0   → minimise mean log|H| in band (deepest expected bandgap)
    β > 0   → upper confidence bound: conservative estimate that accounts
               for GP uncertainty; ensures the bandgap is real even if the
               metamodel is slightly off (recommended when adding more data
               is expensive)

    Returns
    -------
    float — lower value = deeper bandgap = better
    """
    rho = float(rho_scalar)
    if not (rho_min <= rho <= rho_max):
        return 1e10   # hard penalty outside training domain

    mag = predict_magnitude_full(rho, freq_band, model)
    return float(np.mean(mag["mean_log"] + beta * mag["std_log"]))
# end bandgap_objective

def compute_bandgap_metrics(pred: dict,
                            freq_values: np.ndarray,
                            target_band: tuple) -> dict:
    """
    Compute bandgap quality metrics from a full predicted FRF.

    Metrics
    -------
    depth_dB        : 20·log10(mean|H|_outside / mean|H|_inside)
                      → larger = deeper bandgap
    attenuation_dB  : 20·log10(max|H|_outside  / max|H|_inside)
                      → largest peak outside vs inside
    relative_depth  : mean|H|_outside / mean|H|_inside  (linear ratio)
    mean_in_band    : geometric mean of |H| in band
    mean_out_band   : geometric mean of |H| outside band
    """
    f_lo, f_hi   = target_band
    in_band      = (freq_values >= f_lo) & (freq_values <= f_hi)
    out_band     = ~in_band
    eps          = 1e-30

    mag = pred["magnitude"]

    # Geometric means (in log space, then exp)
    geom_in   = np.exp(np.mean(np.log(mag[in_band]  + eps)))
    geom_out  = np.exp(np.mean(np.log(mag[out_band] + eps)))
    max_in    = mag[in_band].max()
    max_out   = mag[out_band].max()

    depth_dB       = 20.0 * np.log10(geom_out  / (geom_in  + eps))
    attenuation_dB = 20.0 * np.log10(max_out   / (max_in   + eps))
    relative_depth = geom_out / (geom_in + eps)

    return dict(
        depth_dB        = depth_dB,
        attenuation_dB  = attenuation_dB,
        relative_depth  = relative_depth,
        geom_mean_in    = geom_in,
        geom_mean_out   = geom_out,
        max_in_band     = max_in,
        max_out_band    = max_out,
    )
# end compute_bandgap_metrics

# ═════════════════════════════════════════════════════════════════════════════
# 6.  Coarse grid scan  (landscape visualisation + global optimum estimate)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("STEP 4 — Coarse Grid Scan of Objective Landscape")
print("═" * 60)

rho_grid   = np.linspace(rho_min, rho_max, N_GRID)
obj_grid   = np.zeros(N_GRID)

print(f"  Evaluating objective at {N_GRID} ρ values...")
for k, rho in enumerate(rho_grid):
    obj_grid[k] = bandgap_objective(rho, model, freq_band, beta=BETA)
    if (k + 1) % 100 == 0:
        print(f"  Progress: {k+1}/{N_GRID}")
    # end if
# end for

idx_best_coarse = np.argmin(obj_grid)
rho_best_coarse = rho_grid[idx_best_coarse]
print(f"  Coarse best ρ : {rho_best_coarse:.4f}  "
      f"(obj = {obj_grid[idx_best_coarse]:.4f})")
print("Finished")
print("═" * 60)


# ═════════════════════════════════════════════════════════════════════════════
# 7.  Refined scalar optimisation  (scipy bounded Brent)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("STEP 5 — Refined Scalar Optimisation")
print("═" * 60)

# Search in a window around the coarse best (avoids local minima far from it)
# Expand window slightly beyond the coarse grid spacing
window = (rho_max - rho_min) / N_GRID * 5   # ±5 grid spacings
rho_lo_search = max(rho_min, rho_best_coarse - window * 20)
rho_hi_search = min(rho_max, rho_best_coarse + window * 20)

result = minimize_scalar(
    fun=bandgap_objective,
    bounds=(rho_lo_search, rho_hi_search),
    method="bounded",
    args=(model, freq_band, BETA),
    options={"xatol": 1e-6, "maxiter": 500},
)

rho_opt     = result.x
obj_opt     = result.fun
converged   = result.success

print(f"  Optimisation converged : {converged}")
print(f"  Optimal ρ*             : {rho_opt:.6f}")
print(f"  Objective at ρ*        : {obj_opt:.6f}")
print(f"  (mean log|H| in band   : {obj_opt:.6f}  →  "
      f"geometric mean |H| = {np.exp(obj_opt):.4e})")
print("Finished")
print("═" * 60)


# ═════════════════════════════════════════════════════════════════════════════
# 8.  Compute full FRF at optimal ρ and report metrics
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("STEP 6 — Computing Full FRF at Optimal ρ* and Bandgap Metrics")
print("═" * 60)

pred_opt = predict_frf_full(rho_opt, freq_all, model)
metrics  = compute_bandgap_metrics(pred_opt, freq_all, (BAND_LOW, BAND_HIGH))

print(f"\n  ── Bandgap metrics at ρ* = {rho_opt:.6f} ──")
print(f"  Target band                : [{BAND_LOW:.1f}, {BAND_HIGH:.1f}] kHz")
print(f"  Geom. mean |H| in band     : {metrics['geom_mean_in']:.4e}")
print(f"  Geom. mean |H| outside band: {metrics['geom_mean_out']:.4e}")
print(f"  Bandgap depth              : {metrics['depth_dB']:.2f} dB")
print(f"  Max attenuation            : {metrics['attenuation_dB']:.2f} dB")
print(f"  Linear attenuation ratio   : {metrics['relative_depth']:.1f}×")

print("Finished")
print("═" * 60)


# ═════════════════════════════════════════════════════════════════════════════
# 9.  Visualisation
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("STEP 7 — Generating Plots")
print("═" * 60)


# ── Plot A: Objective landscape ───────────────────────────────────────────────
def plot_objective_landscape(
    rho_grid:        np.ndarray,
    obj_grid:        np.ndarray,
    rho_opt:         float,
    obj_opt:         float,
    rho_training:    np.ndarray,
    target_band:     tuple,
    beta:            float,
    figsize:         tuple = (10, 4),
) -> plt.Figure:
    """Objective function over the full ρ domain with optimal point marked."""
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(rho_grid, obj_grid, color="steelblue", lw=1.5,
            label="Objective (mean log|U| in band)")
    ax.fill_between(rho_grid, obj_grid.min() - 0.5, obj_grid,
                    alpha=0.10, color="steelblue")

    # Mark training ρ values
    for rho_tr in rho_training:
        ax.axvline(rho_tr, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.axvline(rho_training[0], color="gray", lw=0.8, ls="--", alpha=0.6,
               label="Training ρ values")

    # Mark optimum
    ax.scatter([rho_opt], [obj_opt], color="crimson", s=80, zorder=6,
               label=f"Optimum  ρ* = {rho_opt:.4f}")
    ax.axvline(rho_opt, color="crimson", lw=1.2, ls="-", alpha=0.7)

    beta_label = f"  (β = {beta})" if beta > 0 else ""
    ax.set_title(f"Bandgap Objective Landscape  — band [{target_band[0]:.0f}, "
                 f"{target_band[1]:.0f}] kHz{beta_label}")
    ax.set_xlabel("Relative density ρ",fontsize=18)
    ax.set_ylabel("mean log|U|  in target band\n(lower = deeper bandgap)", fontsize=18)
    ax.legend(fontsize=12)
    fig.tight_layout()
    return fig
# end plot_objective_landscape

fig_land = plot_objective_landscape(
    rho_grid, obj_grid, rho_opt, obj_opt,
    model["meta"]["rho_values"], (BAND_LOW, BAND_HIGH), BETA,
)
fig_land.savefig("bandgap_objective_landscape.png", dpi=150, bbox_inches="tight")
plt.close(fig_land)
print("  Saved bandgap_objective_landscape.png")


# ── Plot B: Full FRF at ρ* ────────────────────────────────────────────────────
def plot_optimised_frf(
    pred:        dict,
    freq_values: np.ndarray,
    rho_opt:     float,
    target_band: tuple,
    metrics:     dict,
    figsize:     tuple = (14, 9),
) -> plt.Figure:
    """Four-panel diagnostic of the FRF at the optimal ρ*."""
    f_lo, f_hi = target_band
    f          = freq_values

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(2, 1, hspace=0.40, wspace=0.32)

    # Shade the target band on all axes
    def shade_band(ax):
        ax.axvspan(f_lo, f_hi, alpha=0.12, color="gold", label="Target band")

    # # ── (a) Real-valued FRF ───────────────────────────────────────────────────
    # ax0 = fig.add_subplot(gs[0, 0])
    # shade_band(ax0)
    # ax0.fill_between(f, pred["H_lower"], pred["H_upper"],
    #                  alpha=0.25, color="steelblue", label="±1σ")
    # ax0.plot(f, pred["H_pred"], color="steelblue", lw=1.2,
    #          label=f"U(ρ*={rho_opt:.4f}, f)")
    # ax0.axhline(0, color="k", lw=0.5, ls=":")
    # ax0.set_title(f"Predicted Displacement at (ρ* = {rho_opt:.4f})", fontsize=24)
    # ax0.set_xlabel("Frequency [kHz]",fontsize=18)
    # ax0.set_ylabel("U [m]",fontsize=18)
    # ax0.legend(fontsize=12)

    # ── (b) Magnitude |H| — log scale ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    shade_band(ax1)
    mag_lo = np.maximum(pred["magnitude"] - pred["magnitude_std"], 1e-30)
    mag_hi = pred["magnitude"] + pred["magnitude_std"]
    ax1.fill_between(f, mag_lo, mag_hi, alpha=0.25, color="steelblue",
                     label="±1σ")
    ax1.semilogy(f, pred["magnitude"], color="steelblue", lw=1.2,
                 label="|U|")

    # Horizontal lines for in-band / out-band geometric means
    # ax1.axhline(metrics["geom_mean_in"],  color="crimson", lw=1.0, ls="--",
    #             label=f"Geom. mean in band  ({metrics['geom_mean_in']:.2e})")
    # ax1.axhline(metrics["geom_mean_out"], color="green",   lw=1.0, ls="--",
    #             label=f"Geom. mean outside  ({metrics['geom_mean_out']:.2e})")

    ax1.set_title(f"|U|  [log scale] — "
                  f"depth = {metrics['depth_dB']:.1f} dB",fontsize=24)
    ax1.set_xlabel("Frequency [kHz]",fontsize=18)
    ax1.set_ylabel("|U|",fontsize=18)
    ax1.legend(fontsize=12)

    # ── (c) log|H| with GP uncertainty ───────────────────────────────────────
    # ax2 = fig.add_subplot(gs[1, 0])
    # shade_band(ax2)
    # ax2.fill_between(f,
    #                  pred["mean_log"] - pred["std_log"],
    #                  pred["mean_log"] + pred["std_log"],
    #                  alpha=0.25, color="darkorange", label="±1σ (log space)")
    # ax2.plot(f, pred["mean_log"], color="darkorange", lw=1.2,
    #          label="mean log|Ĥ|")
    # ax2.set_title("GP mean log|H|  — direct channel 1 output")
    # ax2.set_xlabel("Frequency (kHz)")
    # ax2.set_ylabel("log|H(ρ*, f)|")
    # ax2.legend(fontsize=8)

    # ── (d) Sign probability ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    shade_band(ax3)
    ax3.plot(f, pred["prob_positive"], color="steelblue", lw=1.2,
             label="P(Phase = 0)")
    ax3.axhline(0.5, color="k", lw=0.5, ls="--")
    ax3.set_ylim(-0.08, 1.08)
    ax3.set_title("Phase GP",fontsize=24)
    ax3.set_xlabel("Frequency [kHz]",fontsize=18)
    ax3.set_ylabel("Probability",fontsize=18)
    ax3.legend(fontsize=12)

    fig.suptitle(
        f"Optimised TPMS Lattice  |  ρ* = {rho_opt:.4f}  |  "
        f"Bandgap [{f_lo:.0f}–{f_hi:.0f}] kHz  |  "
        f"Depth = {metrics['depth_dB']:.1f} dB",
        fontsize=11, fontweight="bold",
    )
    return fig
# end plot_optimised_frf

fig_frf = plot_optimised_frf(
    pred_opt, freq_all, rho_opt, (BAND_LOW, BAND_HIGH), metrics)
fig_frf.savefig("optimised_frf.png", dpi=150, bbox_inches="tight")
plt.close(fig_frf)
print("  Saved optimised_frf.png")


# ── Plot C: FRF surface comparison  (training + optimum highlighted) ──────────
def plot_frf_surface_with_optimum(
    model:       dict,
    freq_values: np.ndarray,
    rho_opt:     float,
    pred_opt:    dict,
    target_band: tuple,
    n_surface:   int = 30,
    figsize:     tuple = (14, 5),
) -> plt.Figure:
    """
    Left:  training FRFs (from Abaqus)
    Right: GP predictions at a dense ρ grid, with ρ* highlighted in red
    """
    rho_v     = model["meta"]["rho_values"]
    rho_dense = np.linspace(rho_v.min(), rho_v.max(), n_surface)
    f_lo, f_hi = target_band

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize, sharey=False)
    cmap = plt.cm.viridis

    # Left: training FRFs (not stored in model — mark only ρ positions)
    # (If you want to re-plot training FRFs, pass frf_matrix here)
    ax0.text(0.5, 0.5,
             "Training FRFs not stored in model.\n"
             "Pass frf_matrix to this function to display.",
             ha="center", va="center", transform=ax0.transAxes,
             fontsize=9, color="gray")
    ax0.set_title("Training FRFs (Abaqus)")
    ax0.set_xlabel("Frequency (kHz)")
    ax0.set_ylabel("H(ρ, f)")

    # Right: GP predictions at dense ρ grid
    for k, rho_s in enumerate(rho_dense):
        color   = cmap(k / max(n_surface - 1, 1))
        pred_s  = predict_frf_full(rho_s, freq_values, model)
        alpha   = 0.5 if abs(rho_s - rho_opt) > 1e-3 else 1.0
        lw      = 0.7 if abs(rho_s - rho_opt) > 1e-3 else 2.0
        # ax1.plot(freq_values, pred_s["H_pred"], color=color,
        #          lw=lw, alpha=alpha)
        ax1.semilogy(freq_values, pred_s["magnitude"], color=color, lw=lw, alpha=alpha)
    # end for

    # Overlay ρ* in red
    # ax1.plot(freq_values, pred_opt["H_pred"], color="crimson",
    #          lw=2.5, zorder=5, label=f"ρ* = {rho_opt:.4f}")
    ax1.semilogy(freq_values, pred_opt["magnitude"], color="crimson",
             lw=2.5, zorder=5, label=f"ρ* = {rho_opt:.4f}")
    ax1.axvspan(f_lo, f_hi, alpha=0.12, color="gold", label="Target band")
    # ax1.axhline(0, color="k", lw=0.4, ls=":")
    ax1.set_title(f"GP predictions",fontsize=24)
    ax1.set_xlabel("Frequency (kHz)",fontsize=18)
    ax1.set_ylabel("|U|",fontsize=18)
    ax1.legend(fontsize=12)

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(rho_v.min(), rho_v.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=[ax0, ax1], label="Relative density ρ", shrink=0.85)
    # fig.suptitle("FRF Surface H(ρ, f)", fontsize=11, fontweight="bold")
    return fig
# end plot_frf_surface_with_optimum

fig_surf = plot_frf_surface_with_optimum(
    model, freq_all, rho_opt, pred_opt, (BAND_LOW, BAND_HIGH), N_SURFACE)
fig_surf.savefig("frf_surface_with_optimum.png", dpi=150, bbox_inches="tight")
plt.close(fig_surf)
print("  Saved frf_surface_with_optimum.png")

print("Finished")
print("═" * 60)


# ═════════════════════════════════════════════════════════════════════════════
# 10.  Final summary
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("OPTIMISATION SUMMARY")
print("═" * 60)
print(f"  Target frequency band     : [{BAND_LOW:.1f}, {BAND_HIGH:.1f}] kHz")
print(f"  Optimal relative density  : ρ* = {rho_opt:.6f}")
print(f"  Bandgap depth             : {metrics['depth_dB']:.2f} dB")
print(f"  Max attenuation (peak)    : {metrics['attenuation_dB']:.2f} dB")
print(f"  Linear attenuation ratio  : {metrics['relative_depth']:.1f}×")
print(f"  Geom. mean |H| in band    : {metrics['geom_mean_in']:.4e}")
print(f"  Geom. mean |H| outside    : {metrics['geom_mean_out']:.4e}")
print()
print("  Output files:")
print("    bandgap_objective_landscape.png")
print("    optimised_frf.png")
print("    frf_surface_with_optimum.png")
print("═" * 60)


# ═════════════════════════════════════════════════════════════════════════════
# NOTES
# ═════════════════════════════════════════════════════════════════════════════
#
# Multiple local minima
# ─────────────────────
# The objective landscape may have multiple local minima (e.g., different
# resonance structures at different ρ values can each produce a bandgap).
# The coarse grid scan guards against this, but if you suspect multiple minima,
# increase N_GRID or run minimize_scalar from several starting points and take
# the global best.
#
# UCB weight β
# ────────────
# With β = 0, the optimiser trusts the GP mean exactly.
# With β = 1–2, the objective penalises regions of high GP uncertainty,
# pushing the solution toward ρ values close to training data — safer but
# potentially more conservative.  A good rule of thumb:
#   β = 0   if LOO errors are small (< 5%)
#   β = 1   if LOO errors are moderate (5–15%)
#   β = 2   if you are in an extrapolation regime
#
# Adding out-of-phase constraint (future work)
# ────────────────────────────────────────────
# To enforce that H(ρ*, f) < 0 in the target band (out-of-phase response),
# add a sign penalty to the objective:
#
#   sign_pen = np.mean(np.maximum(0.0, pred_sign_in_band))  # penalise +1 signs
#   obj_total = mean_log_in_band + lambda_sign * sign_pen
#
# Alternatively use a constrained optimiser (scipy.optimize.minimize with
# constraints) that enforces pred_sign = -1 across the full target band.
#
# Next step: Bayesian optimisation with EI
# ────────────────────────────────────────
# If the current GP surrogate accuracy is insufficient, switch to Expected
# Improvement to jointly improve the surrogate and find the optimum:
#
#   EI(ρ) = E[max(f_best - f(ρ), 0)]  where f is the bandgap objective
#
# This selects ρ values that are both informative (high uncertainty) and
# promising (low objective mean), balancing exploration vs exploitation.
























#!/usr/bin/env python3
# """
# The University of Texas at San Antonio
# Klesse School of Engineering and Integrated Design
# Department of Mechanical, Aerospace and Industrial Engineering

# Juan David Navarro PhD
# David Restrepo PhD

# TPMS Lattice Bandgap Optimization
# ===================================
# Loads the trained two-channel GP metamodel and finds the relative density ρ*
# that produces the deepest bandgap in a user-specified frequency range.

# Optimization problem:
#     ρ* = argmin_{ρ ∈ [ρ_min, ρ_max]}  mean_{f ∈ [f_lo, f_hi]}  log|Ĥ(ρ, f)|

# Rationale for objective:
#   - The magnitude GP is trained on log|H| → the objective is smooth and
#     directly from the GP mean (no exponentiation needed)
#   - Minimising mean(log|H|) in the band = minimising the geometric mean of
#     |H| = deepest possible bandgap
#   - beta > 0 adds the predictive std (upper confidence bound), making the
#     result conservative: the bandgap is guaranteed even under GP uncertainty

# Created: 05/27/2026
# Modifications:
#   - 06/11/2026: Patched CombinationKernel._reduce abstract method before
#                 unpickling to handle GPJax 0.14.0 / Equinox 0.13.x mismatch.
#                 Added portable .eqx re-save after successful pickle load so
#                 future runs use load_model() without the patch.
# """

# # ═════════════════════════════════════════════════════════════════════════════
# # 0.  Compatibility patch  ← MUST run before any gpjax import
# # ═════════════════════════════════════════════════════════════════════════════
# #
# # GPJax 0.14 made CombinationKernel._reduce abstract.  The .pkl file was
# # saved with an older version where it was concrete, so pickle cannot
# # instantiate the class without this fix.
# #
# # The patch:
# #   1. Removes '_reduce' from __abstractmethods__ so Python lets pickle
# #      call __new__ on CombinationKernel (and its subclass ProductKernel).
# #   2. Provides a trivial no-op body so any accidental call to _reduce
# #      after loading does not raise AttributeError.
# #
# # This only affects the pickle-load path.  Once the model is re-saved in
# # portable .eqx format (Step 3b below), the patch is no longer needed and
# # can be removed on subsequent runs.
# # ─────────────────────────────────────────────────────────────────────────────
# from gpjax.kernels.base import CombinationKernel as _CK

# if '_reduce' in _CK.__abstractmethods__:
#     _CK.__abstractmethods__ = frozenset(
#         m for m in _CK.__abstractmethods__ if m != '_reduce'
#     )
#     _CK._reduce = lambda self, x: x   # no-op placeholder


# # ═════════════════════════════════════════════════════════════════════════════
# # 1.  Imports
# # ═════════════════════════════════════════════════════════════════════════════
# print("\n" + "═" * 60)
# print("STEP 1 — Initiating Imports")

# import pickle
# import numpy as np
# import jax
# import jax.numpy as jnp
# from jax import config
# from jax.scipy.special import ndtr
# import gpjax as gpx
# import equinox as eqx
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from scipy.optimize import minimize_scalar

# config.update("jax_enable_x64", True)

# print("Finished")
# print("═" * 60)


# # ═════════════════════════════════════════════════════════════════════════════
# # 2.  User settings  ← edit this block only
# # ═════════════════════════════════════════════════════════════════════════════
# print("\n" + "═" * 60)
# print("STEP 2 — Loading User Settings")

# # ── Paths ─────────────────────────────────────────────────────────────────────
# # Original pickle (used only on first run to migrate to .eqx format)
# MODEL_PATH   = "/work/ljk354/FRF_Optim/Learning_GP/CUDA_Outputs_25/CUDA_gp_model.pkl"

# # Portable prefix — on first run the script writes these three files:
# #   <MODEL_PREFIX>_mag_post.eqx
# #   <MODEL_PREFIX>_sign_post.eqx
# #   <MODEL_PREFIX>_data.npz
# # On subsequent runs it loads them directly (no pickle, no patch needed).
# MODEL_PREFIX = "/work/ljk354/FRF_Optim/Learning_GP/CUDA_Outputs_25/CUDA_gp_model"

# # Target bandgap frequency range [kHz]
# BAND_LOW  = 150.0   # lower edge of desired bandgap [kHz]
# BAND_HIGH = 250.0   # upper edge of desired bandgap [kHz]

# # Optimisation settings
# N_GRID    = 500     # coarse grid points for landscape scan
# BETA      = 0.0     # UCB weight on predictive std (0 = mean only; 1–2 = conservative)
#                     # beta > 0 ensures the bandgap is present even accounting
#                     # for GP uncertainty — useful when adding more Abaqus runs
#                     # is expensive

# # Visualisation: additional ρ values to plot for the FRF surface
# N_SURFACE = 30      # number of ρ values for the surface comparison plot

# print(f"  Target band   : [{BAND_LOW}, {BAND_HIGH}] kHz")
# print(f"  UCB weight β  : {BETA}")
# print("Finished")
# print("═" * 60)


# # ═════════════════════════════════════════════════════════════════════════════
# # 3.  Load model
# # ═════════════════════════════════════════════════════════════════════════════
# print("\n" + "═" * 60)
# print("STEP 3 — Loading GP Model")

# # ── Architecture builders (must match training script exactly) ────────────────

# def _build_mag_posterior(n_datapoints: int) -> gpx.gps.ConjugatePosterior:
#     """Rebuild the magnitude GP architecture."""
#     k_rho = gpx.kernels.Matern52(active_dims=[0],
#                                   lengthscale=jnp.array(0.1),
#                                   variance=jnp.array(1.0))
#     k_f   = gpx.kernels.Matern52(active_dims=[1],
#                                   lengthscale=jnp.array(50.0),
#                                   variance=jnp.array(1.0))
#     prior      = gpx.Prior(mean_function=gpx.mean_functions.Zero(),
#                            kernel=gpx.kernels.ProductKernel(kernels=[k_rho, k_f]))
#     likelihood = gpx.likelihoods.Gaussian(num_datapoints=n_datapoints)
#     return prior * likelihood


# def _build_sign_posterior(n_datapoints: int) -> gpx.gps.NonConjugatePosterior:
#     """Rebuild the sign GP architecture."""
#     k_rho = gpx.kernels.Matern52(active_dims=[0],
#                                   lengthscale=jnp.array(0.1),
#                                   variance=jnp.array(1.0))
#     k_f   = gpx.kernels.Matern52(active_dims=[1],
#                                   lengthscale=jnp.array(50.0),
#                                   variance=jnp.array(1.0))
#     prior      = gpx.Prior(mean_function=gpx.mean_functions.Zero(),
#                            kernel=gpx.kernels.ProductKernel(kernels=[k_rho, k_f]))
#     likelihood = gpx.likelihoods.Bernoulli(num_datapoints=n_datapoints)
#     return prior * likelihood


# # ── load_model: portable .eqx path (preferred) ───────────────────────────────

# def load_model(prefix: str) -> dict:
#     """
#     Load the two-channel GP metamodel saved with save_model_portable().

#     Reads:
#       <prefix>_mag_post.eqx   — magnitude posterior leaf arrays
#       <prefix>_sign_post.eqx  — sign posterior leaf arrays
#       <prefix>_data.npz       — datasets + metadata
#     """
#     data    = np.load(prefix + "_data.npz")
#     mag_ds  = gpx.Dataset(X=jnp.array(data["mag_X"]),
#                           y=jnp.array(data["mag_y"]))
#     sign_ds = gpx.Dataset(X=jnp.array(data["sign_X"]),
#                           y=jnp.array(data["sign_y"]))

#     meta = dict(
#         log_abs_H_mean = float(data["log_abs_H_mean"][0]),
#         log_abs_H_std  = float(data["log_abs_H_std"][0]),
#         rho_values     = data["rho_values"],
#         freq_values    = data["freq_values"],
#     )
#     freq_values = data["freq_values"]

#     mag_post  = _build_mag_posterior(mag_ds.n)
#     mag_post  = eqx.tree_deserialise_leaves(prefix + "_mag_post.eqx",  mag_post)

#     sign_post = _build_sign_posterior(sign_ds.n)
#     sign_post = eqx.tree_deserialise_leaves(prefix + "_sign_post.eqx", sign_post)

#     rho_v = meta["rho_values"]
#     frqs  = freq_values
#     print(f"  Prefix         : {prefix}")
#     print(f"  Training ρ     : {np.round(rho_v, 4)}")
#     print(f"  ρ range        : [{rho_v.min():.4f}, {rho_v.max():.4f}]")
#     print(f"  Frequency range: [{frqs.min():.0f}, {frqs.max():.0f}] kHz  "
#           f"({len(frqs)} points)")

#     mk0, mk1 = mag_post.prior.kernel.kernels
#     sk0, sk1 = sign_post.prior.kernel.kernels
#     print(f"  Mag GP   — ℓ_ρ={mk0.lengthscale:.5f}  ℓ_f={mk1.lengthscale:.2f} kHz  "
#           f"σ_noise={mag_post.likelihood.obs_stddev:.4e}")
#     print(f"  Sign GP  — ℓ_ρ={sk0.lengthscale:.5f}  ℓ_f={sk1.lengthscale:.2f} kHz")

#     return dict(
#         mag_post    = mag_post,
#         sign_post   = sign_post,
#         mag_ds      = mag_ds,
#         sign_ds     = sign_ds,
#         meta        = meta,
#         freq_values = freq_values,
#     )


# # ── load_from_pickle: migration path (first run only) ────────────────────────

# def load_from_pickle(pkl_path: str) -> dict:
#     """
#     Load the model from the original .pkl file.
#     Requires the CombinationKernel patch applied at the top of this script.
#     """
#     print(f"  Loading from pickle : {pkl_path}")
#     with open(pkl_path, "rb") as f:
#         loaded = pickle.load(f)
#     print("  Pickle loaded successfully.")
#     print(f"  Keys found          : {list(loaded.keys())}")
#     return loaded


# def resave_portable(loaded: dict, prefix: str) -> None:
#     """
#     Re-save a pickle-loaded model in portable .eqx + .npz format.
#     After this runs once, subsequent runs use load_model() with no patch.
#     """
#     eqx.tree_serialise_leaves(prefix + "_mag_post.eqx",  loaded["mag_post"])
#     eqx.tree_serialise_leaves(prefix + "_sign_post.eqx", loaded["sign_post"])

#     np.savez(prefix + "_data.npz",
#         mag_X          = np.array(loaded["mag_ds"].X),
#         mag_y          = np.array(loaded["mag_ds"].y),
#         sign_X         = np.array(loaded["sign_ds"].X),
#         sign_y         = np.array(loaded["sign_ds"].y),
#         log_abs_H_mean = np.array([loaded["meta"]["log_abs_H_mean"]]),
#         log_abs_H_std  = np.array([loaded["meta"]["log_abs_H_std"]]),
#         rho_values     = loaded["meta"]["rho_values"],
#         freq_values    = loaded["freq_values"],
#     )
#     print(f"  Portable model saved to: {prefix}_{{mag_post,sign_post}}.eqx  "
#           f"and  {prefix}_data.npz")
#     print("  Future runs can use load_model() directly — no patch needed.")


# # ── Decide which loading path to take ────────────────────────────────────────
# import os

# _eqx_ready = (
#     os.path.isfile(MODEL_PREFIX + "_mag_post.eqx")  and
#     os.path.isfile(MODEL_PREFIX + "_sign_post.eqx") and
#     os.path.isfile(MODEL_PREFIX + "_data.npz")
# )

# if _eqx_ready:
#     # ── Fast path: portable files already exist ───────────────────────────────
#     print("  Found portable .eqx files — loading directly (no pickle patch needed).")
#     model = load_model(MODEL_PREFIX)
# else:
#     # ── Migration path: first run, load pickle and re-save ────────────────────
#     print("  Portable .eqx files not found — loading from pickle and migrating.")
#     _pkl_data = load_from_pickle(MODEL_PATH)
#     resave_portable(_pkl_data, MODEL_PREFIX)
#     # Now load properly via the standard path
#     model = load_model(MODEL_PREFIX)

# rho_min   = model["meta"]["rho_values"].min()
# rho_max   = model["meta"]["rho_values"].max()
# freq_all  = model["freq_values"]

# # Validate target band against model frequency range
# assert BAND_LOW  >= freq_all.min(), \
#     f"BAND_LOW={BAND_LOW} kHz is below model range ({freq_all.min()} kHz)"
# assert BAND_HIGH <= freq_all.max(), \
#     f"BAND_HIGH={BAND_HIGH} kHz exceeds model range ({freq_all.max()} kHz)"
# assert BAND_LOW  <  BAND_HIGH, "BAND_LOW must be less than BAND_HIGH"

# band_mask = (freq_all >= BAND_LOW) & (freq_all <= BAND_HIGH)
# freq_band = freq_all[band_mask]
# print(f"  Band contains  : {band_mask.sum()} frequency points")
# print("Finished")
# print("═" * 60)


# # ═════════════════════════════════════════════════════════════════════════════
# # 4.  Prediction utilities  (self-contained; no import from training script)
# # ═════════════════════════════════════════════════════════════════════════════

# def _build_X_test(rho_star: float, freq_values: np.ndarray) -> jnp.ndarray:
#     """Stack (ρ*, f) input pairs for the GP."""
#     n_f = len(freq_values)
#     return jnp.stack([
#         jnp.full((n_f,), float(rho_star)),
#         jnp.array(freq_values, dtype=jnp.float64),
#     ], axis=-1)


# def predict_magnitude_full(rho_star: float,
#                             freq_values: np.ndarray,
#                             model: dict) -> dict:
#     """
#     Predict magnitude channel at (ρ*, freq_values).

#     Returns dict with:
#         magnitude     : |H| in physical units
#         magnitude_std : 1σ uncertainty in |H|  (delta method)
#         mean_log      : GP mean of log|H|       (used in objective)
#         std_log       : GP std  of log|H|       (used in UCB objective)
#     """
#     X_test = _build_X_test(rho_star, freq_values)

#     latent_dist     = model["mag_post"].predict(X_test, train_data=model["mag_ds"])
#     predictive_dist = model["mag_post"].likelihood(latent_dist)

#     mean_std = np.array(predictive_dist.mean).ravel()
#     std_std  = np.sqrt(np.array(predictive_dist.variance).ravel())

#     mean_log = mean_std * model["meta"]["log_abs_H_std"] + model["meta"]["log_abs_H_mean"]
#     std_log  = std_std  * model["meta"]["log_abs_H_std"]

#     magnitude     = np.exp(mean_log)
#     magnitude_std = magnitude * std_log

#     return dict(magnitude=magnitude, magnitude_std=magnitude_std,
#                 mean_log=mean_log, std_log=std_log)


# def predict_sign_full(rho_star: float,
#                       freq_values: np.ndarray,
#                       model: dict) -> dict:
#     """
#     Predict sign channel at (ρ*, freq_values).

#     Returns dict with:
#         prob_positive : P(H > 0) at each frequency
#         pred_sign     : predicted sign ∈ {-1., +1.}
#     """
#     X_test = _build_X_test(rho_star, freq_values)

#     latent_dist = model["sign_post"].predict(X_test, train_data=model["sign_ds"])

#     mu    = jnp.array(latent_dist.mean).ravel()
#     sigma = jnp.sqrt(jnp.array(latent_dist.variance).ravel())

#     kappa         = 1.0 / jnp.sqrt(1.0 + jnp.pi * sigma**2 / 8.0)
#     prob_positive = np.array(ndtr(kappa * mu))
#     pred_sign     = np.where(prob_positive >= 0.5, 1.0, -1.0)

#     return dict(prob_positive=prob_positive, pred_sign=pred_sign)


# def predict_frf_full(rho_star: float,
#                      freq_values: np.ndarray,
#                      model: dict) -> dict:
#     """
#     Predict the full real-valued FRF at a new relative density ρ*.

#     Returns
#     -------
#     dict with keys:
#         H_pred        : (n_f,)  Reconstructed FRF (real-valued, signed)
#         H_upper/lower : (n_f,)  ±1σ envelopes (sign applied)
#         magnitude     : (n_f,)  |H|
#         magnitude_std : (n_f,)  uncertainty in |H|
#         mean_log      : (n_f,)  GP mean of log|H|
#         std_log       : (n_f,)  GP std of log|H|
#         prob_positive : (n_f,)  P(sign = +1)
#         pred_sign     : (n_f,)  ∈ {-1., +1.}
#     """
#     mag  = predict_magnitude_full(rho_star, freq_values, model)
#     sign = predict_sign_full(rho_star, freq_values, model)

#     H_pred  = sign["pred_sign"] * mag["magnitude"]
#     H_upper = sign["pred_sign"] * (mag["magnitude"] + mag["magnitude_std"])
#     H_lower = sign["pred_sign"] * (mag["magnitude"] - mag["magnitude_std"])

#     return dict(**mag, **sign,
#                 H_pred=H_pred, H_upper=H_upper, H_lower=H_lower)


# # ═════════════════════════════════════════════════════════════════════════════
# # 5.  Bandgap objective and metrics
# # ═════════════════════════════════════════════════════════════════════════════

# def bandgap_objective(rho_scalar: float,
#                       model: dict,
#                       freq_band: np.ndarray,
#                       beta: float = 0.0) -> float:
#     """
#     Scalar objective to MINIMISE for the bandgap optimisation.

#     obj(ρ) = mean_{f ∈ band}[ log|Ĥ(ρ,f)| + β · std(log|Ĥ(ρ,f)|) ]

#     Interpretation
#     --------------
#     β = 0   → minimise mean log|H| in band (deepest expected bandgap)
#     β > 0   → upper confidence bound: conservative estimate that accounts
#                for GP uncertainty; ensures the bandgap is real even if the
#                metamodel is slightly off (recommended when adding more data
#                is expensive)

#     Returns
#     -------
#     float — lower value = deeper bandgap = better
#     """
#     rho = float(rho_scalar)
#     if not (rho_min <= rho <= rho_max):
#         return 1e10   # hard penalty outside training domain

#     mag = predict_magnitude_full(rho, freq_band, model)
#     return float(np.mean(mag["mean_log"] + beta * mag["std_log"]))


# def compute_bandgap_metrics(pred: dict,
#                             freq_values: np.ndarray,
#                             target_band: tuple) -> dict:
#     """
#     Compute bandgap quality metrics from a full predicted FRF.

#     Metrics
#     -------
#     depth_dB        : 20·log10(mean|H|_outside / mean|H|_inside)
#                       → larger = deeper bandgap
#     attenuation_dB  : 20·log10(max|H|_outside  / max|H|_inside)
#                       → largest peak outside vs inside
#     relative_depth  : mean|H|_outside / mean|H|_inside  (linear ratio)
#     mean_in_band    : geometric mean of |H| in band
#     mean_out_band   : geometric mean of |H| outside band
#     """
#     f_lo, f_hi   = target_band
#     in_band      = (freq_values >= f_lo) & (freq_values <= f_hi)
#     out_band     = ~in_band
#     eps          = 1e-30

#     mag = pred["magnitude"]

#     geom_in   = np.exp(np.mean(np.log(mag[in_band]  + eps)))
#     geom_out  = np.exp(np.mean(np.log(mag[out_band] + eps)))
#     max_in    = mag[in_band].max()
#     max_out   = mag[out_band].max()

#     depth_dB       = 20.0 * np.log10(geom_out  / (geom_in  + eps))
#     attenuation_dB = 20.0 * np.log10(max_out   / (max_in   + eps))
#     relative_depth = geom_out / (geom_in + eps)

#     return dict(
#         depth_dB        = depth_dB,
#         attenuation_dB  = attenuation_dB,
#         relative_depth  = relative_depth,
#         geom_mean_in    = geom_in,
#         geom_mean_out   = geom_out,
#         max_in_band     = max_in,
#         max_out_band    = max_out,
#     )


# # ═════════════════════════════════════════════════════════════════════════════
# # 6.  Coarse grid scan  (landscape visualisation + global optimum estimate)
# # ═════════════════════════════════════════════════════════════════════════════
# print("\n" + "═" * 60)
# print("STEP 4 — Coarse Grid Scan of Objective Landscape")
# print("═" * 60)

# rho_grid   = np.linspace(rho_min, rho_max, N_GRID)
# obj_grid   = np.zeros(N_GRID)

# print(f"  Evaluating objective at {N_GRID} ρ values...")
# for k, rho in enumerate(rho_grid):
#     obj_grid[k] = bandgap_objective(rho, model, freq_band, beta=BETA)
#     if (k + 1) % 100 == 0:
#         print(f"  Progress: {k+1}/{N_GRID}")

# idx_best_coarse = np.argmin(obj_grid)
# rho_best_coarse = rho_grid[idx_best_coarse]
# print(f"  Coarse best ρ : {rho_best_coarse:.4f}  "
#       f"(obj = {obj_grid[idx_best_coarse]:.4f})")
# print("Finished")
# print("═" * 60)


# # ═════════════════════════════════════════════════════════════════════════════
# # 7.  Refined scalar optimisation  (scipy bounded Brent)
# # ═════════════════════════════════════════════════════════════════════════════
# print("\n" + "═" * 60)
# print("STEP 5 — Refined Scalar Optimisation")
# print("═" * 60)

# window = (rho_max - rho_min) / N_GRID * 5
# rho_lo_search = max(rho_min, rho_best_coarse - window * 20)
# rho_hi_search = min(rho_max, rho_best_coarse + window * 20)

# result = minimize_scalar(
#     fun=bandgap_objective,
#     bounds=(rho_lo_search, rho_hi_search),
#     method="bounded",
#     args=(model, freq_band, BETA),
#     options={"xatol": 1e-6, "maxiter": 500},
# )

# rho_opt     = result.x
# obj_opt     = result.fun
# converged   = result.success

# print(f"  Optimisation converged : {converged}")
# print(f"  Optimal ρ*             : {rho_opt:.6f}")
# print(f"  Objective at ρ*        : {obj_opt:.6f}")
# print(f"  (mean log|H| in band   : {obj_opt:.6f}  →  "
#       f"geometric mean |H| = {np.exp(obj_opt):.4e})")
# print("Finished")
# print("═" * 60)


# # ═════════════════════════════════════════════════════════════════════════════
# # 8.  Compute full FRF at optimal ρ and report metrics
# # ═════════════════════════════════════════════════════════════════════════════
# print("\n" + "═" * 60)
# print("STEP 6 — Computing Full FRF at Optimal ρ* and Bandgap Metrics")
# print("═" * 60)

# pred_opt = predict_frf_full(rho_opt, freq_all, model)
# metrics  = compute_bandgap_metrics(pred_opt, freq_all, (BAND_LOW, BAND_HIGH))

# print(f"\n  ── Bandgap metrics at ρ* = {rho_opt:.6f} ──")
# print(f"  Target band                : [{BAND_LOW:.1f}, {BAND_HIGH:.1f}] kHz")
# print(f"  Geom. mean |H| in band     : {metrics['geom_mean_in']:.4e}")
# print(f"  Geom. mean |H| outside band: {metrics['geom_mean_out']:.4e}")
# print(f"  Bandgap depth              : {metrics['depth_dB']:.2f} dB")
# print(f"  Max attenuation            : {metrics['attenuation_dB']:.2f} dB")
# print(f"  Linear attenuation ratio   : {metrics['relative_depth']:.1f}×")

# print("Finished")
# print("═" * 60)


# # ═════════════════════════════════════════════════════════════════════════════
# # 9.  Visualisation
# # ═════════════════════════════════════════════════════════════════════════════
# print("\n" + "═" * 60)
# print("STEP 7 — Generating Plots")
# print("═" * 60)


# def plot_objective_landscape(
#     rho_grid:        np.ndarray,
#     obj_grid:        np.ndarray,
#     rho_opt:         float,
#     obj_opt:         float,
#     rho_training:    np.ndarray,
#     target_band:     tuple,
#     beta:            float,
#     figsize:         tuple = (10, 4),
# ) -> plt.Figure:
#     """Objective function over the full ρ domain with optimal point marked."""
#     fig, ax = plt.subplots(figsize=figsize)

#     ax.plot(rho_grid, obj_grid, color="steelblue", lw=1.5,
#             label="Objective (mean log|H| in band)")
#     ax.fill_between(rho_grid, obj_grid.min() - 0.5, obj_grid,
#                     alpha=0.10, color="steelblue")

#     for rho_tr in rho_training:
#         ax.axvline(rho_tr, color="gray", lw=0.8, ls="--", alpha=0.6)
#     ax.axvline(rho_training[0], color="gray", lw=0.8, ls="--", alpha=0.6,
#                label="Training ρ values")

#     ax.scatter([rho_opt], [obj_opt], color="crimson", s=80, zorder=6,
#                label=f"Optimum  ρ* = {rho_opt:.4f}")
#     ax.axvline(rho_opt, color="crimson", lw=1.2, ls="-", alpha=0.7)

#     beta_label = f"  (β = {beta})" if beta > 0 else ""
#     ax.set_title(f"Bandgap Objective Landscape  — band [{target_band[0]:.0f}, "
#                  f"{target_band[1]:.0f}] kHz{beta_label}")
#     ax.set_xlabel("Relative density ρ")
#     ax.set_ylabel("mean log|H|  in target band\n(lower = deeper bandgap)")
#     ax.legend(fontsize=9)
#     fig.tight_layout()
#     return fig


# fig_land = plot_objective_landscape(
#     rho_grid, obj_grid, rho_opt, obj_opt,
#     model["meta"]["rho_values"], (BAND_LOW, BAND_HIGH), BETA,
# )
# fig_land.savefig("bandgap_objective_landscape.png", dpi=150, bbox_inches="tight")
# plt.close(fig_land)
# print("  Saved bandgap_objective_landscape.png")


# def plot_optimised_frf(
#     pred:        dict,
#     freq_values: np.ndarray,
#     rho_opt:     float,
#     target_band: tuple,
#     metrics:     dict,
#     figsize:     tuple = (14, 9),
# ) -> plt.Figure:
#     """Four-panel diagnostic of the FRF at the optimal ρ*."""
#     f_lo, f_hi = target_band
#     f          = freq_values

#     fig = plt.figure(figsize=figsize)
#     gs  = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.32)

#     def shade_band(ax):
#         ax.axvspan(f_lo, f_hi, alpha=0.12, color="gold", label="Target band")

#     ax0 = fig.add_subplot(gs[0, 0])
#     shade_band(ax0)
#     ax0.fill_between(f, pred["H_lower"], pred["H_upper"],
#                      alpha=0.25, color="steelblue", label="±1σ")
#     ax0.plot(f, pred["H_pred"], color="steelblue", lw=1.2,
#              label=f"Ĥ(ρ*={rho_opt:.4f}, f)")
#     ax0.axhline(0, color="k", lw=0.5, ls=":")
#     ax0.set_title(f"Predicted FRF  (ρ* = {rho_opt:.4f})")
#     ax0.set_xlabel("Frequency (kHz)")
#     ax0.set_ylabel("H(ρ*, f)")
#     ax0.legend(fontsize=8)

#     ax1 = fig.add_subplot(gs[0, 1])
#     shade_band(ax1)
#     mag_lo = np.maximum(pred["magnitude"] - pred["magnitude_std"], 1e-30)
#     mag_hi = pred["magnitude"] + pred["magnitude_std"]
#     ax1.fill_between(f, mag_lo, mag_hi, alpha=0.25, color="steelblue", label="±1σ")
#     ax1.semilogy(f, pred["magnitude"], color="steelblue", lw=1.2, label="|Ĥ|")
#     ax1.axhline(metrics["geom_mean_in"],  color="crimson", lw=1.0, ls="--",
#                 label=f"Geom. mean in band  ({metrics['geom_mean_in']:.2e})")
#     ax1.axhline(metrics["geom_mean_out"], color="green",   lw=1.0, ls="--",
#                 label=f"Geom. mean outside  ({metrics['geom_mean_out']:.2e})")
#     ax1.set_title(f"|H(ρ*, f)|  [log scale] — "
#                   f"depth = {metrics['depth_dB']:.1f} dB")
#     ax1.set_xlabel("Frequency (kHz)")
#     ax1.set_ylabel("|H(ρ*, f)|")
#     ax1.legend(fontsize=7)

#     ax2 = fig.add_subplot(gs[1, 0])
#     shade_band(ax2)
#     ax2.fill_between(f,
#                      pred["mean_log"] - pred["std_log"],
#                      pred["mean_log"] + pred["std_log"],
#                      alpha=0.25, color="darkorange", label="±1σ (log space)")
#     ax2.plot(f, pred["mean_log"], color="darkorange", lw=1.2, label="mean log|Ĥ|")
#     ax2.set_title("GP mean log|H|  — direct channel 1 output")
#     ax2.set_xlabel("Frequency (kHz)")
#     ax2.set_ylabel("log|H(ρ*, f)|")
#     ax2.legend(fontsize=8)

#     ax3 = fig.add_subplot(gs[1, 1])
#     shade_band(ax3)
#     ax3.plot(f, pred["prob_positive"], color="steelblue", lw=1.2,
#              label="P(sign = +1)")
#     ax3.axhline(0.5, color="k", lw=0.5, ls="--")
#     ax3.set_ylim(-0.08, 1.08)
#     ax3.set_title("Sign GP — P(H > 0)")
#     ax3.set_xlabel("Frequency (kHz)")
#     ax3.set_ylabel("Probability")
#     ax3.legend(fontsize=8)

#     fig.suptitle(
#         f"Optimised TPMS Lattice  |  ρ* = {rho_opt:.4f}  |  "
#         f"Bandgap [{f_lo:.0f}–{f_hi:.0f}] kHz  |  "
#         f"Depth = {metrics['depth_dB']:.1f} dB",
#         fontsize=11, fontweight="bold",
#     )
#     return fig


# fig_frf = plot_optimised_frf(
#     pred_opt, freq_all, rho_opt, (BAND_LOW, BAND_HIGH), metrics)
# fig_frf.savefig("optimised_frf.png", dpi=150, bbox_inches="tight")
# plt.close(fig_frf)
# print("  Saved optimised_frf.png")


# def plot_frf_surface_with_optimum(
#     model:       dict,
#     freq_values: np.ndarray,
#     rho_opt:     float,
#     pred_opt:    dict,
#     target_band: tuple,
#     n_surface:   int = 30,
#     figsize:     tuple = (14, 5),
# ) -> plt.Figure:
#     """
#     Left:  placeholder (training FRFs not stored in model)
#     Right: GP predictions at a dense ρ grid, with ρ* highlighted in red
#     """
#     rho_v      = model["meta"]["rho_values"]
#     rho_dense  = np.linspace(rho_v.min(), rho_v.max(), n_surface)
#     f_lo, f_hi = target_band

#     fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize, sharey=False)
#     cmap = plt.cm.viridis

#     ax0.text(0.5, 0.5,
#              "Training FRFs not stored in model.\n"
#              "Pass frf_matrix to this function to display.",
#              ha="center", va="center", transform=ax0.transAxes,
#              fontsize=9, color="gray")
#     ax0.set_title("Training FRFs (Abaqus)")
#     ax0.set_xlabel("Frequency (kHz)")
#     ax0.set_ylabel("H(ρ, f)")

#     for k, rho_s in enumerate(rho_dense):
#         color  = cmap(k / max(n_surface - 1, 1))
#         pred_s = predict_frf_full(rho_s, freq_values, model)
#         alpha  = 0.5 if abs(rho_s - rho_opt) > 1e-3 else 1.0
#         lw     = 0.7 if abs(rho_s - rho_opt) > 1e-3 else 2.0
#         ax1.plot(freq_values, pred_s["H_pred"], color=color, lw=lw, alpha=alpha)

#     ax1.plot(freq_values, pred_opt["H_pred"], color="crimson",
#              lw=2.5, zorder=5, label=f"ρ* = {rho_opt:.4f}")
#     ax1.axvspan(f_lo, f_hi, alpha=0.12, color="gold", label="Target band")
#     ax1.axhline(0, color="k", lw=0.4, ls=":")
#     ax1.set_title(f"GP predictions  ({n_surface} ρ values, ρ* in red)")
#     ax1.set_xlabel("Frequency (kHz)")
#     ax1.set_ylabel("Ĥ(ρ, f)")
#     ax1.legend(fontsize=8)

#     sm = plt.cm.ScalarMappable(
#         cmap=cmap, norm=plt.Normalize(rho_v.min(), rho_v.max()))
#     sm.set_array([])
#     fig.colorbar(sm, ax=[ax0, ax1], label="Relative density ρ", shrink=0.85)
#     fig.suptitle("FRF Surface H(ρ, f)", fontsize=11, fontweight="bold")
#     return fig


# fig_surf = plot_frf_surface_with_optimum(
#     model, freq_all, rho_opt, pred_opt, (BAND_LOW, BAND_HIGH), N_SURFACE)
# fig_surf.savefig("frf_surface_with_optimum.png", dpi=150, bbox_inches="tight")
# plt.close(fig_surf)
# print("  Saved frf_surface_with_optimum.png")

# print("Finished")
# print("═" * 60)


# # ═════════════════════════════════════════════════════════════════════════════
# # 10.  Final summary
# # ═════════════════════════════════════════════════════════════════════════════
# print("\n" + "═" * 60)
# print("OPTIMISATION SUMMARY")
# print("═" * 60)
# print(f"  Target frequency band     : [{BAND_LOW:.1f}, {BAND_HIGH:.1f}] kHz")
# print(f"  Optimal relative density  : ρ* = {rho_opt:.6f}")
# print(f"  Bandgap depth             : {metrics['depth_dB']:.2f} dB")
# print(f"  Max attenuation (peak)    : {metrics['attenuation_dB']:.2f} dB")
# print(f"  Linear attenuation ratio  : {metrics['relative_depth']:.1f}×")
# print(f"  Geom. mean |H| in band    : {metrics['geom_mean_in']:.4e}")
# print(f"  Geom. mean |H| outside    : {metrics['geom_mean_out']:.4e}")
# print()
# print("  Output files:")
# print("    bandgap_objective_landscape.png")
# print("    optimised_frf.png")
# print("    frf_surface_with_optimum.png")
# print("═" * 60)


# # ═════════════════════════════════════════════════════════════════════════════
# # NOTES
# # ═════════════════════════════════════════════════════════════════════════════
# #
# # Patch lifecycle
# # ───────────────
# # The CombinationKernel patch in Section 0 is only needed on the FIRST run,
# # when the model is still stored as a .pkl file.  Once resave_portable() has
# # written the three .eqx/.npz files, the _eqx_ready branch is taken and the
# # patch is never exercised again (though it is harmless to leave in place).
# #
# # Multiple local minima
# # ─────────────────────
# # The objective landscape may have multiple local minima.  The coarse grid
# # scan guards against this, but increase N_GRID or run minimize_scalar from
# # several starting points if you suspect multiple minima.
# #
# # UCB weight β
# # ────────────
# # β = 0   if LOO errors are small (< 5%)
# # β = 1   if LOO errors are moderate (5–15%)
# # β = 2   if you are in an extrapolation regime
# #
# # Next step: Bayesian optimisation with EI
# # ────────────────────────────────────────
# # If GP surrogate accuracy is insufficient, switch to Expected Improvement:
# #   EI(ρ) = E[max(f_best - f(ρ), 0)]
# # This balances exploration (high uncertainty) vs exploitation (low mean).