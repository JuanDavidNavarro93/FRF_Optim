"""
The University of Texas at San Antonio
Klesse School of Engineering and Integrated Design
Department of Mechanical, Aerospace and Industrial Engineering 

Juan David Navarro PhD
David Restrepo PhD

This python code computes a Two-Channel GP Metamodel for TPMS Lattices using
their FRF as training data. The metamodel uses two channels:

Channel 1 — Magnitude GP : Gaussian process regression on log|H(ρ, f)|
                           (Gaussian likelihood, conjugate posterior)

Channel 2 — Sign GP      : Gaussian process classification on sign(H(ρ, f))
                           mapped to {0, 1} (Bernoulli likelihood,
                           non-conjugate posterior via Laplace approx.)

The purpose of the two channel segmentation is to enable normalization of the 
FRF magnitude and enable calculation of phase

Reconstruction:
    Ĥ(ρ*, f) = pred_sign(ρ*, f) · exp(ŷ₁(ρ*, f))

Kernel (both channels):
    k((ρ,f),(ρ',f')) = k_ρ(ρ,ρ') · k_f(f,f')   — Matérn 5/2 in each dim

The training inputs form a grid (n_ρ × n_th x n_f), which enables Kronecker
structure for efficient inference (see note at bottom of file).

Created: 05/27/2026

Modifications: This version of the code exploits GPU Parallelization
Ma110ri3 parallelization 
"""

# ═════════════════════════════════════════════════════════════════════════════
# 1.  Imports
# ═════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

print("\n" + "═" * 60)
print("STEP 1 — Initiate Imports")
 
import numpy as np
import jax
import jax.numpy as jnp
import gpjax as gpx
import optax
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from jax import config
from jax.scipy.special import ndtr   # standard normal CDF — used for probit

import sys
import os
 
config.update("jax_enable_x64", True)

print(f"JAX devices available: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# Force GPU if available, else warn
if jax.default_backend() != "gpu":
    print("[WARNING] No GPU detected — running on CPU.")
# end if

print('Finished')
print("═" * 60)

# ═════════════════════════════════════════════════════════════════════════════
# 2.  Reading Data
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("STEP 2 Starting Data reading")

Train_iter = 2000
n_rho = 25
rho_0 = 0.2
rho_values  = np.linspace(rho_0, 1.0-rho_0, 25)[:n_rho]
thick_values = np.arange(0.0,6.0,1.0)
# dd = 7
dd = 11
# freq_values = np.arange(100.0, 601.0, dd, dtype=np.float64)
freq_values = np.arange(1.0, 800.0, dd, dtype=np.float64)
frf_matrix  = np.zeros((np.shape(rho_values)[0]*np.shape(thick_values)[0],np.shape(freq_values)[0]))

print(np.shape(frf_matrix))

cont0 = 0
for cont1 in range(n_rho):
    # filn = '/work/ljk354/FRF_Optim/Training/CAE/Results_T1D' + str(cont1+1) + '.txt'
    filn = '/work/ljk354/FRF_Optim/Learning_GP/Data/Results_T1D' + str(cont1+1) + '.txt'
    # frf_matrix[cont0,:] = np.loadtxt(filn, skiprows=3)[99:600:dd,1].T
    frf_matrix[cont0,:] = np.loadtxt(filn, skiprows=3)[::dd,1].T
    cont0 += 1
    for cont2 in range(np.shape(thick_values)[0]-1):
        print(cont2)
        filn = '/work/ljk354/FRF_Optim/Learning_GP/Data_Th/Results_T1D' + str(cont1+1) + 'H' + str(2*cont2+1) + '.txt'
        # frf_matrix[cont0,:] = np.loadtxt(filn, skiprows=3)[99:600:dd,1].T
        frf_matrix[cont0,:] = np.loadtxt(filn, skiprows=3)[::dd,1].T
        cont0 += 1
    # end for
# end for

print('Finished')
print("═" * 60)

# ═════════════════════════════════════════════════════════════════════════════
# 3. Leave-one-out cross-validation
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("STEP 3 — Starting Leave-One-Out Cross-Validation")
print("═" * 60)

def prepare_data(
    rho_values:  np.ndarray,   # (n_rho,)      relative density values
    thick_values: np.ndarray,   # (n_thick,)    thickness values
    frf_matrix:  np.ndarray,   # (n_rho*n_thick, n_f)  real-valued H(ρ, f) from Abaqus
    freq_values: np.ndarray,   # (n_f,)        frequency values in kHz
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
    """
    Flatten the FRF grid and build both channel targets.
 
    Returns
    -------
    X       : (N, 2)   Input pairs [ρ, f],  N = n_rho × n_f
    y_mag   : (N, 1)   Standardised log|H| — target for Channel 1
    y_sign  : (N, 1)   Binary sign ∈ {0., 1.} — target for Channel 2
                       (0 → H < 0,  1 → H > 0)
    meta    : dict     Normalisation stats needed for inverse transforms
    """
    n_rho_x_thick, n_f = frf_matrix.shape
    n_rho = n_rho_x_thick // len(thick_values)
    n_thick = len(thick_values)
    assert rho_values.shape[0] == n_rho, "rho_values length mismatch"
    assert freq_values.shape[0] == n_f,  "freq_values length mismatch"
 
    # Build flattened input grid (row-major: ρ varies slowly, f varies fast)
    X = np.zeros((n_rho_x_thick * n_f, 3), dtype=np.float64)
    cont0 = 0
    for cont1 in range(n_rho):
        for cont2 in range(n_thick):
            for cont3 in range(n_f): 
                X[cont0, 0] = rho_values[cont1]
                X[cont0, 1] = thick_values[cont2]
                X[cont0, 2] = freq_values[cont3]
                cont0 += 1
            # end for
        # end for
    # end for
    # rho_grid, f_grid = np.meshgrid(rho_values, freq_values, indexing="ij")
    # X = np.stack([rho_grid.ravel(), f_grid.ravel()], axis=-1).astype(np.float64)
    H_flat = frf_matrix.ravel().astype(np.float64)
    # Guard: zero FRF values cannot be log-transformed.
    # For undamped FRF at 1 kHz resolution this should not happen, but check.
    zero_mask = H_flat == 0.0
    if zero_mask.any():
        n_zeros = zero_mask.sum()
        print(f"[WARNING] {n_zeros} zero FRF values detected — "
              f"replacing with ±ε before log transform.")
        eps = np.finfo(np.float64).tiny
        H_flat[zero_mask & (H_flat >= 0)] =  eps
        H_flat[zero_mask & (H_flat <  0)] = -eps
    # end if
    
    # ── Channel 1: standardised log-magnitude ─────────────────────────────────
    log_abs_H      = np.log(np.abs(H_flat))
    log_abs_H_mean = log_abs_H.mean()
    log_abs_H_std  = log_abs_H.std()
    y_mag = ((log_abs_H - log_abs_H_mean) / log_abs_H_std).reshape(-1, 1)
    
    # ── Channel 2: binary sign  (0 → negative,  1 → positive) ────────────────
    y_sign = ((np.sign(H_flat) + 1.0) / 2.0).reshape(-1, 1)
 
    meta = dict(
        log_abs_H_mean = log_abs_H_mean,
        log_abs_H_std  = log_abs_H_std,
        n_rho          = n_rho,
        n_thick        = n_thick,
        n_f            = n_f,
        rho_values     = rho_values,
        thick_values   = thick_values,
        freq_values    = freq_values,
    )
    
    # return jnp.array(X), jnp.array(y_mag), jnp.array(y_sign), meta

    device = jax.devices("gpu")[0] if jax.default_backend() == "gpu" else jax.devices()[0]
    return (
    jax.device_put(jnp.array(X), device),
    jax.device_put(jnp.array(y_mag), device),
    jax.device_put(jnp.array(y_sign), device),
    meta,
    )
# end prepare_data

def build_product_kernel(
    rho_lengthscale: float = 0.1,   # initial ℓ_ρ  (≈ half the ρ range is a good start)
    thickness_lengthscale: float = 1.0, # initial ℓ_thickness (≈ half the thickness range)
    f_lengthscale:   float = 50.0,  # initial ℓ_f  in kHz
    variance:        float = 1.0,   # overall amplitude (absorbed into k_ρ)
    ) -> gpx.kernels.AbstractKernel:
    """
    Product kernel k_ρ(ρ,ρ') · k_f(f,f') with Matérn 5/2 in each dimension.
 
    active_dims ensures each sub-kernel only reads its own input column:
      - k_ρ reads column 0  (relative density)
      - k_f reads column 1  (frequency in kHz)
 
    Note: variance is set on k_ρ only; k_f variance is fixed to 1 to avoid
    redundant parameterisation in the product.
    """
    k_rho = gpx.kernels.Matern52(
        active_dims=[0],
        lengthscale=jnp.array(rho_lengthscale),
        variance=jnp.array(variance),
    )
    k_thickness = gpx.kernels.Matern52(
        active_dims=[1],
        lengthscale=jnp.array(thickness_lengthscale),
        variance=jnp.array(1.0),
    )
    k_f = gpx.kernels.Matern52(
        active_dims=[2],
        lengthscale=jnp.array(f_lengthscale),
        variance=jnp.array(1.0),
    )
    
    return gpx.kernels.ProductKernel(kernels=[k_rho, k_thickness, k_f])
# end build_product_kernel

def train_magnitude_gp(
    X_train:      jnp.ndarray,
    y_mag:        jnp.ndarray,
    num_iters:    int   = 1000,
    learning_rate: float = 0.01,
    key:          jax.Array = None,
    ) -> tuple:
    """
    Train Channel 1: conjugate GP (Gaussian likelihood) on standardised log|H|.
 
    Returns
    -------
    opt_posterior : optimised ConjugatePosterior
    history       : list of MLL values during training
    dataset       : gpx.Dataset (needed for predictions)
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    # end if
 
    dataset = gpx.Dataset(X=X_train, y=y_mag)
 
    prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=build_product_kernel(),
    )
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=dataset.n)
    posterior  = prior * likelihood                # ConjugatePosterior
 
    # # objective  = gpx.objectives.ConjugateMLL(negative=True)
    # objective  = -gpx.objectives.conjugate_mll(posterior=posterior, 
    #                                           data=dataset)
 
    # print(f"  [Ch1] Initial  MLL = {-objective:.4f}")
    
    objective = lambda model, data: (-gpx.objectives.conjugate_mll(model, data))
    
    initial_mll = -objective(posterior, dataset)

    print(f"  [Ch1] Initial  MLL = {initial_mll:.4f}")
 
    optimiser = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=learning_rate),
    )
 
    opt_posterior, history = gpx.fit(
        model=posterior,
        objective=objective,
        train_data=dataset,
        optim=optimiser,
        num_iters=num_iters,
        safe=True,
        key=key,
    )
     
    k0, k1, k2 = opt_posterior.prior.kernel.kernels
    # print(f"  [Ch1] ℓ_ρ = {k0.lengthscale:.5f} | "
    #       f"ℓ_f = {k1.lengthscale:.2f} kHz | "
    #       f"σ² (noise) = {opt_posterior.likelihood.obs_stddev**2:.2e}")   
 
    return opt_posterior, history, dataset
# end train_magnitude_gp

def train_sign_gp(
    X_train:      jnp.ndarray,
    y_sign:       jnp.ndarray,
    num_iters:    int   = 1000,
    learning_rate: float = 0.01,
    key:          jax.Array = None,
    ) -> tuple:
    """
    Train Channel 2: GP classifier (Bernoulli likelihood, probit link).
 
    Optimisation uses the log posterior density (Laplace-style: kernel
    hyperparameters and latent values are jointly optimised).
 
    Returns
    -------
    opt_posterior : optimised NonConjugatePosterior
    history       : list of log-posterior values during training
    dataset       : gpx.Dataset (needed for predictions)
    """
    if key is None:
        key = jax.random.PRNGKey(1)
    # end if
 
    dataset = gpx.Dataset(X=X_train, y=y_sign)
 
    prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=build_product_kernel(),
    )
    likelihood = gpx.likelihoods.Bernoulli(num_datapoints=dataset.n)
    posterior  = prior * likelihood                # NonConjugatePosterior
 
    # objective  = gpx.objectives.LogPosteriorDensity(negative=True)
    # print(f"  [Ch2] Initial  log-posterior = {-objective(posterior, dataset):.4f}")
    
    objective = lambda model, data: (-gpx.objectives.log_posterior_density(model, data))
    
    initial_log_posterior = -objective(posterior, dataset)

    print(f"  [Ch2] Initial  log-posterior = {initial_log_posterior:.4f}")
 
    optimiser = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=learning_rate),
    )
 
    opt_posterior, history = gpx.fit(
        model=posterior,
        objective=objective,
        train_data=dataset,
        optim=optimiser,
        num_iters=num_iters,
        safe=True,
        key=key,
    )
 
    k0, k1, k2 = opt_posterior.prior.kernel.kernels
    print(f"  [Ch2] Final    log-posterior = {-history[-1]:.4f}")
    # print(f"  [Ch2] ℓ_ρ = {k0.lengthscale:.5f} | "
    #       f"ℓ_f = {k1.lengthscale:.2f} kHz")
 
    return opt_posterior, history, dataset
# end train_sign_gp 

# @jax.jit
def jit_predict_magnitude(opt_posterior, train_dataset, X_test):
    latent_dist = opt_posterior.predict(X_test, train_data=train_dataset)
    predictive_dist = opt_posterior.likelihood(latent_dist)
    return predictive_dist.mean, predictive_dist.variance
# end jit_predict_magnitude

def predict_magnitude(
    opt_posterior,
    train_dataset: gpx.Dataset,
    X_test:        jnp.ndarray,
    meta:          dict,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict |H| at X_test locations.
 
    Returns
    -------
    magnitude     : (n_test,)  Predicted |H| (physical units)
    magnitude_std : (n_test,)  Predictive std of |H| via delta method
    """
    latent_dist     = opt_posterior.predict(X_test, train_data=train_dataset)
    predictive_dist = opt_posterior.likelihood(latent_dist)
 
    # mean_std = np.array(predictive_dist.mean()).ravel()
    # std_std  = np.array(predictive_dist.stddev()).ravel()
    mean_std = np.array(predictive_dist.mean).ravel()
    std_std = np.sqrt(np.array(predictive_dist.variance).ravel())
 
    # Un-standardise → log|H| space
    mean_log = mean_std * meta["log_abs_H_std"] + meta["log_abs_H_mean"]
    std_log  = std_std  * meta["log_abs_H_std"]
 
    # Exponentiate → |H| space; delta method: std(|H|) ≈ |H| · std(log|H|)
    magnitude     = np.exp(mean_log)
    magnitude_std = magnitude * std_log
 
    return magnitude, magnitude_std
#end predict_magnitude

def predict_sign(
    opt_posterior,
    train_dataset: gpx.Dataset,
    X_test:        jnp.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict P(sign = +1) and the most-probable sign at X_test locations.
 
    The probit-marginalised predictive probability is:
        p(y=1|x*) ≈ Φ(κ · μ*)     κ = 1/√(1 + π σ*²/8)
    where μ*, σ* are the latent GP predictive mean and std.
 
    Returns
    -------
    prob_positive : (n_test,)  Probability that H > 0
    pred_sign     : (n_test,)  Predicted sign ∈ {-1., +1.}
    """
    latent_dist = opt_posterior.predict(X_test, train_data=train_dataset)
 
    # mu    = jnp.array(latent_dist.mean()).ravel()
    # sigma = jnp.array(latent_dist.stddev()).ravel()
    mu    = jnp.array(latent_dist.mean).ravel()
    sigma = jnp.sqrt(jnp.array(latent_dist.variance).ravel())
 
    # Probit-marginalised probability (works well for both probit & logistic links)
    kappa         = 1.0 / jnp.sqrt(1.0 + jnp.pi * sigma**2 / 8.0)
    prob_positive = np.array(ndtr(kappa * mu))
    pred_sign     = np.where(prob_positive >= 0.5, 1.0, -1.0)
 
    return prob_positive, pred_sign
# end predict_sign

def predict_frf(
    rho_star:     float,
    thick_star:   float,
    freq_values:  np.ndarray,
    mag_post,
    sign_post,
    mag_ds:       gpx.Dataset,
    sign_ds:      gpx.Dataset,
    meta:         dict,
    ) -> dict:
    """
    Predict the full real-valued FRF at a new relative density ρ*.
 
    Reconstruction:  Ĥ = pred_sign · magnitude
    Uncertainty:     ±magnitude_std propagated through sign
                     (sign uncertainty shown separately via prob_positive)
 
    Returns a dict with keys:
        H_pred        : (n_f,)  Reconstructed FRF (real-valued, signed)
        H_upper       : (n_f,)  +1σ envelope of |H|, with predicted sign applied
        H_lower       : (n_f,)  -1σ envelope of |H|, with predicted sign applied
        magnitude     : (n_f,)  Predicted |H|
        magnitude_std : (n_f,)  Uncertainty in |H|  (1σ)
        prob_positive : (n_f,)  P(sign = +1) at each frequency
        pred_sign     : (n_f,)  Predicted sign ∈ {-1., +1.}
    """
    n_f    = len(freq_values)
    X_test = jnp.stack(
        [jnp.full((n_f,), float(rho_star)), jnp.full((n_f,), float(thick_star)), jnp.array(freq_values, dtype=jnp.float64)],
        axis=-1,
    )
 
    magnitude, magnitude_std = predict_magnitude(mag_post, mag_ds, X_test, meta)
    prob_positive, pred_sign  = predict_sign(sign_post, sign_ds, X_test)
    
    H_pred  = pred_sign * magnitude
    H_upper = pred_sign * (magnitude + magnitude_std)
    H_lower = pred_sign * (magnitude - magnitude_std)
 
    return dict(
        H_pred        = H_pred,
        H_upper       = H_upper,
        H_lower       = H_lower,
        magnitude     = magnitude,
        magnitude_std = magnitude_std,
        prob_positive = prob_positive,
        pred_sign     = pred_sign,
    )
# end predict_frf

def loo_cross_validation(
    rho_values:    np.ndarray,
    thick_values:  np.ndarray,
    frf_matrix:    np.ndarray,
    freq_values:   np.ndarray,
    num_iters:     int   = 1000,
    learning_rate: float = 0.01,
    ) -> list[dict]:
    """
    Leave-one-out CV over the ρ dimension.
 
    For each fold i:
      - Train on the other (n_rho - 1) ρ values
      - Predict H at ρ_i across all frequencies
      - Compute MAE, RMSE, and sign classification accuracy
 
    Returns a list of result dicts, one per fold.
    """
    n_rho = len(rho_values)
    n_thick = len(thick_values)
    results = []
 
    for i in range(n_rho):
        rho_held_out = rho_values[i]
        print(f"\n{'─'*60}")
        print(f"LOO fold {i+1}/{n_rho}  —  held-out ρ = {rho_held_out:.4f}")
        print(f"{'─'*60}")
 
        # Split
        train_idx = [j for j in range(n_rho) if j != i]
        rho_tr    = rho_values[train_idx]
        H_tr      = frf_matrix[train_idx, :]
        H_test    = frf_matrix[i, :]
 
        # Prepare and train on n_rho-1 density values
        key_mag, key_sign = jax.random.split(jax.random.PRNGKey(i * 100))
        X_tr, y_mag, y_sign, meta = prepare_data(rho_tr, H_tr, freq_values)
        mag_post,  _, mag_ds  = train_magnitude_gp(X_tr, y_mag,  num_iters,
                                                    learning_rate, key_mag)
        sign_post, _, sign_ds = train_sign_gp(     X_tr, y_sign, num_iters,
                                                    learning_rate, key_sign)
 
        # Predict at held-out ρ
        pred = predict_frf(rho_held_out, freq_values,
                           mag_post, sign_post, mag_ds, sign_ds, meta)
        
        # Metrics
        mae      = np.mean(np.abs(pred["H_pred"] - H_test))
        rmse     = np.sqrt(np.mean((pred["H_pred"] - H_test) ** 2))
        sign_acc = np.mean(pred["pred_sign"] == np.sign(H_test))
        rel_err  = np.mean(np.abs(pred["H_pred"] - H_test)
                           / (np.abs(H_test) + 1e-30))  # avoid division by 0
 
        print(f"  MAE:            {mae:.4e}")
        print(f"  RMSE:           {rmse:.4e}")
        print(f"  Mean rel. error:{rel_err * 100:.2f}%")
        print(f"  Sign accuracy:  {sign_acc * 100:.1f}%")
 
        results.append(dict(
            fold          = i,
            rho_held_out  = rho_held_out,
            H_test        = H_test,
            pred          = pred,
            mae           = mae,
            rmse          = rmse,
            rel_err       = rel_err,
            sign_acc      = sign_acc,
        ))
    # end for
 
    # Summary
    print(f"\n{'═'*60}")
    print("LOO Summary")
    print(f"{'═'*60}")
    print(f"  Mean MAE:            {np.mean([r['mae']      for r in results]):.4e}")
    print(f"  Mean RMSE:           {np.mean([r['rmse']     for r in results]):.4e}")
    print(f"  Mean rel. error:     {np.mean([r['rel_err']  for r in results])*100:.2f}%")
    print(f"  Mean sign accuracy:  {np.mean([r['sign_acc'] for r in results])*100:.1f}%")
    
    return results
# end loo_cross_validation

# loo_results = loo_cross_validation(
#     rho_values, thick_values, frf_matrix, freq_values,
#     num_iters=500,       # increase to 1000–2000 for production runs
#     learning_rate=0.01,
# )

print('Finished')
print("═" * 60)

# ═════════════════════════════════════════════════════════════════════════════
# 4. Plot each LOO fold
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("STEP 4 — Starting Plot of each LOO fold")
print("═" * 60)

def plot_loo_result(
    result:      dict,
    freq_values: np.ndarray,
    figsize:     tuple = (14, 9),
    ) -> plt.Figure:
    """
    Four-panel diagnostic plot for a single LOO fold:
      (a) Predicted vs actual FRF  (real-valued, with ±1σ band)
      (b) Predicted vs actual |H|  (log scale)
      (c) Predicted sign probability P(sign=+1) vs actual sign boundaries
      (d) Prediction error H_pred - H_true
    """
    pred        = result["pred"]
    H_test      = result["H_test"]
    rho_held    = result["rho_held_out"]
    f           = freq_values
 
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)
 
    # ── (a) Real-valued FRF ──────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.fill_between(f, pred["H_lower"], pred["H_upper"],
                     alpha=0.25, color="steelblue", label="±1σ")
    ax0.plot(f, pred["H_pred"], color="steelblue", lw=1.2, label="GP prediction")
    ax0.plot(f, H_test,         color="crimson",   lw=1.0,
             ls="--", alpha=0.85, label="Abaqus (truth)")
    ax0.axhline(0, color="k", lw=0.5, ls=":")
    ax0.set_title(f"Real-valued FRF  (ρ = {rho_held:.4f})")
    ax0.set_xlabel("Frequency (kHz)")
    ax0.set_ylabel("H(ρ, f)")
    ax0.legend(fontsize=8)
 
    # ── (b) Magnitude |H| — log scale ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.fill_between(f,
                     np.maximum(pred["magnitude"] - pred["magnitude_std"], 1e-20),
                     pred["magnitude"] + pred["magnitude_std"],
                     alpha=0.25, color="steelblue", label="±1σ")
    ax1.semilogy(f, pred["magnitude"], color="steelblue", lw=1.2, label="GP |Ĥ|")
    ax1.semilogy(f, np.abs(H_test),   color="crimson",   lw=1.0,
                 ls="--", alpha=0.85, label="|H| truth")
    ax1.set_title(f"|H(ρ, f)|  — log scale")
    ax1.set_xlabel("Frequency (kHz)")
    ax1.set_ylabel("|H(ρ, f)|")
    ax1.legend(fontsize=8)
 
    # ── (c) Sign probability ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    true_sign_pos = np.sign(H_test) > 0
    ax2.plot(f, pred["prob_positive"], color="steelblue", lw=1.2,
             label="P(sign = +1)")
    ax2.scatter(f[true_sign_pos],  np.ones(true_sign_pos.sum())  * 1.03,
                s=3, c="green",  marker="|", label="True sign +1")
    ax2.scatter(f[~true_sign_pos], np.zeros((~true_sign_pos).sum()) - 0.03,
                s=3, c="red",    marker="|", label="True sign −1")
    ax2.axhline(0.5, color="k", lw=0.5, ls="--")
    ax2.set_ylim(-0.08, 1.08)
    ax2.set_title("Sign GP — P(sign = +1)")
    ax2.set_xlabel("Frequency (kHz)")
    ax2.set_ylabel("Probability")
    ax2.legend(fontsize=8)
 
    # ── (d) Pointwise error ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    error = pred["H_pred"] - H_test
    ax3.plot(f, error, color="darkorange", lw=0.9)
    ax3.axhline(0, color="k", lw=0.5, ls=":")
    ax3.set_title(f"Pointwise error  (MAE = {result['mae']:.3e})")
    ax3.set_xlabel("Frequency (kHz)")
    ax3.set_ylabel("Ĥ − H")
 
    fig.suptitle(
        f"LOO fold {result['fold']+1}  —  ρ = {rho_held:.4f}  |  "
        f"Sign accuracy: {result['sign_acc']*100:.1f}%",
        fontsize=11, fontweight="bold",
    )
    return fig
# # end plot_loo_result

# Plot each LOO fold
foldr_out = '800kHz_CUDA_Outputs_Thickness'
if not(os.path.isdir(foldr_out)):
    os.mkdir(foldr_out)
# ned if
os.chdir(foldr_out)

# for res in loo_results:
#     fig = plot_loo_result(res, freq_values)
#     fig.savefig(f"CUDA_loo_fold_{res['fold']+1}.png", dpi=350, bbox_inches="tight")
#     plt.close(fig)
#     print(f"  Saved CUDA_loo_fold_{res['fold']+1}.png")
# # end for

print('Finished')
print("═" * 60)

# ═════════════════════════════════════════════════════════════════════════════
# 5. Plot each LOO fold
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("STEP 5 — Starting Full Model Training")
print("═" * 60)

def train_full_model(
    rho_values:    np.ndarray,
    thick_values:  np.ndarray,
    frf_matrix:    np.ndarray,
    freq_values:   np.ndarray,
    num_iters:     int   = 1000,
    learning_rate: float = 0.01,
    key:           jax.Array = None,
    ) -> dict:
    """
    Train both GP channels on the complete dataset.
 
    Returns a model dict that can be passed to predict_frf.
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    key_mag, key_sign = jax.random.split(key)
 
    print("── Preparing data ──")
    X_train, y_mag, y_sign, meta = prepare_data(rho_values, thick_values, frf_matrix, freq_values)
 
    print("── Training Channel 1: Magnitude GP ──")
    mag_post, mag_hist, mag_ds = train_magnitude_gp(
        X_train, y_mag, num_iters, learning_rate, key_mag)
 
    print("── Training Channel 2: Sign GP ──")
    sign_post, sign_hist, sign_ds = train_sign_gp(
        X_train, y_sign, num_iters, learning_rate, key_sign)
 
    return dict(
        mag_post   = mag_post,
        sign_post  = sign_post,
        mag_ds     = mag_ds,
        sign_ds    = sign_ds,
        mag_hist   = mag_hist,
        sign_hist  = sign_hist,
        meta       = meta,
        freq_values = freq_values,
    )
# end train_full_model

model = train_full_model(
    rho_values, thick_values, frf_matrix, freq_values,
    num_iters=Train_iter,
    learning_rate=0.01,
)

print('Finished')
print("═" * 60)

# ═════════════════════════════════════════════════════════════════════════════
# 6. Plot training history
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("STEP 6 — Starting Plot training history")
print("═" * 60)

def plot_training_history(
    mag_hist:  list,
    sign_hist: list,
    figsize:   tuple = (12, 4),
    ) -> plt.Figure:
    """Plot MLL / log-posterior convergence for both channels."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
 
    ax0.plot(-np.array(mag_hist), color="steelblue")
    ax0.set_title("Channel 1 — Magnitude GP\nMarginal Log-Likelihood")
    ax0.set_xlabel("Iteration")
    ax0.set_ylabel("MLL")
 
    ax1.plot(-np.array(sign_hist), color="darkorange")
    ax1.set_title("Channel 2 — Phase GP\nLog Posterior Density")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Log posterior")
 
    fig.suptitle("Training convergence", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig
# end plot_training_history

# Training history
fig_hist = plot_training_history(model["mag_hist"], model["sign_hist"])
fig_hist.savefig("CUDA_training_history.png", dpi=350, bbox_inches="tight")
plt.close(fig_hist)

print('Finished')
print("═" * 60)

# ═════════════════════════════════════════════════════════════════════════════
# 7. Predict at a new density & visualise the FRF surface
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("STEP 7 — Starting FRF Surface Visualisation")
print("═" * 60)

# def plot_frf_surface(
#     rho_values:    np.ndarray,
#     thick_values:  np.ndarray,
#     frf_matrix:    np.ndarray,
#     freq_values:   np.ndarray,
#     model:         dict,
#     rho_dense:     np.ndarray = None,
#     thick_dense:   np.ndarray = None,
#     figsize:       tuple = (14, 5),
#     ) -> plt.Figure:
#     """
#     Compare the GP-interpolated FRF surface H(ρ, f) against the training data.
 
#     Left:  training FRFs stacked (waterfall view)
#     Right: GP predictions at intermediate ρ values not seen during training
#     """
#     if rho_dense is None:
#         rho_dense = np.linspace(rho_values.min(), rho_values.max(), 20)
#     # end if
#     if thick_dense is None:
#         thick_dense = np.linspace(thick_values.min(), thick_values.max(), 20)
#     # end if
#     fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize, sharey=False)
 
#     # Training data waterfall
#     cmap = plt.cm.viridis
#     for k, (rho, thick, H_row) in enumerate(zip(rho_values, thick_values, frf_matrix)):
#         color = cmap(k / max(len(rho_values)*len(thick_values) - 1, 1))
#         ax0.plot(freq_values, np.log10(np.abs(H_row)), color=color, lw=0.9,
#                  label=f"ρ = {rho:.3f}, th = {thick:.2f}")
#     # ax0.axhline(0, color="k", lw=0.4, ls=":")
#     ax0.set_title("Training FRFs (FEM)",fontsize=24)
#     ax0.set_xlabel("Frequency [kHz]",fontsize=18)
#     ax0.set_ylabel("Log_{10}(|U|) [m]",fontsize=18)
#     # ax0.legend(fontsize=8)
#     ax0.set_ylim([-10,-2])
 
#     # GP predictions at dense ρ grid
#     for k, (rho_s, thick_s) in enumerate(zip(rho_dense, thick_dense)):
#         color = cmap(k / max(len(rho_dense)*len(thick_dense) - 1, 1))
#         pred  = predict_frf(
#             rho_s, thick_s, freq_values,
#             model["mag_post"], model["sign_post"],
#             model["mag_ds"],   model["sign_ds"],
#             model["meta"],
#         )
#         ax1.plot(freq_values, np.log10(np.abs(pred["H_pred"])), color=color, lw=0.7, alpha=0.8)
#     # ax1.axhline(0, color="k", lw=0.4, ls=":")
#     ax1.set_title(f"GP predictions",fontsize=24)
#     ax1.set_xlabel("Frequency [kHz]",fontsize=18)
#     ax1.set_ylabel("Log_{10}(|U|) [m]",fontsize=18)
#     ax1.set_ylim([-10,-2])
 
#     sm = plt.cm.ScalarMappable(
#         cmap=cmap,
#         norm=plt.Normalize(rho_values.min(), rho_values.max()),
#     )
#     sm.set_array([])
#     fig.colorbar(sm, ax=[ax0, ax1], label="Relative density ρ", shrink=0.85)
#     # fig.suptitle("FRF Surface H(ρ, f)", fontsize=11, fontweight="bold")
#     return fig

# # end plot_frf_surface

def plot_frf_surface(
    rho_values: np.ndarray,
    thick_values: np.ndarray,
    frf_matrix: np.ndarray,      # (n_rho, n_thick, n_freq)
    freq_values: np.ndarray,
    model: dict,
    rho_dense: np.ndarray = None,
    thick_dense: np.ndarray = None,
    rho_fixed: float = None,
    thick_fixed: float = None,
    figsize=(15,10),
):
    """
    Four-panel visualization.

        (1) Training data varying rho (fixed thickness)
        (2) GP varying rho (fixed thickness)

        (3) Training data varying thickness (fixed rho)
        (4) GP varying thickness (fixed rho)
    """

    if rho_dense is None:
        rho_dense = np.linspace(rho_values.min(), rho_values.max(), 25)
    # end if

    if thick_dense is None:
        thick_dense = np.linspace(thick_values.min(), thick_values.max(), 25)
    # end if

    if rho_fixed is None:
        rho_fixed = rho_values[len(rho_values)//2]
    # end if

    if thick_fixed is None:
        thick_fixed = thick_values[len(thick_values)//2]
    # end if

    rho_idx = np.argmin(np.abs(rho_values-rho_fixed))
    thick_idx = np.argmin(np.abs(thick_values-thick_fixed))

    fig, axs = plt.subplots(2,2,figsize=figsize)

    cmap = plt.cm.viridis
    
    ###############################################################
    # (0) Create frf_tensor
    ###############################################################
    frf_tensor = np.zeros((len(rho_values), len(thick_values), len(freq_values)))
    cont0 = 0
    for cont1 in range(len(rho_values)):
        for cont2 in range(len(thick_values)):
            frf_tensor[cont1, cont2, :] = frf_matrix[cont0, :]
            cont0 += 1
        # end for
    # end for

    ###############################################################
    # (1) Training data : varying rho
    ###############################################################

    ax = axs[0,0]

    for i,rho in enumerate(rho_values):

        color = cmap(i/max(len(rho_values)-1,1))

        H = frf_tensor[i,thick_idx,:]

        ax.plot(
            freq_values,
            np.log10(np.abs(H)),
            color=color,
            lw=1,
            label=f"{rho:.2f}"
        )

    ax.set_title(f"Training Data\nVarying Relative Density\nThickness={thick_values[thick_idx]:.2f}")
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel(r"$\log_{10}|U|$")

    ###############################################################
    # (2) GP : varying rho
    ###############################################################

    ax = axs[0,1]

    for i,rho in enumerate(rho_dense):

        color = cmap(i/max(len(rho_dense)-1,1))

        pred = predict_frf(
            rho,
            thick_values[thick_idx],
            freq_values,
            model["mag_post"],
            model["sign_post"],
            model["mag_ds"],
            model["sign_ds"],
            model["meta"],
        )

        ax.plot(
            freq_values,
            np.log10(np.abs(pred["H_pred"])),
            color=color,
            lw=0.8,
        )

    ax.set_title("GP Prediction\nVarying Relative Density")
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel(r"$\log_{10}|U|$")

    ###############################################################
    # (3) Training data : varying thickness
    ###############################################################

    ax = axs[1,0]

    for j,t in enumerate(thick_values):

        color = cmap(j/max(len(thick_values)-1,1))

        H = frf_tensor[rho_idx,j,:]

        ax.plot(
            freq_values,
            np.log10(np.abs(H)),
            color=color,
            lw=1,
            label=f"{t:.2f}"
        )

    ax.set_title(f"Training Data\nVarying Thickness\nDensity={rho_values[rho_idx]:.2f}")
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel(r"$\log_{10}|U|$")

    ###############################################################
    # (4) GP : varying thickness
    ###############################################################

    ax = axs[1,1]

    for j,t in enumerate(thick_dense):

        color = cmap(j/max(len(thick_dense)-1,1))

        pred = predict_frf(
            rho_values[rho_idx],
            t,
            freq_values,
            model["mag_post"],
            model["sign_post"],
            model["mag_ds"],
            model["sign_ds"],
            model["meta"],
        )

        ax.plot(
            freq_values,
            np.log10(np.abs(pred["H_pred"])),
            color=color,
            lw=0.8,
        )

    ax.set_title("GP Prediction\nVarying Thickness")
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel(r"$\log_{10}|U|$")

    ###############################################################

    for ax in axs.flat:
        ax.set_ylim([-10,-2])

    fig.tight_layout()

    return fig
# end plot_frf_surface

# fig_surf = plot_frf_surface(
#     rho_values, thick_values, frf_matrix, freq_values, model,
#     rho_dense=np.linspace(0.1, 0.5, 25), thick_dense=np.linspace(0.5, 2.0, 25)
# )

fig_surf = plot_frf_surface(
    rho_values,
    thick_values,
    frf_matrix,          # shape = (n_rho, n_thickness, n_freq)
    freq_values,
    model,
    rho_dense=np.linspace(rho_values.min(), rho_values.max(), 25),
    thick_dense=np.linspace(thick_values.min(), thick_values.max(), 25),
)

fig_surf.savefig("CUDA_frf_surface.png", dpi=350, bbox_inches="tight")
plt.close(fig_surf)

print("\nDone frf_surface.png")
print("═" * 60)

# ═════════════════════════════════════════════════════════════════════════════
# 8. Predict at a new density & visualise the Phase surface
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("STEP 8 — Starting Phase Surface Visualisation")
print("═" * 60)

def plot_phase_surface(
    rho_values: np.ndarray,
    thick_values: np.ndarray,
    frf_matrix: np.ndarray,
    freq_values: np.ndarray,
    model: dict,
    rho_dense: np.ndarray=None,
    thick_dense: np.ndarray=None,
    rho_fixed: float=None,
    thick_fixed: float=None,
    figsize=(15,10),
):
    """
    Four-panel visualization of phase.

        (1) Training phase varying rho
        (2) GP phase varying rho

        (3) Training phase varying thickness
        (4) GP phase varying thickness
    """

    if rho_dense is None:
        rho_dense = np.linspace(rho_values.min(), rho_values.max(),25)
    # end if

    if thick_dense is None:
        thick_dense = np.linspace(thick_values.min(), thick_values.max(),25)
    # end if

    if rho_fixed is None:
        rho_fixed = rho_values[len(rho_values)//2]
    # end if

    if thick_fixed is None:
        thick_fixed = thick_values[len(thick_values)//2]
    # end if

    rho_idx   = np.argmin(np.abs(rho_values-rho_fixed))
    thick_idx = np.argmin(np.abs(thick_values-thick_fixed))

    fig,axs = plt.subplots(2,2,figsize=figsize)

    cmap = plt.cm.viridis

    ###############################################################
    # (0) Create frf_tensor
    ###############################################################
    frf_tensor = np.zeros((len(rho_values), len(thick_values), len(freq_values)))
    cont0 = 0
    for cont1 in range(len(rho_values)):
        for cont2 in range(len(thick_values)):
            frf_tensor[cont1, cont2, :] = frf_matrix[cont0, :]
            cont0 += 1
        # end for
    # end for

    ############################################################
    # Training : varying rho
    ############################################################

    ax = axs[0,0]

    for i,rho in enumerate(rho_values):

        H = frf_tensor[i,thick_idx,:]

        phase = np.where(H>=0,0.0,180.0)

        ax.plot(
            freq_values,
            phase,
            color=cmap(i/max(len(rho_values)-1,1)),
            lw=1,
        )

    ax.set_title(f"Training Phase\nVarying Relative Density\nThickness={thick_values[thick_idx]:.2f}")
    ax.set_ylabel("Phase [deg]")

    ############################################################
    # GP : varying rho
    ############################################################

    ax = axs[0,1]

    for i,rho in enumerate(rho_dense):

        pred = predict_frf(
            rho,
            thick_values[thick_idx],
            freq_values,
            model["mag_post"],
            model["sign_post"],
            model["mag_ds"],
            model["sign_ds"],
            model["meta"],
        )

        phase = np.where(pred["pred_sign"]>0,0.0,180.0)

        ax.plot(
            freq_values,
            phase,
            color=cmap(i/max(len(rho_dense)-1,1)),
            lw=1,
        )

    ax.set_title("GP Phase\nVarying Relative Density")

    ############################################################
    # Training : varying thickness
    ############################################################

    ax = axs[1,0]

    for j,t in enumerate(thick_values):

        H = frf_tensor[rho_idx,j,:]

        phase = np.where(H>=0,0.0,180.0)

        ax.plot(
            freq_values,
            phase,
            color=cmap(j/max(len(thick_values)-1,1)),
            lw=1,
        )

    ax.set_title(f"Training Phase\nVarying Thickness\nDensity={rho_values[rho_idx]:.2f}")
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("Phase [deg]")

    ############################################################
    # GP : varying thickness
    ############################################################

    ax = axs[1,1]

    for j,t in enumerate(thick_dense):

        pred = predict_frf(
            rho_values[rho_idx],
            t,
            freq_values,
            model["mag_post"],
            model["sign_post"],
            model["mag_ds"],
            model["sign_ds"],
            model["meta"],
        )

        phase = np.where(pred["pred_sign"]>0,0.0,180.0)

        ax.plot(
            freq_values,
            phase,
            color=cmap(j/max(len(thick_dense)-1,1)),
            lw=1,
        )

    ax.set_title("GP Phase\nVarying Thickness")
    ax.set_xlabel("Frequency [kHz]")

    ############################################################

    for ax in axs.flat:

        ax.set_ylim([-10,190])
        ax.set_yticks([0,180])
        ax.grid(alpha=0.25)

    fig.tight_layout()

    return fig
# end plot_phase_surface

fig_phase = plot_phase_surface(
    rho_values,
    thick_values,
    frf_matrix,
    freq_values,
    model,
    rho_dense=np.linspace(rho_values.min(), rho_values.max(),25),
    thick_dense=np.linspace(thick_values.min(), thick_values.max(),25),
)

fig_phase.savefig(
    "CUDA_phase_surface.png",
    dpi=350,
    bbox_inches="tight",
)
plt.close(fig_phase)

print("\nDone. phase_surface.png")
print("═" * 60)

# ═════════════════════════════════════════════════════════════════════════════
# 8. Save model
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("STEP 8 — Saving model")
print("═" * 60)

# import equinox as eqx

# eqx.tree_serialise_leaves(
#     "mag_posterior.eqx",
#     model["mag_post"],
# )

# eqx.tree_serialise_leaves(
#     "sign_posterior.eqx",
#     model["sign_post"],
# )

# Saving the model using pkl
import pickle
with open("CUDA_gp_model.pkl", "wb") as f:
    pickle.dump(model, f)
# end with

print("\nFinished saving model")
print("═" * 60)