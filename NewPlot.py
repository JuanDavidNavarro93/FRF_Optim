import equinox as eqx
import numpy as np
import jax.numpy as jnp
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gpjax as gpx

import sys

# ── Patch CombinationKernel ────────────────────────────────────────────────
from gpjax.kernels.base import CombinationKernel
CombinationKernel.__abstractmethods__ = frozenset(
    m for m in CombinationKernel.__abstractmethods__ if m != '_reduce'
)
# Provide a no-op so any call to _reduce doesn't crash post-load
CombinationKernel._reduce = lambda self, x: x

# ── Unpickle ───────────────────────────────────────────────────────────────
MODEL_PATH   = "800kHz_CUDA_Outputs_Thickness/CUDA_gp_model.pkl"
MODEL_PREFIX = "800kHz_CUDA_Outputs_Thickness/CUDA_gp_model"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Pickle loaded! Keys:", list(model.keys()))

print(model)