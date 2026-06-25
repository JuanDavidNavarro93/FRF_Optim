# from __future__ import annotations

print("\n" + "═" * 60)
print("STEP 1 — Initiate Imports")
 
import numpy as np
# import jax
# import jax.numpy as jnp
# import gpjax as gpx
# import optax
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# from jax import config
# from jax.scipy.special import ndtr   # standard normal CDF — used for probit

import sys
 
# config.update("jax_enable_x64", True)

print('Finished')
print("═" * 60)

#======================================

fig = plt.figure(figsize=(21,4.5))
gs  = gridspec.GridSpec(1, 3, hspace=0.38, wspace=0.32)
ax0 = fig.add_subplot(gs[0])
ax0.plot(np.array([1,2,3]),np.array([3,1,2]), color="steelblue", label='algo 1')
ax0.axhline(0, color="k", lw=0.5, ls=":")
ax0.set_title(f"Real-valued FRF  (ρ = )")
ax0.set_xlabel("Frequency (kHz)")
ax0.set_ylabel("H(ρ, f)")
ax0.legend(fontsize=8)

fig.savefig(f"temp.png", dpi=150, bbox_inches="tight")
