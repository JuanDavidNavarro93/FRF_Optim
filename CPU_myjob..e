/work/ljk354/FRF_Optim/GP_Jax/NewPlot.py:160: UserWarning: Explicitly requested dtype float64 requested in array is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  [jnp.full((n_f,), float(rho_star)), jnp.full((n_f,), float(thick_star)), jnp.array(freq_values, dtype=jnp.float64)],
Traceback (most recent call last):
  File "/work/ljk354/FRF_Optim/GP_Jax/NewPlot.py", line 246, in <module>
    pred = predict_frf(
           ^^^^^^^^^^^^
  File "/work/ljk354/FRF_Optim/GP_Jax/NewPlot.py", line 164, in predict_frf
    magnitude, magnitude_std = predict_magnitude(mag_post, mag_ds, X_test, meta)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/ljk354/FRF_Optim/GP_Jax/NewPlot.py", line 79, in predict_magnitude
    latent_dist     = opt_posterior.predict(X_test, train_data=train_dataset)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ljk354/.local/lib/python3.12/site-packages/equinox/_module/_prebuilt.py", line 46, in __call__
    return self.__func__(self.__self__, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ljk354/.local/lib/python3.12/site-packages/gpjax/gps.py", line 596, in predict
    noise = self.likelihood.noise_vector(train_data.n)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ljk354/.local/lib/python3.12/site-packages/equinox/_module/_prebuilt.py", line 46, in __call__
    return self.__func__(self.__self__, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ljk354/.local/lib/python3.12/site-packages/gpjax/likelihoods.py", line 377, in noise_vector
    return jnp.full(n, jnp.square(_val(self.obs_stddev)))
                                  ^^^^^^^^^^^^^^^^^^^^^
  File "/home/ljk354/.local/lib/python3.12/site-packages/gpjax/likelihoods.py", line 51, in _val
    return x.unwrap() if isinstance(x, AbstractUnwrappable) else x
           ^^^^^^^^^^
  File "/home/ljk354/.local/lib/python3.12/site-packages/equinox/_module/_prebuilt.py", line 46, in __call__
    return self.__func__(self.__self__, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ljk354/.local/lib/python3.12/site-packages/gpjax/parameters.py", line 53, in unwrap
    return biject_to(self._constraint)(self._unconstrained)
                                       ^^^^^^^^^^^^^^^^^^^
  File "/home/ljk354/.local/lib/python3.12/site-packages/equinox/_module/_module.py", line 586, in __getattribute__
    out = super().__getattribute__(name)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'NonNegativeReal' object has no attribute '_unconstrained'. Did you mean: '_constraint'?
