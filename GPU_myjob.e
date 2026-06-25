Traceback (most recent call last):
  File "/work/ljk354/FRF_Optim/GP_Jax/Bivar_Cuda_GP.py", line 724, in <module>
    model = train_full_model(
        rho_values, thick_values,frf_matrix, freq_values,
        num_iters=200,
        learning_rate=0.01,
    )
  File "/work/ljk354/FRF_Optim/GP_Jax/Bivar_Cuda_GP.py", line 704, in train_full_model
    mag_post, mag_hist, mag_ds = train_magnitude_gp(
                                 ~~~~~~~~~~~~~~~~~~^
        X_train, y_mag, num_iters, learning_rate, key_mag)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/ljk354/FRF_Optim/GP_Jax/Bivar_Cuda_GP.py", line 262, in train_magnitude_gp
    print(f"  [Ch1] Initial  MLL = {initial_mll:.4f}")
                                   ^^^^^^^^^^^^^^^^^
  File "/home/ljk354/.conda/envs/CUDA_GP/lib/python3.14/site-packages/jax/_src/array.py", line 333, in __format__
    return format(self._value if self.ndim else self._value[()], format_spec)
                                                ^^^^^^^^^^^
  File "/home/ljk354/.conda/envs/CUDA_GP/lib/python3.14/site-packages/jax/_src/profiler.py", line 384, in wrapper
    return func(*args, **kwargs)
  File "/home/ljk354/.conda/envs/CUDA_GP/lib/python3.14/site-packages/jax/_src/array.py", line 641, in _value
    npy_value, did_copy = self._single_device_array_to_np_array_did_copy()
                          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
jax.errors.JaxRuntimeError: INTERNAL: jaxlib/gpu/solver_handle_pool.cc:37: operation gpusolverDnCreate(&handle) failed: cuSolver internal error
