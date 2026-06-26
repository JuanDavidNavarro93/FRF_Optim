  0%|          | 0/50 [00:00<?, ?it/s]Compiling...:   0%|          | 0/50 [00:00<?, ?it/s]Running:  20%|██        | 10/50 [00:03<00:13,  3.06it/s]Running:  20%|██        | 10/50 [00:03<00:13,  3.06it/s, Value=13070.30]Running:  40%|████      | 20/50 [00:10<00:16,  1.84it/s, Value=13070.30]Running:  40%|████      | 20/50 [00:10<00:16,  1.84it/s, Value=12671.93]Running:  60%|██████    | 30/50 [00:17<00:12,  1.64it/s, Value=12671.93]Running:  60%|██████    | 30/50 [00:17<00:12,  1.64it/s, Value=12322.03]Running:  80%|████████  | 40/50 [00:24<00:06,  1.55it/s, Value=12322.03]Running:  80%|████████  | 40/50 [00:24<00:06,  1.55it/s, Value=12031.94]Running: 100%|██████████| 50/50 [00:31<00:00,  1.51it/s, Value=12031.94]Running: 100%|██████████| 50/50 [00:31<00:00,  1.51it/s, Value=11813.74]Running: 100%|██████████| 50/50 [00:36<00:00,  1.36it/s, Value=11813.74]
  0%|          | 0/50 [00:00<?, ?it/s]Compiling...:   0%|          | 0/50 [00:00<?, ?it/s]Running:  20%|██        | 10/50 [00:10<00:43,  1.10s/it]Running:  20%|██        | 10/50 [00:10<00:43,  1.10s/it, Value=26072.86]Running:  40%|████      | 20/50 [00:21<00:31,  1.06s/it, Value=26072.86]Running:  40%|████      | 20/50 [00:21<00:31,  1.06s/it, Value=21374.00]Running:  60%|██████    | 30/50 [00:31<00:20,  1.05s/it, Value=21374.00]Running:  60%|██████    | 30/50 [00:31<00:20,  1.05s/it, Value=19854.00]Running:  80%|████████  | 40/50 [00:42<00:10,  1.04s/it, Value=19854.00]Running:  80%|████████  | 40/50 [00:42<00:10,  1.04s/it, Value=18863.92]Running: 100%|██████████| 50/50 [00:52<00:00,  1.04s/it, Value=18863.92]Running: 100%|██████████| 50/50 [00:52<00:00,  1.04s/it, Value=18037.61]Running: 100%|██████████| 50/50 [01:00<00:00,  1.22s/it, Value=18037.61]
Traceback (most recent call last):
  File "/work/ljk354/FRF_Optim/GP_Jax/Bivar_Cuda_GP.py", line 1015, in <module>
    fig_surf = plot_frf_surface(
        rho_values,
    ...<5 lines>...
        thick_dense=np.linspace(thick_values.min(), thick_values.max(), 25),
    )
  File "/work/ljk354/FRF_Optim/GP_Jax/Bivar_Cuda_GP.py", line 899, in plot_frf_surface
    H = frf_tensor[i,thick_idx,:]
        ~~~~~~~~~~^^^^^^^^^^^^^^^
IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed
