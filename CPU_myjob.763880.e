Traceback (most recent call last):
  File "/work/ljk354/FRF_Optim/GP_Jax/MyOptimization.py", line 224, in <module>
    model = load_model(MODEL_PREFIX)
            ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/ljk354/FRF_Optim/GP_Jax/MyOptimization.py", line 179, in load_model
    data    = np.load(prefix + ".pkl")
              ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ljk354/.local/lib/python3.12/site-packages/numpy/lib/_npyio_impl.py", line 486, in load
    raise ValueError(
ValueError: This file contains pickled (object) data. If you trust the file you can load it unsafely using the `allow_pickle=` keyword argument or `pickle.load()`.
