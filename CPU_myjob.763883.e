/home/ljk354/.local/lib/python3.12/site-packages/gpjax/dataset.py:43: UserWarning: X is not of type float64. Got X.dtype=float32. This may lead to numerical instability. 
  _check_precision(self.X, self.y)
/home/ljk354/.local/lib/python3.12/site-packages/gpjax/dataset.py:43: UserWarning: y is not of type float64.Got y.dtype=float32. This may lead to numerical instability.
  _check_precision(self.X, self.y)
Traceback (most recent call last):
  File "/home/ljk354/.local/lib/python3.12/site-packages/equinox/_serialisation.py", line 57, in _f
    return f(*xs)
           ^^^^^^
  File "/home/ljk354/.local/lib/python3.12/site-packages/equinox/_serialisation.py", line 347, in __deserialise
    return spec(f, y)
           ^^^^^^^^^^
  File "/home/ljk354/.local/lib/python3.12/site-packages/equinox/_serialisation.py", line 148, in default_deserialise_filter_spec
    return jnp.load(f)  # pyright: ignore[reportArgumentType]
           ^^^^^^^^^^^
  File "/home/ljk354/.local/lib/python3.12/site-packages/jax/_src/numpy/lax_numpy.py", line 207, in load
    out = np.load(file, *args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ljk354/.local/lib/python3.12/site-packages/numpy/lib/_npyio_impl.py", line 460, in load
    raise EOFError("No data left in file")
EOFError: No data left in file

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/ljk354/.local/lib/python3.12/site-packages/equinox/_serialisation.py", line 57, in _f
    return f(*xs)
           ^^^^^^
  File "/home/ljk354/.local/lib/python3.12/site-packages/equinox/_serialisation.py", line 349, in _deserialise
    return _ordered_tree_map(__deserialise, x, is_leaf=is_leaf)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ljk354/.local/lib/python3.12/site-packages/equinox/_serialisation.py", line 68, in _ordered_tree_map
    return treedef.unflatten(_f(*xs) for xs in zip(*all_leaves))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ljk354/.local/lib/python3.12/site-packages/equinox/_serialisation.py", line 68, in <genexpr>
    return treedef.unflatten(_f(*xs) for xs in zip(*all_leaves))
                             ^^^^^^^
  File "/home/ljk354/.local/lib/python3.12/site-packages/equinox/_serialisation.py", line 66, in _f
    raise exc from e
equinox._serialisation.TreePathError: Error at leaf with path (GetAttrKey(name='prior'), GetAttrKey(name='kernel'), GetAttrKey(name='kernels'), SequenceKey(idx=0), GetAttrKey(name='variance'), GetAttrKey(name='_unconstrained'))

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/work/ljk354/FRF_Optim/GP_Jax/MyOptimization.py", line 224, in <module>
    model = load_model(MODEL_PREFIX)
            ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/ljk354/FRF_Optim/GP_Jax/MyOptimization.py", line 195, in load_model
    mag_post  = eqx.tree_deserialise_leaves(prefix + "_mag_post.eqx", mag_post)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ljk354/.local/lib/python3.12/site-packages/equinox/_serialisation.py", line 351, in tree_deserialise_leaves
    out = _ordered_tree_map(_deserialise, filter_spec, like)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ljk354/.local/lib/python3.12/site-packages/equinox/_serialisation.py", line 68, in _ordered_tree_map
    return treedef.unflatten(_f(*xs) for xs in zip(*all_leaves))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ljk354/.local/lib/python3.12/site-packages/equinox/_serialisation.py", line 68, in <genexpr>
    return treedef.unflatten(_f(*xs) for xs in zip(*all_leaves))
                             ^^^^^^^
  File "/home/ljk354/.local/lib/python3.12/site-packages/equinox/_serialisation.py", line 62, in _f
    raise exc from e
equinox._serialisation.TreePathError: Error at leaf with path (GetAttrKey(name='prior'), GetAttrKey(name='kernel'), GetAttrKey(name='kernels'), SequenceKey(idx=0), GetAttrKey(name='variance'), GetAttrKey(name='_unconstrained'))
