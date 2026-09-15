# Supporting utilities

## Frame sources

### Frame

```{eval-rst}
.. autoclass:: revelsMD.frame_sources.Frame
   :members:
   :show-inheritance:
```

### contiguous\_blocks

```{eval-rst}
.. autofunction:: revelsMD.frame_sources.contiguous_blocks
```

### interleaved\_blocks

```{eval-rst}
.. autofunction:: revelsMD.frame_sources.interleaved_blocks
```

## Cell geometry

```{eval-rst}
.. autofunction:: revelsMD.cell.is_orthorhombic
```

```{eval-rst}
.. autofunction:: revelsMD.cell.cartesian_to_fractional
```

```{eval-rst}
.. autofunction:: revelsMD.cell.fractional_to_cartesian
```

```{eval-rst}
.. autofunction:: revelsMD.cell.wrap_fractional
```

```{eval-rst}
.. autofunction:: revelsMD.cell.apply_minimum_image
```

```{eval-rst}
.. autofunction:: revelsMD.cell.inscribed_sphere_radius
```

```{eval-rst}
.. autofunction:: revelsMD.cell.cells_are_compatible
```

## Statistics

### WelfordAccumulator3D

```{eval-rst}
.. autoclass:: revelsMD.statistics.WelfordAccumulator3D
   :members:
   :show-inheritance:
```

### compute\_lambda\_weights

```{eval-rst}
.. autofunction:: revelsMD.statistics.compute_lambda_weights
```

### combine\_estimators

```{eval-rst}
.. autofunction:: revelsMD.statistics.combine_estimators
```

## Backends

### get\_backend

```{eval-rst}
.. autofunction:: revelsMD.backends.get_backend
```

### get\_fft\_workers

```{eval-rst}
.. autofunction:: revelsMD.backends.get_fft_workers
```
