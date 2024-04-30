# Stable API

```{eval-rst}
.. currentmodule:: neuralgcm
```

## PressureLevelModel

The user-facing API for NeuralGCM models centers around `PressureLevelModel`:

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    PressureLevelModel
```

### Constructor

Use this class method to create a new model:

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    PressureLevelModel.from_checkpoint
```

### Properties

These properties describe the coordinate system on which a model is defined:

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    PressureLevelModel.timestep
    PressureLevelModel.data_coords
    PressureLevelModel.model_coords
```

### Learned methods

These method use trained model parameters to convert from input variables
defined on data coordinates (i.e., pressure levels) to internal model state
variables defined on model coordinates (i.e., sigma levels) and back.

`advance` and `unroll` allow for stepping forward in time.

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    PressureLevelModel.encode
    PressureLevelModel.decode
    PressureLevelModel.advance
    PressureLevelModel.unroll
```

### Unit conversion

The internal state of NeuralGCM models uses non-dimensional units and
"simulation time," instead of SI and `numpy.datetime64`. These utilities allow
for converting arrays back and forth, including inside JAX code:

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    PressureLevelModel.to_nondim_units
    PressureLevelModel.from_nondim_units
    PressureLevelModel.datetime64_to_sim_time
    PressureLevelModel.sim_time_to_datetime64
```

### Xarray conversion

Xarray is convenient for data preparation and evaluation, but is not compatible
with JAX. Use these methods to convert between `xarray.Dataset` objects and
inputs/outputs from learned methods:

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    PressureLevelModel.inputs_from_xarray
    PressureLevelModel.forcings_from_xarray
    PressureLevelModel.data_from_xarray
    PressureLevelModel.data_to_xarray
```

## Demo dataset & models

These constructors are useful for testing purposes, to avoid the need to load
large datasets from cloud storage. Instead, they rely on small test datasets
packaged with the `neuralgcm` code.

For non-testing purposes, see the model checkpoints from the paper in the
{doc}`./inference_demo`.

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    demo.load_data
    demo.load_checkpoint_tl63_stochastic
```
