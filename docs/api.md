# Stable API

```{eval-rst}
.. currentmodule:: neuralgcm
```

## PressureLevelModel

The user-facing API for NeuralGCM models centers around `PressureLevelModel`:

```{eval-rst}
.. autoclass:: PressureLevelModel
```

### Constructor

Use this class method to create a new model:

```{eval-rst}
.. automethod:: PressureLevelModel.from_checkpoint
```

### Properties

These properties describe the coordinate system and variables for which a model
is defined:

```{eval-rst}
.. autoproperty:: PressureLevelModel.timestep
.. autoproperty:: PressureLevelModel.data_coords
.. autoproperty:: PressureLevelModel.model_coords
.. autoproperty:: PressureLevelModel.input_variables
.. autoproperty:: PressureLevelModel.forcing_variables
```

(learned-methods)=
### Learned methods

These method use trained model parameters to convert from input variables
defined on data coordinates (i.e., pressure levels) to internal model state
variables defined on model coordinates (i.e., sigma levels) and back.

`advance` and `unroll` allow for stepping forward in time.

```{eval-rst}
.. automethod:: PressureLevelModel.encode
.. automethod:: PressureLevelModel.decode
.. automethod:: PressureLevelModel.advance
.. automethod:: PressureLevelModel.unroll
```

### Unit conversion

The internal state of NeuralGCM models uses non-dimensional units and
"simulation time," instead of SI and `numpy.datetime64`. These utilities allow
for converting arrays back and forth, including inside JAX code:

```{eval-rst}
.. automethod:: PressureLevelModel.to_nondim_units
.. automethod:: PressureLevelModel.from_nondim_units
.. automethod:: PressureLevelModel.datetime64_to_sim_time
.. automethod:: PressureLevelModel.sim_time_to_datetime64
```

### Xarray conversion

Xarray is convenient for data preparation and evaluation, but is not compatible
with JAX. Use these methods to convert between `xarray.Dataset` objects and
inputs/outputs from learned methods:

```{eval-rst}
.. automethod:: PressureLevelModel.inputs_from_xarray
.. automethod:: PressureLevelModel.forcings_from_xarray
.. automethod:: PressureLevelModel.data_to_xarray
```

## Demo dataset & models

These constructors are useful for testing purposes, to avoid the need to load
large datasets from cloud storage. Instead, they rely on small test datasets
packaged with the `neuralgcm` code.

For non-testing purposes, see the model checkpoints from the paper in the
{doc}`./inference_demo`.

```{eval-rst}
.. autofunction:: neuralgcm.demo.load_data
.. autofunction:: neuralgcm.demo.load_checkpoint_tl63_stochastic
```
