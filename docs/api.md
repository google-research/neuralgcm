# Stable API

```{eval-rst}
.. currentmodule:: neuralgcm
```

## Model

The user-facing API for NeuralGCM models centers around `Model`:

```{eval-rst}
.. autoclass:: Model
```

### Constructor

Use this class method to create a new model:

```{eval-rst}
.. automethod:: Model.from_checkpoint
```

### Properties

These properties describe the coordinate system and variables for which a model
is defined:

```{eval-rst}
.. autoproperty:: Model.timestep
.. autoproperty:: Model.data_coords
.. autoproperty:: Model.model_coords
.. autoproperty:: Model.input_variables
.. autoproperty:: Model.forcing_variables
```

### Learned methods

These method use trained model parameters to convert from input variables
defined on data coordinates (i.e., pressure levels) to internal model state
variables defined on model coordinates (i.e., sigma levels) and back.

`advance` and `unroll` allow for stepping forward in time.

```{eval-rst}
.. automethod:: Model.encode
.. automethod:: Model.decode
.. automethod:: Model.advance
.. automethod:: Model.unroll
```

### Unit conversion

The internal state of NeuralGCM models uses non-dimensional units and
"simulation time," instead of SI and `numpy.datetime64`. These utilities allow
for converting arrays back and forth, including inside JAX code:

```{eval-rst}
.. automethod:: Model.to_nondim_units
.. automethod:: Model.from_nondim_units
.. automethod:: Model.datetime64_to_sim_time
.. automethod:: Model.sim_time_to_datetime64
```

### Xarray conversion

Xarray is convenient for data preparation and evaluation, but is not compatible
with JAX. Use these methods to convert between `xarray.Dataset` objects and
inputs/outputs from learned methods:

```{eval-rst}
.. automethod:: Model.inputs_from_xarray
.. automethod:: Model.forcings_from_xarray
.. automethod:: Model.data_from_xarray
.. automethod:: Model.data_to_xarray
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
