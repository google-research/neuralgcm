# Installation

For best performance, we recommend running NeuralGCM models on a computer with
an attached GPU or TPU. Otherwise, performance will be very slow.

You can install NeuralGCM from source using pip, which should automatically
install its dependencies, including [JAX](https://github.com/google/jax):
```
pip install git+https://github.com/google-research/neuralgcm
```

Note that NeuralGCM requires the development version of
[Dinosaur](https://github.com/google-research/dinosuar), so if you update
NeuralGCM you will probably need to update Dinosaur, too.
