# NeuralGCM documentation

## Overview

The NeuralGCM codebase consists of a handful of different components, suitable
for reproducing and extending results from our paper,
[Neural General Circulation Models for Weather and Climate](https://arxiv.org/abs/2311.07222):

1. **Dynamics**: The atmospheric dynamical core is distributed in the separate
   [Dinosaur](https://github.com/google-research/dinosaur) package.
2. **ML modules**: [Haiku](https://github.com/google-deepmind/dm-haiku) modules
   for defining neural network layers.
3. **ML training**: Pseudo-code for training NeuralGCM models can be found in
   the [`reference_code` subdirectory](https://github.com/google-research/neuralgcm/tree/main/neuralgcm/reference_code).
4. **ML inference**: Code for running forecasts with pre-trained models,
   encapsulated in the {py:class}`~neuralgcm.PressureLevelModel` class.
5. **Evaluation**: Code for evaluating NeuralGCM weather forecasts, along with
   archived re-forecasts for 2020, can be found in the
   [WeatherBench2](https://github.com/google-research/weatherbench2) project.

The documentation here focuses mostly on our API for *inference* (i.e., running
trained NeuralGCM atmospheric models), which we believe is the most immediately
useful part of the NeuralGCM code for third parties. It is also a part of our
code that we can commit to supporting in roughly its current form.

We would love to support training, modifying and fine-tuning NeuralGCM models,
but with the present codebase based on Haiku and
[Gin](https://github.com/google/gin-config) this is much trickier than it needs
to be. We are currently (in May 2024) refactoring the modeling code to improve
usability -- stay tuned!

## Contents

```{toctree}
:maxdepth: 1
installation.md
inference_demo.ipynb
datasets.ipynb
trained_models.ipynb
api.md
```