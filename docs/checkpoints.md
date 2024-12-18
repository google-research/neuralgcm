# Pre-trained model checkpoints

A collection of pre-trained model checkpoints are available for download [on Google Cloud Storage](https://console.cloud.google.com/storage/browser/neuralgcm/models) at `gs://neuralgcm/models/`. All checkpoints are distributed by Google under the Creative Commons [Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/) license.

| Reference | Model Name | Path |
| -- | -- | -- |
| [NeuralGCMs for weather and climate](https://www.nature.com/articles/s41586-024-07744-y) | 0.7° deterministic  | `v1/deterministic_0_7_deg.pkl` |
| | 1.4° deterministic  | `v1/deterministic_1_4_deg.pkl` |
| | 2.8° deterministic  | `v1/deterministic_2_8_deg.pkl` |
| | 1.4° stochastic     | `v1/stochastic_1_4_deg.pkl` |
| [NeuralGCMs optimized to predict satellite-based precipitation observations](https://arxiv.org/abs/2412.11973) | 2.8° stochastic (precipitation)  | `v1_precip/stochastic_precip_2_8_deg.pkl` |
| | 2.8° stochastic (evaporation)  | `v1_precip/stochastic_evap_2_8_deg.pkl` |

The best model to use depends on your use-case:

- For weather forecasts, we generally recommend using the 1.4° stochastic model, which has the best performance [on most metrics](https://sites.research.google/weatherbench/), especially for 5+ day forecasts. If you care most about shorter lead times, consider using the 0.7° deterministic model, which is considerably slower but somewhat more accurate for the first few days.

- For seasonal to climate timescales, the higher resolution NeuralGCM models are not stable. You can reliably use the unmodified 1.4° stochastic model for ~6 months, or the 1.4° deterministic model for ~2 years (stability can be enhanced by [fixing mean log surface pressure](./checkpoint_modifications.ipynb)). For multi-decadal climate simulations, the 2.8° stochastic models offer the bonus of predicting precipitation and evaporation separately. Note that in all of these cases, you will need to supply your own forcing data for sea surface temperature and sea ice concentration, because NeuralGCM currently only models the atmosphere.

- For cases where you need to predict precipitation, you can use either the 2.8° stochastic (precipitation) or the 2.8° stochastic (evaporation) model. These models each predict both precipitation and evaporation (the difference is [how they are trained](https://arxiv.org/abs/2412.11973)) and offer different trade-offs: the precipitation model produces realistic cumulative precipitation values at temporal resolutions of 6 hours or greater, but struggles with accuracy at finer (sub-6-hour) resolutions, and the evaporation model can produce slightly negative precipitation rates. For weather forecasts, these models are slightly less accurate than the 1.4° stochastic model, but they are reliably stable for 20 year atmosphere-only climate simulations.

The NeuralGCM team is currently focused on developing improved finer-resolution stochastic models that predict important surface observations (such as precipitation) and that have suitable stability for seasonal forecasts and climate simulation studies.

When new models are released, we will update this page and announce them on the NeuralGCM [mailing list](https://groups.google.com/g/neuralgcm-announce).
