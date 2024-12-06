# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Coordinate systems that describe how data & model states are discretized."""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Iterable

from dinosaur import coordinate_systems as dinosaur_coordinates
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import vertical_interpolation
import jax
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental import typing
import numpy as np
from penzai.core import struct


SphericalHarmonicsImpl = spherical_harmonic.SphericalHarmonicsImpl
RealSphericalHarmonics = spherical_harmonic.RealSphericalHarmonics
FastSphericalHarmonics = spherical_harmonic.FastSphericalHarmonics
P = jax.sharding.PartitionSpec


@dataclasses.dataclass
class ArrayKey:
  """Wrapper for a numpy array to make it hashable."""

  value: np.ndarray

  def __eq__(self, other):
    return (
        isinstance(self, ArrayKey)
        and self.value.dtype == other.value.dtype
        and self.value.shape == other.value.shape
        and (self.value == other.value).all()
    )

  def __ne__(self, other):
    return not self == other

  def __hash__(self) -> int:
    return hash((self.value.shape, self.value.tobytes()))


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class TimeDelta(cx.Coordinate):
  """Coordinates that discretize data along static relative time."""

  time: np.ndarray = dataclasses.field(metadata={'pytree_node': False})
  offset: float = dataclasses.field(
      default=0.0, metadata={'pytree_node': False}
  )

  @property
  def dims(self):
    return ('timedelta',)

  @property
  def shape(self):
    return self.time.shape

  @property
  def fields(self):
    return {'timedelta': cx.wrap(self.time, self)}

  @classmethod
  def as_index(cls, axis_size: int, offset: float = 0.0) -> TimeDelta:
    return cls(time=np.arange(axis_size), offset=offset)

  def _components(self):
    return (self.offset, ArrayKey(self.time))

  def tree_flatten(self):
    """Flattens TimeDelta."""
    aux_data = self._components()
    return (), aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, leaves):
    """Unflattens TimeDelta."""
    del leaves  # unused
    offset, array_key = aux_data
    return cls(offset=offset, time=array_key.value)

  def __eq__(self, other):
    return (
        isinstance(other, TimeDelta)
        and self._components() == other._components()
    )

  def __ne__(self, other):
    return not self == other

  def __hash__(self) -> int:
    return hash(self._components())


#
# Grid-like and spherical harmonic coordinate systems
#


# TODO(dkochkov) Consider leaving out spherical_harmonics_impl from repr.
@struct.pytree_dataclass
class LonLatGrid(cx.Coordinate, struct.Struct):
  """Coordinates that discretize data as point values on lon-lat grid."""

  longitude_nodes: int = dataclasses.field(metadata={'pytree_node': False})
  latitude_nodes: int = dataclasses.field(metadata={'pytree_node': False})
  latitude_spacing: str = dataclasses.field(
      default='gauss', metadata={'pytree_node': False}
  )
  longitude_offset: float = dataclasses.field(
      default=0.0, metadata={'pytree_node': False}
  )
  radius: float | None = dataclasses.field(
      default=None, metadata={'pytree_node': False}
  )
  spherical_harmonics_impl: SphericalHarmonicsImpl = dataclasses.field(
      default=FastSphericalHarmonics,
      kw_only=True,
      metadata={'pytree_node': False},
  )
  longitude_wavenumbers: int = dataclasses.field(
      default=0, repr=False, kw_only=True, metadata={'pytree_node': False}
  )
  total_wavenumbers: int = dataclasses.field(
      default=0, repr=False, kw_only=True, metadata={'pytree_node': False}
  )
  _ylm_grid: spherical_harmonic.Grid = dataclasses.field(
      init=False, repr=False, metadata={'pytree_node': False}
  )

  def __post_init__(self):
    ylm_grid = spherical_harmonic.Grid(
        longitude_wavenumbers=self.longitude_wavenumbers,
        total_wavenumbers=self.total_wavenumbers,
        longitude_nodes=self.longitude_nodes,
        latitude_nodes=self.latitude_nodes,
        latitude_spacing=self.latitude_spacing,
        longitude_offset=self.longitude_offset,
        radius=self.radius,
        spherical_harmonics_impl=self.spherical_harmonics_impl,
    )
    object.__setattr__(self, '_ylm_grid', ylm_grid)

  @property
  def ylm_grid(self) -> spherical_harmonic.Grid:
    return self._ylm_grid

  @property
  def dims(self):
    return ('longitude', 'latitude')

  @property
  def shape(self):
    return self.ylm_grid.nodal_shape

  @property
  def fields(self):
    return {
        k: cx.wrap(v, cx.SelectedAxis(self, i))
        for i, (k, v) in enumerate(zip(self.dims, self.ylm_grid.nodal_axes))
    }

  def to_spherical_harmonic_grid(
      self,
      longitude_wavenumbers: int | None = None,
      total_wavenumbers: int | None = None,
      spherical_harmonics_impl: SphericalHarmonicsImpl | None = None,
  ):
    """Constructs a `SphericalHarmonicGrid` from the `LonLatGrid`."""
    if longitude_wavenumbers is None:
      longitude_wavenumbers = self.longitude_wavenumbers
    if total_wavenumbers is None:
      total_wavenumbers = self.total_wavenumbers
    if spherical_harmonics_impl is None:
      spherical_harmonics_impl = self.spherical_harmonics_impl
    return SphericalHarmonicGrid(
        longitude_wavenumbers=longitude_wavenumbers,
        total_wavenumbers=total_wavenumbers,
        longitude_offset=self.longitude_offset,
        radius=self.radius,
        spherical_harmonics_impl=spherical_harmonics_impl,
        longitude_nodes=self.longitude_nodes,
        latitude_nodes=self.latitude_nodes,
        latitude_spacing=self.latitude_spacing,
    )

  @classmethod
  def from_dinosaur_grid(
      cls,
      ylm_grid: spherical_harmonic.Grid,
  ):
    return cls(
        longitude_nodes=ylm_grid.longitude_nodes,
        latitude_nodes=ylm_grid.latitude_nodes,
        latitude_spacing=ylm_grid.latitude_spacing,
        longitude_offset=ylm_grid.longitude_offset,
        radius=ylm_grid.radius,
        total_wavenumbers=ylm_grid.total_wavenumbers,
        longitude_wavenumbers=ylm_grid.longitude_wavenumbers,
        spherical_harmonics_impl=ylm_grid.spherical_harmonics_impl,
    )

  @classmethod
  def construct(
      cls,
      max_wavenumber: int,
      gaussian_nodes: int,
      latitude_spacing: str = 'gauss',
      longitude_offset: float = 0.0,
      radius: float | None = None,
      spherical_harmonics_impl: SphericalHarmonicsImpl = FastSphericalHarmonics,
  ) -> LonLatGrid:
    """Constructs a `LonLatGrid` compatible with max_wavenumber and nodes.

    Args:
      max_wavenumber: maximum wavenumber to resolve.
      gaussian_nodes: number of nodes on the Gaussian grid between the equator
        and a pole.
      latitude_spacing: either 'gauss' or 'equiangular'. This determines the
        spacing of nodal grid points in the latitudinal (north-south) direction.
      longitude_offset: the value of the first longitude node, in radians.
      radius: radius of the sphere. If `None` a default values of `1` is used.
      spherical_harmonics_impl: class providing an implementation of spherical
        harmonics.

    Returns:
      Constructed LonLatGrid object.
    """
    ylm_grid = spherical_harmonic.Grid(
        longitude_wavenumbers=max_wavenumber + 1,
        total_wavenumbers=max_wavenumber + 2,
        longitude_nodes=4 * gaussian_nodes,
        latitude_nodes=2 * gaussian_nodes,
        latitude_spacing=latitude_spacing,
        longitude_offset=longitude_offset,
        spherical_harmonics_impl=spherical_harmonics_impl,
        radius=radius,
    )
    return cls.from_dinosaur_grid(ylm_grid=ylm_grid)

  # The factory methods below return "standard" grids that appear in the
  # literature. See, e.g. https://doi.org/10.5194/tc-12-1499-2018 and
  # https://www.ecmwf.int/en/forecasts/documentation-and-support/data-spatial-coordinate-systems

  # The number in these names correspond to the maximum resolved wavenumber,
  # which is one less than the number of wavenumbers used in the Grid
  # constructor. An additional total wavenumber is added because the top
  # wavenumber is clipped from the initial state and each calculation of
  # explicit tendencies.

  # The names for these factory methods (including capilatization) are
  # standard in the literature.
  # pylint:disable=invalid-name

  # T* grids can model quadratic terms without aliasing, because the maximum
  # total wavenumber is <= 2/3 of the number of latitudinal nodes. ECMWF
  # sometimes calls these "TQ" (truncated quadratic) grids.

  @classmethod
  def T21(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=21, gaussian_nodes=16, **kwargs)

  @classmethod
  def T31(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=31, gaussian_nodes=24, **kwargs)

  @classmethod
  def T42(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=42, gaussian_nodes=32, **kwargs)

  @classmethod
  def T85(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=85, gaussian_nodes=64, **kwargs)

  @classmethod
  def T106(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=106, gaussian_nodes=80, **kwargs)

  @classmethod
  def T119(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=119, gaussian_nodes=90, **kwargs)

  @classmethod
  def T170(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=170, gaussian_nodes=128, **kwargs)

  @classmethod
  def T213(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=213, gaussian_nodes=160, **kwargs)

  @classmethod
  def T340(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=340, gaussian_nodes=256, **kwargs)

  @classmethod
  def T425(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=425, gaussian_nodes=320, **kwargs)

  # TL* grids do not truncate any frequencies, and hence can only model linear
  # terms exactly. ECMWF used "TL" (truncated linear) grids for semi-Lagrangian
  # advection (which eliminates quadratic terms) up to 2016, which it switched
  # to "cubic" grids for resolutions above TL1279:
  # https://www.ecmwf.int/sites/default/files/elibrary/2016/17262-new-grid-ifs.pdf

  @classmethod
  def TL31(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=31, gaussian_nodes=16, **kwargs)

  @classmethod
  def TL47(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=47, gaussian_nodes=24, **kwargs)

  @classmethod
  def TL63(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=63, gaussian_nodes=32, **kwargs)

  @classmethod
  def TL95(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=95, gaussian_nodes=48, **kwargs)

  @classmethod
  def TL127(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=127, gaussian_nodes=64, **kwargs)

  @classmethod
  def TL159(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=159, gaussian_nodes=80, **kwargs)

  @classmethod
  def TL179(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=179, gaussian_nodes=90, **kwargs)

  @classmethod
  def TL255(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=255, gaussian_nodes=128, **kwargs)

  @classmethod
  def TL639(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=639, gaussian_nodes=320, **kwargs)

  @classmethod
  def TL1279(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=1279, gaussian_nodes=640, **kwargs)


@struct.pytree_dataclass
class SphericalHarmonicGrid(cx.Coordinate, struct.Struct):
  """Coordinates that discretize data as spherical harmonic coefficients."""

  longitude_wavenumbers: int = dataclasses.field(
      metadata={'pytree_node': False}
  )
  total_wavenumbers: int = dataclasses.field(metadata={'pytree_node': False})
  longitude_offset: float = dataclasses.field(
      default=0.0, metadata={'pytree_node': False}
  )
  radius: float | None = dataclasses.field(
      default=None, metadata={'pytree_node': False}
  )
  spherical_harmonics_impl: SphericalHarmonicsImpl = dataclasses.field(
      default=FastSphericalHarmonics,
      kw_only=True,
      metadata={'pytree_node': False},
  )
  longitude_nodes: int = dataclasses.field(
      default=0, repr=False, kw_only=True, metadata={'pytree_node': False}
  )
  latitude_nodes: int = dataclasses.field(
      default=0, repr=False, kw_only=True, metadata={'pytree_node': False}
  )
  latitude_spacing: str = dataclasses.field(
      default='gauss', repr=False, kw_only=True, metadata={'pytree_node': False}
  )
  _ylm_grid: spherical_harmonic.Grid = dataclasses.field(
      init=False, repr=False, metadata={'pytree_node': False}
  )

  def __post_init__(self):
    ylm_grid = spherical_harmonic.Grid(
        longitude_wavenumbers=self.longitude_wavenumbers,
        total_wavenumbers=self.total_wavenumbers,
        longitude_offset=self.longitude_offset,
        radius=self.radius,
        spherical_harmonics_impl=self.spherical_harmonics_impl,
        longitude_nodes=self.longitude_nodes,
        latitude_nodes=self.latitude_nodes,
        latitude_spacing=self.latitude_spacing,
    )
    object.__setattr__(self, '_ylm_grid', ylm_grid)

  @property
  def ylm_grid(self) -> spherical_harmonic.Grid:
    return self._ylm_grid

  @property
  def dims(self):
    return ('longitude_wavenumber', 'total_wavenumber')

  @property
  def shape(self) -> tuple[int, ...]:
    return self.ylm_grid.modal_shape

  @property
  def fields(self):
    return {
        k: cx.wrap(v, cx.SelectedAxis(self, i))
        for i, (k, v) in enumerate(zip(self.dims, self.ylm_grid.modal_axes))
    }

  def to_lon_lat_grid(
      self,
      longitude_nodes: int | None = None,
      latitude_nodes: int | None = None,
      latitude_spacing: str | None = None,
  ):
    """Constructs a `LonLatGrid` from the `SphericalHarmonicGrid`."""
    if longitude_nodes is None:
      longitude_nodes = self.longitude_nodes
    if latitude_nodes is None:
      latitude_nodes = self.latitude_nodes
    if latitude_spacing is None:
      latitude_spacing = self.latitude_spacing
    return LonLatGrid(
        longitude_nodes=longitude_nodes,
        latitude_nodes=latitude_nodes,
        latitude_spacing=latitude_spacing,
        longitude_offset=self.longitude_offset,
        radius=self.radius,
        spherical_harmonics_impl=self.spherical_harmonics_impl,
        longitude_wavenumbers=self.longitude_wavenumbers,
        total_wavenumbers=self.total_wavenumbers,
    )

  @classmethod
  def from_dinosaur_grid(
      cls,
      ylm_grid: spherical_harmonic.Grid,
  ):
    return cls(
        longitude_wavenumbers=ylm_grid.longitude_wavenumbers,
        total_wavenumbers=ylm_grid.total_wavenumbers,
        longitude_offset=ylm_grid.longitude_offset,
        radius=ylm_grid.radius,
        spherical_harmonics_impl=ylm_grid.spherical_harmonics_impl,
        longitude_nodes=ylm_grid.longitude_nodes,
        latitude_nodes=ylm_grid.latitude_nodes,
        latitude_spacing=ylm_grid.latitude_spacing,
    )

  @classmethod
  def with_wavenumbers(
      cls,
      longitude_wavenumbers: int,
      dealiasing: str = 'quadratic',
      latitude_spacing: str = 'gauss',
      longitude_offset: float = 0.0,
      spherical_harmonics_impl: SphericalHarmonicsImpl = (
          FastSphericalHarmonics
      ),
      radius: float | None = None,
  ) -> SphericalHarmonicGrid:
    """Constructs a `SphericalHarmonicGrid` by specifying only wavenumbers."""
    # The number of nodes is chosen for de-aliasing.
    order = {'linear': 2, 'quadratic': 3, 'cubic': 4}[dealiasing]
    longitude_nodes = order * longitude_wavenumbers + 1
    latitude_nodes = math.ceil(longitude_nodes / 2)
    ylm_grid = spherical_harmonic.Grid(
        longitude_wavenumbers=longitude_wavenumbers,
        total_wavenumbers=longitude_wavenumbers + 1,
        longitude_nodes=longitude_nodes,
        latitude_nodes=latitude_nodes,
        latitude_spacing=latitude_spacing,
        longitude_offset=longitude_offset,
        spherical_harmonics_impl=spherical_harmonics_impl,
        radius=radius,
    )
    return cls.from_dinosaur_grid(ylm_grid=ylm_grid)

  @classmethod
  def construct(
      cls,
      max_wavenumber: int,
      gaussian_nodes: int,
      latitude_spacing: str = 'gauss',
      longitude_offset: float = 0.0,
      radius: float | None = None,
      spherical_harmonics_impl: SphericalHarmonicsImpl = (
          FastSphericalHarmonics
      ),
  ) -> SphericalHarmonicGrid:
    """Constructs a `SphericalHarmonicGrid` with max_wavenumber.

    Args:
      max_wavenumber: maximum wavenumber to resolve.
      gaussian_nodes: number of nodes on the Gaussian grid between the equator
        and a pole.
      latitude_spacing: either 'gauss' or 'equiangular'. This determines the
        spacing of nodal grid points in the latitudinal (north-south) direction.
      longitude_offset: the value of the first longitude node, in radians.
      radius: radius of the sphere. If `None` a default values of `1` is used.
      spherical_harmonics_impl: class providing an implementation of spherical
        harmonics.

    Returns:
      Constructed SphericalHarmonicGrid object.
    """
    ylm_grid = spherical_harmonic.Grid(
        longitude_wavenumbers=max_wavenumber + 1,
        total_wavenumbers=max_wavenumber + 2,
        longitude_nodes=4 * gaussian_nodes,
        latitude_nodes=2 * gaussian_nodes,
        latitude_spacing=latitude_spacing,
        longitude_offset=longitude_offset,
        spherical_harmonics_impl=spherical_harmonics_impl,
        radius=radius,
    )
    return cls.from_dinosaur_grid(ylm_grid=ylm_grid)

  # The factory methods below return "standard" grids that appear in the
  # literature. See, e.g. https://doi.org/10.5194/tc-12-1499-2018 and
  # https://www.ecmwf.int/en/forecasts/documentation-and-support/data-spatial-coordinate-systems

  # The number in these names correspond to the maximum resolved wavenumber,
  # which is one less than the number of wavenumbers used in the Grid
  # constructor. An additional total wavenumber is added because the top
  # wavenumber is clipped from the initial state and each calculation of
  # explicit tendencies.

  # The names for these factory methods (including capilatization) are
  # standard in the literature.
  # pylint:disable=invalid-name

  # T* grids can model quadratic terms without aliasing, because the maximum
  # total wavenumber is <= 2/3 of the number of latitudinal nodes. ECMWF
  # sometimes calls these "TQ" (truncated quadratic) grids.

  @classmethod
  def T21(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=21, gaussian_nodes=16, **kwargs)

  @classmethod
  def T31(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=31, gaussian_nodes=24, **kwargs)

  @classmethod
  def T42(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=42, gaussian_nodes=32, **kwargs)

  @classmethod
  def T85(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=85, gaussian_nodes=64, **kwargs)

  @classmethod
  def T106(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=106, gaussian_nodes=80, **kwargs)

  @classmethod
  def T119(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=119, gaussian_nodes=90, **kwargs)

  @classmethod
  def T170(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=170, gaussian_nodes=128, **kwargs)

  @classmethod
  def T213(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=213, gaussian_nodes=160, **kwargs)

  @classmethod
  def T340(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=340, gaussian_nodes=256, **kwargs)

  @classmethod
  def T425(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=425, gaussian_nodes=320, **kwargs)

  # TL* grids do not truncate any frequencies, and hence can only model linear
  # terms exactly. ECMWF used "TL" (truncated linear) grids for semi-Lagrangian
  # advection (which eliminates quadratic terms) up to 2016, which it switched
  # to "cubic" grids for resolutions above TL1279:
  # https://www.ecmwf.int/sites/default/files/elibrary/2016/17262-new-grid-ifs.pdf

  @classmethod
  def TL31(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=31, gaussian_nodes=16, **kwargs)

  @classmethod
  def TL47(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=47, gaussian_nodes=24, **kwargs)

  @classmethod
  def TL63(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=63, gaussian_nodes=32, **kwargs)

  @classmethod
  def TL95(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=95, gaussian_nodes=48, **kwargs)

  @classmethod
  def TL127(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=127, gaussian_nodes=64, **kwargs)

  @classmethod
  def TL159(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=159, gaussian_nodes=80, **kwargs)

  @classmethod
  def TL179(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=179, gaussian_nodes=90, **kwargs)

  @classmethod
  def TL255(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=255, gaussian_nodes=128, **kwargs)

  @classmethod
  def TL639(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=639, gaussian_nodes=320, **kwargs)

  @classmethod
  def TL1279(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=1279, gaussian_nodes=640, **kwargs)


#
# Vertical level coordinates
#


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class SigmaLevels(cx.Coordinate):
  """Coordinates that discretize data as fraction of the surface pressure."""

  boundaries: np.ndarray = dataclasses.field(metadata={'pytree_node': False})
  sigma_levels: sigma_coordinates.SigmaCoordinates = dataclasses.field(
      init=False,
      repr=False,
      compare=False,
      metadata={'pytree_node': False},
  )

  def __init__(self, boundaries: Iterable[float] | np.ndarray):
    boundaries = np.asarray(boundaries)
    self.boundaries = boundaries
    self.__post_init__()

  def __post_init__(self):
    sigma_levels = sigma_coordinates.SigmaCoordinates(
        boundaries=self.boundaries
    )
    object.__setattr__(self, 'sigma_levels', sigma_levels)

  @property
  def dims(self):
    return ('sigma',)

  @property
  def shape(self) -> tuple[int, ...]:
    return self.sigma_levels.centers.shape

  @property
  def fields(self):
    return {'sigma': cx.wrap(self.sigma_levels.centers, self)}

  def asdict(self) -> dict[str, Any]:
    return {k: v.tolist() for k, v in dataclasses.asdict(self).items()}

  def _components(self):
    return (ArrayKey(self.boundaries),)

  def tree_flatten(self):
    """Flattens SigmaLevels."""
    aux_data = self._components()
    return (), aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, leaves):
    """Unflattens SigmaLevels."""
    del leaves  # unused
    (boundaries_key,) = aux_data
    return cls(boundaries=boundaries_key.value)

  def __eq__(self, other):
    return (
        isinstance(other, SigmaLevels)
        and self._components() == other._components()
    )

  def __ne__(self, other):
    return not self == other

  def __hash__(self) -> int:
    return hash(self._components())

  @classmethod
  def from_dinosaur_sigma_levels(
      cls,
      sigma_levels: sigma_coordinates.SigmaCoordinates,
  ):
    return cls(boundaries=sigma_levels.boundaries)

  @classmethod
  def equidistant(
      cls,
      layers: int,
  ) -> SigmaLevels:
    sigma_levels = sigma_coordinates.SigmaCoordinates.equidistant(layers)
    boundaries = sigma_levels.boundaries
    return cls(boundaries=boundaries)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class PressureLevels(cx.Coordinate):
  """Coordinates that discretize data per pressure levels."""

  centers: np.ndarray = dataclasses.field(metadata={'pytree_node': False})
  pressure_levels: vertical_interpolation.PressureCoordinates = (
      dataclasses.field(
          init=False,
          repr=False,
          compare=False,
          metadata={'pytree_node': False},
      )
  )

  def __init__(self, centers: Iterable[float] | np.ndarray):
    centers = np.asarray(centers)
    self.centers = centers
    self.__post_init__()

  def __post_init__(self):
    pressure_levels = vertical_interpolation.PressureCoordinates(
        centers=self.centers
    )
    object.__setattr__(self, 'pressure_levels', pressure_levels)

  @property
  def dims(self):
    return ('pressure',)

  @property
  def shape(self) -> tuple[int, ...]:
    return self.centers.shape

  @property
  def fields(self):
    return {'pressure': cx.wrap(self.centers, self)}

  def asdict(self) -> dict[str, Any]:
    return {k: v.tolist() for k, v in dataclasses.asdict(self).items()}

  def _components(self):
    return (ArrayKey(self.centers),)

  def tree_flatten(self):
    """Flattens PressureLevels."""
    aux_data = self._components()
    return (), aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, leaves):
    """Unflattens PressureLevels."""
    del leaves  # unused
    (centers,) = aux_data
    return cls(centers=centers.value)

  def __eq__(self, other):
    return (
        isinstance(other, PressureLevels)
        and self._components() == other._components()
    )

  def __ne__(self, other):
    return not self == other

  def __hash__(self) -> int:
    return hash(self._components())

  @classmethod
  def from_dinosaur_pressure_levels(
      cls,
      pressure_levels: vertical_interpolation.PressureCoordinates,
  ):
    return cls(centers=pressure_levels.centers)


@struct.pytree_dataclass
class LayerLevels(cx.Coordinate, struct.Struct):
  """Coordinates that discretize data by index of unstructured layer."""

  n_layers: int = dataclasses.field(metadata={'pytree_node': False})

  @property
  def dims(self):
    return ('layer_index',)

  @property
  def shape(self) -> tuple[int, ...]:
    return (self.n_layers,)

  @property
  def fields(self):
    return {'layer_index': cx.wrap(np.arange(self.n_layers), self)}


#
# Solver-specific coordinate combinations
#


@struct.pytree_dataclass
class DinosaurCoordinates(cx.CartesianProduct, struct.Struct):
  """Coordinate that is product of horizontal & vertical coorinates.

  This combined coordinate object is useful for compactly keeping track of the
  full coordinate system of the Dinosaur dynamic core or pressure-level
  representation of the spherical shell data.
  """

  coordinates: tuple[cx.Coordinate, ...] = dataclasses.field(
      init=False, metadata={'pytree_node': False}
  )
  horizontal: LonLatGrid | SphericalHarmonicGrid = dataclasses.field(
      metadata={'pytree_node': False}
  )
  vertical: SigmaLevels | PressureLevels | LayerLevels = dataclasses.field(
      metadata={'pytree_node': False}
  )
  dycore_partition_spec: jax.sharding.PartitionSpec = dataclasses.field(
      metadata={'pytree_node': False}, default=P('z', 'x', 'y')
  )
  physics_partition_spec: jax.sharding.PartitionSpec = dataclasses.field(
      metadata={'pytree_node': False}, default=P(None, ('x', 'z'), 'y')
  )

  def __init__(
      self,
      horizontal,
      vertical,
      dycore_partition_spec: jax.sharding.PartitionSpec = P('z', 'x', 'y'),
      physics_partition_spec: jax.sharding.PartitionSpec = P(
          None, ('x', 'z'), 'y'
      ),
  ):
    super().__init__(coordinates=(vertical, horizontal))
    self.horizontal = horizontal
    self.vertical = vertical
    self.dycore_partition_spec = dycore_partition_spec
    self.physics_partition_spec = physics_partition_spec

  @property
  def dims(self):
    return self.vertical.dims + self.horizontal.dims

  @property
  def shape(self) -> tuple[int, ...]:
    return self.vertical.shape + self.horizontal.shape

  @property
  def fields(self):
    return self.vertical.fields | self.horizontal.fields

  @property
  def dinosaur_coords(self):
    """Returns the CoordinateSystem object from the Dinosaur package."""
    # TODO(dkochkov) Either make spmd_mesh an argument or ideally add
    # ShardingMesh object to the new API to hold sharding information.
    spmd_mesh = None  # make this an argument and change property to a method.
    horizontal, vertical = self.horizontal, self.vertical
    horizontal = horizontal.ylm_grid
    if isinstance(vertical, SigmaLevels):
      vertical = vertical.sigma_levels
    elif isinstance(vertical, PressureLevels):
      vertical = vertical.pressure_levels
    elif isinstance(vertical, LayerLevels):
      pass
    else:
      raise ValueError(f'Unsupported vertical {vertical=}')
    return dinosaur_coordinates.CoordinateSystem(
        horizontal=horizontal, vertical=vertical, spmd_mesh=spmd_mesh
    )

  @property
  def dinosaur_grid(self):
    return self.dinosaur_coords.horizontal

  @classmethod
  def from_dinosaur_coords(
      cls,
      coords: dinosaur_coordinates.CoordinateSystem,
  ):
    """Constructs instance from coordinates in Dinosaur package."""
    horizontal = LonLatGrid.from_dinosaur_grid(coords.horizontal)
    if isinstance(coords.vertical, sigma_coordinates.SigmaCoordinates):
      vertical = SigmaLevels.from_dinosaur_sigma_levels(coords.vertical)
    elif isinstance(
        coords.vertical, vertical_interpolation.PressureCoordinates
    ):
      vertical = PressureLevels.from_dinosaur_pressure_levels(coords.vertical)
    else:
      raise ValueError(f'Unsupported vertical {coords.vertical=}')
    return cls(horizontal=horizontal, vertical=vertical)


#
# Helper functions.
#
# TODO(dkochkov) Refactor/remove helpers below to coordax.coords.


def consistent_coords(*inputs) -> cx.Coordinate:
  """Returns the unique coordinate, or raises a ValueError."""
  if not all([cx.is_positional_prefix_field(f) for f in inputs]):
    raise ValueError(f'{inputs=} are not in positional prefix order.')
  dim_names = {tuple(f.named_shape.keys()) for f in inputs}
  if len(dim_names) != 1:
    raise ValueError(f'Found non-unique {dim_names=} in inputs.')
  (dim_names,) = dim_names
  coords = {
      cx.compose_coordinates(*[f.coords[k] for k in dim_names]) for f in inputs
  }
  if len(coords) != 1:
    raise ValueError(f'Found non-unique {coords=} in inputs.')
  (coords,) = coords
  return coords


def modal_shape_to_nodal_shape(
    modal_shape: typing.Pytree,
    grid: SphericalHarmonicGrid,
) -> typing.Pytree:
  """Returns the nodal shape corresponding to the given modal shape."""
  # Note: here modal_shape is expected to be a pytree with
  # typing.ShapeDtypeStruct leaves, so that eval_shape works as expected.
  to_nodal_shape = lambda x: jax.eval_shape(grid.ylm_grid.to_nodal, x)
  return jax.tree.map(to_nodal_shape, modal_shape)


def nodal_shape_to_modal_shape(
    nodal_shape: typing.Pytree,
    grid: SphericalHarmonicGrid,
) -> typing.Pytree:
  """Returns the modal shape corresponding to the given nodal shape."""
  # Note: here nodal_shape is expected to be a pytree with
  # typing.ShapeDtypeStruct leaves, so that eval_shape works as expected.
  to_modal_shape = lambda x: jax.eval_shape(grid.ylm_grid.to_modal, x)
  return jax.tree.map(to_modal_shape, nodal_shape)


def split_field_attrs(pytree):
  """Splits pytree of `Field` into data, sim_time and spec."""
  is_field = lambda x: isinstance(x, cx.Field)
  fields, treedef = jax.tree.flatten(pytree, is_leaf=is_field)
  data = jax.tree.unflatten(treedef, [x.data for x in fields])
  coords = consistent_coords(*fields)
  return data, coords
