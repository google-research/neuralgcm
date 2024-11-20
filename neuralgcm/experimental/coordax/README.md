# ``Coordax`` - Coordinate Axes for scientific computing in JAX

``Coordax`` makes it easy to associate array dimensions with coordinates in the context of scientific simulation codes written in JAX. This allows for efficient and expressive manipulation of data defined on structured grids, enabling operations like differentiation and interpolation with respect to physical coordinates.

**Key features**

1. Supports computation on locally-positional axes via coordinate map (``cmap``)
2. Manages coordinate objects that carry discretization details and custom methods
3. Compatible with lossless conversion to and from ``Xarray`` representation.

Current Field API and implementation builds on top of [``penzai.core.named_axes``](https://penzai.readthedocs.io/en/stable/notebooks/named_axes.html) package.

``Coordax`` is particularly well-suited for scientific simulations where it is crucial to propagate discretization details and associated objects throughout the computation, such as Earth system modeling of fluid dynamics. For other use cases, we believe [``penzai.core.named_axes``](https://penzai.readthedocs.io/en/stable/notebooks/named_axes.html) is a better fit.
