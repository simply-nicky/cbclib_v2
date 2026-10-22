"""Differentiable parameter states for local scaling and scan-wide merging."""
from typing import Generic, Literal, TypeVar, overload, Tuple, Type
from typing_extensions import Self
import pandas as pd
from .._src.annotations import AnyNamespace, IntArray, RealArray, Shape
from .._src.array_api import asnumpy, broadcast_to
from .._src.data_container import ArrayContainer, DataContainer, IndexLookup
from .._src.state import State, field
from ..indexer.cbc_setup import BaseSetup
from .cbc_data import (LegendreBasis, MergedList, PoissonData, StreakData, StreakPoints,
                       WeightedLinearEstimator)

class PedestalState(State, DataContainer):
    """Hold additive background corrections for predicted streaks.

    Attributes:
        pedestal: Additive photon-count pedestal for each streak, with shape
            ``(n_streaks,)``.
    """
    pedestal : RealArray # (n_streaks,)

    @classmethod
    def default(cls, n_streaks: int, xp: AnyNamespace) -> 'PedestalState':
        """Construct a zero pedestal for each streak."""
        return cls(pedestal=xp.zeros((n_streaks,)))

    @property
    def n_streaks(self) -> int:
        """Return the number of independently fitted predicted streaks."""
        return len(self.pedestal)

    @property
    def shape(self) -> Shape:
        return self.pedestal.shape

    def pedestal_at(self, points: StreakPoints) -> RealArray:
        """Return the pedestal associated with each measured point."""
        return self.pedestal[points.streak_id]

class StreakState(State, DataContainer):
    """Hold local photometric parameters for predicted streaks.

    Attributes:
        asinh_hkl: Inverse hyperbolic sine of the streak intensity coefficient for each
            unnormalised profile, with shape ``(n_streaks,)``.
        log_sigma: Natural logarithm of the profile width for each pattern, with shape
            ``(n_patterns,)``.
    """
    asinh_hkl   : RealArray # (n_streaks,)
    log_sigma   : RealArray # (n_patterns,)

    @classmethod
    def default(cls, n_streaks: int, n_patterns: int, sigma: float, xp: AnyNamespace
                ) -> 'StreakState':
        """Construct zero intensity values at a fixed initial width."""
        return cls(asinh_hkl=xp.zeros((n_streaks,)),
                   log_sigma=xp.full((n_patterns,), xp.log(sigma)))

    @classmethod
    def from_data(cls: Type[Self], data: StreakData, sigma: float, weights: RealArray,
                  xp: AnyNamespace) -> Self:
        """Initialise signed profile coefficients by their information-weighted median."""
        previous = xp.zeros((data.n_streaks,))
        log_sigma = xp.full((data.n_patterns,), xp.log(sigma))
        profile = data.modelled.profile(log_sigma[data.points.index], xp)
        estimator = WeightedLinearEstimator(group_id=data.points.streak_id, weights=weights)
        intensity = estimator.fit_median(profile, data.signal, previous)
        return cls(asinh_hkl=xp.arcsinh(intensity), log_sigma=log_sigma)

    @overload
    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, xp: AnyNamespace, *,
                         index: Literal[False]=False) -> 'StreakState': ...

    @overload
    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, xp: AnyNamespace, *,
                         index: Literal[True]) -> Tuple[IntArray, 'StreakState']: ...

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, xp: AnyNamespace, *,
                         index: bool=False
                         ) -> 'StreakState' | Tuple[IntArray, 'StreakState']:
        lookup = IndexLookup.build(xp.asarray(df['index']))
        state = cls(asinh_hkl=xp.asarray(df['asinh_hkl']),
                    log_sigma=xp.asarray(df['log_sigma'])[lookup.indices()])
        if index:
            return lookup.unique, state
        return state

    @property
    def n_streaks(self) -> int:
        """Return the number of independently fitted predicted streaks."""
        return self.asinh_hkl.size

    @property
    def n_patterns(self) -> int:
        """Return the number of patterns in the indexed data."""
        return self.log_sigma.size

    def intensity_at(self, points: StreakPoints) -> RealArray:
        """Return the signed profile coefficient associated with each measured point."""
        xp = self.__array_namespace__()
        return xp.sinh(self.asinh_hkl[points.streak_id])

    def log_sigma_at(self, points: StreakPoints) -> RealArray:
        """Return the log profile width associated with each measured point."""
        return self.log_sigma[points.index]

    def to_dataframe(self, index: IntArray, frames: IntArray) -> pd.DataFrame:
        """Export photometric parameters for each predicted streak."""
        if index.size != self.n_streaks or frames.size != self.n_patterns:
            raise ValueError("Index and frame arrays must match the streak and pattern sizes.")
        return pd.DataFrame({'index': asnumpy(frames[index]),
                             'asinh_hkl': asnumpy(self.asinh_hkl),
                             'log_sigma': asnumpy(self.log_sigma[index])})

class RefineStreakState(StreakState):
    """Hold refined position and broadening parameters for predicted streaks.

    Attributes:
        asinh_hkl: Inverse hyperbolic sine of the streak intensity coefficient for each
            unnormalised profile, with shape ``(n_streaks,)``.
        log_sigma: Natural logarithm of the profile width for each pattern,
            with shape ``(n_patterns,)``.
        displacement: Detector-plane ``(x, y)`` displacement in pixels for each streak,
            with shape ``(n_streaks, 2)``.
        pixel_size: Pixel size in metres.
    """
    displacement : RealArray # (n_streaks, 2), detector-plane in pixels
    pixel_size   : float = field(static=True)

    @classmethod
    def default(cls, n_streaks: int, n_patterns: int, sigma: float, pixel_size: float,
                xp: AnyNamespace) -> 'RefineStreakState':
        """Construct zero intensity and displacement values at a fixed initial width."""
        state = StreakState.default(n_streaks, n_patterns, sigma, xp)
        return cls.from_streaks(state, pixel_size)

    @classmethod
    def from_streaks(cls, streaks: StreakState, pixel_size: float) -> 'RefineStreakState':
        """Construct a refined state with zero displacement."""
        xp = streaks.__array_namespace__()
        return cls(asinh_hkl=streaks.asinh_hkl, log_sigma=streaks.log_sigma,
                   displacement=xp.zeros((streaks.n_streaks, 2)), pixel_size=pixel_size)

    @overload
    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, pixel_size: float,
                         xp: AnyNamespace, *, index: Literal[False]=False,
                         ) -> 'RefineStreakState': ...

    @overload
    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, pixel_size: float,
                         xp: AnyNamespace, *, index: Literal[True]
                         ) -> Tuple[IntArray, 'RefineStreakState']: ...

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, pixel_size: float,
                         xp: AnyNamespace, *, index: bool=False
                         ) -> 'RefineStreakState' | Tuple[IntArray, 'RefineStreakState']:
        frames, streaks = StreakState.import_dataframe(df, index=True, xp=xp)
        displacement = xp.stack((xp.asarray(df['displacement_x']),
                                 xp.asarray(df['displacement_y'])), axis=-1)
        state = cls(asinh_hkl=streaks.asinh_hkl, log_sigma=streaks.log_sigma,
                    displacement=displacement, pixel_size=pixel_size)
        if index:
            return frames, state
        return state

    def displacement_at(self, points: StreakPoints) -> RealArray:
        """Return the detector-plane displacement associated with each measured point."""
        return self.pixel_size * self.displacement[points.streak_id]

    def points_at(self, points: StreakPoints) -> RealArray:
        """Return detector points translated by their per-streak displacement."""
        return points.points + self.displacement_at(points)

    def to_dataframe(self, index: IntArray, frames: IntArray) -> pd.DataFrame:
        """Export photometric and displacement parameters for each predicted streak."""
        df = super().to_dataframe(index, frames)
        df['displacement_x'] = asnumpy(self.displacement[..., 0])
        df['displacement_y'] = asnumpy(self.displacement[..., 1])
        return df

class FullState(State, DataContainer):
    streaks : StreakState
    setup   : BaseSetup

T_Data = TypeVar("T_Data", bound=MergedList | PoissonData)

class BaseMergeState(DataContainer, Generic[T_Data]):
    log_scale: RealArray # (n_patterns,)
    asinh_hkl: RealArray # (n_reflections,)

    def normalise(self) -> Self:
        """Set the geometric mean scale to one while preserving every prediction.

        This fixes one gauge only; disconnected components must be fitted independently.
        Empty states are unchanged. Index gaps participate in the gauge convention.
        """
        if not self.log_scale.size:
            return self
        xp = self.__array_namespace__()
        shift = xp.mean(self.log_scale)
        return self.replace(log_scale=self.log_scale - shift,
                            asinh_hkl=xp.arcsinh(xp.sinh(self.asinh_hkl) * xp.exp(shift)))

class MergeState(State, BaseMergeState[MergedList]):
    """Shared intensities and positive illumination scales for one connected component.

    Attributes:
        log_scale: Natural log scales, shape ``(n_patterns,)``, including index gaps.
        asinh_hkl: Inverse hyperbolic sine of signed intensities, shape
            ``(n_reflections,)``, ordered by :attr:`MergedList.hkl`.
    """
    log_scale: RealArray # (n_patterns,)
    asinh_hkl: RealArray # (n_reflections,)

    @classmethod
    def from_data(cls, data: MergedList) -> 'MergeState':
        xp = data.__array_namespace__()
        I_hkl = data.reflection_estimator.median(data.I_hkl, xp)
        return cls(log_scale=xp.zeros((data.n_patterns,)),
                   asinh_hkl=xp.arcsinh(I_hkl)).normalise()

    def intensity_at(self, data: MergedList) -> RealArray:
        """Return shared intensities per merging observation or per scan streak."""
        xp = self.__array_namespace__()
        return xp.sinh(self.asinh_hkl)[data.reflection_id]

    def scale_at(self, data: MergedList) -> RealArray:
        """Return positive scales per merging observation or per scan streak."""
        xp = self.__array_namespace__()
        return xp.exp(self.log_scale)[data.index]

class PupilState(State, ArrayContainer):
    """Represent smooth pupil illumination with a fixed integrated intensity.

    The amplitude is one plus an expansion in nonconstant orthonormal shifted
    Legendre modes. Squaring it and dividing by ``1 + sum(coefficients**2)``
    gives a nonnegative intensity with unit integral on the unit-square pupil.
    The corresponding integral in physical pupil coordinates equals the pupil
    area. The intensity polynomial has twice the amplitude's maximum degree.

    Attributes:
        coefficients: Nonconstant amplitude coefficients, shape
            ``(1, n_modes - 1)`` for shared illumination or
            ``(n_patterns, n_modes - 1)`` for independent pattern illumination.
            Columns follow the supplied basis modes after the constant mode.

    Example:
        Construct a shared pupil and evaluate it for two patterns::

            from cbclib_v2.annotations import NumPy as xp
            from cbclib_v2.scaler import LegendreFactory, PupilState

            basis = LegendreFactory.create(1, xp)
            pupil = PupilState.default(basis, n_patterns=1, xp=xp)
            pupil = pupil.replace(coefficients=xp.asarray([[0.2, -0.1]]))
            points = xp.asarray([[0.25, 0.75], [0.6, 0.4]])
            intensity = pupil.at(points, xp.asarray([0, 1]), basis)

        Use ``n_patterns=2`` to fit independent coefficients for these patterns.
    """
    coefficients: RealArray

    @classmethod
    def default(cls, basis: 'LegendreBasis', n_patterns: int, xp: AnyNamespace
                ) -> 'PupilState':
        """Construct uniform illumination shared or independent across patterns.

        Args:
            basis: Orthonormal pupil basis including the constant mode first.
            n_patterns: Number of coefficient rows; one shares the pupil.
            xp: Array namespace for the fitted coefficients.
        """
        return cls(coefficients=xp.zeros((n_patterns, basis.modes.shape[0] - 1)))

    @property
    def shape(self) -> Shape:
        return self.coefficients.shape[:-1]

    @property
    def n_modes(self) -> int:
        """Return the number of fitted nonconstant pupil modes."""
        return self.coefficients.shape[-1]

    def coefficients_at(self, index: IntArray) -> RealArray:
        """Return amplitude coefficients for compact pattern indices.

        Args:
            index: Pattern indices with shape ``(...)``.

        Returns:
            Coefficients with shape ``(..., n_modes)``. A shared pupil
            selects the same coefficient row for every pattern.
        """
        xp = self.__array_namespace__()
        return broadcast_to(self.coefficients, index, (self.n_modes,), xp)

    def at(self, points: RealArray, index: IntArray, basis: 'LegendreBasis'
           ) -> RealArray:
        """Evaluate the analytically normalised intensity without an aperture mask.

        Args:
            points: Unit-square pupil coordinates ``(u, v)``, shape ``(..., 2)``.
                Coordinates outside the pupil are evaluated without clipping.
            index: Compact pattern indices, shape ``(...)``.
            basis: Orthonormal basis matching the coefficient columns. Its arrays,
                points and indices must share the coefficients' array namespace.

        Returns:
            Nonnegative intensities with shape ``(...)`` and unit integral over
            ``[0, 1] ** 2`` for each pattern. All normalisation is algebraic.
        """
        xp = self.__array_namespace__()
        coefficients = self.coefficients_at(index)
        amplitude = 1.0 + xp.sum(basis.evaluate(points)[..., 1:] * coefficients, axis=-1)
        integral = 1.0 + xp.sum(coefficients**2, axis=-1)
        return amplitude**2 / integral

class PoissonState(State, BaseMergeState[PoissonData]):
    """Shared intensities and positive illumination scales for one connected component.

    Attributes:
        log_scale: Natural log scales, shape ``(n_patterns,)``, including index gaps.
        asinh_hkl: Inverse hyperbolic sine of signed intensities, shape
            ``(n_reflections,)``, ordered by :attr:`PoissonData.hkl`.
        log_sigma: Natural log standard deviations, shape ``(n_patterns,)``.
        pupil: Smooth pupil illumination with fixed integrated intensity, shape
            ``(n_patterns or 1,)``.
    """
    log_scale   : RealArray     # (n_patterns,)
    asinh_hkl   : RealArray     # (n_reflections,)
    log_sigma   : RealArray     # (n_patterns,)
    pupil       : PupilState    # (n_patterns or 1,)

    @classmethod
    def from_data(cls, data: PoissonData, log_sigma: RealArray, pupil: PupilState
                  ) -> 'PoissonState':
        xp = data.__array_namespace__()
        profile = data.modelled.profile(log_sigma[data.points.index], xp)
        I_hkl = data.reflection_estimator.weighted_mean(data.signal, profile, xp)
        return cls(log_scale=xp.zeros(data.n_patterns), asinh_hkl=xp.arcsinh(I_hkl),
                   log_sigma=log_sigma, pupil=pupil).normalise()

    def intensity_at(self, data: PoissonData) -> RealArray:
        """Return shared intensities per merging observation or per scan streak."""
        xp = self.__array_namespace__()
        return xp.sinh(self.asinh_hkl)[data.reflection_id[data.points.streak_id]]

    def scale_at(self, data: PoissonData) -> RealArray:
        """Return positive scales per merging observation or per scan streak."""
        xp = self.__array_namespace__()
        return xp.exp(self.log_scale)[data.points.index]

    def log_sigma_at(self, data: PoissonData) -> RealArray:
        """Return the profile width associated with each measured point."""
        return self.log_sigma[data.points.index]

    def pupil_at(self, data: PoissonData) -> RealArray:
        """Return smooth polynomial pupil illumination, including outside the aperture."""
        return self.pupil.at(data.modelled.pupil_xy, data.points.index, data.basis)
