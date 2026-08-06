import pandas as pd
from .._src.annotations import AnyNamespace, IntArray, RealArray
from .._src.array_api import add_at, asnumpy, safe_log, safe_divide
from .._src.crystfel import Detector
from .._src.data_container import ArrayContainer, DataContainer, IndexedContainer
from .._src.data_processing import CrystData
from .._src.state import field, State
from ..indexer.cbc_data import AnyPoints, Miller
from ..indexer.cbc_pupil import BasePupil, SourcePlane
from ..indexer.cbc_setup import ResolvedSetup
from .cbc_symmetry import PointGroup

class IntensityModel(State, ArrayContainer):
    kout        : RealArray
    source      : SourcePlane
    pupil       : BasePupil

    @property
    def kin(self) -> RealArray:
        """Return incident-vector candidates associated with the observed rays."""
        return self.kout - self.source.q

    @property
    def distance_sq(self) -> RealArray:
        """Return the summed squared distances to the source plane and pupil support."""
        xp = self.__array_namespace__()
        dist_src = xp.sum((self.kin - self.source.project(self.kin))**2, axis=-1)
        dist_ppl = xp.sum((self.kin - self.pupil.project(self.kin))**2, axis=-1)
        return xp.asarray(dist_src + dist_ppl)

    def log_radial(self, log_sigma: RealArray, xp: AnyNamespace) -> RealArray:
        """Return Gaussian attenuation by source-plane and pupil-support distance."""
        inv_sigma_sq = xp.exp(-2.0 * log_sigma)
        return -0.5 * self.distance_sq * inv_sigma_sq

    def log_profile(self, log_sigma: RealArray, xp: AnyNamespace) -> RealArray:
        """Return the continuous source-plane and pupil-support log profile."""
        return self.log_radial(log_sigma, xp)

class StreakPoints(State, ArrayContainer):
    index       : IntArray  # (n_points,) frame index
    streak_id   : IntArray  # (n_points,) streak index
    points      : RealArray # (n_points, 2) (x, y) coordinates in meters

    @property
    def x(self) -> RealArray:
        return self.points[..., 0]

    @property
    def y(self) -> RealArray:
        return self.points[..., 1]

class StreakIndices(State, IndexedContainer):
    index       : IntArray  # (n_points,) frame index
    streak_id   : IntArray  # (n_points,) streak index
    y           : IntArray # (n_points,) y pixel coordinates
    x           : IntArray # (n_points,) x pixel coordinates

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame, xp: AnyNamespace) -> 'StreakIndices':
        """Import pattern points from :method:`~cbclib_v2.Patterns.pattern_dataframe`."""
        df = df.drop_duplicates(subset=['index', 'y', 'x'], keep=False)
        index = xp.asarray(df['index'].to_numpy())
        streak_id = xp.asarray(df['streak_id'].to_numpy())
        y = xp.asarray(df['y'].to_numpy())
        x = xp.asarray(df['x'].to_numpy())
        return cls(index=index, y=y, x=x, streak_id=streak_id)

    @property
    def xy(self) -> IntArray:
        xp = self.__array_namespace__()
        return xp.stack((self.x, self.y), axis=-1)

    def to_points(self, detector: Detector) -> StreakPoints:
        return StreakPoints(index=self.index, streak_id=self.streak_id,
                            points=detector.pixel_size * self.xy)

class PhotonCounts(State, ArrayContainer):
    I0          : IntArray
    background  : RealArray

    @classmethod
    def import_data(cls, data: CrystData, indices: StreakIndices, xp: AnyNamespace
                    ) -> 'PhotonCounts':
        if data.is_empty(data.whitefield):
            raise ValueError("Data does not contain whitefield counts.")
        I0 = xp.asarray(data.data[(indices.index, indices.y, indices.x)])
        background = xp.asarray(data.whitefield[(indices.index, indices.y, indices.x)])
        return cls(I0=I0, background=background)

    @property
    def signal(self) -> RealArray:
        xp = self.__array_namespace__()
        return xp.clip(self.I0 - self.background, 0.0, xp.inf)

class ReflectionsMap(State, DataContainer):
    """Map observed streaks to pattern-local symmetry-equivalent reflections.

    Attributes:
        reflection_id: Contiguous merged-reflection index for every input streak,
            with shape ``(n_streaks,)``.
        n_reflections: Number of pattern-local merged reflections.
    """

    reflection_id : IntArray
    n_reflections : int = field(static=True)

    @classmethod
    def from_miller(cls, miller: Miller, point_group: PointGroup,
                    xp: AnyNamespace) -> 'ReflectionsMap':
        """Map Miller indices to pattern-local intensity-equivalence orbits.

        Args:
            miller: Indexed Miller indices for the observed streaks.
            point_group: Crystallographic point group used for merging.
            xp: Array namespace used for the calculation.

        Returns:
            Streak-to-reflection mapping and the number of merged reflections.
        """
        index = xp.reshape(xp.broadcast_to(miller.index, miller.hkl.shape[:-1]), (-1,))
        hkl = xp.reshape(miller.hkl_indices, (-1, 3))
        canonical = point_group.canonical(hkl, xp)
        keys = xp.concat((index[:, None], canonical), axis=-1)
        keys, reflection_id = xp.unique(keys, return_inverse=True, axis=0)
        return cls(reflection_id=xp.asarray(reflection_id), n_reflections=keys.shape[0])

    def __len__(self) -> int:
        return self.n_reflections

    def at(self, points: StreakPoints) -> IntArray:
        """Return the merged-reflection index associated with each measured point."""
        return self.reflection_id[points.streak_id]

    def canonical(self, miller: Miller, point_group: PointGroup, xp: AnyNamespace
                  ) -> Miller:
        """Return one canonical Miller index for each mapped reflection.

        The output rows follow ``reflection_id`` order, so row ``r`` is aligned with scaler
        state and result arrays at row ``r``.

        Args:
            miller: Unmerged Miller indices used to construct this reflection map.
            point_group: Crystallographic point group used to construct this map.
            xp: Array namespace used for canonicalization.

        Returns:
            Pattern-local canonical Miller indices with shape ``(n_reflections,)``.
        """
        index = xp.reshape(xp.broadcast_to(miller.index, miller.hkl.shape[:-1]), (-1,))
        hkl = xp.reshape(miller.hkl_indices, (-1, 3))
        canonical = point_group.canonical(hkl, xp)
        _, representatives = xp.unique(self.reflection_id, return_index=True)
        return Miller(index=xp.asarray(index[representatives]),
                      hkl=xp.asarray(canonical[representatives]))

class ScalerData(State, DataContainer):
    """Hold observed streak data and its shared-intensity mapping."""

    points      : StreakPoints
    counts      : PhotonCounts
    miller      : Miller
    reflections : ReflectionsMap

    @classmethod
    def import_data(cls, data: CrystData, streak_ids: StreakIndices, miller: Miller,
                    point_group: PointGroup, detector: Detector, xp: AnyNamespace
                    ) -> 'ScalerData':
        points = streak_ids.to_points(detector)
        counts = PhotonCounts.import_data(data, streak_ids, xp)
        reflections = ReflectionsMap.from_miller(miller, point_group, xp)
        return cls(points=points, counts=counts, miller=miller, reflections=reflections)

class ScalerState(State, DataContainer):
    log_hkl        : RealArray    # (n_reflections,)
    log_sigma_kin  : RealArray    # (n_frames,)

    @classmethod
    def default(cls, n_reflections: int, n_frames: int, sigma: float, xp: AnyNamespace
                ) -> 'ScalerState':
        log_hkl = xp.zeros((n_reflections,))
        log_sigma_kin = xp.full((n_frames,), xp.log(sigma))
        return cls(log_hkl=log_hkl, log_sigma_kin=log_sigma_kin)

    @classmethod
    def from_data(cls, data: ScalerData, setup: ResolvedSetup, sigma: float
                  ) -> 'ScalerState':
        xp = setup.__array_namespace__()
        reflection_id = data.reflections.at(data.points)
        sums = xp.zeros((len(data.reflections),))
        counts = xp.zeros((len(data.reflections),))
        sums = add_at(sums, reflection_id, data.counts.signal)
        counts = add_at(counts, reflection_id, xp.ones(data.points.shape[0]))
        log_hkl = safe_log(safe_divide(sums, counts, xp), xp)

        log_sigma_kin = xp.full((len(setup.xtal),), xp.log(sigma))
        return cls(log_hkl=log_hkl, log_sigma_kin=log_sigma_kin)

    def log_sigma_at(self, points: AnyPoints) -> RealArray:
        return self.log_sigma_kin[points.index]

    def log_at(self, points: StreakPoints, reflections: ReflectionsMap) -> RealArray:
        return self.log_hkl[reflections.at(points)]

class ScalerResult(State, DataContainer):
    """Hold fitted reflection intensities and their conditional uncertainties.

    Attributes:
        miller: Pattern-local canonical Miller indices, with shape
            ``(n_reflections,)``.
        I_hkl: Fitted intensities, with shape ``(n_reflections,)``.
        sigma_hkl: Conditional Poisson standard errors of ``I_hkl``, with shape
            ``(n_reflections,)``.
    """

    miller    : Miller
    I_hkl     : RealArray
    sigma_hkl : RealArray

    @classmethod
    def from_data(cls, data: ScalerData, point_group: PointGroup, I_hkl: RealArray,
                  sigma_hkl: RealArray, xp: AnyNamespace) -> 'ScalerResult':
        """Construct a result with canonical Miller indices from scaling data.

        Canonical indices are merged independently within each pattern in the same order as
        :meth:`ReflectionsMap.from_miller`.

        Args:
            data: Scaling observations containing the unmerged Miller indices.
            point_group: Crystallographic point group used for intensity merging.
            I_hkl: Fitted intensities with shape ``(n_reflections,)``.
            sigma_hkl: Conditional standard errors with shape ``(n_reflections,)``.
            xp: Array namespace used for canonicalization.

        Returns:
            Self-contained scaling result aligned with the merged reflection order.
        """
        miller = data.reflections.canonical(data.miller, point_group, xp)
        return cls(miller=miller, I_hkl=I_hkl, sigma_hkl=sigma_hkl)

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame, xp: AnyNamespace) -> 'ScalerResult':
        """Import canonical Miller indices, intensities, and uncertainties from a dataframe."""
        hkl = xp.stack((xp.asarray(df['h'].to_numpy()), xp.asarray(df['k'].to_numpy()),
                        xp.asarray(df['l'].to_numpy())), axis=-1)
        miller = Miller(index=xp.asarray(df['index'].to_numpy()), hkl=hkl)
        return cls(miller=miller, I_hkl=xp.asarray(df['I_hkl'].to_numpy()),
                   sigma_hkl=xp.asarray(df['sigma_hkl'].to_numpy()))

    def to_dataframe(self) -> pd.DataFrame:
        """Export canonical Miller indices, intensities, and uncertainties to a dataframe."""
        return pd.DataFrame({'index': asnumpy(self.miller.index),
                             'h': asnumpy(self.miller.h),
                             'k': asnumpy(self.miller.k),
                             'l': asnumpy(self.miller.l),
                             'I_hkl': asnumpy(self.I_hkl),
                             'sigma_hkl': asnumpy(self.sigma_hkl)})
