import pandas as pd
from .._src.annotations import AnyNamespace, IntArray, RealArray
from .._src.array_api import add_at, safe_log, safe_divide
from .._src.crystfel import Detector
from .._src.data_container import ArrayContainer, DataContainer, IndexedContainer
from .._src.data_processing import CrystData
from .._src.state import State
from ..indexer.cbc_data import AnyPoints, Miller
from ..indexer.cbc_pupil import BasePupil, SourcePlane
from ..indexer.cbc_setup import ResolvedSetup

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

class PatternData(State, DataContainer):
    points      : StreakPoints  # (n_points,)
    counts      : PhotonCounts  # (n_points,)
    miller      : Miller        # (n_streaks,)

class ScalerData(State, ArrayContainer):
    points      : StreakPoints  # (n_points,)
    counts      : PhotonCounts  # (n_points,)
    modelled    : IntensityModel   # (n_points,)

class ScalerState(State, DataContainer):
    log_hkl        : RealArray    # (n_streaks,)
    log_sigma_kin  : RealArray    # (n_frames,)

    @classmethod
    def default(cls, n_streaks: int, n_frames: int, sigma: float, xp: AnyNamespace
                ) -> 'ScalerState':
        log_hkl = xp.zeros((n_streaks,))
        log_sigma_kin = xp.full((n_frames,), xp.log(sigma))
        return cls(log_hkl=log_hkl, log_sigma_kin=log_sigma_kin)

    @classmethod
    def from_data(cls, data: PatternData, setup: ResolvedSetup, sigma: float
                  ) -> 'ScalerState':
        xp = setup.__array_namespace__()
        sums = xp.zeros((data.miller.shape[0],))
        counts = xp.zeros((data.miller.shape[0],))
        sums = add_at(sums, data.points.streak_id, data.counts.signal)
        counts = add_at(counts, data.points.streak_id, xp.ones(data.points.shape[0]))
        log_hkl = safe_log(safe_divide(sums, counts, xp), xp)

        log_sigma_kin = xp.full((len(setup.xtal),), xp.log(sigma))
        return cls(log_hkl=log_hkl, log_sigma_kin=log_sigma_kin)

    def log_sigma_at(self, points: AnyPoints) -> RealArray:
        return self.log_sigma_kin[points.index]

    def log_at(self, points: StreakPoints) -> RealArray:
        return self.log_hkl[points.streak_id]
