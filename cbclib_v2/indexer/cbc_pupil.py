from typing import Tuple
from .._src.array_api import array_namespace, kxy_to_k, project_to_rect, safe_divide, safe_sqrt
from .._src.state import State
from .._src.data_container import ArrayContainer, DataContainer
from ..annotations import RealArray, Shape
from .cbc_setup import Rectangle

class Edge(State, ArrayContainer):
    tau     : RealArray     # (..., 4, 2) edge tangents
    origin  : RealArray     # (..., 4, 2) edge origins

    @property
    def shape(self) -> Shape:
        return self.tau.shape[:-2]

    @property
    def normal(self) -> RealArray:
        xp = self.__array_namespace__()
        return xp.stack((self.tau[..., 1], -self.tau[..., 0]), axis=-1)

    def to_points(self, t: RealArray) -> 'EdgePoints':
        return EdgePoints(tau=self.tau[..., None, :], origin=self.origin[..., None, :], t=t)

class EdgePoints(State, DataContainer):
    tau     : RealArray     # (4, 1, 2) edge tangents
    origin  : RealArray     # (4, 1, 2) edge origins
    t       : RealArray     # (M, N, 4, 2) parameters along each edge

    @property
    def xy(self) -> RealArray:
        return self.origin + self.t[..., None] * self.tau

    @property
    def points(self) -> RealArray:
        return kxy_to_k(self.xy, self.__array_namespace__())

    def project(self) -> 'EdgePoints':
        """Project supporting-line points onto their finite edge segments."""
        xp = self.__array_namespace__()
        return self.replace(t=xp.clip(self.t, 0.0, 1.0))

    def distance(self) -> RealArray:
        """Return in-plane distances to the corresponding finite edge segments."""
        xp = self.__array_namespace__()
        delta = self.xy - self.project().xy
        return safe_sqrt(xp.sum(delta**2, axis=-1), xp)

class RectangularPupil(Rectangle):
    roi : RealArray     # (n_samples, 4) rectangle coordinates (ky_0, ky_1, kx_0, kx_1)

    @property
    def shape(self) -> Shape:
        return self.roi.shape[:-1]

    @property
    def edges(self) -> Edge:
        xp = self.__array_namespace__()
        origins = xp.stack((
            xp.stack((self.x1, self.y0), axis=-1),  # bottom-right
            xp.stack((self.x1, self.y1), axis=-1),  # top-right
            xp.stack((self.x0, self.y1), axis=-1),  # top-left
            xp.stack((self.x0, self.y0), axis=-1),  # bottom-left
        ), axis=-2)
        tau = xp.stack((
            xp.stack((xp.zeros_like(self.x0), self.y1 - self.y0), axis=-1),  # up
            xp.stack((self.x0 - self.x1, xp.zeros_like(self.y0)), axis=-1),  # left
            xp.stack((xp.zeros_like(self.x0), self.y0 - self.y1), axis=-1),  # down
            xp.stack((self.x1 - self.x0, xp.zeros_like(self.y0)), axis=-1),  # right
        ), axis=-2)
        return Edge(tau=tau, origin=origins)

    def project(self, kin: RealArray) -> RealArray:
        """Project incident vectors onto the rectangular spherical support."""
        xp = self.__array_namespace__()
        xy = project_to_rect(kin[..., :2], self.min, self.max, xp)
        return kxy_to_k(xy, xp)

    def distance(self, kin: RealArray) -> RealArray:
        """Return 3D Euclidean distances to the spherical pupil support."""
        xp = self.__array_namespace__()
        delta = kin - self.project(kin)
        return safe_sqrt(xp.sum(delta**2, axis=-1), xp)

class SourcePlane(State, ArrayContainer):
    """Laue plane associated with a reciprocal-lattice vector."""
    q       : RealArray     # (n_points, 3) reciprocal-lattice vectors
    q_mag   : RealArray     # (n_points,) squared magnitudes of reciprocal-lattice vectors

    @classmethod
    def from_q(cls, q: RealArray) -> 'SourcePlane':
        """Construct a source plane from a reciprocal-lattice vector."""
        xp = array_namespace(q)
        return cls(q=q, q_mag=xp.sum(q**2, axis=-1))

    @property
    def shape(self) -> Shape:
        return self.q.shape[:-1]

    def residual(self, kin: RealArray) -> RealArray:
        """Return the unnormalised signed Laue-plane residual."""
        xp = self.__array_namespace__()
        return xp.asarray(xp.sum(kin * self.q, axis=-1) + 0.5 * self.q_mag)

    def distance(self, kin: RealArray) -> RealArray:
        """Return perpendicular distances from incident vectors to the Laue plane."""
        xp = self.__array_namespace__()
        normal = xp.sqrt(self.q_mag)
        return safe_divide(xp.abs(self.residual(kin)), normal, xp)

    def project(self, kin: RealArray) -> RealArray:
        """Orthogonally project incident vectors onto the Laue plane."""
        xp = self.__array_namespace__()
        scale = safe_divide(self.residual(kin), self.q_mag, xp)
        return kin - scale[..., None] * self.q

class PupilIntersection(State, DataContainer):
    """Candidate intersections between an Ewald circle and pupil supporting lines.
    """
    source : SourcePlane    # (n_points,) reciprocal-lattice vectors
    edges : Edge            # (n_points or 1, n_pupils) or (n_points or 1,) finite pupil edges

    @classmethod
    def from_edge(cls, q: RealArray, edges: Edge) -> 'PupilIntersection':
        """Construct an intersection from a reciprocal vector and finite pupil edges."""
        return cls(source=SourcePlane.from_q(q), edges=edges)

    def solutions(self) -> EdgePoints:
        xp = self.__array_namespace__()
        source = self.source.expand_dims(axis=range(self.source.ndim, self.edges.ndim + 1))
        origin_dot_q = xp.sum(self.edges.origin * source.q[..., :2], axis=-1)
        f1 = -0.5 * source.q_mag - origin_dot_q
        f2 = xp.sum(self.edges.tau * source.q[..., :2], axis=-1)

        qz_sq = source.q[..., 2]**2
        a = f2 * f2 + qz_sq * xp.sum(self.edges.tau**2, axis=-1)
        b = f1 * f2 - qz_sq * xp.sum(self.edges.origin * self.edges.tau, axis=-1)
        c = f1 * f1 - qz_sq * (1 - xp.sum(self.edges.origin**2, axis=-1))

        delta = xp.where(b * b > a * c, b * b - a * c, 0.0)

        sqrt_delta = safe_sqrt(delta, xp)
        t0 = safe_divide(b - sqrt_delta, a, xp)
        t1 = safe_divide(b + sqrt_delta, a, xp)
        return self.edges.to_points(xp.stack((t0, t1), axis=-1))

    def select(self, solutions: EdgePoints) -> Tuple[RealArray, RealArray]:
        """Select the two Ewald-valid roots nearest to the pupil support."""
        xp = self.__array_namespace__()
        proj = solutions.project()
        axes = tuple(range(self.source.q.ndim - 1, solutions.points.ndim - 1))
        source = self.source.expand_dims(axis=axes)

        q_distance = source.distance(solutions.points)
        distance = xp.sqrt(xp.sum((solutions.xy - proj.xy)**2, axis=-1))
        score = distance + q_distance

        root_indices = xp.argmin(score, axis=-1)
        kin = xp.take_along_axis(proj.points, root_indices[..., None, None], axis=-2)[..., 0, :]
        score = xp.take_along_axis(score, root_indices[..., None], axis=-1)[..., 0]

        edge_indices = xp.argsort(score, axis=-1)[..., :2]
        kin = xp.take_along_axis(kin, edge_indices[..., None], axis=-2)
        score = xp.take_along_axis(score, edge_indices, axis=-1)
        return kin, score

def intersect(q: RealArray, edges: Edge) -> Tuple[RealArray, RealArray]:
    """Return the two source-line endpoints nearest to finite pupil edges."""
    intersection = PupilIntersection.from_edge(q=q, edges=edges)
    solutions = intersection.solutions()
    return intersection.select(solutions)
