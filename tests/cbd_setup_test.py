import pytest
from cbclib_v2 import default_rng
from cbclib_v2.annotations import Generator, NDArray, NumPyNamespace, NumPy, RealArray
from cbclib_v2.indexer import (CBDPoints, ConvexPolygon, EdgePoints, FixedApertureSetup,
                               FixedPupilSetup, FixedSetup, LinePoints, Miller, MillerWithRLP,
                               PupilIntersection, Rectangle, RefinerModel, ResolvedGeometry,
                               ResolvedSetup, RotationState, SimulatedVectors, SourcePlane,
                               XtalCell, XtalState)
from cbclib_v2.test_util import TestSetup, check_close

class TestCBDSetup():
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def rng(self, xp: NumPyNamespace) -> Generator[NDArray]:
        return default_rng(42, xp)

    @pytest.fixture
    def initial(self, xp: NumPyNamespace) -> FixedSetup:
        return FixedSetup(TestSetup.xtal(xp), TestSetup.fixed_geometry())

    @pytest.fixture
    def resolved(self, initial: FixedSetup, xp: NumPyNamespace) -> ResolvedSetup:
        return initial.resolve(xp)

    @pytest.fixture
    def geometry(self, resolved: ResolvedSetup) -> ResolvedGeometry:
        if isinstance(resolved.geometry, ResolvedGeometry):
            return resolved.geometry
        raise ValueError(f"ResolvedGeometry expected, but {type(resolved.geometry)} found.")

    def skew_symmetric(self, vec: RealArray, xp: NumPyNamespace) -> RealArray:
        return xp.linalg.cross(xp.eye(vec.shape[-1]), vec[..., None, :])

    def rodriguez_formula(self, angles: RealArray, xp: NumPyNamespace) -> RealArray:
        axis = xp.stack([xp.sin(angles[..., 1]) * xp.cos(angles[..., 2]),
                         xp.sin(angles[..., 1]) * xp.sin(angles[..., 2]),
                         xp.cos(angles[..., 1])], axis=-1)
        skew = self.skew_symmetric(axis, xp)
        I = 1.0 * xp.broadcast_to(xp.eye(skew.shape[-1]), skew.shape)
        S = xp.sin(angles[..., 0])[..., None, None]
        C = xp.cos(angles[..., 0])[..., None, None]
        return I + S * skew + (1.0 - C) * (skew @ skew)

    @pytest.fixture
    def xtal(self, initial: FixedSetup) -> XtalState:
        return initial.xtal

    @pytest.fixture
    def ormatrix(self, xtal: XtalState) -> RotationState:
        return xtal.orientation_matrix

    @pytest.fixture
    def cell(self, xtal: XtalState) -> XtalCell:
        return xtal.unit_cell

    @pytest.fixture(params=[0.3,])
    def q_abs(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture(params=[200,])
    def num_points(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def atol(self) -> float:
        return 2e-6

    @pytest.fixture
    def miller(self, rng: Generator[NDArray], q_abs: float, num_points: int, model: RefinerModel,
               resolved: ResolvedSetup, xp: NumPyNamespace) -> Miller:
        miller = model.hkl_in_aperture(q_abs, resolved, xp)
        idxs = rng.choice(miller.hkl.shape[0], size=(num_points,))
        return miller[idxs]

    @pytest.fixture
    def rlp(self, miller: Miller, model: RefinerModel,
            initial: FixedSetup, xp: NumPyNamespace) -> MillerWithRLP:
        return model.xtal.hkl_to_q(miller, initial.xtal, xp)

    @pytest.fixture
    def laue(self, rlp: MillerWithRLP, model: RefinerModel, resolved: ResolvedSetup,
             xp: NumPyNamespace) -> SimulatedVectors:
        pupil = model.lens.pupil(resolved.geometry, xp)
        return model.lens.source_lines(rlp, pupil, xp)

    @pytest.fixture
    def points(self, laue: SimulatedVectors, model: RefinerModel, resolved: ResolvedSetup,
               atol: float, xp: NumPyNamespace) -> CBDPoints:
        return model.kout_to_points(laue, resolved.geometry, xp, atol=atol)

    def text_xtal_to_cell(self, xtal: XtalState, ormatrix: RotationState,
                          cell: XtalCell, xp: NumPyNamespace):
        basis = cell.to_basis()
        check_close(xp.linalg.det(ormatrix.matrix), xp.array(1.0))
        check_close(ormatrix @ basis.basis, xtal.basis)

    def test_xtal_to_spherical(self, xtal: XtalState):
        r, theta, phi = xtal.to_spherical()
        basis = XtalState.import_spherical(r, theta, phi).basis
        check_close(xtal.basis, basis)

    def text_reciprocate_xtal(self, xtal: XtalState):
        check_close(xtal.basis, xtal.reciprocate().reciprocate().basis)

    def test_cell_to_xtal(self, cell: XtalCell):
        new_cell = cell.to_basis().unit_cell
        check_close(cell.angles, new_cell.angles)
        check_close(cell.lengths, new_cell.lengths)

    def test_hkl_and_q(self, miller: Miller, rlp: MillerWithRLP,
                       model: RefinerModel, initial: FixedSetup, xp: NumPyNamespace):
        rlp = model.xtal.q_to_hkl(rlp, initial.xtal, xp)
        assert xp.all(rlp.hkl_indices == miller.hkl_indices)

    def test_laue(self, laue: SimulatedVectors, model: RefinerModel,
                  geometry: ResolvedGeometry, xp: NumPyNamespace):
        check_close(xp.broadcast_to(laue.q[..., None, :], laue.kout.shape),
                    laue.kout - laue.kin)
        valid = laue.distance == 0.0
        kin = model.lens.project_to_pupil(laue.kin, laue.index, geometry, xp)
        check_close(laue.kin, kin)
        q = xp.broadcast_to(laue.q[..., None, :], laue.kin.shape)
        check_close(xp.sum(laue.kin[valid] * q[valid], axis=-1),
                    -0.5 * xp.sum(q[valid]**2, axis=-1))

    def test_points_and_kout(self, laue: SimulatedVectors, points: CBDPoints,
                             model: RefinerModel, geometry: ResolvedGeometry, atol: float,
                             xp: NumPyNamespace):
        valid = xp.isclose(laue.distance, 0.0, atol=atol)
        points = model.kout_to_points(laue, geometry, xp, atol=atol)
        kout = model.points_to_kout(points, geometry, xp)
        check_close(kout[valid], laue.kout[valid])
        assert xp.all(xp.isnan(points.points[~valid]))

    def test_first_order_defocus_correction(self, laue: SimulatedVectors, points: CBDPoints,
                                            model: RefinerModel, geometry: ResolvedGeometry,
                                            atol: float, xp: NumPyNamespace):
        valid = xp.isclose(laue.distance, 0.0, atol=atol)
        kout_zero = model.points_to_kout(LinePoints(points.index, points.points), geometry, xp)

        kout_first = model.points_to_kout(points, geometry, xp)

        assert not xp.allclose(kout_zero[valid], laue.kout[valid])
        check_close(kout_first[valid], laue.kout[valid])

    def test_rotation_to_tilt(self, ormatrix: RotationState, xp: NumPyNamespace):
        tilt = ormatrix.to_tilt()
        check_close(tilt.to_rotation().matrix, self.rodriguez_formula(tilt.angles, xp))
        check_close(ormatrix.matrix, tilt.to_rotation().matrix)

    def test_rotation_to_tilt_over_axis(self, ormatrix: RotationState):
        tilt_over_axis = ormatrix.to_tilt().to_tilt_over_axis()
        check_close(ormatrix.matrix, tilt_over_axis.to_tilt().to_rotation().matrix)

    def test_fixed_pupil_state_from_resolved(self, resolved: ResolvedSetup, xp: NumPyNamespace):
        restored = FixedPupilSetup.from_resolved(resolved)
        converted = restored.resolve(xp)

        check_close(converted.xtal.basis, resolved.xtal.basis)
        check_close(converted.geometry.foc_pos, resolved.geometry.foc_pos)
        check_close(converted.geometry.pupil_roi, resolved.geometry.pupil_roi)
        if isinstance(converted.geometry, ResolvedGeometry) and \
           isinstance(resolved.geometry, ResolvedGeometry):
            check_close(converted.geometry.defocus, resolved.geometry.defocus)

    def test_fixed_aperture_state_from_resolved(self, resolved: ResolvedSetup, xp: NumPyNamespace):
        restored = FixedApertureSetup.from_resolved(resolved)
        converted = restored.resolve(xp)

        check_close(converted.xtal.basis, resolved.xtal.basis)
        check_close(converted.geometry.foc_pos, resolved.geometry.foc_pos)
        check_close(converted.geometry.pupil_roi, resolved.geometry.pupil_roi)
        if isinstance(converted.geometry, ResolvedGeometry) and \
           isinstance(resolved.geometry, ResolvedGeometry):
            check_close(converted.geometry.defocus, resolved.geometry.defocus)

class TestPupilProjection:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def edge_parameters(self, xp: NumPyNamespace) -> RealArray:
        return xp.broadcast_to(xp.asarray([-0.5, 0.5, 1.5]), (4, 3))

    @pytest.fixture
    def rectangle(self, xp: NumPyNamespace) -> Rectangle:
        return Rectangle(roi=xp.asarray([0.0, 2.0, 0.0, 4.0]))

    @pytest.fixture
    def rectangle_points(self, rectangle: Rectangle,
                         edge_parameters: RealArray) -> EdgePoints:
        return rectangle.edges.to_points(edge_parameters)

    @pytest.fixture
    def polygon(self, xp: NumPyNamespace) -> ConvexPolygon:
        return ConvexPolygon(center=xp.zeros(2), lengths=xp.ones(4))

    @pytest.fixture
    def polygon_points(self, polygon: ConvexPolygon,
                       edge_parameters: RealArray) -> EdgePoints:
        return polygon.edges.to_points(edge_parameters)

    @pytest.fixture
    def spherical_pupil(self, xp: NumPyNamespace) -> Rectangle:
        return Rectangle(roi=xp.asarray([-0.2, 0.2, -0.1, 0.1]))

    @pytest.fixture
    def incident_vectors(self, xp: NumPyNamespace) -> RealArray:
        return xp.asarray([[0.0, 0.0, 1.0], [0.2, 0.0, xp.sqrt(0.96)]])

    def check_edge_projection(self, points: EdgePoints, xp: NumPyNamespace) -> None:
        projected = points.project()
        expected_t = xp.clip(points.t, 0.0, 1.0)
        displacement = points.xy - projected.xy
        expected_distance = xp.sqrt(xp.sum(displacement**2, axis=-1))

        # Projection onto a finite edge clamps its line parameter and measures displacement.
        check_close(projected.t, expected_t)
        check_close(points.distance(), expected_distance)

    def test_rectangle_distance(self, rectangle_points: EdgePoints,
                                xp: NumPyNamespace) -> None:
        self.check_edge_projection(rectangle_points, xp)

    def test_polygon_projection(self, polygon_points: EdgePoints,
                                xp: NumPyNamespace) -> None:
        self.check_edge_projection(polygon_points, xp)

    def test_spherical_projection(self, spherical_pupil: Rectangle,
                                  incident_vectors: RealArray,
                                  xp: NumPyNamespace) -> None:
        projected = spherical_pupil.project(incident_vectors)
        expected_xy = xp.clip(incident_vectors[..., :2], spherical_pupil.min,
                              spherical_pupil.max)
        expected_z = xp.sqrt(1.0 - xp.sum(expected_xy**2, axis=-1))
        expected = xp.concat((expected_xy, expected_z[..., None]), axis=-1)

        # Pupil projection clips transverse coordinates while preserving the unit sphere.
        check_close(projected, expected)
        check_close(xp.sum(projected**2, axis=-1), xp.ones(projected.shape[:-1]))
        check_close(spherical_pupil.distance(incident_vectors),
                    xp.sqrt(xp.sum((incident_vectors - projected)**2, axis=-1)))

class TestSourcePlane:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def q(self, xp: NumPyNamespace) -> RealArray:
        return xp.asarray([0.2, 0.0, 0.0])

    @pytest.fixture
    def source(self, q: RealArray) -> SourcePlane:
        return SourcePlane.from_q(q=q)

    @pytest.fixture
    def kin(self, xp: NumPyNamespace) -> RealArray:
        return xp.asarray([0.0, 0.0, 1.0])

    @pytest.fixture
    def zero_source(self, q: RealArray, xp: NumPyNamespace) -> SourcePlane:
        return SourcePlane.from_q(q=xp.zeros_like(q))

    @pytest.fixture
    def batched_source(self, xp: NumPyNamespace) -> SourcePlane:
        q = xp.asarray([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0]])
        return SourcePlane.from_q(q=q)

    @pytest.fixture
    def batched_kin(self, xp: NumPyNamespace) -> RealArray:
        return xp.zeros((2, 3, 3))

    @pytest.fixture
    def expanded_source(self, batched_source: SourcePlane) -> SourcePlane:
        return batched_source.expand_dims(axis=1)

    def test_normalised_distance(self, source: SourcePlane, kin: RealArray,
                                 xp: NumPyNamespace) -> None:
        projected = source.project(kin)
        displacement = kin - projected

        # Orthogonal projection lands on the Laue plane by moving parallel to its normal.
        check_close(source.residual(projected), xp.zeros_like(source.distance(kin)))
        check_close(xp.linalg.cross(displacement, source.q), xp.zeros_like(source.q))
        check_close(source.distance(kin), xp.sqrt(xp.sum(displacement**2, axis=-1)))

    def test_zero_q(self, zero_source: SourcePlane, kin: RealArray,
                    xp: NumPyNamespace) -> None:
        # A zero reciprocal vector defines no plane correction or perpendicular distance.
        check_close(zero_source.project(kin), kin)
        check_close(zero_source.distance(kin), xp.zeros_like(zero_source.distance(kin)))

    def test_expand_dims(self, expanded_source: SourcePlane, batched_kin: RealArray,
                         xp: NumPyNamespace) -> None:
        projected = expanded_source.project(batched_kin)
        displacement = batched_kin - projected

        # Expanding the plane batch axis preserves the projection law under broadcasting.
        check_close(expanded_source.residual(projected),
                    xp.zeros_like(expanded_source.distance(batched_kin)))
        check_close(expanded_source.distance(batched_kin),
                    xp.sqrt(xp.sum(displacement**2, axis=-1)))

class TestPupilIntersection:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def outside_pupil(self, xp: NumPyNamespace) -> Rectangle:
        return Rectangle(roi=xp.asarray([-0.2, 0.2, 0.0, 0.2]))

    @pytest.fixture
    def outside_intersection(self, outside_pupil: Rectangle,
                             xp: NumPyNamespace) -> PupilIntersection:
        return PupilIntersection.from_edge(q=xp.asarray([0.2, 0.0, 0.0]),
                                           edges=outside_pupil.edges)

    @pytest.fixture
    def outside_solutions(self, outside_pupil: Rectangle,
                          xp: NumPyNamespace) -> EdgePoints:
        parameters = xp.asarray([[0.5, 0.25], [1.5, 0.5],
                                 [0.5, 0.25], [-0.5, 0.5]])
        return outside_pupil.edges.to_points(parameters)

    @pytest.fixture
    def parallel_pupil(self, xp: NumPyNamespace) -> Rectangle:
        return Rectangle(roi=xp.asarray([-0.1, 0.1, -0.2, 0.2]))

    @pytest.fixture
    def parallel_q(self, xp: NumPyNamespace) -> RealArray:
        return xp.asarray([[0.0, 0.1, 0.0],
                           [0.0, 0.2, 0.0],
                           [0.0, 0.22, 0.0]])

    @pytest.fixture
    def parallel_intersection(self, parallel_pupil: Rectangle, parallel_q: RealArray,
                              xp: NumPyNamespace) -> PupilIntersection:
        edges = parallel_pupil.edges.replace(
            tau=xp.broadcast_to(parallel_pupil.edges.tau, parallel_q.shape[:-1] + (4, 2)),
            origin=xp.broadcast_to(parallel_pupil.edges.origin,
                                   parallel_q.shape[:-1] + (4, 2)),
        )
        return PupilIntersection.from_edge(q=parallel_q, edges=edges)

    def test_outside_score(self, outside_pupil: Rectangle,
                           outside_intersection: PupilIntersection,
                           outside_solutions: EdgePoints, xp: NumPyNamespace) -> None:
        projected = outside_solutions.project()
        edge_distance = xp.sqrt(xp.sum((outside_solutions.xy - projected.xy)**2, axis=-1))
        plane_distance = outside_intersection.source.distance(outside_solutions.points)
        candidate_score = edge_distance + plane_distance
        expected_score = xp.sort(xp.min(candidate_score, axis=-1), axis=-1)[..., :2]

        kin, score = outside_intersection.select(outside_solutions)

        # Selection minimizes the sum of finite-edge and Laue-plane violations.
        check_close(score, expected_score)
        assert xp.all(kin[..., :2] >= outside_pupil.min)
        assert xp.all(kin[..., :2] <= outside_pupil.max)

    def test_parallel_plane(self, parallel_pupil: Rectangle, parallel_q: RealArray,
                            parallel_intersection: PupilIntersection,
                            xp: NumPyNamespace) -> None:
        kin, score = parallel_intersection.select(parallel_intersection.solutions())
        plane_y = -0.5 * xp.sum(parallel_q**2, axis=-1) / parallel_q[..., 1]
        expected_y = xp.clip(plane_y, parallel_pupil.y0, parallel_pupil.y1)
        expected_x = xp.stack((parallel_pupil.x0, parallel_pupil.x1), axis=-1)
        expected_x = xp.broadcast_to(expected_x, kin[..., 0].shape)
        expected_score = xp.broadcast_to(xp.abs(plane_y - expected_y)[..., None], score.shape)

        # A parallel Laue plane intersects at both x boundaries or clips to the nearest y edge.
        check_close(xp.sort(kin[..., 0], axis=-1), xp.sort(expected_x, axis=-1))
        check_close(kin[..., 1], xp.broadcast_to(expected_y[..., None], score.shape))
        check_close(score, expected_score)
        check_close(xp.sum(kin**2, axis=-1), xp.ones(kin.shape[:-1]))
