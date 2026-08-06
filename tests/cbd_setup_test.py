import pytest
from cbclib_v2 import default_rng
from cbclib_v2.annotations import Generator, NDArray, NumPyNamespace, NumPy, RealArray
from cbclib_v2.indexer import (CBDPoints, ConvexPolygon, FixedApertureSetup, FixedPupilSetup,
                               FixedSetup, Miller, MillerWithRLP, PupilIntersection, Rectangle,
                               RefinerModel, ResolvedGeometry, ResolvedSetup, RotationState,
                               SimulatedVectors, SourcePlane, XtalCell, XtalState)
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
                  resolved: ResolvedSetup, xp: NumPyNamespace):
        check_close(xp.broadcast_to(laue.q, laue.kout.shape), laue.kout - laue.kin)
        valid = laue.distance == 0.0
        kin = model.lens.project_to_pupil(laue.kin, laue.index, resolved.geometry, xp)
        check_close(laue.kin, kin)
        q = xp.broadcast_to(laue.q, laue.kin.shape)
        check_close(xp.sum(laue.kin[valid] * q[valid], axis=-1),
                    -0.5 * xp.sum(q[valid]**2, axis=-1))

    def test_points_and_kout(self, laue: SimulatedVectors, points: CBDPoints,
                             model: RefinerModel, resolved: ResolvedSetup, atol: float,
                             xp: NumPyNamespace):
        valid = xp.isclose(laue.distance, 0.0, atol=atol)
        smp_pos = model.kin_to_sample(points.kin, points.index, resolved.geometry, xp)
        kout = model.points_to_kout(points, smp_pos, xp)
        check_close(kout[valid], laue.kout[valid])
        assert xp.all(xp.isnan(points.points[~valid]))

    def test_first_order_defocus_correction(self, laue: SimulatedVectors, points: CBDPoints,
                                            model: RefinerModel, resolved: ResolvedSetup,
                                            atol: float, xp: NumPyNamespace):
        valid = xp.isclose(laue.distance, 0.0, atol=atol)
        smp_pos = model.smp_center(points.index, resolved.geometry, xp)
        kout_zero = model.points_to_kout(points, smp_pos, xp)

        kin_zero = kout_zero - points.q
        smp_pos = model.kin_to_sample(kin_zero, points.index, resolved.geometry, xp)
        kout_first = model.points_to_kout(points, smp_pos, xp)

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
        if isinstance(converted.geometry, ResolvedGeometry) and isinstance(resolved.geometry, ResolvedGeometry):
            check_close(converted.geometry.defocus, resolved.geometry.defocus)

    def test_fixed_aperture_state_from_resolved(self, resolved: ResolvedSetup, xp: NumPyNamespace):
        restored = FixedApertureSetup.from_resolved(resolved)
        converted = restored.resolve(xp)

        check_close(converted.xtal.basis, resolved.xtal.basis)
        check_close(converted.geometry.foc_pos, resolved.geometry.foc_pos)
        check_close(converted.geometry.pupil_roi, resolved.geometry.pupil_roi)
        if isinstance(converted.geometry, ResolvedGeometry) and isinstance(resolved.geometry, ResolvedGeometry):
            check_close(converted.geometry.defocus, resolved.geometry.defocus)

class TestPupilProjection:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    def test_rectangle_distance(self, xp: NumPyNamespace) -> None:
        pupil = Rectangle(roi=xp.asarray([0.0, 2.0, 0.0, 4.0]))
        points = pupil.edges.to_points(xp.broadcast_to(xp.asarray([-0.5, 0.5, 1.5]),
                                                       (4, 3)))

        projected = points.project()

        check_close(projected.t, xp.broadcast_to(xp.asarray([0.0, 0.5, 1.0]), (4, 3)))
        check_close(points.distance(),
                    xp.asarray([[1.0, 0.0, 1.0], [2.0, 0.0, 2.0],
                                [1.0, 0.0, 1.0], [2.0, 0.0, 2.0]]))

    def test_polygon_projection(self, xp: NumPyNamespace) -> None:
        pupil = ConvexPolygon(center=xp.zeros(2), lengths=xp.ones(4))
        points = pupil.edges.to_points(xp.broadcast_to(xp.asarray([-0.5, 0.5, 1.5]),
                                                       (4, 3)))

        projected = points.project()

        check_close(projected.t, xp.broadcast_to(xp.asarray([0.0, 0.5, 1.0]), (4, 3)))
        check_close(points.distance(),
                    xp.broadcast_to(xp.asarray([1.0, 0.0, 1.0]), (4, 3)))

    def test_spherical_projection(self, xp: NumPyNamespace) -> None:
        pupil = Rectangle(roi=xp.asarray([-0.2, 0.2, -0.1, 0.1]))
        kin = xp.asarray([[0.0, 0.0, 1.0], [0.2, 0.0, xp.sqrt(0.96)]])

        projected = pupil.project(kin)

        check_close(projected[0], kin[0])
        check_close(projected[1], xp.asarray([0.1, 0.0, xp.sqrt(0.99)]))
        check_close(pupil.distance(kin), xp.sqrt(xp.sum((kin - projected)**2, axis=-1)))

class TestSourcePlane:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    def test_normalised_distance(self, xp: NumPyNamespace) -> None:
        source = SourcePlane.from_q(q=xp.asarray([0.2, 0.0, 0.0]))
        kin = xp.asarray([0.0, 0.0, 1.0])

        check_close(source.distance(kin), xp.asarray(0.1))
        check_close(source.project(kin), xp.asarray([-0.1, 0.0, 1.0]))

    def test_zero_q(self, xp: NumPyNamespace) -> None:
        source = SourcePlane.from_q(q=xp.zeros(3))
        kin = xp.asarray([0.0, 0.0, 1.0])

        check_close(source.distance(kin), xp.asarray(0.0))
        check_close(source.project(kin), kin)

    def test_expand_dims(self, xp: NumPyNamespace) -> None:
        source = SourcePlane.from_q(q=xp.asarray([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0]]))
        kin = xp.zeros((2, 3, 3))

        with pytest.raises(ValueError):
            source.distance(kin)

        expanded = source.expand_dims(axis=1)

        assert expanded.q.shape == (2, 1, 3)
        assert expanded.q_mag.shape == (2, 1)
        assert expanded.distance(kin).shape == (2, 3)

class TestPupilIntersection:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    def test_outside_score(self, xp: NumPyNamespace) -> None:
        pupil = Rectangle(roi=xp.asarray([-0.2, 0.2, 0.0, 0.2]))
        intersection = PupilIntersection.from_edge(q=xp.asarray([0.2, 0.0, 0.0]),
                                                   edges=pupil.edges)
        parameters = xp.asarray([[0.5, 0.25], [1.5, 0.5],
                                 [0.5, 0.25], [-0.5, 0.5]])

        kin, score = intersection.select(pupil.edges.to_points(parameters))

        check_close(score, xp.asarray([[0.1, 0.1]]))
        assert xp.all(kin[..., :2] >= pupil.min)
        assert xp.all(kin[..., :2] <= pupil.max)
        assert xp.all(intersection.source.distance(kin) > 0.0)

    def test_parallel_plane(self, xp: NumPyNamespace) -> None:
        pupil = Rectangle(roi=xp.asarray([-0.1, 0.1, -0.2, 0.2]))
        q = xp.asarray([[0.0, 0.1, 0.0],
                        [0.0, 0.2, 0.0],
                        [0.0, 0.22, 0.0]])
        edges = pupil.edges.replace(
            tau=xp.broadcast_to(pupil.edges.tau, (3, 4, 2)),
            origin=xp.broadcast_to(pupil.edges.origin, (3, 4, 2)),
        )
        intersection = PupilIntersection.from_edge(q=q, edges=edges)

        kin, score = intersection.select(intersection.solutions())

        check_close(xp.sort(kin[..., 0], axis=-1),
                    xp.broadcast_to(xp.asarray([-0.2, 0.2]), (3, 2)))
        check_close(kin[..., 1], xp.asarray([[-0.05, -0.05],
                                             [-0.1, -0.1],
                                             [-0.1, -0.1]]))
        check_close(score, xp.asarray([[0.0, 0.0], [0.0, 0.0], [0.01, 0.01]]))
