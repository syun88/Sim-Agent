"""Collision shape checks; these do not claim simulator-runtime validation."""
import numpy as np
import trimesh
from backend.app.robots.collision_geometry import joint_bore_parts, faithful_convex_parts, inset_authored_boundary, float32_convex_piece


def test_bearing_clearance_removes_only_internal_cylinder_and_preserves_outer_case():
    box = trimesh.creation.box(extents=[.04, .04, .02])
    pieces = joint_bore_parts(box, np.zeros(3), np.array([0., 0., 1.]), .005)
    assert len(pieces) == 16 and all(p.is_convex for p in pieces)
    assert not any(p.contains([[0., 0., 0.]])[0] for p in pieces)
    assert any(p.contains([[.015, .001, 0.]])[0] for p in pieces)
    removed = box.volume - sum(p.volume for p in pieces)
    assert np.isclose(removed, np.pi * .005**2 * .02, rtol=.06)


def test_refinement_keeps_separated_material_separate():
    a = trimesh.creation.box(extents=[.01, .02, .02]);a.apply_translation([-.015, 0., 0.])
    b = a.copy();b.apply_translation([.03, 0., 0.])
    visual = trimesh.util.concatenate([a, b])
    parts = faithful_convex_parts(visual.convex_hull, visual, linear_tolerance=None)
    assert len(parts) == 2
    assert not any(p.contains([[0., 0., 0.]])[0] for p, _ in parts)
    assert np.isclose(sum(p.volume for p, _ in parts), visual.volume)


def test_skin_does_not_shrink_an_internal_partition_face():
    visual = trimesh.creation.box(extents=[.04, .04, .04])
    piece = trimesh.creation.box(extents=[.02, .04, .04]);piece.apply_translation([-.01, 0., 0.])
    inset = inset_authored_boundary(piece, visual, .001)
    np.testing.assert_allclose(inset.bounds[:, 0], [-.019, 0.], atol=1e-7)


def test_skin_expands_a_cavity_instead_of_only_insetting_the_outer_hull():
    outer = trimesh.creation.box(extents=[.04, .04, .02])
    hole = trimesh.creation.box(extents=[.02, .02, .04])
    visual = trimesh.boolean.difference([outer, hole], engine='manifold')
    wall = trimesh.creation.box(extents=[.01, .04, .02]);wall.apply_translation([.015, 0., 0.])
    inset = inset_authored_boundary(wall, visual, .001)
    np.testing.assert_allclose(inset.bounds[:, 0], [.011, .019], atol=1e-7)


def test_export_hull_has_no_collapsed_faces_at_physics_precision():
    mesh = trimesh.creation.box(extents=[.03, .03, .03])
    vertices = np.vstack([mesh.vertices, mesh.vertices[0]+1e-12])
    broken = trimesh.Trimesh(vertices, np.vstack([mesh.faces, [0, 1, len(vertices)-1]]), process=False)
    result = float32_convex_piece(broken)
    assert result.is_watertight and np.all(result.area_faces>0)
    np.testing.assert_allclose(result.bounds, mesh.bounds, atol=2e-9)
