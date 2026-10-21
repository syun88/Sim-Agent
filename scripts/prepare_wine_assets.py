"""Download the redistributable CC0 vessels and reproduce their simulation meshes.

The original GLBs remain unchanged. Open bottle visuals omit the cork; collision
surfaces are closed revolutions of the measured inner/outer profiles, preserving
the cavity and open mouth rather than convexifying the vessel into a solid lump.
"""
from pathlib import Path
import hashlib
import json

import httpx
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1] / "assets" / "wine"
SOURCES = {
    "bottle": ("wine-bottle-original-drink-vessel-dff39232", "dc1a5eaaa98fc6c23aed4f5a92be09072620e3a0891ae9c8b5775e08ee256e50"),
    "glass": ("wine-glass-original-drink-vessel-98bf1e17", "85df13d7e2f3323315fea4b8174bc5fe7cc9b3b2a6f9da384b214ca434c4606e"),
}
BOTTLE_PROFILE = [(0, 0), (.034, 0), (.0375, .012), (.0375, .208), (.031, .233), (.0125, .257), (.0125, .300),
                  (.010, .300), (.010, .257), (.0285, .233), (.035, .208), (.035, .012), (0, .007)]
GLASS_BOWL = [(.090, .009), (.094, .0148), (.096, .0178), (.098, .02061), (.100, .02307), (.102, .025),
              (.10667, .02874), (.11133, .03235), (.116, .0356), (.12067, .03827), (.12533, .04015),
              (.130, .041), (.1355, .04129), (.141, .04154), (.1465, .04174), (.152, .04188),
              (.1575, .04197), (.163, .042), (.16833, .04162), (.17367, .04062), (.179, .03924),
              (.18433, .03769), (.18967, .0362), (.195, .035), (.19917, .03426), (.20333, .03357),
              (.2075, .03292), (.21167, .03229), (.21583, .03165), (.22, .031)]


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    provenance = []
    for name, (slug, digest) in SOURCES.items():
        path = ROOT / f"{name}.glb"
        if not path.exists():
            response = httpx.get(f"https://www.innerscene.com/api/library/{slug}/download", follow_redirects=True, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source checksum changed: {name}")
        scene = trimesh.load(path, force="scene")
        meshes = []
        scale = .70 if name == "bottle" else .65
        offset = .085 if name == "bottle" else 0.
        for node in scene.graph.nodes_geometry:
            if name == "bottle" and "cork" in node.lower():
                continue
            transform, key = scene.graph[node]
            mesh = scene.geometry[key].copy()
            mesh.apply_transform(transform)
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
            mesh.apply_scale(scale)
            mesh.vertices[:, 2] -= offset
            meshes.append(mesh)
        visual = trimesh.util.concatenate(meshes)
        # The native rasterizer sorts transparent objects, not individual faces.
        # Nested inner/outer walls therefore blend in triangle order and produce
        # a speckled bowl. Use a single outward-facing visual skin; the separate
        # watertight collision shell below retains the full physical cavity.
        centers = visual.triangles_center
        radial_normal = (centers[:, :2] * visual.face_normals[:, :2]).sum(axis=1)
        interior_wall = radial_normal < -1e-6
        floor_band = (centers[:, 2] < .02) if name == "bottle" else (
            (centers[:, 2] > .0585) & (centers[:, 2] < .07))
        interior_floor = floor_band & (visual.face_normals[:, 2] > .999)
        visual.update_faces(~(interior_wall | interior_floor))
        visual.remove_unreferenced_vertices()
        # MJCF supplies the colors. Avoid exporting two unrelated materials to
        # the same material.mtl and leaving the bottle's material unresolved.
        visual.export(ROOT / f"{name}_open.obj", include_texture=False)
        if name == "glass":
            z = visual.triangles_center[:, 2]
            for part_name, mask in (("base", z < .005), ("stem", (z >= .005) & (z < .0585)),
                                    ("bowl", z >= .0585)):
                part = visual.copy()
                part.update_faces(mask)
                part.remove_unreferenced_vertices()
                part.export(ROOT / f"glass_{part_name}.obj", include_texture=False)
        if name == "bottle":
            profile = BOTTLE_PROFILE
        else:
            outer = [(0, 0), (.02905, 0), (.035, .002), (.03255, .005), (.01225, .007), (.004, .0077), (.004, .09)]
            outer += [(r, z) for z, r in GLASS_BOWL]
            inner = [(r - .0018, z) for z, r in reversed(GLASS_BOWL[2:])]
            profile = outer + inner + [(0, .094)]
        collision = trimesh.creation.revolve(np.asarray(profile), sections=64)
        collision.apply_scale(scale)
        collision.vertices[:, 2] -= offset
        assert collision.is_watertight and collision.is_winding_consistent
        collision.export(ROOT / f"{name}_collision.obj")
        provenance.append({"name": name, "source": f"https://www.innerscene.com/tools/library/3d-parts/{slug}",
                           "license": "CC0-1.0", "original_sha256": digest, "scale": scale,
                           "origin_z_offset_m": offset, "collision_volume_m3": collision.volume,
                           "collision_bounds_m": collision.bounds.tolist(),
                           "visual_representation": "single outward skin; back faces culled"})
    for path in ROOT.glob("*.obj"):
        path.write_text(path.read_text().rstrip() + "\n")
    (ROOT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
