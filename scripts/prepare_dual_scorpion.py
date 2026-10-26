"""Prepare the DualScorpion description the way the registry attaches it.

The upstream URDF ships visual meshes only: 18 links, 17 visuals, **no collision elements at
all**, and each visual carries a non-zero origin because the meshes are baked at their assembly
position (for example ds_urdf_v4 at [0.259, 0.003, -0.076]). Two consequences drive this script:

- Collision geometry has to be created, not copied. Each link's mesh is decomposed with CoACD into
  convex pieces and every piece is written with the **visual's own origin and scale**. Skipping that
  is the documented failure of this repo: "a proxy written without them is a thousand times too
  large and offset (measured: the declaration then read a 16 m overlap)". A convex hull of the whole
  mesh is not an option either - hulling a concave link adds volume (measured at 4.3x on
  OpenManipulator-X).
- The mesh filenames say `package://assembly_1/meshes/...`, which no importer can resolve, so the
  meshes are copied next to the URDF and the references are rewritten relative to it.

The adjacent-pair declaration is measured, not guessed: `_collision_declaration` from the
OpenManipulator-X preparation is reused as-is, so the two packages cannot drift apart.
"""
import argparse
import copy
import json
import pathlib
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

DEFAULT_RAW = (pathlib.Path.home() / "Documents" / "dual-scorpion-sim" / "third_party"
               / "momoiorg_dual_scorpion" / "dual_scorpion_Full_urdf")
DEFAULT_OUT = pathlib.Path.home() / "Documents" / "dual_scorpion"


def _clean(mesh):
    solid = mesh.copy()
    solid.merge_vertices()
    solid.update_faces(solid.nondegenerate_faces())
    solid.update_faces(solid.unique_faces())
    trimesh.repair.fix_winding(solid)
    trimesh.repair.fix_inversion(solid)
    trimesh.repair.fix_normals(solid)
    return solid


def _rigid(extents):
    """A box that matches the mesh's own extents, for meshes CoACD cannot split."""
    return trimesh.creation.box(extents=extents)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=pathlib.Path, default=DEFAULT_RAW,
                        help="upstream package holding urdf/ and meshes/")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT,
                        help="where the prepared URDF and meshes are written")
    parser.add_argument("--concavity", type=float, default=.05,
                        help="CoACD concavity threshold, lower is more pieces")
    parser.add_argument("--max-pieces", type=int, default=8,
                        help="CoACD cap per link mesh")
    parser.add_argument("--overlap-threshold-mm", type=float, default=.1,
                        help="declare an adjacent overlap at or above this depth")
    args = parser.parse_args()

    import coacd
    import os
    os.environ.setdefault("GENESIS_SHOW_VIEWER", "0")

    source = args.raw / "urdf" / "dual_scorpion.urdf"
    if not source.is_file():
        raise SystemExit(f"no URDF at {source}")
    args.out.mkdir(parents=True, exist_ok=True)
    mesh_out = args.out / "meshes"
    mesh_out.mkdir(exist_ok=True)
    for stl in (args.raw / "meshes").glob("*"):
        shutil.copy(stl, mesh_out / stl.name)

    tree = ET.parse(source)
    root = tree.getroot()
    pieces_total, replaced, skipped = 0, 0, []
    for link in root.findall("link"):
        visuals = link.findall("visual")
        if not visuals:
            continue
        for element in link.findall("collision"):
            link.remove(element)
        for visual in visuals:
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            name = pathlib.Path(mesh.get("filename")).name
            origin = visual.find("origin")
            scale = mesh.get("scale") or "1 1 1"
            mesh.set("filename", f"meshes/{name}")
            local = mesh_out / name
            if not local.is_file():
                skipped.append(f"{link.get('name')}:{name}")
                continue
            solid = _clean(trimesh.load(local, force="mesh", process=False))
            # The pieces must be expressed in the coordinates the mesh itself uses, because each
            # one is written under the visual's own origin: that is what keeps the collision
            # geometry on the part instead of beside it.
            decomposed = []
            try:
                decomposed = coacd.run_coacd(
                    coacd.Mesh(np.asarray(solid.vertices, np.float64), np.asarray(solid.faces, np.int32)),
                    threshold=args.concavity, max_convex_hull=args.max_pieces,
                    preprocess_mode="auto", preprocess_resolution=50, resolution=1000,
                    mcts_iterations=60, seed=0)
            except Exception as error:
                print(f"  CoACD failed on {link.get('name')}: {type(error).__name__}", flush=True)
            if not decomposed:
                decomposed = [(np.asarray(solid.vertices, float), np.asarray(solid.faces, np.int64))]
            for index, (vertices, faces) in enumerate(decomposed):
                piece = trimesh.Trimesh(np.asarray(vertices, float), np.asarray(faces, np.int64), process=True)
                if not len(piece.faces):
                    continue
                relative = f"{link.get('name')}_proxy_{index}.stl"
                piece.export(mesh_out / relative)
                collision = ET.SubElement(link, "collision")
                if origin is not None:
                    collision.append(copy.deepcopy(origin))
                geometry = ET.SubElement(collision, "geometry")
                ET.SubElement(geometry, "mesh", filename=f"meshes/{relative}", scale=scale)
                pieces_total += 1
            replaced += 1

    if skipped:
        raise SystemExit("meshes referenced but not found: " + ", ".join(skipped))
    ET.indent(tree)
    destination = args.out / "dual_scorpion.urdf"
    tree.write(destination, encoding="utf-8")

    collisions = sum(len(link.findall("collision")) for link in root.findall("link"))
    print(f"  links={len(root.findall('link'))} collision_elements={collisions} pieces={pieces_total}",
          flush=True)

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from prepare_open_manipulator_x import _collision_declaration
    declaration = args.out / "collision_config.json"
    declaration.unlink(missing_ok=True)
    settings = _collision_declaration(destination, args.overlap_threshold_mm)
    if settings:
        # The runtime request is all-or-nothing: the engine turns adjacent collision off wholesale,
        # so a partial set would drop a constraint the private queries still honour (the import
        # refuses a partial declaration outright). Name every adjacent pair that carries collision
        # geometry, and keep the measured overlaps in the file as the record of why.
        from backend.app.robots.urdf import URDFModel
        model = URDFModel(str(destination))
        adjacent = {tuple(sorted((joint['parent'], joint['child']))) for joint in model.joints}
        settings["runtime_excluded_adjacent_pairs"] = [list(pair) for pair in sorted(
            {pair for pair in adjacent
             if all(len(model.links[link].findall('collision')) for link in pair)})]
        declaration.write_text(json.dumps(settings, indent=2) + "\n")
        print("  collision_config.json written (%d pairs)" % len(settings["runtime_excluded_adjacent_pairs"]),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
