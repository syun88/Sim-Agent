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
A supplied initial_state.json is preserved; otherwise a floor-mounted, collision-free setup
is sampled with terminal-joint apertures unchanged for discovery, but also checked with the
runtime's opened grippers. Non-adjacent collisions stay enabled.
Preparation does not certify motor tracking or physical grasping; those require runtime checks.
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


def _derive_home(destination, budget=2000):
    """Choose a collision-free setup without disabling any non-adjacent pair."""
    from backend.app.robots.registry import import_robot
    from backend.app.robots.collision_planner import PrivateCollisionQueries

    profile = import_robot(str(destination))
    queries = PrivateCollisionQueries(profile, table_top=0.)
    opened = {}
    for gripper in profile['grippers'].values():
        opened.update({gripper['joint']: float(gripper['open_q']),
                       **{name: float(gripper['open_q'] * k) for name, k in gripper['followers'].items()}})
    # Preserve terminal-joint apertures: moving them before gripper discovery can fold
    # the finger out of its inference frame and make both grippers disappear on import.
    parents = {j['parent'] for j in queries.urdf.joints}
    movable = [j for j in profile['joints'] if j['child'] in parents]
    rng = np.random.default_rng(43)
    for attempt in range(budget + 1):
        if attempt == 0:
            candidate = dict(profile['home'])
        else:
            candidate = {**profile['home'], **{j['name']: float(rng.uniform(j['lower'], j['upper']))
                                             for j in movable}}
        # reset() opens the jaws, independently of their authored discovery posture. A setup
        # clear only with the authored aperture can start in collision (measured: 8 mm).
        if (not queries.contacts(candidate, tolerance=-.0005)
                and not queries.contacts({**candidate, **opened}, tolerance=-.0005)):
            return {**candidate, 'mount': 'floor'}
    raise RuntimeError(f'No collision-free setup in {budget} sampled postures')


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
    parser.add_argument("--skin-mm", type=float, default=.5,
                        help="pull each collision piece inside the authored surface by this much")
    parser.add_argument("--overlap-threshold-mm", type=float, default=.1,
                        help="declare an adjacent overlap at or above this depth")
    args = parser.parse_args()

    import coacd
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from backend.app.robots.collision_geometry import inset_authored_boundary, float32_convex_piece
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
    pieces_total, skipped = 0, []
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
            # Preserve an already closed solid. Voxel preprocessing can inflate its boundary
            # (base_link grew ~3 mm at resolution 50), blocking every non-adjacent IK check.
            solid = trimesh.load(local, force="mesh", process=True)
            if not solid.is_volume:
                solid = _clean(solid)
            # The pieces must be expressed in the coordinates the mesh itself uses, because each
            # one is written under the visual's own origin: that is what keeps the collision
            # geometry on the part instead of beside it.
            decomposed = []
            try:
                decomposed = coacd.run_coacd(
                    coacd.Mesh(np.asarray(solid.vertices, np.float64), np.asarray(solid.faces, np.int32)),
                    threshold=args.concavity, max_convex_hull=args.max_pieces,
                    preprocess_mode="off" if solid.is_volume else "auto",
                    preprocess_resolution=50, resolution=1000,
                    mcts_iterations=60, seed=0)
            except Exception as error:
                raise RuntimeError(f"CoACD failed on {link.get('name')}: {error}") from error
            if not decomposed:
                raise RuntimeError(f"No collision pieces for {link.get('name')}")
            kept = 0
            for index, (vertices, faces) in enumerate(decomposed):
                # Weld and orient the decomposed solid before clipping its authored boundary.
                piece = _clean(trimesh.Trimesh(np.asarray(vertices, float), np.asarray(faces, np.int64),
                                               process=True))
                if not len(piece.faces) or not piece.is_volume:
                    print(f"  omitted non-volume piece: {link.get('name')}:{index}", flush=True)
                    continue
                # Skin authored faces, not internal partition faces; this is not a cure for
                # voxel inflation or proof that a concave link's whole convex hull is faithful.
                inset = inset_authored_boundary(piece, solid, args.skin_mm / 1000.)
                if inset is None or not len(inset.faces):
                    print(f"  omitted skin-only piece: {link.get('name')}:{index}", flush=True)
                    continue
                piece = float32_convex_piece(inset)
                if piece is None:
                    print(f"  omitted zero-volume piece: {link.get('name')}:{index}", flush=True)
                    continue
                relative = f"{link.get('name')}_proxy_{index}.stl"
                piece.export(mesh_out / relative)
                collision = ET.SubElement(link, "collision")
                if origin is not None:
                    collision.append(copy.deepcopy(origin))
                geometry = ET.SubElement(collision, "geometry")
                ET.SubElement(geometry, "mesh", filename=f"meshes/{relative}", scale=scale)
                pieces_total += 1
                kept += 1
            if not kept:
                raise RuntimeError(f"{link.get('name')}: no collision piece survived")

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
    # A floor mount is independent of wheel support. Keep an authored setup, if supplied;
    # otherwise derive one only after the adjacent-pair declaration is complete.
    setup = args.out / 'initial_state.json'
    derive_home = not setup.is_file()
    if derive_home:
        setup.write_text(json.dumps({'mount': 'floor'}) + '\n')
    declaration = args.out / "collision_config.json"
    declaration.unlink(missing_ok=True)
    try:
        settings = _collision_declaration(destination, args.overlap_threshold_mm)
        if settings:
            # The engine disables adjacent collision wholesale, so declare every collidable
            # adjacent pair. Non-adjacent pairs remain checked, including at the setup posture.
            from backend.app.robots.urdf import URDFModel
            model = URDFModel(str(destination))
            adjacent = {tuple(sorted((joint['parent'], joint['child']))) for joint in model.joints}
            settings["runtime_excluded_adjacent_pairs"] = [list(pair) for pair in sorted(
                {pair for pair in adjacent
                 if all(len(model.links[link].findall('collision')) for link in pair)})]
            declaration.write_text(json.dumps(settings, indent=2) + "\n")
            print("  collision_config.json written (%d pairs)" % len(settings["runtime_excluded_adjacent_pairs"]),
                  flush=True)
        if derive_home:
            setup.write_text(json.dumps(_derive_home(destination), indent=2) + '\n')
        else:
            # An authored setup is preserved, but must satisfy the same execution clearance.
            _derive_home(destination, budget=0)
    except Exception:
        if derive_home:
            setup.unlink(missing_ok=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
