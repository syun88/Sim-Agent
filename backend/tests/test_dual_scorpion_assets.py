import hashlib
import json
import xml.etree.ElementTree as ET

from scripts.prepare_dual_scorpion import (
    ASSET_ROOT,
    SOURCE_METADATA,
    build_runtime_package,
    verify_source_package,
)


def test_dual_scorpion_assets_match_the_pinned_upstream_package():
    metadata = verify_source_package()
    assert metadata["source_repo"] == "https://github.com/momoiorg-repository/dual_scorpion"
    assert metadata["source_revision"] == "1f64c14bfe67f957b3c718ca9f9b06965502bd51"
    assert metadata["license"] == "Apache-2.0"
    assert (SOURCE_METADATA.parent / metadata["license_file"]).is_file()
    assert hashlib.sha256((ASSET_ROOT / "scorpion.urdf").read_bytes()).hexdigest() == metadata["source_urdf_sha256"]


def test_dual_scorpion_urdf_mesh_references_resolve_and_keep_both_arms():
    root = ET.parse(ASSET_ROOT / "scorpion.urdf").getroot()
    assert root.get("name") == "scorpion"
    assert len(root.findall("link")) == 17
    active_joints = [joint for joint in root.findall("joint") if joint.get("type") != "fixed"]
    assert len(active_joints) == 16
    names = {joint.get("name") for joint in active_joints}
    for side in ("L", "R"):
        assert {f"joint{i}_{side}" for i in range(8)} <= names

    references = [mesh.get("filename") for mesh in root.findall(".//mesh")]
    assert references
    assert all((ASSET_ROOT / reference).is_file() for reference in references)


def test_runtime_overlay_explicitly_applies_the_upstream_adjacent_exclusions(tmp_path):
    runtime_root, exclusions = build_runtime_package(tmp_path / "runtime-source")
    source_config = json.loads((ASSET_ROOT / "collision_config.json").read_text())
    runtime_config = json.loads((runtime_root / "collision_config.json").read_text())

    documented = {tuple(sorted(pair)) for pair in source_config["excluded_adjacent_pairs"]}
    declared = {tuple(sorted(pair)) for pair in runtime_config["runtime_excluded_adjacent_pairs"]}
    assert documented == declared == set(map(tuple, exclusions))
    assert len(declared) == 16
    assert (runtime_root / "scorpion.urdf").resolve() == (ASSET_ROOT / "scorpion.urdf").resolve()
