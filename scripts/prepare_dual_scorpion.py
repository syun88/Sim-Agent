"""Verify and register the pinned upstream Dual Scorpion Genesis model.

The upstream Genesis subtree already contains the authored visual and collision meshes.
Keep those source assets untouched. This script verifies their pinned hashes, creates a
machine-local package overlay that promotes the upstream documented adjacent-pair exclusions
to Sim-Agent's explicit runtime key, then imports that package. Import and structure checks do
not certify physical grasp success or hardware safety.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = (ROOT / "third_party/momoiorg_dual_scorpion/dual_scorpion_Full_urdf/genesis").resolve()
THIRD_PARTY_ROOT = ASSET_ROOT.parents[1]
SOURCE_METADATA = THIRD_PARTY_ROOT / "SOURCE.json"
EXPECTED_URDF_SHA256 = "94097ff255ded808ccb287344d9d4fc28618388debcb2808e7a1bbd904b02a48"


def verify_source_package():
    metadata = json.loads(SOURCE_METADATA.read_text())
    if metadata["source_repo"] != "https://github.com/momoiorg-repository/dual_scorpion":
        raise RuntimeError("unexpected Dual Scorpion source repository")
    if metadata["source_revision"] != "1f64c14bfe67f957b3c718ca9f9b06965502bd51":
        raise RuntimeError("unexpected upstream revision")
    if metadata["source_urdf_sha256"] != EXPECTED_URDF_SHA256:
        raise RuntimeError("unexpected pinned URDF digest")

    entries = []
    for line in (ASSET_ROOT / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        rel_path = PurePosixPath(relative)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise RuntimeError(f"unsafe path in asset checksum manifest: {relative}")
        path = ASSET_ROOT.joinpath(*rel_path.parts)
        if not path.is_file():
            raise RuntimeError(f"upstream asset is missing: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"upstream asset hash mismatch: {relative}")
        entries.append(path)

    if len(entries) != metadata["included_files"]:
        raise RuntimeError(f"expected {metadata['included_files']} source files, found {len(entries)}")
    total_bytes = sum(path.stat().st_size for path in entries)
    if total_bytes != metadata["included_bytes"]:
        raise RuntimeError(f"source byte count mismatch: {total_bytes}")
    urdf = ASSET_ROOT / "scorpion.urdf"
    if hashlib.sha256(urdf.read_bytes()).hexdigest() != EXPECTED_URDF_SHA256:
        raise RuntimeError("pinned URDF hash mismatch")
    return metadata


def build_runtime_package(destination):
    """Link immutable upstream assets and write an explicit Sim-Agent collision overlay."""
    sys.path.insert(0, str(ROOT))
    from backend.app.robots.urdf import URDFModel

    model = URDFModel(ASSET_ROOT / "scorpion.urdf")
    adjacent = {
        tuple(sorted((joint["parent"], joint["child"])))
        for joint in model.joints
    }
    collidable = {
        pair for pair in adjacent
        if all(model.links[name].findall("collision") for name in pair)
    }
    source_config = json.loads((ASSET_ROOT / "collision_config.json").read_text())
    documented = {
        tuple(sorted((str(pair[0]), str(pair[1]))))
        for pair in source_config.get("excluded_adjacent_pairs", [])
    }
    if documented != collidable:
        raise RuntimeError(
            "upstream documented adjacent exclusions do not exactly cover collidable adjacent pairs"
        )

    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("scorpion.urdf", "initial_state.json", "meshes"):
        target = destination / name
        source = ASSET_ROOT / name
        if target.is_symlink():
            target.unlink()
        elif target.exists():
            raise RuntimeError(f"refusing to replace non-generated runtime source: {target}")
        target.symlink_to(source, target_is_directory=source.is_dir())

    overlay = dict(source_config)
    overlay["runtime_excluded_adjacent_pairs"] = [list(pair) for pair in sorted(documented)]
    config_path = destination / "collision_config.json"
    temporary = config_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(overlay, indent=2) + "\n")
    temporary.replace(config_path)
    return destination, sorted(documented)


def main():
    metadata = verify_source_package()
    runtime_root, runtime_exclusions = build_runtime_package(ROOT / "outputs/dual_scorpion_source")
    sys.path.insert(0, str(ROOT))
    from backend.app.robots.registry import import_robot

    profile = import_robot(str(runtime_root))
    chains = profile["chains"]
    grippers = profile.get("grippers", {})
    if profile["links"] != 17 or len(profile["joints"]) != 16:
        raise RuntimeError("the imported upstream model is not the pinned 17-link / 16-joint shape")
    if set(chains) != {"left", "right"} or set(grippers) != {"left", "right"}:
        raise RuntimeError("both upstream arms and grippers must be discovered")
    if any(len(chains[side]["joints"]) != 7 for side in ("left", "right")):
        raise RuntimeError("each arm must expose its seven arm joints")
    actual_exclusions = {
        tuple(sorted(pair)) for pair in profile["collision_config"]["runtime_excluded_adjacent_pairs"]
    }
    if actual_exclusions != set(map(tuple, runtime_exclusions)):
        raise RuntimeError("the imported runtime did not receive all upstream adjacent exclusions")

    result = {
        "source_repo": metadata["source_repo"],
        "source_revision": metadata["source_revision"],
        "urdf_sha256": metadata["source_urdf_sha256"],
        "robot_id": profile["id"],
        "links": profile["links"],
        "active_joints": len(profile["joints"]),
        "arms": {side: chains[side]["joints"] for side in ("left", "right")},
        "grippers": sorted(grippers),
        "runtime_excluded_adjacent_pairs": len(runtime_exclusions),
        "profile_directory": profile["directory"],
        "warnings": profile.get("warnings", []),
        "scope": "source integrity, runtime collision overlay, and URDF import structure; no physical task claim",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
