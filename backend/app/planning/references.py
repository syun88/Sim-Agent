"""Literal object references shared by parsing and independent intent checks.

The hand-tuned patterns below exist because the boundary rules are subtle (red must not match inside
"red wine", 青 must not match inside 青いゾーン). They stay authoritative for the objects they name. Any
other object in `planning/catalog.py` gets a pattern from the names the catalog declares, so a cup added to
the catalog can be referred to without touching this file.
"""
import re

from backend.app.planning import catalog

HAND_TUNED_PATTERNS = {
    "red_cube": r"(?<![a-z0-9_])red(?![ _-]*wine)(?:[ _-]*cube)?(?![a-z0-9_])|赤(?!ワイン)(?:い|の)?(?:キューブ|ブロック|箱)?|あか",
    "blue_cube": r"(?<![a-z0-9_])blue(?:[ _-]*cube)?(?![a-z0-9_])|青(?:い|の)?(?:キューブ|ブロック|箱)?|あお",
    "wine_bottle": r"ワイン(?:ボトル|の?瓶)?|ボトル|瓶|\bwine(?:[ _-]*bottle)?\b|\bbottle\b",
    "wine_glass": r"グラス|ワイングラス|\b(?:wine[ _-]*)?glass\b",
}


def _patterns() -> dict[str, str]:
    patterns = catalog.alias_patterns(catalog.load().objects)
    patterns.update(HAND_TUNED_PATTERNS)
    return patterns


OBJECT_PATTERNS = _patterns()


def object_mentions(text, state):
    available = {item["id"] for item in state["objects"]}
    matches = []
    for name, pattern in OBJECT_PATTERNS.items():
        if name not in available:
            continue
        for match in re.finditer(pattern, text, re.I):
            if re.match(r"(?:い|の|[\s_-])*(?:ゾーン|エリア|zone|target)", text[match.end():], re.I):
                continue
            matches.append((match.start(), match.end(), name))
    return sorted(matches)


def named_objects(text, state):
    return list(dict.fromkeys(name for _, _, name in object_mentions(text, state)))
