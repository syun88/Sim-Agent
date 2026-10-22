"""Literal object references shared by parsing and independent intent checks."""
import re

OBJECT_PATTERNS = {
    "red_cube": r"(?<![a-z0-9_])red(?![ _-]*wine)(?:[ _-]*cube)?(?![a-z0-9_])|赤(?!ワイン)(?:い|の)?(?:キューブ|ブロック|箱)?|あか",
    "blue_cube": r"(?<![a-z0-9_])blue(?:[ _-]*cube)?(?![a-z0-9_])|青(?:い|の)?(?:キューブ|ブロック|箱)?|あお",
    "wine_bottle": r"ワイン(?:ボトル|の?瓶)?|ボトル|瓶|\bwine(?:[ _-]*bottle)?\b|\bbottle\b",
    "wine_glass": r"グラス|ワイングラス|\b(?:wine[ _-]*)?glass\b",
}


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
