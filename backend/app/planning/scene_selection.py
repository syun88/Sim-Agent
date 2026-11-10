"""Select the predeclared scene before language grounding, without executing text."""
import re


WINE = r"ワイン|ボトル|グラス|瓶|\bwine\b|\bbottle\b|\bglass\b|\bpour\b|注(?:ぐ|いで)"
CUBE = r"キューブ|ブロック|\bcube\b|\bred\b|\bblue\b|[赤青](?:い|を|の)"


def scene_for_instruction(instruction: str, current: str = "cubes") -> str:
    # Both kinds of object in one sentence need both in one scene.
    wants_wine = bool(re.search(WINE, instruction, re.I))
    wants_cubes = bool(re.search(CUBE, instruction, re.I))
    if wants_wine and wants_cubes:
        return "mixed"
    if wants_wine:
        return "wine"
    if wants_cubes:
        return "cubes"
    return current
