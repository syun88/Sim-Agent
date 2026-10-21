"""Select the predeclared scene before language grounding, without executing text."""
import re


def scene_for_instruction(instruction: str, current: str = "cubes") -> str:
    if re.search(r"ワイン|ボトル|グラス|瓶|\bwine\b|\bbottle\b|\bglass\b|\bpour\b|注(?:ぐ|いで)", instruction, re.I):
        return "wine"
    if re.search(r"キューブ|ブロック|\bcube\b|\bred\b|\bblue\b|[赤青](?:い|を|の)", instruction, re.I):
        return "cubes"
    return current
