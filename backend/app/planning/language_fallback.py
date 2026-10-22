"""Small, explicitly labelled Japanese/English fallback for the typed task language.

Unrecognised clauses fail closed. Free paraphrases go to the configured language model;
this grammar never silently executes just the first recognisable part of a sentence.
"""
from __future__ import annotations

import re
import unicodedata

from backend.app.planning.task_program import GroundingError, TaskAction, TaskProgram
from backend.app.planning.references import named_objects
from backend.app.planning.spatial_language import placement_references, strip_preservation

VERBS = re.compile(
    r"(?P<throw>投げ(?:て|る|ろ)|放り(?:投げて|投げる)|上[にへ](?:軽く|ぽんと|そっと|少し|ちょっと|\s)*放(?:って|つ)|\bthrow\b|\btoss\b)"
    r"|(?P<pour>注(?:いで|ぐ|げ)|そそ(?:いで|ぐ)|\bpour\b)"
    r"|(?P<handover>受け渡(?:して|す)|渡(?:した|して|す|せ)|手渡(?:して|す)|\bhandover\b|\bhand[ -]over\b|\bpass(?:es)?\b|\bhand(?=\s+(?:it|the|that)))"
    r"|(?P<place>置(?:いた|いて|く|け)|(?:乗|載)せ(?:て|る)|重ね(?:て|る)|積(?:んで|む)|入(?:れて|れる)|運(?:んで|ぶ)|\bplace\b|\bput(?:s)?\b|\bstack\b|\bset\s+down\b|\bleave\b)"
    r"|(?P<pick>掴(?:んで|む|め)|つか(?:んで|む)|持ち上げ(?:て|る)|握(?:って|る)|取(?:って|る)|と(?:って|る)|\bpick(?:s)?(?:\s+up)?\b|\bgrab\b|\bgrasp\b|\btake(?:s)?\b|\blift\b)"
    r"|(?P<approach>近づ(?:いて|けて|く)|アプローチ|\bapproach\b)"
    r"|(?P<open>開(?:いて|けて|く|け)|\bopen\b)"
    r"|(?P<close>閉(?:じて|めて|じる|める)|\bclose\b)"
    r"|(?P<home>初期姿勢|ホーム|\bhome\b)"
    r"|(?P<move>動か(?:して|す)|移動(?:して|する)?|\bmove\b)", re.I,
)
ARMS = {"left": r"左(?:手|腕|アーム|ハンド)?|\bleft(?:\s+(?:hand|arm))?\b", "right": r"右(?:手|腕|アーム|ハンド)?|\bright(?:\s+(?:hand|arm))?\b"}


def arm_mentions(text: str) -> list[str]:
    matches = [(m.start(), side) for side, pattern in ARMS.items() for m in re.finditer(pattern, text, re.I)]
    return [side for _, side in sorted(matches)]


def target_in(text: str) -> str | None:
    if re.search(r"グラス|\bglass\b", text, re.I):
        return "wine_glass"
    if re.search(r"そこに|同じ場所|\bthere\b|\bsame (?:place|target)\b", text):
        return "there"
    for target, pattern in {
        "blue_zone": r"blue[ _]*(?:zone|target)|青(?:い|の)?(?:ゾーン|エリア)",
        "green_zone": r"green[ _]*(?:zone|target)|緑(?:の|色の)?(?:ゾーン|エリア)",
        "center": r"中央|真ん中|\bcent(?:er|re)\b|\bmiddle\b",
        "left_side": r"左(?:手)?(?:の)?(?:ところ|側|方|場所)|左に|\bleft(?:\s+(?:side|area))?(?!\s+(?:hand|arm))\b",
        "right_side": r"右(?:手)?(?:の)?(?:ところ|側|方|場所)|右に|\bright(?:\s+(?:side|area))?(?!\s+(?:hand|arm))\b",
    }.items():
        if re.search(pattern, text, re.I):
            return target
    return None


def direction_in(text: str) -> str | None:
    for direction, pattern in {
        "up": r"上[にへ]|\bup\b", "down": r"下[にへ]|\bdown\b",
        "right": r"右[にへ]|\bright\b(?!\s+(?:hand|arm))", "left": r"左[にへ]|\bleft\b(?!\s+(?:hand|arm))",
        "forward": r"前[にへ]|前方|\bforward\b|\bfront\b", "back": r"後ろ[にへ]|奥に|\bback(?:ward)?s?\b",
    }.items():
        if re.search(pattern, text, re.I):
            return direction
    return None


def parse_fallback(instruction: str, state: dict) -> TaskProgram:
    text = unicodedata.normalize("NFKC", instruction).lower().strip()
    joints = state.get("robot", {}).get("joints", [])
    for joint in joints:
        name = joint["name"]
        matched = re.fullmatch(r"(?:set\s+)?" + re.escape(name.lower()) +
            r"\s*(?:を|to|=)?\s*(-?\d+(?:\.\d+)?)\s*(?:度|degrees?|deg|°)\s*(?:にして|に動かして|にする|へ|に|please)?[。.!\s]*", text)
        if matched:
            return TaskProgram(actions=[TaskAction(op="joint", joint=name, position_deg=float(matched.group(1)))])
    if re.search(r"\b(?:yellow|purple|orange|black|white)\b|黄色|紫色|白い|黒い|緑(?:の|色の)?(?:キューブ|cube)|green\s+cube|\b(?:ball|sphere|cylinder)\b", text):
        raise GroundingError("This scene has only a red cube and a blue cube")
    # Bounded noun-only requests: the omitted predicate is unambiguously pour.
    # Full matching prevents an unknown trailing task from being silently dropped.
    number = r"\d+(?:\.\d+)?"
    noun_pour = re.fullmatch(
        rf"(?:赤)?ワイン(?:を)?\s*(?:(?P<before>{number})\s*ml(?:くらい|ほど)?[、,\s]*)?"
        rf"(?:そこの|その|あの|ここの)?グラス[にへ]\s*"
        rf"(?:(?P<after>{number})\s*ml(?:くらい|ほど)?|少しだけ|少し|ちょっと)?\s*お願い(?:します)?[。.!！\s]*"
        r"(?:(?:終わったら|その後|あとで)?[、,\s]*ボトル(?:も|を)"
        r"(?:(?:元の(?:場所|ところ)|元)[にへ])?戻して(?:おいて|ね)?[。.!！\s]*)?", text)
    if noun_pour:
        amount = noun_pour.group("before") or noun_pour.group("after")
        return TaskProgram(actions=[TaskAction(op="pour", object="wine_bottle", target="wine_glass",
                                               volume_ml=float(amount) if amount else 20.)],
                           assumptions=[] if amount else ["Unspecified small pour means 20 mL"])
    # A colour or ID by itself is a useful informal pick request; prose without a verb isn't.
    if re.fullmatch(r"(?:red|blue)(?:[ _-]*cube)?|[赤青](?:い|の)?(?:キューブ|ブロック|箱)?", text):
        return TaskProgram(actions=[TaskAction(op="pick", object=named_objects(text, state)[0])])
    text = strip_preservation(text, state)
    verbs = list(VERBS.finditer(text))
    if not verbs:
        raise GroundingError("No supported action understood; describe which cube, hand and destination")
    actions: list[TaskAction] = []
    last_object: str | None = None
    last_arm: str | None = None
    english = not bool(re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", text))
    # Japanese predicates end clauses. English predicates start clauses; split at sequencing
    # connectors first so that 'right hand picks ...' retains its explicit subject.
    segments: list[tuple[str, str]] = []
    if english:
        clauses = re.split(r"[,;.]+|\b(?:and then|then|after that|next(?!\s+to\b)|afterwards)\b|\band\s+(?=(?:pick|grab|pass|hand|put|place|move|open|close|take|lift|throw|toss|pour)\b)", text)
        for clause in clauses:
            if re.search(r"\b(?:do\s+not|don't|never)\b", clause, re.I):
                continue
            found = list(VERBS.finditer(clause))
            if len(found) > 1:
                raise GroundingError("Fallback cannot resolve this compound clause; use the language model or separate with 'then'")
            if not found:
                if clause.strip(" , "):
                    raise GroundingError(f"Unrecognised clause: {clause.strip()}")
                continue
            segments.append((found[0].lastgroup, clause))
    else:
        previous = 0
        for verb in verbs:
            segments.append((verb.lastgroup, text[previous:verb.end()]))
            previous = verb.end()
        remainder = text[previous:]
        if re.sub(r"ください|下さい|お願い(?:します)?|です|ね|よ|[\s、。,.!！?？]", "", remainder):
            raise GroundingError(f"Unrecognised instruction suffix: {remainder}")
    for op, segment in segments:
        names = named_objects(segment, state)
        relative = placement_references(segment, state) if op == "place" else []
        if len(relative) > 1:
            raise GroundingError("Resolve compound spatial references with the language model")
        relation = relative[0]["relation"] if relative else None
        if relative:
            # The last noun can be the support/reference, while the moved cube
            # is the held/discourse object from the preceding clause.
            anchor = relative[0]["target"]
            names = [name for name in names if name != anchor]
        if (op == "place" and "wine_bottle" in names and "wine_glass" in names
                and re.search(r"ワイン|\bwine\b", segment)
                and not re.search(r"(?:ワイン)?(?:ボトル|瓶)を|(?:put|place)\s+(?:the\s+)?(?:wine\s+)?bottle\b", segment)):
            op = "pour"
        if op == "pour":
            names = [n for n in names if n != "wine_glass"]
        arms = arm_mentions(segment)
        reference = bool(re.search(r"\bit\b|\bthat\b|それ|さっきの|その(?:物|キューブ|箱)", segment))
        obj = names[0] if names else ("it" if reference else None)
        if obj is None:
            if re.search(r"左(?:側)?の(?:キューブ|箱)|\bleft(?:most)?\s+cube", segment): obj = "left_cube"
            if re.search(r"右(?:側)?の(?:キューブ|箱)|\bright(?:most)?\s+cube", segment): obj = "right_cube"
        arm = arms[0] if arms else None
        target = relative[0]["target"] if relative else target_in(segment)
        if op == "move" and target and not re.search(r"\d", segment):
            op = "place"
        if op in {"pick", "place", "handover", "approach", "throw", "pour"}:
            if len(names) > 1 or (op == "pick" and re.search(r"\bboth\b|\ball\b|\btwo\b|2つ|両方|すべて", segment)):
                if op != "pick" and op != "place":
                    raise GroundingError("Specify one object for each handover")
                for name in names or [x["id"] for x in state["objects"]]:
                    actions.append(TaskAction(op=op, object=name, target=target))
                continue
            if obj is None:
                if last_object:
                    obj = "it"
                elif re.search(r"左.*(?:持って|握って|held)|held.*left", segment):
                    obj = "held:left"
                elif re.search(r"右.*(?:持って|握って|held)|held.*right", segment):
                    obj = "held:right"
                else:
                    obj = None
        receiver = None
        if op == "handover":
            # 'to left' / '左手に' names the receiver, not the acting hand.
            receiver = arms[-1] if arms else ("left" if last_arm == "right" else "right" if last_arm else None)
            arm = arms[0] if len(arms) > 1 else last_arm
            if arm is None and receiver:
                arm = "right" if receiver == "left" else "left"
        if op == "place":
            # Destination words do not change the acting hand. An explicit 'with left hand'
            # or '左手で' can; otherwise continue with the current holder.
            explicit = re.search(r"(left|right)\s+(?:hand|arm)|([左右])(?:手|腕)で", segment)
            arm = None if target else arm
            if explicit and ("with" in segment or "で" in segment):
                arm = "left" if explicit.group(0).startswith(("left", "左")) else "right"
        distance = None
        direction = None
        if op == "move":
            direction = direction_in(segment)
            match = re.search(r"(\d+(?:\.\d+)?)\s*(mm|ミリ|cm|センチ|m|メートル)?", segment)
            distance = float(match.group(1)) if match else 5.0
            if match and match.group(2) in {"mm", "ミリ"}: distance /= 10
            if match and match.group(2) in {"m", "メートル"}: distance *= 100
        action = TaskAction(op=op, object=obj, arm=arm, receiver=receiver, target=target,
                            direction=direction, distance_cm=distance, relation=relation)
        if op == "throw":
            action.direction = direction_in(segment) or "up"
            match = re.search(r"(\d+(?:\.\d+)?)\s*(?:cm|センチ)", segment)
            action.height_cm = float(match.group(1)) if match else 3.
        if op == "pour":
            action.object = obj or ("wine_bottle" if "wine_bottle" in {o["id"] for o in state["objects"]} else None)
            action.target = "wine_glass"
            match = re.search(r"(\d+(?:\.\d+)?)\s*(?:ml|ミリリットル)", segment, re.I)
            action.volume_ml = float(match.group(1)) if match else 20.
        actions.append(action)
        if names:
            last_object = names[0]
        last_arm = receiver if op == "handover" else (arm or last_arm)
    return TaskProgram(actions=actions)
