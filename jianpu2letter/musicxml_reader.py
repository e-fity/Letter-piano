# -*- coding: utf-8 -*-
"""
解析 MusicXML（score-partwise）为中间结构。

只用标准库 xml.etree.ElementTree，无第三方依赖。
支持内容：调号、拍号、divisions、音符（音高/时值/休止/延音 tie/连音 slur/歌词）、
小节线（反复记号）。多声部默认取指定声部（第 1 声部）。
"""

import xml.etree.ElementTree as ET


def _local(tag):
    """去掉可能的命名空间前缀，返回本地标签名。"""
    return tag.rsplit("}", 1)[-1]


def _children(elem, tag):
    return [c for c in list(elem) if _local(c.tag) == tag]


def _child(elem, tag):
    for c in list(elem):
        if _local(c.tag) == tag:
            return c
    return None


def _text(elem):
    return (elem.text or "").strip() if elem is not None else ""


def _int(elem, default=0):
    t = _text(elem)
    try:
        return int(t)
    except (TypeError, ValueError):
        return default


def _float(elem, default=0.0):
    t = _text(elem)
    try:
        return float(t)
    except (TypeError, ValueError):
        return default


# 调号 fifths（升降号数）→ 大调主音音名（也即简谱「1=X」的 X）
# 小调按关系大调处理：同一 fifths 的关系大调主音即「1」。
MAJOR_KEYS_BY_FIFTHS = {
    0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#", 7: "C#",
    -1: "F", -2: "Bb", -3: "Eb", -4: "Ab", -5: "Db", -6: "Gb", -7: "Cb",
}


def _parse_repeat(el):
    """从小节线元素解析反复方向，返回 'forward' / 'backward' / None。"""
    rep = _child(el, "repeat")
    if rep is not None:
        return rep.get("direction")
    return None


def _parse_note(el):
    note = {
        "rest": False,
        "chord": False,
        "grace": False,
        "step": None,       # C D E F G A B
        "alter": 0,         # -2..2（重升重降）
        "octave": None,     # MusicXML 八度（C4 = 中央C）
        "duration": 0.0,    # 以当前 divisions 为单位的时值
        "tie_start": False,
        "tie_stop": False,
        "slur_start": [],   # 连音 start 的 number 列表
        "slur_stop": [],    # 连音 stop 的 number 列表
        "lyrics": [],       # 歌词音节列表
    }
    for c in list(el):
        tag = _local(c.tag)
        if tag == "rest":
            note["rest"] = True
        elif tag == "chord":
            note["chord"] = True
        elif tag == "grace":
            note["grace"] = True
        elif tag == "duration":
            note["duration"] = _float(c)
        elif tag == "pitch":
            step = _child(c, "step")
            alter = _child(c, "alter")
            octave = _child(c, "octave")
            if step is not None:
                note["step"] = _text(step)
            if alter is not None:
                note["alter"] = _int(alter)
            if octave is not None:
                note["octave"] = _int(octave)
        elif tag == "tie":
            note["tie_" + c.get("type", "start")] = True
        elif tag == "lyric":
            text = _child(c, "text")
            note["lyrics"].append(_text(text) if text is not None else "")
        elif tag == "notations":
            for sub in list(c):
                st = _local(sub.tag)
                if st == "slur":
                    num = sub.get("number", "1")
                    try:
                        num = int(num)
                    except ValueError:
                        num = 1
                    if sub.get("type") == "stop":
                        note["slur_stop"].append(num)
                    else:
                        note["slur_start"].append(num)
                elif st == "tied":
                    note["tie_" + sub.get("type", "start")] = True
    return note


def _parse_measure(m_el):
    measure = {
        "number": m_el.get("number", ""),
        "divisions": None,
        "key_fifths": None,
        "key_mode": "major",
        "beats": None,
        "beat_type": None,
        "notes": [],
        "barline_before": None,  # repeat direction at left
        "barline_after": None,   # repeat direction / style at right
    }
    for el in list(m_el):
        tag = _local(el.tag)
        if tag == "attributes":
            div = _child(el, "divisions")
            if div is not None:
                measure["divisions"] = _int(div)
            key = _child(el, "key")
            if key is not None:
                fifths = _child(key, "fifths")
                if fifths is not None:
                    measure["key_fifths"] = _int(fifths)
                mode = _child(key, "mode")
                if mode is not None:
                    measure["key_mode"] = _text(mode) or "major"
            time = _child(el, "time")
            if time is not None:
                beats = _child(time, "beats")
                beat_type = _child(time, "beat-type")
                if beats is not None:
                    measure["beats"] = _int(beats)
                if beat_type is not None:
                    measure["beat_type"] = _int(beat_type)
        elif tag == "note":
            measure["notes"].append(_parse_note(el))
        elif tag == "barline":
            loc = el.get("location", "right")
            rep = _parse_repeat(el)
            style = _text(_child(el, "bar-style"))
            if loc == "left":
                measure["barline_before"] = rep
            else:
                measure["barline_after"] = rep or style or "single"
    return measure


def _parse_title(root):
    # movement-title 是 score-partwise 的直接子元素
    for tag in ("movement-title", "work-title"):
        el = _child(root, tag)
        if el is not None and _text(el):
            return _text(el)
    # work-title 位于 <work> 内
    work = _child(root, "work")
    if work is not None:
        wt = _child(work, "work-title")
        if wt is not None and _text(wt):
            return _text(wt)
    # 兜底：<credit><credit-words>
    for el in root.iter():
        if _local(el.tag) == "credit-words":
            return _text(el)
    return ""


def parse(path, part_index=0):
    """
    解析 MusicXML 文件。

    参数：
      path       MusicXML 文件路径
      part_index 声部索引（默认 0 = 第一个 <part>）

    返回 dict：
      title, key_fifths, key_mode, beats, beat_type, divisions, measures
    """
    tree = ET.parse(path)
    root = tree.getroot()

    parts = _children(root, "part")
    if part_index >= len(parts):
        raise ValueError(f"声部索引 {part_index} 超出范围（共 {len(parts)} 个声部）")
    part = parts[part_index]

    score = {
        "title": _parse_title(root),
        "key_fifths": 0,
        "key_mode": "major",
        "beats": 4,
        "beat_type": 4,
        "divisions": 1,
        "measures": [],
    }

    for m_el in _children(part, "measure"):
        measure = _parse_measure(m_el)
        # 收集全局默认值（首个非 None 的调号/拍号/divisions）
        if measure["key_fifths"] is not None and score["key_fifths"] == 0:
            score["key_fifths"] = measure["key_fifths"]
            score["key_mode"] = measure["key_mode"]
        if measure["beats"] is not None:
            score["beats"] = measure["beats"]
            score["beat_type"] = measure["beat_type"]
        if measure["divisions"] is not None:
            score["divisions"] = measure["divisions"]
        score["measures"].append(measure)

    if score["divisions"] <= 0:
        score["divisions"] = 1
    if score["beats"] <= 0:
        score["beats"] = 4
    if score["beat_type"] <= 0:
        score["beat_type"] = 4

    return score
