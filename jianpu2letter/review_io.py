# -*- coding: utf-8 -*-
"""OCR 审核 JSON 的保存、校验与重建。"""

import json
import os

from . import textscore

FORMAT = "jianpu2letter-review-v1"


def _bar_text(measure):
    """小节线 → 文本记号。映射统一放在 textscore.bar_to_token()，这里只做转发，
    避免两处各写一套、日久走样。
    """
    return textscore.bar_to_token(measure)


def save_review(path, measures, title, key_mark, meter, source_image=""):
    """把 OCR note 流保存为便于人工修改的 JSON。"""
    groups = []
    current_row = object()
    current = []
    for measure in measures:
        row = measure.get("row", 0)
        if current and row != current_row:
            groups.append(current)
            current = []
        current_row = row
        current.append(measure)
    if current:
        groups.append(current)

    lines = []
    for line_no, group in enumerate(groups, 1):
        notes_out = []
        note_no = 0
        for measure in group:
            measure_notes = measure.get("notes", [])
            for index, note in enumerate(measure_notes):
                note_no += 1
                markup = note.get("markup") or {}
                notes_out.append({
                    "id": f"L{line_no:02d}N{note_no:03d}",
                    "n": note.get("n"),
                    "o": int(note.get("o", 0)),
                    "beams": int(markup.get("beams", 0)),
                    "dotted": bool(markup.get("dotted", False)),
                    "extend": int(markup.get("extend", 0)),
                    "ext_xs": [round(float(v), 1) for v in (note.get("ext_xs") or [])],
                    # 连音：模型里 OCR 路径的连音只存成**相邻对**，所以有 slur_start
                    # 就等于"与下一音有连音弧"。以前这里硬写 False，
                    # 结果连音在"从审核 JSON 重渲染"那一步会整个丢掉。
                    "slur_to_next": bool(note.get("slur_start")),
                    # 连音弧编号：模型里"两条相邻弧"与"一条长弧"靠**编号**区分
                    #（见 ocr_jianpu 里 `slur_id` 那段），只存布尔会丢掉这个信息。
                    # 0 = 这个音不成对（无连音）。
                    "slur_id": (int(note["slur_start"][0])
                                if note.get("slur_start") else 0),
                    "lyric": note.get("lyric") or "",
                    "paren_before": bool(note.get("paren_before", False)),
                    "paren_after": bool(note.get("paren_after", False)),
                    # 重音记号 `>`（数字上方的小 V）。必须**存也读**，
                    # 否则"从审核 JSON 重渲染"时它会像歌词/conf/slur 那样静默丢失。
                    "accent": bool(note.get("accent", False)),
                    # 上方另外三种记号（同样**存也读**，否则重渲染时静默丢失）
                    "breath": bool(note.get("breath", False)),
                    "triplet": bool(note.get("triplet", False)),
                    "house": str(note.get("house") or ""),
                    # 房子**横线**的两个端点（本节首音 / 末音）。同样必须**存也读**。
                    "house_start": bool(note.get("house_start", False)),
                    "house_stop": bool(note.get("house_stop", False)),
                    # 连音弧的**开放端**（跨行弧的残段）：行末那截 / 行首那截。
                    # 同样必须**存也读**。要在 `slur_to_next` 之后追加这两行，
                    # 别覆盖它。
                    "slur_left": bool(note.get("slur_left", False)),
                    "slur_right": bool(note.get("slur_right", False)),
                    "bar_after": _bar_text(measure) if index == len(measure_notes) - 1 else "",
                    "bar_x": (round(float(measure["bar_x"]), 1)
                              if index == len(measure_notes) - 1
                              and measure.get("bar_x") is not None else None),
                    "bar_before": "||:" if (index == 0 and measure.get("repeat_before") == "forward") else "",
                    "source_x": round(float(note.get("cx", 0.0)), 1),
                    # 逐音置信度（0..1；null = 拿不到分数）。只读，供渲染图加黄底。
                    "conf": (None if note.get("conf") is None
                             else round(float(note["conf"]), 3)),
                })
        lines.append({
            "line": line_no,
            # 这一行的「字符序列」文本（与渲染图的字符格子同源）：
            # 不用看图就能肉眼比对原谱，也能整行复制去对照。
            "text": textscore.to_text(group, with_line_prefix=False),
            "notes": notes_out,
        })

    payload = {
        "format": FORMAT,
        "metadata": {
            "title": title,
            "key": key_mark,
            "meter": meter,
            "source_image": os.path.abspath(source_image) if source_image else "",
        },
        "_instructions": [
            "对照同目录的 ocr_overlay.png；图中 LxxNxxx 对应此 JSON 的 note.id。",
            "text: 本行识别出的**字符序列**（只读，供肉眼比对原谱，改音符后不会自动刷新）。",
            "n: 0=休止，1..7=简谱数字，null=未确定。",
            "o: -1=低音点，0=中音，1=高音点；可用 -3..3。",
            "beams: 数字下方减时线条数（0..3）。",
            "dotted: 右侧附点（true/false）。",
            "extend: 数字后延时横线条数（0..4）。",
            "slur_to_next: 此音与下一音是否有连音弧。",
            "slur_id: 连音弧编号（0=无）。**两条相邻的弧编号不同**，所以不要把它统一成 1，"
            "否则两条弧会被当成一条长弧画出来。",
            "accent: 重音记号 `>`（画在数字上方的小 V）。",
            "breath / triplet: `v` 换气 / `³` 三连音（都画在数字上方）。",
            "house: 房子标号 \"1\"/\"2\"/\"3\"（原谱是 `1.` `2.` `3.` 那种反复段落标记）。",
            "house_start / house_stop: 房子**横线**的两端（本节首音标 start、末音标 stop），"
            "渲染时按这两个音的位置连线。",
            "slur_left / slur_right: 连音弧的**开放端**——弧跨行断开时，"
            "行首那截标 slur_left、行末那截标 slur_right，渲染时画到行边界为止。",
            "bar_after: 空字符串、|、||、:|| 或 ||:。不要修改 id/source_x。",
            "conf: 逐音置信度 0..1（null=拿不到分数）。只读；低于 0.70 的音会在"
            "渲染图里加黄底标出，想改这个阈值就调 ocr_jianpu.CONF_WARN。",
        ],
        "lines": lines,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return path


def _as_int(value, name, minimum, maximum, note_id):
    if isinstance(value, bool):
        raise ValueError(f"{note_id}: {name} 不能是布尔值")
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{note_id}: {name} 必须是整数") from exc
    if not (minimum <= value <= maximum):
        raise ValueError(f"{note_id}: {name} 必须在 {minimum}..{maximum}，得到 {value}")
    return value


def load_review(path):
    """读取并严格校验审核 JSON，返回 metadata, measures。"""
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("format") != FORMAT:
        raise ValueError(f"不是受支持的审核 JSON（format 应为 {FORMAT!r}）")

    metadata = payload.get("metadata") or {}
    title = str(metadata.get("title") or os.path.splitext(os.path.basename(path))[0])
    key_mark = str(metadata.get("key") or "1=C")
    if not key_mark.startswith("1="):
        key_mark = "1=" + key_mark
    meter = str(metadata.get("meter") or "4/4")

    measures = []
    seen_ids = set()
    for line_index, line in enumerate(payload.get("lines") or [], 1):
        current = []
        measure_index = 0
        source_notes = line.get("notes") if isinstance(line, dict) else None
        if not isinstance(source_notes, list):
            raise ValueError(f"第 {line_index} 行缺少 notes 数组")

        for raw in source_notes:
            if not isinstance(raw, dict):
                raise ValueError(f"第 {line_index} 行包含非对象音符")
            note_id = str(raw.get("id") or f"L{line_index:02d}N???")
            if note_id in seen_ids:
                raise ValueError(f"重复音符 id：{note_id}")
            seen_ids.add(note_id)

            n = raw.get("n")
            if n is not None:
                n = _as_int(n, "n", 0, 7, note_id)
            o = _as_int(raw.get("o", 0), "o", -3, 3, note_id)
            beams = _as_int(raw.get("beams", 0), "beams", 0, 3, note_id)
            extend = _as_int(raw.get("extend", 0), "extend", 0, 6, note_id)
            dotted = bool(raw.get("dotted", False))
            q = (1.0 / (2 ** beams)) * (1.5 if dotted else 1.0) + extend

            # 连音弧的**编号**必须读回来：模型里"两条相邻弧"和"一条长弧"是靠
            # 编号区分的（见 ocr_jianpu 里 `slur_id` 那段），一律回读成 `[1]`
            # 会把两条相邻的弧重新并成一条。`slur_to_next` 是旧字段（布尔），
            # 新写的 JSON 带 `slur_id`；两种都兼容。
            _slur_id_raw = int(raw.get("slur_id") or 0)
            _slur_start = ([_as_int(_slur_id_raw, "slur_id", 1, 999, note_id)]
                           if _slur_id_raw > 0
                           else ([1] if bool(raw.get("slur_to_next", False)) else []))

            note = {
                "review_id": note_id,
                "rest": n == 0,
                "n": n,
                "o": o,
                "q": round(q, 4),
                "markup": {"beams": beams, "dotted": dotted, "extend": extend},
                "lyric": str(raw.get("lyric") or ""),
                "slur_start": _slur_start,
                "slur_stop": [],
                "cx": float(raw.get("source_x", len(current))),
                "ext_xs": [float(v) for v in (raw.get("ext_xs") or [])],
                "paren_before": bool(raw.get("paren_before", False)),
                "paren_after": bool(raw.get("paren_after", False)),
                "accent": bool(raw.get("accent", False)),
                "breath": bool(raw.get("breath", False)),
                "triplet": bool(raw.get("triplet", False)),
                "house": str(raw.get("house") or ""),
                "house_start": bool(raw.get("house_start", False)),
                "house_stop": bool(raw.get("house_stop", False)),
                "slur_left": bool(raw.get("slur_left", False)),
                "slur_right": bool(raw.get("slur_right", False)),
                # 置信度必须读回来：否则"从审核 JSON 重渲染"这一步会把它丢掉，
                # 黄底就没了（歌词那次就是踩了同一个坑——字段没回读）。
                "conf": (None if raw.get("conf") is None else float(raw["conf"])),
            }
            # 反复开始记号（||:）写在「该小节第一个音」上：映射到上一小节的
            # bar_after="forward"，字母谱即渲染为 `||:`。
            if str(raw.get("bar_before") or "").strip() and measures:
                measures[-1]["repeat_after"] = "forward"
            if current and current[-1].get("slur_start"):
                # **沿用上一个音那份编号**，不要写死 `[1]`——否则"两条相邻弧"
                # 会共用编号、被 `_slur_chains()` 并成一条长弧。
                note["slur_stop"] = list(current[-1]["slur_start"])
            current.append(note)

            bar_after = str(raw.get("bar_after") or "")
            if bar_after:
                measure_index += 1
                measures.append({
                    "number": f"{line_index}-{measure_index}",
                    "row": line_index,
                    "bar_x": (float(raw["bar_x"]) if raw.get("bar_x") is not None else None),
                    "repeat_before": None,
                    "repeat_after": "backward" if bar_after == ":||" else (
                        "forward" if bar_after == "||:" else (
                            "double" if bar_after == "||" else "single")),
                    "notes": current,
                })
                current = []

        if current:
            measure_index += 1
            measures.append({
                "number": f"{line_index}-{measure_index}",
                "row": line_index,
                "bar_x": None,
                "repeat_before": None,
                "repeat_after": "single",
                "notes": current,
            })

    if not measures:
        raise ValueError("审核 JSON 没有可生成的音符")
    return {"title": title, "key": key_mark, "meter": meter}, measures
