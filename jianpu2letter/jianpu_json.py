# -*- coding: utf-8 -*-
"""解析字母琴网站的简谱 JSON（schema: jianpu-score-json）。

结构（与网站 game.js 使用方式一致）：
  {
    "schema": {"name": "jianpu-score-json", ...},
    "musical_setup": {"key_mark": "1=♭E", "meter": {"printed": "4/4"}, "tempo_qpm": 100},
    "measures": [
      {"id": "m1", "region": "...",
       "events": [{"event_index": 0, "n": 6, "o": 0, "q": 1.0,
                   "beam": 1, "dot": false, "ext": 0, "r": false,
                   "triplet": null, "keyboard": {"key": "j"}}],
       "connections": [{"type": "tie"|"slur", "from":.., "to":..}]},
      ...
    ]
  }

输出与 solfege.build_stream 兼容的 measures（供 letterscore / jianpu_render 使用）。
"""

import json


def _is_site_score(payload):
    schema = payload.get("schema") or {}
    return schema.get("name") == "jianpu-score-json" or "measures" in payload


def _load_payload(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _key_mark(payload):
    setup = payload.get("musical_setup") or {}
    mark = setup.get("key_mark") or "1=C"
    # 网站写 "1=♭E"，转成 ASCII
    mark = (mark.replace("♭", "b").replace("♯", "#")
                .replace("♮", "").replace(" ", ""))
    return mark


def parse(path):
    """读取网站简谱 JSON → (metadata, measures)。"""
    payload = _load_payload(path)
    if not _is_site_score(payload):
        raise ValueError("不是受支持的网站简谱 JSON（缺少 measures / jianpu-score-json）")

    setup = payload.get("musical_setup") or {}
    document = payload.get("document") or {}
    metadata = {
        "title": document.get("title") or "",
        "key": _key_mark(payload),
        "meter": (setup.get("meter") or {}).get("printed") or "4/4",
        "tempo": setup.get("tempo_qpm"),
    }

    measures_out = []
    for m_index, measure in enumerate(payload.get("measures") or []):
        events = measure.get("events") or []

        # 连音/延音：type=slur → 连音弧；type=tie → 合并时值（与网站 game.js 一致）
        slur_from, slur_to = set(), set()
        tie_targets = set()
        for conn in measure.get("connections") or []:
            if conn.get("type") == "slur":
                slur_from.add(conn.get("from"))
                slur_to.add(conn.get("to"))
            elif conn.get("type") == "tie":
                tie_targets.add(conn.get("to"))

        notes = []
        for event in events:
            beams = int(event.get("beam") or 0)
            dotted = bool(event.get("dot"))
            extend = int(event.get("ext") or 0)
            q = event.get("q")
            if q is None:
                q = (1.0 / (2 ** beams)) * (1.5 if dotted else 1.0) + extend
            is_rest = bool(event.get("r"))
            n = None if is_rest else int(event["n"])
            o = int(event.get("o") or 0)
            idx = event.get("event_index")

            # 延音（tie）目标：并入上一个同音高音符
            if idx in tie_targets and notes and not is_rest and notes[-1]["n"] == n \
                    and notes[-1]["o"] == o:
                notes[-1]["q"] = round(notes[-1]["q"] + float(q), 4)
                mk = notes[-1]["markup"]
                mk["extend"] = max(0, int(round(notes[-1]["q"])) - 1)
                continue

            note = {
                "rest": is_rest,
                "n": 0 if is_rest else n,
                "o": max(-3, min(3, o)),
                "q": round(float(q), 4),
                "markup": {"beams": beams, "dotted": dotted, "extend": extend},
                "lyric": "",
                "slur_start": [1] if idx in slur_from else [],
                "slur_stop": [1] if idx in slur_to else [],
                "cx": None,
                "paren_before": False,
                "paren_after": False,
            }
            notes.append(note)

        if not notes:
            continue
        measures_out.append({
            "number": measure.get("id") or f"m{m_index + 1}",
            "row": None,              # 网站数据无谱面行概念，交给排版按行宽换行
            "repeat_before": None,
            "repeat_after": "single",
            "notes": notes,
        })

    if not measures_out:
        raise ValueError("网站 JSON 里没有可用的小节")
    return metadata, measures_out
