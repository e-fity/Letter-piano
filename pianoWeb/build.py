# -*- coding: utf-8 -*-
"""把 `_review.json`（或它合并后的那一份）渲染成一个**自包含的字母谱网页**。

用法（在项目根目录下）：

    python pianoWeb/build.py out/4_review.json -o 我的中国心.html
    python pianoWeb/build.py out/演员_review.json -o 演员.html
    python pianoWeb/build.py out/4_review.json -o 4.html --annotations 4.notes.json

- 输入就是现成的 `<图>_review.json`（两阶段流程的产物），**不重跑 OCR、不改原代码**
  （只用 `jianpu2letter` 的 `review_io` / `mapping` 做只读查询）。
- 输出是**单个 HTML 文件**：CSS/JS/数据全内嵌，双击就能看，不需要服务器。

「哪些是自动的、哪些要手填」——
- **自动**（全部来自 JSON 真数据）：字母、音区分色、减时线、附点、延时线、休止、
  小节线/反复记号、括号、重音/换气/三连音/房子、连音弧、歌词、谱头（曲名/调号/拍号）。
- **手填**（`--annotations` 指的一个小 JSON，**没有就整块不显示**）：
  * `subtitle` 副标题；
  * `sections` 分段（"前奏"/"主歌 ×2"/"副歌"/"结尾"）——工具本身不产出分段名；
  * `panels`  注释面板（如"音区校正""交叉校验""判读规则"）——内容逐谱手写，无法自动生成。
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from jianpu2letter import mapping, review_io   # noqa: E402  （只读用）

# 调号 → 中文说法（谱头那行用）
_KEY_CN = {"C": "C调", "Db": "降D调", "D": "D调", "Eb": "降E调", "E": "E调", "F": "F调",
           "F#": "升F调", "Gb": "降G调", "G": "G调", "Ab": "降A调", "A": "A调", "Bb": "降B调",
           "B": "B调", "C#": "升C调"}
_FLAT = {"C": "♭C", "D": "♭D", "E": "♭E", "F": "♭F", "G": "♭G", "A": "♭A", "B": "♭B"}


def _key_label(key_mark):
    """'1=Eb' → '♭E（降E调）'；'1=C' → 'C（C调）'。"""
    letter = str(key_mark).split("=", 1)[-1].strip()
    acc = ""
    if len(letter) > 1 and letter[0] in ("b", "#"):
        acc, letter = letter[0], letter[1:]
    shown = (_FLAT.get(letter, letter) if acc == "b"
             else ("♯" + letter if acc == "#" else letter))
    return f"{shown}（{_KEY_CN.get(acc + letter if acc else letter, letter + '调')}）"


def _band(o):
    """音区分档：低/中/高（±2/±3 用 Shift 扩展，仍归本档）。"""
    o = int(o)
    return "low" if o < 0 else ("mid" if o == 0 else "high")


def _note_cell(raw):
    """一个音的展示单元。

    三种特殊情况（都不带唱名）：
      * 休止      `n == 0`   → 显示 `0`
      * **没认出来** `n is None` → 显示 `Q`，网页上**标红底**（工具在谱面上是画红框的那批）
    """
    n = raw.get("n")
    o = int(raw.get("o", 0) or 0)
    common = {
        "id": raw.get("id") or "",        # 编辑时按它精确定位（**不能按序号**）
        "beams": int(raw.get("beams", 0) or 0),
        "dotted": bool(raw.get("dotted")),
        "extend": int(raw.get("extend", 0) or 0),
        "lyric": raw.get("lyric") or "",
        "barBefore": raw.get("bar_before") or "",
        "barAfter": raw.get("bar_after") or "",
        "parenBefore": bool(raw.get("paren_before")),
        "parenAfter": bool(raw.get("paren_after")),
        "origN": n,                       # 原始值（"恢复原谱"用）
        "origO": o,
    }

    if n == 0:                                     # 休止
        return dict(common, rest=True, unknown=False, ch="0", band="rest")
    if n is None:                                  # 没认出来 → Q + 红底
        return dict(common, rest=False, unknown=True, ch="Q", band="unknown")

    key, mod = mapping.key_for(int(n), o, "mapping2")
    cell = dict(common, rest=False, unknown=False, ch=key, band=_band(o),
                n=int(n), o=o)
    # 映射2 的 ±2/±3 八度靠 Shift：标出来，别让用户按错
    cell["mod"] = {"LSHIFT": "L⇧", "RSHIFT": "R⇧", None: ""}.get(mod, "")
    cell.update(
        accent=bool(raw.get("accent")),
        breath=bool(raw.get("breath")),
        triplet=bool(raw.get("triplet")),
        house=raw.get("house") or "",
        slurToNext=bool(raw.get("slur_to_next")),
        slurLeft=bool(raw.get("slur_left")),
        slurRight=bool(raw.get("slur_right")),
        lowConf=raw.get("conf") is not None and float(raw["conf"]) < 0.70,
    )
    return cell


def _load_annotations(path):
    """旁注文件：subtitle / sections / panels / legend / about。缺省全空。"""
    if not path:
        return {}
    if not os.path.exists(path):
        raise SystemExit(f"旁注文件不存在：{path}")
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def build_data(review_path, annotations, annotations_path=""):
    metadata, _measures = review_io.load_review(review_path)
    with open(review_path, encoding="utf-8") as fp:
        payload = json.load(fp)

    lines = []
    for line in payload.get("lines") or []:
        cells = [_note_cell(raw) for raw in (line.get("notes") or [])]
        if not cells:
            continue
        lines.append({"line": line.get("line"), "cells": cells})

    return {
        "title": metadata.get("title", ""),
        "key": metadata.get("key", "1=C"),
        "keyLabel": _key_label(metadata.get("key", "1=C")),
        "meter": metadata.get("meter", "4/4"),
        "source": os.path.basename(metadata.get("source_image", "") or ""),
        "subtitle": annotations.get("subtitle", ""),
        "about": annotations.get("about", "已按 26 字母对照表逐音转换。点击谱中的字母可以试听。"),
        "legend": annotations.get("legend") or [
            {"cls": "low", "label": "低音 Z-M"},
            {"cls": "mid", "label": "中音 S-K"},
            {"cls": "high", "label": "高音 E-Q"},
            {"cls": "unknown", "label": "Q 没认出来（红底）"},
            {"cls": "ext", "label": "— 延长"},
            {"cls": "rest", "label": "0 休止"},
        ],
        "panels": annotations.get("panels") or [],
        "sections": annotations.get("sections") or [],
        "lines": lines,
        # 编辑功能要用的两样：
        #  * reviewPath —— 后端着原文件路径去改（无后端时只用来显示）
        #  * payload     —— **完整的原始审核 JSON**。页面上改完直接把它 patch 了导出，
        #                   这样"导出改好的 _review.json"不依赖任何后端。
        "reviewPath": os.path.abspath(review_path),
        # 旁注文件路径：后端重跑后重新生成页面数据时要用它，否则分段/面板会丢
        "annotationsPath": (os.path.abspath(annotations_path) if annotations_path else ""),
        "payload": payload,
    }


def render(data, template_path, out_path):
    with open(template_path, encoding="utf-8") as fp:
        template = fp.read()
    html = template.replace(
        "/*__DATA__*/null",
        json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write(html)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="把 _review.json 渲染成自包含的字母谱网页")
    ap.add_argument("review", help="<图>_review.json（或页组合并出来的那份）")
    ap.add_argument("-o", "--out", required=True, help="输出的 .html")
    ap.add_argument("--annotations", default=None,
                    help="可选的旁注 JSON（subtitle / sections / panels）")
    ap.add_argument("--template", default=os.path.join(_HERE, "template.html"))
    args = ap.parse_args(argv)

    data = build_data(args.review, _load_annotations(args.annotations), args.annotations or "")
    render(data, args.template, args.out)
    n_notes = sum(len(l["cells"]) for l in data["lines"])
    print(f"[ok] {args.out}  （{data['title']} {data['key']} {data['meter']}"
          f"；{len(data['lines'])} 行 / {n_notes} 音"
          f"{'；含分段与面板' if data['sections'] or data['panels'] else ''}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
