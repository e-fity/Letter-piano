# -*- coding: utf-8 -*-
"""拍数校验：不靠看图，用「每小节时值必须等于拍号」来定位识别错误。

用法：
  python verify_score.py review1\1_review.json --beats 4
  python verify_score.py review2\2_review.json --beats 4

4/4 拍下，完整小节的时值合计应为 4.0 拍。凡是不等于 4.0 的小节，
基本可以断定该小节内有音符被漏读、错读、或时值（减时线/附点/延时线）标错。
"""

import argparse
import json


def _note_beats(note):
    beams = int(note.get("beats", note.get("beams", 0)) or 0)
    beams = int(note.get("beams", beams) or 0)
    dotted = bool(note.get("dotted", False))
    extend = int(note.get("extend", 0) or 0)
    return (1.0 / (2 ** beams)) * (1.5 if dotted else 1.0) + extend


def _fmt(note):
    if note.get("n") is None:
        return "?"
    s = str(note["n"])
    o = int(note.get("o", 0))
    s += "^" * o if o > 0 else "v" * (-o)
    s += "_" * int(note.get("beams", 0) or 0)
    if note.get("dotted"):
        s += "·"
    s += "-" * int(note.get("extend", 0) or 0)
    return s


def main(path, beats_per_measure=4.0, tolerance=0.01):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("format") != "jianpu2letter-review-v1":
        print(f"[warn] 非审核 JSON，仍尝试按 lines/notes 解析：{path}")

    measures = []
    for line in payload.get("lines", []):
        current = []
        line_no = line.get("line")
        for note in line.get("notes", []):
            current.append(note)
            if note.get("bar_after"):
                measures.append({"line": line_no, "notes": current,
                                 "bar": note["bar_after"]})
                current = []
        if current:
            measures.append({"line": line_no, "notes": current, "bar": "(行尾)"})

    total = 0
    bad = 0
    unknown = 0
    print(f"共 {len(measures)} 个小节，拍号基准 {beats_per_measure} 拍/小节\n")

    for index, measure in enumerate(measures, 1):
        beats = sum(_note_beats(note) for note in measure["notes"])
        total += 1
        tags = []
        if beats - beats_per_measure > tolerance:
            tags.append("超拍")
        elif beats_per_measure - beats < tolerance:
            tags.append("缺拍")
        if any(note.get("n") is None for note in measure["notes"]):
            tags.append("含?")
            unknown += 1

        if tags:
            bad += 1
            notes_text = " ".join(_fmt(n) for n in measure["notes"])
            print(f"[{index:3d}] 行{measure['line']} 拍数={beats:5.2f} "
                  f"{'/'.join(tags):8s} {notes_text}")

    print(f"\n合计 {total} 小节，其中 {bad} 个拍数异常"
          f"（{unknown} 个含未识别 ?）")
    print("提示：拍数异常的小节就是最可能出错的位置；")
    print("      「缺拍」通常是漏读音符或延时线少算，「超拍」通常是多读音符。")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--beats", type=float, default=4.0)
    args = parser.parse_args()
    raise SystemExit(main(args.path, args.beats))
