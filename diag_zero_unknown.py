# -*- coding: utf-8 -*-
"""诊断：把「判成休止 0」和「没认出来(?)」的音，全部拿形状分类器复核一遍。

目的：
  A. 判成 0 的音里，哪些"最像的其实是别的数字"（= 假休止）；
  B. 没认出来的音里，它们"最像哪个数字、分数多少"（看是门槛问题还是字形问题）。

用法：python diag_zero_unknown.py 7.png 14.jpg 1.jpg 3.png
输出：diag_zero_unknown.txt
"""

import sys

import jianpu2letter.ocr_jianpu as oj


def best_digit(block, binary, refs, min_score=0.25, min_margin=0.0):
    """放宽门槛再问一次：这个块最像哪个数字、多少分。"""
    return oj._classify_by_reference(block, binary, refs, min_score=min_score,
                                     min_margin=min_margin, return_score=True)


def main():
    out = []
    for img in sys.argv[1:]:
        measures = oj.ocr_to_stream(img)
        binary = oj._LAST_BINARY
        refs = oj._all_reference_glyphs()
        row_no = 0
        prev = object()
        zeros, unknowns = [], []
        for measure in measures:
            if measure.get("row") != prev:
                prev = measure.get("row")
                row_no += 1
            for note in measure.get("notes", []):
                n, box = note.get("n"), note.get("box")
                if box is None or n not in (0, None):
                    continue
                b = {"x": int(box[0]), "y": int(box[1]), "w": int(box[2]), "h": int(box[3])}
                d, sc = best_digit(b, binary, refs)
                item = (row_no, b, note.get("o"), d, sc)
                (zeros if n == 0 else unknowns).append(item)
        out.append(f"===== {img}")
        out.append(f"-- 判成休止 0 的 {len(zeros)} 个")
        for r, b, o, d, sc in zeros:
            flag = "  ← 假休止?" if (d not in (0, None) and sc >= 0.70) else ""
            out.append(f"   行{r:<3} box=({b['x']},{b['y']},{b['w']},{b['h']}) o={o:<3} "
                       f"最像 {d}（{sc:.2f}）{flag}")
        out.append(f"-- 没认出来(?) 的 {len(unknowns)} 个")
        for r, b, o, d, sc in unknowns:
            out.append(f"   行{r:<3} box=({b['x']},{b['y']},{b['w']},{b['h']}) o={o:<3} "
                       f"最像 {d}（{sc:.2f}）")
    open("diag_zero_unknown.txt", "w", encoding="utf-8").write("\n".join(out))
    print("done")


if __name__ == "__main__":
    main()
