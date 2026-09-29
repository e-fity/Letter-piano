# -*- coding: utf-8 -*-
"""打印简谱图片前几行「数字块」的几何特征（纯文字），用于判断字体是否为空心字。

用法：
  python diagnose_blocks.py 2.png
"""

import sys

import cv2
import numpy as np

from jianpu2letter import ocr_jianpu as oj


def _line_groups(comps):
    rows = []
    for c in sorted(comps, key=lambda c: c["y"] + c["h"] / 2):
        cy = c["y"] + c["h"] / 2
        for r in rows:
            if abs(cy - r["cy"]) < 40:
                r["blocks"].append(c)
                r["cy"] = float(np.mean([b["y"] + b["h"] / 2 for b in r["blocks"]]))
                break
        else:
            rows.append({"cy": cy, "blocks": [c]})
    rows.sort(key=lambda r: r["cy"])
    return rows


def main(path, max_rows=6, max_blocks=16):
    color = cv2.imread(path, cv2.IMREAD_COLOR)
    if color is None:
        print(f"无法读取图片：{path}")
        return 1
    color = cv2.resize(color, None, fx=oj.SCALE, fy=oj.SCALE,
                       interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 9)
    print(f"放大后尺寸：{gray.shape[1]} x {gray.shape[0]}")

    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comps = []
    for i in range(1, len(stats)):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area >= 8 and 25 <= h <= 70 and 0.25 <= w / max(h, 1) <= 1.3:
            comps.append({"x": x, "y": y, "w": w, "h": h, "area": area})

    rows = _line_groups(comps)
    print(f"候选块 {len(comps)} 个，行 {len(rows)} 个\n")

    total = 0
    holes = 0
    for i, r in enumerate(rows, 1):
        if i > max_rows:
            break
        print(f"--- 行{i}（y≈{r['cy']:.0f}，块数 {len(r['blocks'])}）---")
        for b in sorted(r["blocks"], key=lambda c: c["x"])[:max_blocks]:
            roi = binary[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]]
            ink = roi > 0
            fill = float(ink.mean())
            hole = oj._hole_class(b, binary)
            total += 1
            if hole is not None:
                holes += 1
            print(f"  x={b['x']:5d} w={b['w']:3d} h={b['h']:3d} "
                  f"填充率={fill:.2f} 内孔={hole}")

    print(f"\n[汇总] 前几行内孔命中 {holes}/{total}")
    print("提示：若填充率普遍很低（<0.4）且内孔命中比例很高，说明是空心字，")
    print("      内孔判 0 会失效，应改用 OCR 数字。")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python diagnose_blocks.py <图片>")
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1]))
