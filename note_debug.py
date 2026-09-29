# -*- coding: utf-8 -*-
"""
note_debug.py — 简谱音符切割调试脚本（纯文本输出 + 可视化）。

用 OpenCV 连通域把旋律行里的每个数字/符号"切"出来，输出几何信息，
用来确定「数字块 / 音区点 / 减时线 / 附点 / 延时线 / 小节线」的过滤规则。

用法：
  python note_debug.py 1.jpg
"""

import re
import sys

import cv2

# 简谱旋律行允许出现的字符集
MELODY_CHARS = set("0123456789ilI|:()·.—－-_．. ")


def _run_rapidocr(path):
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()
    out = engine(path)
    result = out[0] if isinstance(out, tuple) else out
    tokens = []
    for item in result or []:
        box, text = item[0], item[1]
        score = item[2] if len(item) >= 3 else 1.0
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        tokens.append({
            "x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys),
            "cx": (min(xs) + max(xs)) / 2, "cy": (min(ys) + max(ys)) / 2,
            "text": str(text), "conf": float(score),
        })
    return tokens


def _is_melody(text):
    if re.search(r"[一-鿿]", text):
        return False
    s = text.strip()
    if not s:
        return False
    melody_cnt = sum(1 for ch in s if ch in MELODY_CHARS)
    has_digit = any(ch.isdigit() or ch in "ilI" for ch in s)
    return has_digit and melody_cnt / max(len(s), 1) >= 0.6


def main(path):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 9)

    tokens = _run_rapidocr(path)
    melody = [t for t in tokens if _is_melody(t["text"])]

    # 按 y 聚类成旋律行
    rows = []
    for t in sorted(melody, key=lambda t: t["cy"]):
        for r in rows:
            if abs(t["cy"] - r["cy"]) < 30:
                r["tokens"].append(t)
                r["cy"] = sum(x["cy"] for x in r["tokens"]) / len(r["tokens"])
                break
        else:
            rows.append({"cy": t["cy"], "tokens": [t]})
    rows.sort(key=lambda r: r["cy"])

    # 连通域
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)

    print(f"图片尺寸: {w} x {h}")
    print(f"旋律行数: {len(rows)}")

    for ri, row in enumerate(rows):
        ts = sorted(row["tokens"], key=lambda t: t["x0"])
        joined = " ".join(t["text"] for t in ts)
        y0 = min(t["y0"] for t in ts) - 8
        y1 = max(t["y1"] for t in ts) + 8

        # 该行内连通块（含音区点等小块）
        comps = []
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if y + ch < y0 or y > y1:
                continue
            comps.append((x, y, cw, ch, area))
        comps.sort(key=lambda c: c[0])

        print(f"\n[行{ri:02d}] y={y0:.0f}~{y1:.0f}  块数={len(comps)}")
        print(f"   文本: {joined!r}")

        if ri < 2:  # 前两行详细列块
            for x, y, cw, ch, area in comps:
                print(f"     x={x:5.0f} y={y:5.0f} w={cw:4.0f} h={ch:4.0f} "
                      f"area={area:6.0f} ar={cw / max(ch, 1):.2f}")

    # 可视化：红框标旋律行，绿框标连通块
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for row in rows:
        ts = sorted(row["tokens"], key=lambda t: t["x0"])
        x0 = min(t["x0"] for t in ts); x1 = max(t["x1"] for t in ts)
        y0 = min(t["y0"] for t in ts); y1 = max(t["y1"] for t in ts)
        cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < 30:
            continue
        cv2.rectangle(vis, (x, y), (x + cw, y + ch), (0, 255, 0), 1)

    out = path.rsplit(".", 1)[0] + "_debug_blocks.png"
    cv2.imwrite(out, vis)
    print(f"\n可视化图: {out}（红框=旋律行，绿框=连通块）")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python note_debug.py <图片路径>")
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1]))
