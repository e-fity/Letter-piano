# -*- coding: utf-8 -*-
"""
核心管线自测（仅标准库，无需 matplotlib / paddleocr）。

解析 sample_mother.xml → 首调 → 映射2字母谱文本，比对预期结果。
运行：  python selftest.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jianpu2letter import musicxml_reader, solfege, letterscore

# 预期输出（映射2）
EXPECTED = [
    "||: J · H_ F H | E ⌒ J_ H_ J —",
    "B — N 0 | S 0 S 0 :||",
]


def main():
    score = musicxml_reader.parse("sample_mother.xml")
    measures = solfege.build_stream(score)
    lines = letterscore.measures_to_lines(measures, measures_per_line=2)

    key_mark = solfege.key_mark_from_fifths(score["key_fifths"])
    meter = f"{score['beats']}/{score['beat_type']}"
    print(f"标题: {score['title']}")
    print(f"调号: {key_mark}   拍号: {meter}")
    print("=" * 40)

    ok = True
    for i, want in enumerate(EXPECTED, 1):
        got = lines[i - 1] if i - 1 < len(lines) else ""
        status = "PASS" if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"[{status}] 第{i}行")
        print(f"   实际: {got}")
        if got != want:
            print(f"   期望: {want}")
    print("=" * 40)
    print("全部通过 ✅" if ok else "存在不一致 ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
