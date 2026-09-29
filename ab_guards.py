# -*- coding: utf-8 -*-
"""A/B：同一张图、同一次 OCR，只切换「按类定标签」那两道闸，比较最终数字。

做法：把 `_label_blocks_by_cluster` 包一层——先按当前代码（闸开）跑一遍并记下结果，
再把 values 复位、把两个阈值改成"永不开火"跑第二遍（闸关），最后恢复闸开的结果让流水线
继续。这样一次 OCR 就能拿到严格对照，避开"基线是旧代码"的干扰。

用法：python ab_guards.py [只跑某张图]
"""

import sys
from contextlib import redirect_stdout
import io

import jianpu2letter.ocr_jianpu as oj
from regress import discover_images

_orig = oj._label_blocks_by_cluster
OFF = {"VOTE_MIN_SIM": -2.0, "MERGE_HOST_MIN": 2.0}
report = []


def _labels(row_records):
    return [list(r["values"]) for r in row_records]


def wrapper(row_records, binary, refs, *args, **kwargs):
    before = _labels(row_records)
    changed_on = _orig(row_records, binary, refs, *args, **kwargs)
    on = _labels(row_records)

    for r, v in zip(row_records, before):
        r["values"] = list(v)
    saved = {k: getattr(oj, k) for k in OFF}
    for k, v in OFF.items():
        setattr(oj, k, v)
    with redirect_stdout(io.StringIO()):      # 第二遍的 [fix]/聚类摘要不必重复打
        _orig(row_records, binary, refs, *args, **kwargs)
    off = _labels(row_records)
    for k, v in saved.items():
        setattr(oj, k, v)

    for r, v in zip(row_records, on):        # 恢复"闸开"的结果，流水线继续
        r["values"] = list(v)

    for ri, (o, n) in enumerate(zip(off, on)):
        if o != n:
            report.append((ri + 1, o, n))
    return changed_on


def main():
    oj._label_blocks_by_cluster = wrapper
    oj.VERBOSE_DEBUG = False                 # 只要 A/B 结论，不要逐行诊断

    base = [n for n, _p in discover_images(".", "piano")]
    if len(sys.argv) > 1:
        base = [sys.argv[1]]

    for name in base:
        report.clear()
        try:
            oj.ocr_to_stream(name, debug_dir=None)
        except Exception as exc:             # noqa: BLE001
            print(f"[错误] {name}: {type(exc).__name__}: {exc}")
            continue
        if not report:
            print(f"[一致] {name}：两道闸都没改动任何数字")
            continue
        print(f"[变化] {name}：{len(report)} 行的数字被改动")
        for ri, off, on in report:
            print(f"   行{ri} 闸关: {' '.join('?' if v is None else str(v) for v in off)}")
            print(f"   行{ri} 闸开: {' '.join('?' if v is None else str(v) for v in on)}")


if __name__ == "__main__":
    main()
