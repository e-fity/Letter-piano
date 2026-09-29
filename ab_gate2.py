# -*- coding: utf-8 -*-
"""证明「闸②（弱证据类并入强证据类）」自己能独立兜住同一批错判。

做法：把闸①的门槛调成不可能满足（VOTE_MIN_SIM=-2，投票一律被采纳），于是 18 块那类
一定会被 3 票判成 1；这时只有闸② 能救。看它有没有把该类并回 5、数字分布对不对。

用法：python ab_gate2.py 5.png
"""

import sys

import jianpu2letter.ocr_jianpu as oj

oj.VOTE_MIN_SIM = -2.0            # 闸① 永久放行 → 只剩闸② 起作用
oj.VERBOSE_DEBUG = False

image = sys.argv[1] if len(sys.argv) > 1 else "5.png"
measures = oj.ocr_to_stream(image, debug_dir=None)

dist = {}
for m in measures:
    for n in m.get("notes", []):
        if n.get("n") is not None:
            dist[n["n"]] = dist.get(n["n"], 0) + 1
print(f"[gate2-only] {image} 数字分布：{ {k: dist[k] for k in sorted(dist)} }")
