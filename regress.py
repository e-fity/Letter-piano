# -*- coding: utf-8 -*-
"""
regress.py —— 图片识别回归测试：把当前结果与基线对比，防止"改好一处、打坏另一处"。

用法（在 D:\\Letter piano11 下）：

  python regress.py --save        # 第一次：以当前结果建立基线（写入 regress_baseline.json）
  python regress.py               # 以后每次：跑全部图并与基线对比，列出差异
  python regress.py --only 4.png  # 只跑其中一张
  python regress.py --verbose     # 打印进度的同时显示每张图的行数
  python regress.py --imgdir piano  # 图片所在子目录（默认就是 piano）

⚠ `--save` 是**按图名合并**（覆盖同名项）：要彻底重建基线就先删掉
  `regress_baseline.json` 再 `--save`（否则早已删掉的旧图名会一直留在文件里）。

工作方式：
  - 直接调用库（不写中间文件），对每张图得到「字母谱文本行 + 数字分布 + 行数」；
  - 与基线逐张、逐行比较，报告新增/删除/变化的行；
  - 任何一张图报错都只影响它自己，不会中断整轮。
"""

import argparse
import glob
import json
import os
import sys
import traceback

BASELINE = "regress_baseline.json"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
# `对照谱` 是 AI 大模型转出的对照图（不是待识别的原谱）——用户 2026-09-20 明确说不跑它。
SKIP_PATTERNS = ("_debug", "_debug_blocks", "test_", "crop", "regress", "对照谱")


def discover_images(base_dir, imgdir):
    """找出待测图片（默认在 `<项目>/piano/` 下；该目录不存在时退回项目根目录）。

    返回 `[(名字, 相对路径)]`：**名字**当基线的键（`--only 4.png` 也用它），**路径**喂给 OCR。
    ⚠ 旧版只在项目根目录找 —— 图片搬进 `piano/` 之后一张也找不到，`--save` 直接报
    "没找到待测图片"（用户 2026-09-23 踩到，基线因此停在 09-17 一直没法刷新）。
    """
    folder = os.path.join(base_dir, imgdir)
    if not os.path.isdir(folder):
        folder = base_dir
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "*"))):
        name = os.path.basename(path)
        ext = os.path.splitext(name)[1].lower()
        if ext not in IMAGE_EXTS:
            continue
        if any(p in name for p in SKIP_PATTERNS):
            continue
        out.append((name, os.path.relpath(path, base_dir)))
    return out


def run_one(image_path):
    """跑一张图，返回 {lines, dist, rows} 或抛出异常。"""
    from jianpu2letter import ocr_jianpu, letterscore

    measures = ocr_jianpu.ocr_to_stream(image_path, debug_dir=None)
    lines = letterscore.measures_to_lines(measures, measures_per_line=2)

    dist = {}
    rows = set()
    for m in measures:
        rows.add(m.get("row"))
        for note in m.get("notes", []):
            if note.get("n") is not None:
                dist[note["n"]] = dist.get(note["n"], 0) + 1
    return {
        "lines": lines,
        "dist": {str(k): dist[k] for k in sorted(dist)},
        "rows": len(rows),
    }


def compare(old, new):
    """返回差异描述列表（空 = 完全一致）。"""
    diffs = []
    if old.get("rows") != new.get("rows"):
        diffs.append(f"行数：{old.get('rows')} → {new.get('rows')}")
    if old.get("dist") != new.get("dist"):
        diffs.append(f"数字分布：{old.get('dist')} → {new.get('dist')}")
    old_lines, new_lines = old.get("lines", []), new.get("lines", [])
    if len(old_lines) != len(new_lines):
        diffs.append(f"字母谱文本行数：{len(old_lines)} → {len(new_lines)}")
    for i in range(min(len(old_lines), len(new_lines))):
        if old_lines[i] != new_lines[i]:
            diffs.append(f"第 {i + 1} 行不同：\n     旧: {old_lines[i]}\n     新: {new_lines[i]}")
    return diffs


def main(argv=None):
    parser = argparse.ArgumentParser(prog="regress.py")
    parser.add_argument("--save", action="store_true", help="以当前结果建立/更新基线")
    parser.add_argument("--only", default=None, help="只跑指定图片（文件名）")
    parser.add_argument("--dir", default=None, help="工作目录（默认脚本所在目录）")
    parser.add_argument("--imgdir", default="piano", help="图片所在子目录（默认 piano）")
    parser.add_argument("--verbose", action="store_true", help="显示每张图的行数")
    args = parser.parse_args(argv)

    base_dir = args.dir or os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    sys.path.insert(0, base_dir)

    if args.only:
        _folder = os.path.join(base_dir, args.imgdir)
        _rel = os.path.join(args.imgdir, args.only) \
            if os.path.isdir(_folder) else args.only
        images = [(args.only, _rel)]
    else:
        images = discover_images(base_dir, args.imgdir)
    if not images:
        print("没找到待测图片")
        return 1

    baseline_path = os.path.join(base_dir, BASELINE)
    baseline = {}
    if os.path.exists(baseline_path):
        try:
            with open(baseline_path, "r", encoding="utf-8") as fp:
                baseline = json.load(fp)
        except Exception:  # noqa: BLE001
            print(f"[warn] {BASELINE} 解析失败，将视为无基线")

    print(f"待测图片 {len(images)} 张：{', '.join(n for n, _p in images)}")
    print("-" * 60)

    results = {}
    failed = []
    for name, rel_path in images:
        try:
            res = run_one(rel_path)
            results[name] = res
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"[报错] {name}：{exc.__class__.__name__}: {exc}")
            if args.verbose:
                traceback.print_exc()
            continue

        if args.save:
            print(f"[基线] {name}：{res['rows']} 行，数字分布 {res['dist']}")
            continue

        old = baseline.get(name)
        if old is None:
            print(f"[新图] {name}：无基线可比（{res['rows']} 行）")
            continue
        diffs = compare(old, res)
        if not diffs:
            print(f"[相同] {name}（{res['rows']} 行）")
        else:
            print(f"[变化] {name}：")
            for d in diffs:
                print("   - " + d)

    print("-" * 60)
    if args.save:
        merged = dict(baseline)
        merged.update(results)
        with open(baseline_path, "w", encoding="utf-8") as fp:
            json.dump(merged, fp, ensure_ascii=False, indent=2)
        print(f"已写入基线：{baseline_path}（共 {len(merged)} 张）")
    else:
        changed = sum(1 for n, r in results.items()
                      if n in baseline and compare(baseline[n], r))
        print(f"结果：{len(results)} 张完成，{changed} 张有变化，{len(failed)} 张报错")
        if changed or failed:
            print("提示：若变化是「修好了一处」，核对无误后用 --save 更新基线；"
                  "若是「打坏了别处」，请回退相应改动。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
