# -*- coding: utf-8 -*-
"""一键跑一个文件夹里的所有简谱图片 → 输出到 out/，最后打一张汇总表。

用法（在项目根目录 D:\\Letter piano11 下）：

    python run_all.py                    # 跑 piano/ 全部，输出到 out/
    python run_all.py --dir piano -o out
    python run_all.py --only 14.jpg      # 只跑一张（其余参数照旧）
    python run_all.py --no-render        # 只出文本/XML/原位替换图，不重出两张 png

曲名 / 调号 / 拍号写在 `piano_config.json`（与项目根同级），形如：

    {"14.jpg": {"title": "富士山下", "key": "F", "meter": "4/4"}}

没配的图：曲名 = 文件名、调号 = 1=C、拍号 = 4/4。

每张图的完整控制台输出写在 `out/run_log.txt`；一张失败不影响其它张。
"""

import argparse
import json
import os
import re
import subprocess
import sys

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
# 本工具自己的产物，不是待识别原谱
SKIP_PATTERNS = ("_overlay", "_letter", "_jianpu", "_debug", "test_", "crop")
# 子进程（cli.py）读到页眉调号/曲名时打印的那行，用来回收写回配置
_KEY_LINE = re.compile(r"\[key\] 页眉自动读到调号 (1=[A-Ga-g][#b]?)")
_TITLE_LINE = re.compile(r"\[title\] 自动读到标题: (.+)")


def collect_images(folder, only=None):
    out = []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(IMAGE_EXTS):
            continue
        if any(p in name for p in SKIP_PATTERNS):
            continue
        out.append(name)
    if only:
        out = [n for n in out if n == only] or [only]
    return out


def parse_summary(text):
    """从一张图的控制台输出里抓出汇总用的几个数字。"""
    got = {"low": "-", "dist": "-", "conv": "-"}
    m = re.search(r"置信度低于 70% 的音 (\d+) 个", text)
    if m:
        got["low"] = m.group(1)
    m = re.search(r"数字分布：(.*)$", text, re.M)
    if m:
        got["dist"] = m.group(1).strip()
    m = re.search(r"盖了 (\d+) 个数字；休止 (\d+) 个保持原样；"
                  r"没认出来/疑似误判 (\d+) 处", text)
    if m:
        got["conv"] = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    return got


def main():
    ap = argparse.ArgumentParser(description="一键跑一个文件夹里的所有简谱图片")
    ap.add_argument("--dir", default="piano", help="图片所在文件夹（默认 piano）")
    ap.add_argument("-o", "--outdir", default="out", help="输出文件夹（默认 out）")
    ap.add_argument("--only", default=None, help="只跑这一张（文件名）")
    ap.add_argument("--config", default="piano_config.json",
                    help="曲名/调号/拍号配置（默认 piano_config.json）")
    ap.add_argument("--no-render", action="store_true", help="不重出两张 png")
    ap.add_argument("--letter-png", action="store_true",
                    help="额外渲染键盘字母谱图片（<名字>_letter.png；默认不出）")
    ap.add_argument("--scan-only", action="store_true",
                    help="只跑第一阶段：出 <名字>_review.json + 叠加图 + 字符序列，"
                         "供人工修改 JSON 后再喂回（不再出 png/xml）")
    args = ap.parse_args()

    # 控制台可能是 GBK：本脚本自己也打中文（曲名等），统一按 UTF-8 输出，避免
    # "✓/✗ 之类字符不在 GBK 码表"导致的 UnicodeEncodeError（子进程那边同样处理）
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    root = os.path.dirname(os.path.abspath(__file__))
    cfg = {}
    cfg_path = os.path.join(root, args.config)
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as fp:
            cfg = json.load(fp)
        print(f"[配置] {args.config}："
              f"{len([k for k in cfg if not str(k).startswith('_')])} 张有曲名/调号/拍号"
              f"（另有页组 {len(cfg.get('_groups') or {})} 组）")

    names = collect_images(os.path.join(root, args.dir), args.only)
    if not names:
        print(f"[错误] {args.dir}/ 里没找到图片")
        return 1

    outdir = os.path.join(root, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "run_log.txt")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")     # 子进程统一按 UTF-8 输出

    # 属于「本次会合并的页组」的图：只出 `<图>_review.json`（+叠加图）供人工改，
    # **成品留给合并那一份**（见文件末尾的页组处理）。所以它们按 `--scan-only` 跑。
    groups = cfg.get("_groups") or {}
    grouped = set()
    if isinstance(groups, dict):
        for _pages in groups.values():
            if isinstance(_pages, list) and len(_pages) >= 2 and all(p in names for p in _pages):
                grouped.update(_pages)

    rows, failed = [], []
    cfg_filled, key_missing, title_missing = [], [], []   # 写回配置的项 / 没读到的调号 / 没读到的标题
    with open(log_path, "w", encoding="utf-8") as log:
        for idx, name in enumerate(names, 1):
            meta = cfg.get(name, {})
            cmd = [sys.executable, "-m", "jianpu2letter",
                   os.path.join(args.dir, name), "-o", args.outdir, "--traceback"]
            for flag, key in (("--title", "title"), ("--key", "key"), ("--meter", "meter")):
                if meta.get(key):
                    cmd += [flag, str(meta[key])]
            if args.no_render:
                cmd.append("--no-render")
            if args.letter_png:
                cmd.append("--letter-png")
            if args.scan_only or name in grouped:
                cmd.append("--scan-only")

            tag = meta.get("title") or name
            print(f"[{idx}/{len(names)}] {name}（{tag}）…", flush=True)
            proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True)
            text = proc.stdout.decode("utf-8", "replace") + \
                proc.stderr.decode("utf-8", "replace")
            log.write(f"===== {name} =====\n{text}\n\n")
            if proc.returncode != 0:
                failed.append(name)
                print(f"    [FAIL] 失败（细节见 {args.outdir}/run_log.txt）")
                continue
            got = parse_summary(text)
            rows.append((name, tag, got))
            print(f"    [OK] 低置信 {got['low']} | 字母/休止/红框 {got['conv']} | {got['dist']}")
            # 补齐配置里缺的字段：调号/标题尽量从页眉读，节拍数不自动读（默认 4/4）
            entry = cfg.setdefault(name, {})
            filled = []
            if not meta.get("key"):
                m = _KEY_LINE.search(text)
                if m:
                    entry["key"] = m.group(1)[2:]        # "1=Bb" → "Bb"
                    filled.append(f"key={entry['key']}")
                else:
                    key_missing.append(name)
            if not meta.get("title"):
                m = _TITLE_LINE.search(text)
                if m:
                    entry["title"] = m.group(1).strip()
                    filled.append(f"title={entry['title']}")
                else:
                    entry["title"] = "未读取标题"          # 读不到 → 写占位，提示用户手填
                    title_missing.append(name)
                    filled.append("title=未读取标题")
            if not meta.get("meter"):
                entry["meter"] = "4/4"                    # 节拍数不自动读，默认 4/4
                filled.append("meter=4/4")
            if filled:
                cfg_filled.append((name, "、".join(filled)))

    print()
    print(f"===== 汇总（{len(rows)}/{len(names)} 张成功）" + " =====")
    print(f"{'图':9s} {'曲名':<14s} {'低置信':>5s} {'字母/休止/红框':>14s}  数字分布")
    for name, tag, got in rows:
        print(f"{name:9s} {tag:<14s} {got['low']:>5s} {got['conv']:>14s}  {got['dist']}")
    if failed:
        print(f"\n[失败] {' '.join(failed)}（看 {args.outdir}/run_log.txt）")
    if cfg_filled:
        with open(cfg_path, "w", encoding="utf-8") as fp:
            json.dump(cfg, fp, ensure_ascii=False, indent=2)
            fp.write("\n")
        print(f"\n[配置] 已把读到的信息写回 {args.config}（下次直接用）：")
        for n, s in cfg_filled:
            print(f"   {n}: {s}")
    if key_missing:
        print(f"\n[调号?] 以下图没从页眉读到调号，请手填 {args.config} 的 \"key\"："
              + ", ".join(key_missing))
    if title_missing:
        print(f"\n[标题?] 以下图没从页眉读到标题，请手填 {args.config} 的 \"title\"："
              + ", ".join(title_missing))
    # ---- 页组：同一首曲子的多页合并成一份（`piano_config.json` 顶层 `_groups`）----
    # 组内每页只出了各自的 `_review.json`（+叠加图，见循环里的 `grouped`），
    # 这里把各页合并成一份并跑出**成品**。
    for gname, pages in (groups.items() if isinstance(groups, dict) else []):
        if not isinstance(pages, list) or len(pages) < 2:
            continue
        if not all(p in names for p in pages):
            # 只在「本次确实跑了这组的一部分」时才提醒 —— 只判 `args.only` 会把**跟本次
            # 毫无关系**的组也报一遍（用户 2026-09-28 跑 `--only 41.png` 时，配置里
            # 毫不相干的"年少有为"组被打成"只跑了 2 页中的一部分"，看着像操作错了）。
            if args.only and any(p in names for p in pages):
                print(f"\n[页组] {gname}：--only 只跑了 {len(pages)} 页中的一部分，跳过合并"
                      f"（整组一起跑才会合并）")
            continue
        review_paths = [os.path.join(outdir, os.path.splitext(p)[0] + "_review.json")
                        for p in pages]
        missing = [p for p, rp in zip(pages, review_paths) if not os.path.exists(rp)]
        if missing:
            print(f"\n[页组] {gname}：缺少 {'、'.join(missing)} 的审核 JSON，跳过合并")
            continue
        from jianpu2letter import pagegroup
        merged_path = os.path.join(outdir, gname + "_review.json")
        meta, _measures, seams = pagegroup.merge_reviews(review_paths, merged_path)
        print(f"\n[页组] {gname}：{' + '.join(pages)} → {os.path.basename(merged_path)}"
              f"（{meta.get('title')} {meta.get('key')} {meta.get('meter')}）")
        full = None
        try:
            _n, _d = (int(v) for v in str(meta.get("meter", "4/4")).split("/"))
            full = _n * 4.0 / _d
        except (ValueError, ZeroDivisionError):
            pass
        for s in seams:
            if s["merged"]:
                print(f"   · 第 {s['after_page']}/{s['after_page'] + 1} 页之间：小节被切开"
                      f"（{s['beats_prev']}+{s['beats_next']} 拍）→ 已自动合成一小节")
            elif full and not (abs(s["beats_prev"] - full) < 1e-6
                               and abs(s["beats_next"] - full) < 1e-6):
                print(f"   · [需核对] 第 {s['after_page']}/{s['after_page'] + 1} 页之间："
                      f"两页相邻小节是 {s['beats_prev']} 拍 + {s['beats_next']} 拍"
                      f"（整小节 {full:g} 拍）→ 不是干净的断点，请对照两页叠加图核对")
        # 第二阶段：拿合并后的 JSON 出成品（文件名前缀 = 组名）
        cmd = [sys.executable, "-m", "jianpu2letter", merged_path, "-o", args.outdir]
        if args.no_render:
            cmd.append("--no-render")
        if args.letter_png:
            cmd.append("--letter-png")
        if args.scan_only:
            cmd.append("--scan-only")
        proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True)
        text = proc.stdout.decode("utf-8", "replace") + proc.stderr.decode("utf-8", "replace")
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"===== [页组] {gname} =====\n{text}\n\n")
        for line in text.splitlines():
            if line.startswith(("[ok]", "[warn]", "[skip]")):
                print("   " + line)

    print(f"\n输出目录：{outdir}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
