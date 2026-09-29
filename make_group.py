# -*- coding: utf-8 -*-
"""交互式页组工具：把同一首曲子的两页合并成一份。

用法（在项目根目录下）：

    python make_group.py

运行后**逐个问你两张图片的名字**（例如 `36.jpg`、`37.png`，带不带 `piano/` 都能认），
然后：

1. 把页组写进 `piano_config.json` 顶层的 `_groups`（组名取第 1 页的曲名；
   以后跑 `run_all.py` 也会自动合并这两页）；
2. 每页各跑一次（`--scan-only`），**中间产物写到临时目录**（`_group_tmp/`，跑完删掉）；
3. **合并**成一份 `<组名>_review.json`（行号顺延、元数据取第 1 页、接缝自动判定，
   见 `jianpu2letter/pagegroup.py`）；
4. 用合并后的 JSON 出成品。

**`out2/` 里只留这 5 个文件**（前缀都是组名）：

    <组名>_review.json    合并后的完整审核 JSON（改它再喂回即可重出）
    <组名>_overlay.png    两张单页叠加图**竖着拼成一张**（合并谱没有单一原图，
                          这是最接近"整首原谱 + 字母"的形式）
    <组名>_letter.xml     字母谱 XML
    <组名>_jianpu.png     简谱图
    <组名>_jianpu.txt     逐行字符序列

⚠ 本脚本**只读**用原有代码（`subprocess` 调 `run_all.py` / `jianpu2letter` +
`pagegroup`），**不改任何原有代码**。
"""

import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = "piano"
OUT_DIR = "out2"           # 只放合并后的 5 个文件
TMP_DIR = "_group_tmp"     # 两页的中间产物，跑完删掉
CONFIG = "piano_config.json"
_GROUP_OUT_EXTS = ("_review.json", "_overlay.png", "_letter.xml", "_jianpu.png", "_jianpu.txt")


def _ask_image(prompt, imgs):
    while True:
        raw = input(prompt).strip().strip('"').strip("'")
        if not raw:
            print("  （不能为空，重来）")
            continue
        name = os.path.basename(raw)
        if name not in imgs:
            print(f"  [错误] piano/ 里没有 {name!r}。现有：")
            for n in sorted(imgs, key=lambda x: int(''.join(c for c in x if c.isdigit()) or 0)):
                print(f"     {n}")
            continue
        return name


def _run(cmd):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True)
    return proc.returncode, (proc.stdout.decode("utf-8", "replace")
                             + proc.stderr.decode("utf-8", "replace"))


def _stack_overlays(paths, out_path):
    """把各页叠加图按同一宽度缩放后竖着拼成一张（中间留一条细分隔）。"""
    import cv2
    import numpy as np
    width, parts, gap = 1600, [], None
    for path in paths:
        if not os.path.exists(path):
            continue
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        fx = width / img.shape[1]
        pages = cv2.resize(img, (width, max(1, int(img.shape[0] * fx))),
                           interpolation=cv2.INTER_AREA)
        if parts:
            gap = np.full((8, width, 3), 180, np.uint8)
            parts.append(gap)
        parts.append(pages)
    if not parts:
        return None
    ok, buf = cv2.imencode(".png", np.vstack(parts))     # 走 imencode：中文路径下 imwrite 会失败
    if ok:
        buf.tofile(out_path)
        return out_path
    return None


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    img_root = os.path.join(HERE, IMG_DIR)
    imgs = [n for n in os.listdir(img_root)
            if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))]
    if not imgs:
        print(f"[错误] {IMG_DIR}/ 里没有图片")
        return 1

    print("=" * 56)
    print(" 页组合并 —— 把同一首曲子的两页合并成一份")
    print("=" * 56)
    a = _ask_image("第 1 页的图片名：", imgs)
    b = _ask_image("第 2 页的图片名：", imgs)
    while b == a:
        print("  （两页不能是同一张）")
        b = _ask_image("第 2 页的图片名：", imgs)
    pages = [a, b]

    cfg_path = os.path.join(HERE, CONFIG)
    with open(cfg_path, encoding="utf-8") as fp:
        cfg = json.load(fp, object_pairs_hook=dict)
    gname = str((cfg.get(a) or {}).get("title") or "")
    if gname in ("", "未读取标题"):
        gname = os.path.splitext(a)[0]
    print(f"\n→ 页序：{' + '.join(pages)}    组名：{gname}")
    if input("确认？（直接回车 = 确认，输入 n 取消）").strip().lower() == "n":
        print("已取消。")
        return 0

    # ① 写进配置
    cfg.setdefault("_groups", {})[gname] = pages
    with open(cfg_path, "w", encoding="utf-8") as fp:
        json.dump(cfg, fp, ensure_ascii=False, indent=2)
        fp.write("\n")
    print(f"[配置] 已写入 _groups：{gname} -> {pages}")

    out_root = os.path.join(HERE, OUT_DIR)
    tmp_root = os.path.join(HERE, TMP_DIR)
    shutil.rmtree(tmp_root, ignore_errors=True)
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(tmp_root, exist_ok=True)
    # 清掉 out2 里上次留下的（只删本工具产出的那些名字）
    for f in os.listdir(out_root):
        if f.startswith(tuple(gname + e for e in _GROUP_OUT_EXTS)) or f == "run_log.txt":
            os.remove(os.path.join(out_root, f))

    try:
        # ② 每页各跑一次 —— 中间产物进临时目录
        for p in pages:
            print(f"\n[{p}] 识别中…")
            rc, text = _run([sys.executable, os.path.join(HERE, "run_all.py"),
                             "--only", p, "-o", TMP_DIR, "--scan-only"])
            if rc != 0:
                print(f"[失败] {p}（细节见 {TMP_DIR}/run_log.txt）")
                print(text[-800:])
                return 2
            for line in text.splitlines():
                if line.startswith(("[ok]", "[配置]", "[调号", "[标题", "[FAIL]")):
                    print("   " + line)

        # ③ 合并 → out2/<组名>_review.json
        sys.path.insert(0, HERE)
        from jianpu2letter import pagegroup      # 只读复用
        review_paths = [os.path.join(tmp_root, os.path.splitext(p)[0] + "_review.json")
                        for p in pages]
        missing = [p for p, r in zip(pages, review_paths) if not os.path.exists(r)]
        if missing:
            print(f"[错误] 缺少 {'、'.join(missing)} 的审核 JSON，无法合并")
            return 2
        merged = os.path.join(out_root, gname + "_review.json")
        meta, _measures, seams = pagegroup.merge_reviews(review_paths, merged)
        print(f"\n[页组] 已合并 → {OUT_DIR}/{gname}_review.json")
        print(f"       元数据取第 1 页：{meta.get('title')} {meta.get('key')} {meta.get('meter')}")
        try:
            _num, _den = (int(v) for v in str(meta.get("meter", "4/4")).split("/"))
            full = _num * 4.0 / _den
        except (ValueError, ZeroDivisionError):
            full = None
        for s in seams:
            if s["merged"]:
                print(f"       · 第 {s['after_page']}/{s['after_page'] + 1} 页之间：小节被切开"
                      f"（{s['beats_prev']}+{s['beats_next']} 拍）→ 已自动合成一小节")
            elif full and abs(s["beats_prev"] - full) < 1e-6 and abs(s["beats_next"] - full) < 1e-6:
                print(f"       · 第 {s['after_page']}/{s['after_page'] + 1} 页之间："
                      f"两小节各自 {full:g} 拍（整小节）→ 直接拼接")
            else:
                print(f"       · [需核对] 第 {s['after_page']}/{s['after_page'] + 1} 页之间："
                      f"相邻小节是 {s['beats_prev']} 拍 + {s['beats_next']} 拍"
                      f"（整小节 {full:g} 拍）→ 不是干净的断点，请对着两页的叠加图核对")

        # ④ 用合并后的 JSON 出成品 → out2
        print(f"\n[{gname}] 出成品中…")
        rc, text = _run([sys.executable, "-m", "jianpu2letter", merged, "-o", OUT_DIR])
        if rc != 0:
            print("[失败] 成品生成失败：")
            print(text[-800:])
            return 2
        for line in text.splitlines():
            if line.startswith(("[ok]", "[warn]", "[skip]")):
                print("   " + line)

        # ⑤ 两张单页叠加图 → 竖拼成合并版的 _overlay.png
        ov = _stack_overlays(
            [os.path.join(tmp_root, os.path.splitext(p)[0] + "_overlay.png") for p in pages],
            os.path.join(out_root, gname + "_overlay.png"))
        print(f"   [ok] 合并叠加图：{OUT_DIR}\\{gname}_overlay.png" if ov
              else "   [warn] 单页叠加图缺失，跳过合并叠加图")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)      # 中间产物不留

    print(f"\n===== {OUT_DIR}/ 里的产物（应只有这 5 个）=====")
    for f in sorted(os.listdir(out_root)):
        print(f"   {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
