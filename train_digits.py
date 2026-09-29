# -*- coding: utf-8 -*-
"""
train_digits.py —— 用「网站简谱 JSON（真值）+ 对应图片」训练本字体数字分类器。

原理：
  同一首歌既有图片又有网站整理的 JSON（含每个音的正确唱名），于是可以：
    ① 跑现有流程切出图片里的每个数字块（这一步已可靠）；
    ② 读 JSON 得到每个小节的正确数字序列；
    ③ **按小节对齐**——只有当「图片里该小节切出的块数 == JSON 该小节的音数」
       时才配对（这是强正确信号，配不上直接跳过）；
    ④ 把 (数字块字形特征, 正确数字) 存进 digit_model.npz。
  之后识别新谱时，数字块直接在这个模型里做最近邻——判据从「像不像系统字体的 5」
  变成「像不像你自己的谱里的 5」，从根本上解决字体不同导致的 5/6、0/6 混淆。

用法：
  python train_digits.py            # 按 train_pairs.json 训练（不存在则用内置默认）
  python train_digits.py --list     # 只统计各对的可用样本数，不写模型

train_pairs.json 格式（可自行增删）：
  [{"image": "1.jpg", "json": "上海滩.json"}, ...]

产出：digit_model.npz（放在本目录）。没有它时，识别流程自动回退到原有判据，
      因此不会影响任何已有用法。
"""

import argparse
import json
import os
import sys

import numpy as np

DEFAULT_PAIRS = [
    {"image": "1.jpg", "json": "上海滩.json"},
    {"image": "小星星.png", "json": "小星星.json"},
    {"image": "4.png", "json": "我的中国心.json"},
    {"image": "5.png", "json": "千年等一回.json"},
]


def load_pairs(base_dir):
    path = os.path.join(base_dir, "train_pairs.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:  # noqa: BLE001
            print("[warn] train_pairs.json 解析失败，用内置默认")
    return DEFAULT_PAIRS


def collect_from_pair(image_path, json_path, base_dir):
    """从一对 (图片, 网站JSON) 采集样本，返回 (feats, labels, 说明)。"""
    sys.path.insert(0, base_dir)
    from jianpu2letter import jianpu_json, ocr_jianpu

    # ① 真值：JSON 的小节 → 数字序列（休止符记 0）
    _, truth_measures = jianpu_json.parse(json_path)
    truth = []
    for m in truth_measures:
        seq = []
        for note in m["notes"]:
            seq.append(0 if note.get("rest") else note.get("n"))
        truth.append(seq)

    # ② 图片：跑流程，拿到 measures（含每音的 cx/row）与每行的数字块
    measures = ocr_jianpu.ocr_to_stream(image_path, debug_dir=None)
    rows_info = getattr(ocr_jianpu, "LAST_ROW_INFO", []) or []

    # 行 → 该行的块（按 x 排序）
    blocks_by_row = {}
    for r in rows_info:
        blocks_by_row[r.get("row_index")] = sorted(r["blocks"], key=lambda b: b["x"])

    # 每个 measure 的块（按音序取最近 cx 的块）
    def blocks_of_measure(measure):
        row = measure.get("row")
        pool = list(blocks_by_row.get(row, []))
        picked = []
        for note in measure.get("notes", []):
            cx = note.get("cx")
            if cx is None or not pool:
                picked.append(None)
                continue
            best = min(pool, key=lambda b: abs(b["x"] + b["w"] / 2 - cx))
            pool.remove(best)
            picked.append(best)
        return picked

    feats, labels = [], []
    used_measures = 0
    for i, m in enumerate(measures):
        if i >= len(truth):
            break
        blocks = blocks_of_measure(m)
        seq = truth[i]
        if any(b is None for b in blocks):
            continue
        if len(blocks) != len(seq):
            continue  # 块数与真值音数不一致 → 不配对（避免污染样本）
        for block, digit in zip(blocks, seq):
            if digit is None:
                continue
            feats.append(ocr_jianpu._shape_feature(block, ocr_jianpu._LAST_BINARY))
            labels.append(int(digit))
        used_measures += 1

    note = (f"{os.path.basename(image_path)}：用了 {used_measures} 个小节，"
            f"采到 {len(labels)} 条样本")
    return feats, labels, note


def main(argv=None):
    parser = argparse.ArgumentParser(prog="train_digits.py")
    parser.add_argument("--list", action="store_true", help="只统计，不写模型")
    parser.add_argument("--dir", default=None, help="工作目录（默认脚本所在目录）")
    args = parser.parse_args(argv)

    base_dir = args.dir or os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    sys.path.insert(0, base_dir)

    pairs = load_pairs(base_dir)
    all_feats, all_labels, groups = [], [], []
    for gi, pair in enumerate(pairs):
        img, js = pair.get("image"), pair.get("json")
        if not img or not js or not os.path.exists(img) or not os.path.exists(js):
            print(f"[跳过] {img} ↔ {js}（文件缺失；网站 JSON 需先下载）")
            continue
        try:
            feats, labels, note = collect_from_pair(img, js, base_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[报错] {img}：{exc.__class__.__name__}: {exc}")
            continue
        print("[采集] " + note)
        all_feats.extend(feats)
        all_labels.extend(labels)
        groups.extend([gi] * len(labels))

    if not all_labels:
        print("没有采集到样本（检查图片与 JSON 是否成对、且能正常识别）")
        return 1

    feats_arr = np.stack(all_feats).astype(np.float32)
    labels_arr = np.array(all_labels, dtype=np.int16)

    print("-" * 56)
    dist = {int(d): int((labels_arr == d).sum()) for d in sorted(set(all_labels.tolist()))}
    print(f"样本合计 {len(labels_arr)} 条，数字分布 {dist}")
    for gi, pair in enumerate(pairs):
        n = int((np.array(groups) == gi).sum())
        if n:
            print(f"  来自 {pair.get('image')}：{n} 条")

    if args.list:
        return 0

    out_path = os.path.join(base_dir, "digit_model.npz")
    np.savez_compressed(out_path, feats=feats_arr, labels=labels_arr)
    print(f"已写出数字模型：{out_path}")
    print("之后跑识别时会自动加载它（没有该文件则回退到原有判据）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
