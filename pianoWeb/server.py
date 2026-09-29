# -*- coding: utf-8 -*-
"""简谱图片 → 字母谱「服务」（HTTP 接口）。

    python pianoWeb/server.py                 # 默认 http://127.0.0.1:8000
    python pianoWeb/server.py --port 8080

- 浏览器打开首页就是上传页（选图 → 出 `_overlay.png` 预览 + JSON）。
- 程序调用走 `POST /convert`（multipart 字段 `image`）：
      curl -F "image=@piano/36.jpg" http://127.0.0.1:8000/convert

**输入**：一张简谱图片（页面支持**拖拽**，也可点选）。可选表单字段 `key` / `title` / `meter`
覆盖页眉读数；`key` 的写法随便：`♭A` / `bA` / `A♭` / `Ab` 都会被归一成内部形式 `Ab`
（谱面印降号在字母左上角，本工具内部统一字母在前）。
（⚠ `key` / `meter` 只进元数据，**不改变识别出的字母** —— 字母是「数字→键位」的固定首调映射。）
**输出**：`_overlay.png`（数字原位盖成字母的对照图）+ 一份 `_notes.json`：顶层带
**曲目 / 调号 / 拍号**，`lines` 里逐音给出「原谱数字 n / 八度 o → 转换后字母 letter / Shift shift」。
默认 JSON **只给「行 / 数字 / 字母」**（`n` / `o` / `letter` / `shift`）—— 一个音一个对象，
够做对照用。另有几个开关：
  `?full=1`    —— 给完整字段（时值 beams/dotted/extend、歌词、小节线、位置 source_x…）。
  `?review=1`  —— 另给工具链标准的 `_review.json`（**不含** letter/shift，因为那是算出来的、
                  页面上改完音高就过期，混进去反而误导）。
  `?inline=1`  —— 把 overlay 以 base64 塞进响应；`?debug=1` 附带识别日志。

## 为什么这样接
- 不自己发明 schema：JSON 就是工具链既有的 `_review.json` 结构，**只多两个字段**
  `letter` / `shift` —— 这样页面端（`pianoWeb/`）和命令行两条路看到的是同一套数据。
- `_overlay.png` 由 `overlay_letters.render` 画；它默认读 `ocr_jianpu._LAST_COLOR`
  这个**模块全局**（底图必须是刚识别时那张缩放图），所以下面用一把锁把
  「OCR + 画图」整段串起来 —— 否则并发两个请求会互相串底图。
- 每次请求一个独立工作目录（`<系统临时目录>/jianpu_api/<id>/`），产物互不覆盖。
"""

import argparse
import base64
import io
import json
import os
import pickle
import re
import sys
import tempfile
import threading
import uuid
import contextlib

import cv2
from flask import Flask, jsonify, request, send_file, Response
from werkzeug.utils import secure_filename

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jianpu2letter import mapping, ocr_jianpu, overlay_letters, review_io  # noqa: E402

WORK_ROOT = os.path.join(tempfile.gettempdir(), "jianpu_api")
# 识别 + 画图整段串行：两者都依赖 `ocr_jianpu` 的模块全局（`_LAST_COLOR` 等），
# 并发跑会互相串底图/缩放倍数。单机自用服务，一把锁足够，也避免 cv2 多线程抢内存。
_LOCK = threading.Lock()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024      # 单张上限 32MB


# 项目内部统一「字母 + 升降号在后」：`piano_config.json` 与
# `musicxml_reader.MAJOR_KEYS_BY_FIFTHS` 都这么写（Cb Gb Db Ab Eb Bb F C G D A E B F# C#）。
_VALID_KEYS = ("Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B",
               "F#", "C#")


def _normalize_key(raw):
    """把各种写法归一成内部形式（`Ab` / `F#` / `C`）。

    谱面把降号印在字母**左上角**（`1=♭A`），口述常写成 `bA`；而本工具内部一律
    「字母在后」（`Ab`）。这里四种写法都收：`♭A` / `bA` / `A♭` / `Ab` → `Ab`。
    认不出来的原样返回 —— 宁可让下游报错，也别悄悄换成别的调。
    """
    s = str(raw or "").strip()
    for pre in ("1=", "1＝"):
        if s.startswith(pre):
            s = s[len(pre):]
    s = s.replace("♭", "b").replace("♯", "#").replace("＃", "#")
    if not s:
        return None
    if len(s) >= 2 and s[0].lower() in ("b", "#") and s[1].lower() in "abcdefg":
        name = s[1].upper() + s[0].lower()      # 升降号在前：bA / #F
    elif s[0].lower() in "abcdefg":
        name = s[0].upper() + s[1:]             # 字母在前：Ab / F#
    else:
        name = s.upper()
    return name if name in _VALID_KEYS else s


def _slim_note(x):
    """精简视图里的一个音：`id / n / o / letter`（+ 需要时才有 `shift` / `unknown`）。

    `id` 必须留 —— 页面上的"改这个音/删这个音"要靠它回灌到 `_review.json`。
    `unknown: true` 是给"认不出来的那些"打的显式标记（页面上红底显示）。
    """
    n = x.get("n")
    out = {"id": x.get("id"), "n": n, "o": x.get("o", 0), "letter": x.get("letter")}
    if x.get("shift"):
        out["shift"] = x["shift"]
    if n is None:
        out["unknown"] = True
    return out


def _letter(n, o):
    """原始数字 + 八度 → 字母谱键位。返回 (letter, shift)；(None,None) = 没有音高。

    休止（n=0）在字母谱里仍是 `0`；认不出的块（n 为 None）没有字母。
    """
    if n is None:
        return None, None
    if int(n) == 0:
        return "0", None
    try:
        key, mod = mapping.key_for(int(n), int(o))
        return key, mod
    except Exception:                       # 八度越界等：不回 500，如实标成无字母
        return None, None


def _run(image_path, workdir, stem):
    """跑一趟识别。返回 (measures, scale, log)。

    **把识别用的缩放彩色底图落盘**（`<stem>_base.png`）：`overlay_letters.render` 要一张
    与音符坐标同空间的底图，默认从 `ocr_jianpu._LAST_COLOR` 取 —— 那是**模块全局**，
    而 `/apply`（不重跑 OCR）可能隔了好几轮请求，全局早被别的图覆盖了。落盘才拿得准。
    """
    with _LOCK:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            measures = ocr_jianpu.ocr_to_stream(image_path, debug_dir=None)
            base = ocr_jianpu._LAST_COLOR
            scale = ocr_jianpu._LAST_SCALE
            if base is not None:
                ok, imbuf = cv2.imencode(".png", base)
                if ok:
                    with open(os.path.join(workdir, stem + "_base.png"), "wb") as fp:
                        fp.write(imbuf.tobytes())
        log = buf.getvalue()
    return measures, scale, log


def _notes_json(measures, meta_extra):
    """在既有 review JSON 结构上逐音补 `letter` / `shift`，组成对外响应。"""
    review_path = meta_extra["_review_path"]
    review_io.save_review(
        review_path, measures,
        title=meta_extra["title"], key_mark=meta_extra["key"],
        meter=meta_extra["meter"], source_image=meta_extra["source"])
    import json
    with open(review_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    counted = {"total": 0, "letters": 0, "rests": 0, "unknown": 0}
    for line in payload.get("lines") or []:
        for i, note in enumerate(line.get("notes") or [], 1):
            note["line"] = line.get("line")      # 行号
            note["index"] = i                    # 该行第几个音（id 里的 N 部分）
            ch, shift = _letter(note.get("n"), note.get("o", 0))
            note["letter"] = ch
            note["shift"] = shift              # None / "LSHIFT" / "RSHIFT"
            counted["total"] += 1
            if note.get("n") is None:
                counted["unknown"] += 1
            elif int(note.get("n")) == 0:
                counted["rests"] += 1
            elif ch:
                counted["letters"] += 1
    payload["_instructions"] = list(payload.get("_instructions") or []) + [
        "line / index —— 行号与该行内的序号（与 id 的 L03N029 对应）。",
        "letter —— 该音转换后的字母谱键位（n=0 休止时为 \"0\"；认不出时为 null）。",
        "shift  —— None / \"LSHIFT\" / \"RSHIFT\"：映射2 里 ±2、±3 八度用 Shift 层。",
        "source_x —— 该音在**识别用缩放图**上的 x（overlay 就是画在这张图上的）；"
        "要换回原图坐标除以 OCR 日志里的缩放倍数。",
    ]
    return payload, counted


def _handle(image_bytes, filename, form):
    rid = uuid.uuid4().hex[:12]
    workdir = os.path.join(WORK_ROOT, rid)
    os.makedirs(workdir, exist_ok=True)
    safe = secure_filename(filename) or "upload.png"
    stem = os.path.splitext(safe)[0] or "upload"
    src = os.path.join(workdir, safe)
    with open(src, "wb") as fp:
        fp.write(image_bytes)

    measures, scale, log = _run(src, workdir, stem)

    raw_key = form.get("key") or getattr(ocr_jianpu, "_LAST_KEY_MARK", None) or "C"
    key = "1=" + (_normalize_key(raw_key) or "C")
    title = form.get("title") or getattr(ocr_jianpu, "_LAST_TITLE", None) or stem
    meter = form.get("meter") or "4/4"
    meta = {"title": title, "key": key, "meter": meter, "source": filename,
            "_scale": scale, "_stem": stem,
            "_review_path": os.path.join(workdir, stem + "_review.json")}
    # 把 OCR 的 measures 落盘：`/apply` 要**改它**（而不是改 review JSON）—— 画字母靠
    # 每个音的 `box`（图上外框），而 `box` 只有 OCR 现场有，`review_io.load_review`
    # 从 JSON 读回来的音符没有它，`overlay_letters.render` 会直接跳过 → 重出的图一个字都没有。
    with open(os.path.join(workdir, stem + "_measures.pkl"), "wb") as fp:
        pickle.dump(measures, fp)
    # 记一份给 `/apply` 用：它不重跑 OCR，要靠这些从磁盘把底图和 measures 找回来
    with open(os.path.join(workdir, "_meta.json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False)

    return _finish(measures, workdir, stem, rid, meta, log)


def _iter_notes_with_ids(measures):
    """按 `review_io.save_review` 的规则给每个音编号（`L{行}N{该行内序号}`）。

    必须和它**逐字一致**，否则 `/apply` 拿到的 `noteId` 对不上音。
    """
    groups, current, current_row = [], [], object()
    for measure in measures:
        row = measure.get("row", 0)
        if current and row != current_row:
            groups.append(current)
            current = []
        current_row = row
        current.append(measure)
    if current:
        groups.append(current)
    out = []
    for line_no, group in enumerate(groups, 1):
        note_no = 0
        for measure in group:
            for note in measure.get("notes", []):
                note_no += 1
                out.append(("L%02dN%03d" % (line_no, note_no), note, measure))
    return out


@app.post("/apply")
def apply_edits():
    """按人工修正重出 overlay + `_notes.json`（**不重跑 OCR** —— 图没变，重跑会把修正冲掉）。

    请求体 JSON：
        {"id": "<某次 /convert 返回的 id>",
         "edits": [{"noteId": "L03N029", "op": "set", "n": 5, "o": 0},
                   {"noteId": "L08N020", "op": "remove"}]}
    返回与 `/convert` 同形（多一个 `applied` 汇总），页面据此原地刷新。
    """
    data = request.get_json(silent=True) or {}
    rid = str(data.get("id") or "")
    edits = data.get("edits") or []
    if not re.fullmatch(r"[0-9a-f]{12}", rid):
        return jsonify({"ok": False, "error": "缺少或非法的 id"}), 400
    workdir = os.path.join(WORK_ROOT, rid)
    meta_path = os.path.join(workdir, "_meta.json")
    if not os.path.isfile(meta_path):
        return jsonify({"ok": False,
                        "error": "这一次的结果已经不在临时目录里了（服务重启过或临时文件被清过）"
                                 "——请重新识别一次"}), 404
    with open(meta_path, "r", encoding="utf-8") as fp:
        meta = json.load(fp)
    stem = meta.get("_stem") or "upload"
    measures_path = os.path.join(workdir, stem + "_measures.pkl")
    if not os.path.isfile(measures_path):
        return jsonify({"ok": False, "error": "找不到该次的识别中间结果，请重新识别一次"}), 404
    with open(measures_path, "rb") as fp:
        measures = pickle.load(fp)

    index = {nid: (note, measure) for nid, note, measure in _iter_notes_with_ids(measures)}
    applied = {"set": 0, "removed": 0, "missing": []}
    for e in edits:
        nid = str(e.get("noteId") or "")
        hit = index.get(nid)
        if hit is None:
            applied["missing"].append(nid)
            continue
        note, measure = hit
        if e.get("op") == "remove":
            lst = measure.get("notes") or []
            for i, x in enumerate(lst):
                if x is note:
                    lst.pop(i)
                    break
            applied["removed"] += 1
        elif e.get("op") == "set":
            try:
                n = int(e["n"])
                o = int(e.get("o") or 0)
            except (KeyError, TypeError, ValueError):
                applied["missing"].append(nid)
                continue
            note["n"] = n
            note["o"] = o
            note["rest"] = (n == 0)
            applied["set"] += 1
    with open(measures_path, "wb") as fp:          # 改完存回，页面可连续多轮修改
        pickle.dump(measures, fp)
    return jsonify(_finish(measures, workdir, stem, rid, meta, "", applied=applied))


def _body_lines(payload, full):
    """精简视图：只给「第几行 / 什么数字 / 转换后什么字母」（+ 需要时 shift/unknown）。
    完整的 25 个字段（时值、歌词、小节线、位置…）对"数字↔字母对照"是噪音，要全量加 `?full=1`。"""
    if full:
        return payload.get("lines") or []
    return [{"line": ln.get("line"),
             "notes": [_slim_note(x) for x in (ln.get("notes") or [])]}
            for ln in (payload.get("lines") or [])]


def _finish(measures, workdir, stem, rid, meta, log, applied=None):
    """由 measures 出 overlay + `_notes.json` 并组装响应。`/convert` 与 `/apply` 共用。

    ⚠ 底图取**落盘的那张**（`<stem>_base.png`）而不是全局 `_LAST_COLOR` —— `/apply` 不重跑
    OCR，全局可能已被别的请求覆盖，用它会把字母画到别人的谱上。
    """
    base = ocr_jianpu._imread_unicode(os.path.join(workdir, stem + "_base.png"), cv2.IMREAD_COLOR)
    overlay_path = os.path.join(workdir, stem + "_overlay.png")
    with _LOCK:
        drawn, rests, unknown_n, ok = overlay_letters.render(
            measures, overlay_path, color=base, scale=meta.get("_scale"))

    payload, counted = _notes_json(measures, meta)
    payload.pop("_review_path", None)
    full = request.args.get("full") == "1"
    body_lines = _body_lines(payload, full)
    inst = ["id —— 该音在 `_review.json` 里的 id（`L03N029` = 第 3 行第 29 音）；改/删要靠它。",
            "n —— 原谱数字（0=休止，null=没认出来）。",
            "o —— 八度：-1 低音 / 0 中音 / 1 高音（±2、±3 见 shift）。",
            "letter —— 转换后的字母谱键位（休止为 \"0\"，认不出为 null）。",
            "unknown —— 只出现在**认不出来**的音上（`n` 为 null 且 `letter` 为 null），"
            "用来在前端红底标出；overlay 上对应位置也画了红框。",
            "shift —— null / \"LSHIFT\" / \"RSHIFT\"：映射2 里 ±2、±3 八度靠 Shift 层，"
            "所以 letter 相同时靠它区分音区（只有需要时才出现）。",
            "改/删认不出的音：`POST /apply` {\"id\":…, \"edits\":[{\"noteId\":…,\"op\":\"set\","
            "\"n\":5,\"o\":0}]}（页面上有「应用并重跑」）。",
            "完整字段（时值 beams/dotted/extend、歌词、小节线、位置 source_x…）加 `?full=1`。"]
    doc = {"format": "jianpu2letter-notes-v1",
           "title": meta["title"], "key": meta["key"], "meter": meta["meter"],
           "source": meta["source"], "counts": counted,
           "_instructions": inst, "lines": body_lines}
    notes_path = os.path.join(workdir, stem + "_notes.json")
    with open(notes_path, "w", encoding="utf-8") as fp:
        json.dump(doc, fp, ensure_ascii=False, indent=2)

    files = {"overlay": "/files/%s/%s_overlay.png" % (rid, stem),
             "notes": "/files/%s/%s_notes.json" % (rid, stem)}
    if request.args.get("review") == "1":
        # `_review.json` 是工具链的标准审核文件，能喂回 `python -m jianpu2letter … -o out`
        # 重出成品。它**不含** letter/shift —— 那两个是算出来的，改完音高就会过期，
        # 混进 review 反而误导。默认不导出，要就 `?review=1`。
        files["review"] = "/files/%s/%s_review.json" % (rid, stem)
    body = {"ok": bool(ok), "id": rid, "source": meta["source"],
            "title": meta["title"], "key": meta["key"], "meter": meta["meter"],
            "files": files, "counts": counted,
            "overlayStats": {"drawn": drawn, "rests": rests, "unknown": unknown_n},
            "lines": body_lines, "_instructions": inst}
    if applied is not None:
        body["applied"] = applied
    if request.args.get("inline") == "1":
        with open(overlay_path, "rb") as fp:
            body["overlay"] = {"png_base64": base64.b64encode(fp.read()).decode("ascii")}
    if request.args.get("debug") == "1":
        body["log"] = log
    return body


@app.post("/convert")
def convert():
    f = request.files.get("image") or request.files.get("file")
    if f is None:
        return jsonify({"ok": False, "error": "缺少图片：请用 multipart 字段 image 上传"}), 400
    ext = os.path.splitext(f.filename or "")[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        return jsonify({"ok": False, "error": "不支持的图片类型：%s" % ext}), 400
    try:
        return jsonify(_handle(f.read(), f.filename or "upload.png", request.form))
    except Exception as exc:                # noqa: BLE001
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}), 500


@app.get("/files/<rid>/<path:name>")
def files(rid, name):
    if not re.fullmatch(r"[0-9a-f]{12}", rid):
        return jsonify({"ok": False, "error": "bad id"}), 400
    safe = secure_filename(name)
    path = os.path.join(WORK_ROOT, rid, safe)
    if not os.path.isfile(path):            # secure_filename 挡了 ../，这里再确认一次目录
        return jsonify({"ok": False, "error": "not found"}), 404
    return send_file(path, max_age=0)


PAGE = """<!doctype html><meta charset="utf-8"><title>简谱 → 字母谱</title>
<style>
 body{font:14px/1.6 system-ui,"Microsoft YaHei",sans-serif;max-width:980px;margin:32px auto;padding:0 16px;color:#222}
 h1{font-size:20px;margin:0 0 4px} .hint{color:#666;margin-bottom:18px}
 .row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0}
 input[type=text]{padding:6px 8px;border:1px solid #ccc;border-radius:6px;width:110px}
 button{padding:8px 16px;border:0;border-radius:7px;background:#1a7f37;color:#fff;cursor:pointer}
 button:disabled{background:#999;cursor:default}
 pre{background:#f6f8fa;border:1px solid #e2e5e9;border-radius:8px;padding:12px;overflow:auto;max-height:420px}
 img{max-width:100%;border:1px solid #ddd;border-radius:8px;margin-top:10px}
 .err{color:#b00;white-space:pre-wrap}
.drop{border:2px dashed #c2c8d0;border-radius:10px;padding:22px 16px;text-align:center;
  color:#667;cursor:pointer;transition:.15s;margin:6px 0 12px}
.drop:hover{border-color:#8a93a0}
.drop.over{border-color:#1a7f37;background:#f0fbf3;color:#1a7f37}
.drop input{display:none}
.drop b{color:#1a7f37}
select{padding:7px 8px;border:1px solid #ccc;border-radius:6px;background:#fff;font:inherit}
.chip{display:inline-block;margin:2px 1px;padding:1px 6px;border:1px solid #dde1e6;border-radius:5px;
  font:12px/1.5 ui-monospace,Consolas,monospace;background:#fafbfc}
.chip b{color:#1a7f37;font-weight:600}
.chip.rest{color:#889;background:#f3f4f6}
.chip.unk{background:#ffe3e3;border-color:#e88;color:#a00;font-weight:700}
.linechips{margin:3px 0}
.linechips .ln{display:inline-block;min-width:34px;color:#9aa;font:11px/1.6 sans-serif}
.fixlist{max-height:250px;overflow:auto;border:1px solid #e2e5e9;border-radius:8px;padding:8px;background:#fbfcfd}
.fixlist label{display:inline-block;min-width:150px;font:12px/1.9 ui-monospace,Consolas,monospace}
#ovimg{margin-top:6px}
.fold{margin:10px 0;border:1px solid #e2e5e9;border-radius:8px;background:#fbfcfd}
.fold>summary{cursor:pointer;padding:9px 12px;font-weight:600;color:#334}
.fold[open]>summary{border-bottom:1px solid #e2e5e9}
.fold>div,.fold>pre{margin:0;border:0;border-radius:0 0 8px 8px;max-height:460px}
</style>
<h1>简谱图片 → 字母谱</h1>
<div class="hint">选一张简谱图片，得到 <b>_overlay.png</b>（数字原位盖成字母）和逐音对照 JSON。
 调号/曲名/拍号留空就用页眉自动读到的。</div>
<div class="drop" id="drop">
  <input type="file" id="f" accept=".png,.jpg,.jpeg,.bmp,.webp">
  <div id="dropmsg">把简谱图片<b>拖到这里</b>，或点这里选文件（松手即开始识别）</div>
</div>
<div class="row">
  <label class="hint" for="key">调号</label>
  <select id="key" title="留空就用页眉自动读到的。⚠ 调号只进元数据，不改变识别出的字母">
    <option value="">（留空 = 读页眉）</option>
    <option value="1=Cb">1=Cb（♭C）</option>
    <option value="1=Gb">1=Gb（♭G）</option>
    <option value="1=Db">1=Db（♭D）</option>
    <option value="1=Ab">1=Ab（♭A）</option>
    <option value="1=Eb">1=Eb（♭E）</option>
    <option value="1=Bb">1=Bb（♭B）</option>
    <option value="1=F">1=F</option>
    <option value="1=C">1=C</option>
    <option value="1=G">1=G</option>
    <option value="1=D">1=D</option>
    <option value="1=A">1=A</option>
    <option value="1=E">1=E</option>
    <option value="1=B">1=B</option>
    <option value="1=F#">1=F#（♯F）</option>
    <option value="1=C#">1=C#（♯C）</option>
  </select>
  <input type="text" id="title" placeholder="曲名（留空 = 页眉 / 文件名）">
  <label class="hint" for="meter">拍号</label>
  <input type="text" id="meter" list="meterlist" placeholder="（默认 4/4）"
         title="⚠ 拍号也只进元数据：切小节是按谱面竖线切的">
  <datalist id="meterlist">
    <option>4/4</option><option>2/4</option><option>3/4</option><option>6/8</option>
    <option>3/8</option><option>9/8</option><option>12/8</option><option>2/2</option>
  </datalist>
  <button id="go">开始识别</button>
</div>
<div class="hint" style="margin-top:-6px">调号按<b>五度圈</b>排列（♭C→…→♯C，15 个，12 个音级全覆盖）。
  <b>谱面把降号写在字母左上角（`1=♭A`），本工具内部统一记作字母在后（`Ab`）</b> ——
  下拉里两个写法都标了，提交的是内部写法。调号 / 拍号都只进元数据，<b>不影响识别出的字母</b>。</div>
<div id="msg"></div>
<div id="out" hidden>
  <p><span id="sum"></span></p>
  <p>
    <a id="lOverlay" target="_blank">下载 _overlay.png</a> ｜
    <a id="lNotes" target="_blank">下载 notes JSON</a>
    <a id="lReview" target="_blank" hidden> ｜ 下载 _review.json</a>
  </p>
  <img id="ovimg" alt="_overlay.png">
  <div id="fixbox" hidden>
    <h3>认不出的 <span id="fixN">0</span> 处 —— 勾选后可<b>批量</b>改 / 删</h3>
    <div id="fixList" class="fixlist"></div>
    <div class="row">
      <button type="button" id="selAll">全选</button>
      <button type="button" id="selNone">全不选</button>
      <label class="hint"><input type="radio" name="op" value="remove" checked> 删除</label>
      <label class="hint"><input type="radio" name="op" value="set"> 改为</label>
      <select id="fixLetter"></select>
      <button type="button" id="apply" class="primary">应用并重跑</button>
    </div>
    <p class="hint">「重跑」= 按修正后的数据重出 overlay 和 JSON（秒级，<b>不重新识别</b>）。
      删除会让那个音**连同它的时值**一起消失 —— 对"这个块根本不是音符"（升号 ♯、间奏文字…）正是想要的。</p>
  </div>
  <details class="fold">
    <summary>校验修改（数字 → 字母；认不出来的已红底标出）</summary>
    <div id="chips"></div>
  </details>
  <details class="fold">
    <summary>JSON</summary>
    <pre id="json"></pre>
  </details>
</div>
<script>
const $ = s => document.querySelector(s);

/* 拖拽：把文件塞进 <input type=file> 再统一走 `run()`，两条入口一份逻辑 */
const drop = $("#drop");
$("#f").onchange = () => showPicked();
drop.onclick = () => $("#f").click();
drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", e => {
  e.preventDefault(); drop.classList.remove("over");
  take(e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0], true);
});
/* 拖到页面别处也别让浏览器直接打开图片 */
["dragover", "drop"].forEach(t => document.addEventListener(t, e => e.preventDefault()));

function showPicked(){
  const f = $("#f").files[0];
  $("#dropmsg").innerHTML = f ? ("已选：<b>" + f.name + "</b>") : "把简谱图片<b>拖到这里</b>，或点这里选文件";
}
function take(file, autoRun){
  if(!file) return;
  if(!/\\.(png|jpe?g|bmp|webp)$/i.test(file.name)){ $("#msg").className = "err";
    $("#msg").textContent = "只认图片：.png / .jpg / .jpeg / .bmp / .webp"; return; }
  const dt = new DataTransfer(); dt.items.add(file); $("#f").files = dt.files;
  showPicked();
  if(autoRun) run();
}

let LAST = null;                       // 最近一次结果；/apply 要拿它的 id 与 noteId
const KEYS = [{l:"Z",o:-1,n:1},{l:"X",o:-1,n:2},{l:"C",o:-1,n:3},{l:"V",o:-1,n:4},
              {l:"B",o:-1,n:5},{l:"N",o:-1,n:6},{l:"M",o:-1,n:7},
              {l:"S",o:0,n:1},{l:"D",o:0,n:2},{l:"F",o:0,n:3},{l:"G",o:0,n:4},
              {l:"H",o:0,n:5},{l:"J",o:0,n:6},{l:"K",o:0,n:7},
              {l:"E",o:1,n:1},{l:"R",o:1,n:2},{l:"T",o:1,n:3},{l:"Y",o:1,n:4},
              {l:"U",o:1,n:5},{l:"I",o:1,n:6},{l:"O",o:1,n:7}];
const BAND = {"-1":"低","0":"中","1":"高"};
$("#fixLetter").innerHTML = KEYS.map(k =>
    '<option value="' + k.n + ',' + k.o + '">' + k.l + '（' + BAND[String(k.o)] + k.n + '）</option>')
  .join("") + '<option value="0,0">0（休止）</option>';

function chip(x){
  const cls = x.unknown ? "chip unk" : (x.n === 0 ? "chip rest" : "chip");
  const dig = (x.n === null ? "?" : x.n);
  let let_ = (x.letter === null ? "Q" : x.letter);
  if(x.shift === "LSHIFT") let_ += "⇧L"; else if(x.shift === "RSHIFT") let_ += "⇧R";
  return '<span class="' + cls + '">' + dig + ' → <b>' + let_ + '</b></span>';
}

function render(j){
  LAST = j;
  $("#out").hidden = false;
  $("#sum").textContent = j.title + " ｜ " + j.key + " " + j.meter + " ｜ 共 "
    + j.counts.total + " 音（字母 " + j.counts.letters + " / 休止 " + j.counts.rests
    + " / 认不出 " + j.counts.unknown + "）";
  const t = "?t=" + Date.now();
  $("#lOverlay").href = j.files.overlay + t; $("#ovimg").src = j.files.overlay + t;
  $("#lNotes").href = j.files.notes + t;
  $("#lReview").hidden = !j.files.review;
  if(j.files.review) $("#lReview").href = j.files.review;

  const un = [];
  for(const L of j.lines) for(const x of L.notes) if(x.unknown) un.push(x);
  $("#fixbox").hidden = (un.length === 0);
  $("#fixN").textContent = un.length;
  $("#fixList").innerHTML = un.map(x =>
      '<label><input type="checkbox" value="' + x.id + '"> ' + x.id + '</label>').join("");

  $("#chips").innerHTML = j.lines.map(L =>
      '<div class="linechips"><span class="ln">行' + L.line + '</span>'
      + L.notes.map(chip).join("") + '</div>').join("");
  $("#json").textContent = JSON.stringify({title: j.title, key: j.key, meter: j.meter,
                                           counts: j.counts, lines: j.lines}, null, 1);
}

async function run(){
  const file = $("#f").files[0];
  if(!file){ $("#msg").className = ""; $("#msg").textContent = "先选一张图片（或直接拖进来）"; return; }
  const fd = new FormData();
  fd.append("image", file);
  for(const k of ["key","title","meter"]){ const v = $("#"+k).value.trim(); if(v) fd.append(k, v); }
  $("#go").disabled = true; $("#msg").className = ""; $("#msg").textContent = "识别中…（单张约 10~30 秒）";
  $("#out").hidden = true;
  try{
    const j = await (await fetch("/convert", {method:"POST", body: fd})).json();
    if(!j.ok) throw new Error(j.error || "识别失败");
    $("#msg").textContent = "识别完成";
    render(j);
  }catch(e){
    $("#msg").className = "err"; $("#msg").textContent = String(e.message || e);
  }finally{ $("#go").disabled = false; }
}
$("#go").onclick = run;

$("#selAll").onclick = () => document.querySelectorAll("#fixList input").forEach(c => c.checked = true);
$("#selNone").onclick = () => document.querySelectorAll("#fixList input").forEach(c => c.checked = false);
$("#apply").onclick = async () => {
  if(!LAST) return;
  const ids = Array.from(document.querySelectorAll("#fixList input:checked")).map(c => c.value);
  if(!ids.length){ $("#msg").className = ""; $("#msg").textContent = "先勾选要改/删的（可多选）"; return; }
  const op = document.querySelector('input[name="op"]:checked').value;
  const lv = $("#fixLetter").value.split(",");
  const edits = ids.map(id => op === "remove" ? {noteId: id, op: "remove"}
      : {noteId: id, op: "set", n: +lv[0], o: +lv[1]});
  $("#apply").disabled = true; $("#msg").className = ""; $("#msg").textContent = "重出中…";
  try{
    const j = await (await fetch("/apply", {method:"POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({id: LAST.id, edits})})).json();
    if(!j.ok) throw new Error(j.error || "应用失败");
    const a = j.applied || {};
    $("#msg").textContent = "已应用：" + (a.set || 0) + " 处改、" + (a.removed || 0) + " 处删"
      + ((a.missing || []).length ? "，" + a.missing.length + " 处没找到（已忽略）" : "");
    render(j);
  }catch(e){
    $("#msg").className = "err"; $("#msg").textContent = String(e.message || e);
  }finally{ $("#apply").disabled = false; }
};
</script>"""


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html; charset=utf-8")


@app.get("/healthz")
def healthz():
    return jsonify({"ok": True})


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):          # Windows 控制台 GBK 兜底（同 cli.py）
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="pianoWeb/server.py")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    os.makedirs(WORK_ROOT, exist_ok=True)
    print("简谱→字母谱服务： http://%s:%d/          （工作目录 %s）"
          % (args.host, args.port, WORK_ROOT), flush=True)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
