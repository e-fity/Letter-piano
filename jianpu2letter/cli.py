# -*- coding: utf-8 -*-
"""命令行入口：把 MusicXML 或中文数字谱图片转换成映射2键盘字母谱。

用法示例：
  python convert.py song.musicxml -o out/
  python convert.py jianpu.png -o out/ --key Eb
  python -m jianpu2letter song.xml -o out/
"""

import argparse
import os
import sys

from . import musicxml_reader, solfege, letterscore


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="convert.py",
        description="把 MusicXML 或中文数字谱图片转换成映射2键盘字母谱（XML）与中文数字谱图片。",
    )
    p.add_argument("input", help="输入文件：.musicxml/.xml/.mxl 或图片 .png/.jpg/.jpeg")
    p.add_argument("-o", "--outdir", default=".", help="输出目录（默认当前目录）")
    p.add_argument("--title", default=None, help="覆盖曲名（图片输入建议提供）")
    p.add_argument("--key", default=None, dest="key",
                   help="覆盖调号，如 Eb / F# / C（MusicXML 从调号自动推断；图片从页眉「1=X」"
                        "自动读，读不到才退回 1=C；本参数优先级最高）")
    p.add_argument("--meter", default=None, dest="meter",
                   help="覆盖拍号，如 4/4 / 3/4 / 6/8（图片默认 4/4；MusicXML/网站 JSON 用其自带值）")
    p.add_argument("--tonic-octave", type=int, default=4,
                   help="中音1 所在八度（默认 4，即主音落在中央C八度 C4-B4）")
    p.add_argument("--part", type=int, default=0, help="MusicXML 声部索引（默认 0 = 第一声部）")
    p.add_argument("--measures-per-line", type=int, default=2,
                   help="字母谱每行小节数（默认 2）")
    p.add_argument("--render-measures-per-line", type=int, default=4,
                   help="简谱图片每行小节数（默认 4）")
    p.add_argument("--scan-only", action="store_true",
                   help="图片只执行第一阶段：生成审核 JSON 与叠加图，不生成 XML/成品")
    p.add_argument("--measure-mode", choices=("vline", "beats"), default="vline",
                   help="小节划分方式：vline=按识别到的小节线（默认）；beats=按拍数")
    p.add_argument("--no-render", action="store_true", help="不渲染中文数字谱图片")
    p.add_argument("--letter-png", action="store_true",
                   help="**额外**渲染键盘字母谱图片（<base>_letter.png）。默认不出："
                        "字母谱的数据在 _letter.xml 里，要在图上核对可用 _overlay.png"
                        "（原位盖字母、保留原谱上下文）")
    p.add_argument("--overlay-letters", action="store_true",
                   help="输出「原位替换图」——图片输入**默认就出**，此开关只为兼容旧命令保留")
    p.add_argument("--no-overlay", action="store_true",
                   help="不输出「原位替换图」（在原谱图片上把数字直接盖成字母，不重排）")
    p.add_argument("--ocr-debug-dir", default=None, help="OCR 中间结果输出目录（调优用）")
    p.add_argument("--traceback", action="store_true",
                   help="出错时打印完整调用栈（排查渲染/生成失败用）")
    return p.parse_args(argv)


def _detect_input_type(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        # 两种 JSON：本工具的审核 JSON，或字母琴网站的简谱 JSON
        try:
            import json as _json
            with open(path, "r", encoding="utf-8") as f:
                payload = _json.load(f)
        except Exception:  # noqa: BLE001
            return "review_json"
        if isinstance(payload, dict):
            if (payload.get("schema") or {}).get("name") == "jianpu-score-json" \
                    or "measures" in payload:
                return "site_json"
        return "review_json"
    if ext == ".txt":
        return "jianpu_text"
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
        return "image"
    with open(path, "rb") as f:
        head = f.read(8)
    if head[:2] == b"PK":  # .mxl 是 zip 容器
        return "mxl"
    if head.startswith(b"<?xml") or b"<score" in head or ext in (".xml", ".musicxml"):
        return "musicxml"
    return "image"


def _warn_exc(prefix, exc, show_traceback=False):
    """统一的异常提示：`[warn] 前缀：类型: 消息 @ 文件:行`。

    历史教训：这里原来只打印 `{e}`，于是

        NameError: name '_draw_parens_at' is not defined
        SyntaxError: '(' was never closed

    都只表现为一句含糊的 `[warn] xxx 渲染失败：...`，被当成"就是没生成图"，
    结果"三样产物只出来两样"藏了很久。所以现在**必须带上异常类型和出错位置**。

    SyntaxError 自己的 `.filename/.lineno` 指向真正出错的文件，优先用它；
    其余异常取调用栈最后一帧（也就是真正抛出的那一行）。
    """
    import traceback
    where = ""
    filename = getattr(exc, "filename", None)
    lineno = getattr(exc, "lineno", None)
    if filename and lineno:
        where = f" @ {os.path.basename(filename)}:{lineno}"
    else:
        frames = traceback.extract_tb(exc.__traceback__)
        if frames:
            where = f" @ {os.path.basename(frames[-1].filename)}:{frames[-1].lineno}"
    print(f"[warn] {prefix}：{type(exc).__name__}: {exc}{where}")
    if show_traceback:
        traceback.print_exc()


def _report_low_conf(measures):
    """汇总置信度偏低的音（渲染时会给它们加黄底）。

    单独打成清单，是因为"图有没有出、黄底看不看得清"都不一定——
    至少控制台要能直接看到是哪几个音可疑。
    非 OCR 路径的音没有 conf 字段，这个函数自然什么都不打印。
    """
    try:
        from . import jianpu_render
    except ImportError:
        return
    rows = jianpu_render.low_conf_notes(measures)
    if not rows:
        return
    print(f"[置信度] {len(rows)} 个音低于 {jianpu_render.CONF_WARN:.0%}"
          f"（可疑；渲染时会加黄底）：")
    for row_no, note_no, n, conf in rows:
        digit = "?" if n is None else str(n)
        conf_txt = "无分数" if conf is None else f"{conf:.2f}"
        print(f"  行{row_no} 音{note_no} 数字={digit} 置信度={conf_txt}")


def _dump_char_text(measures, outdir, base, title="", measures_per_line=4,
                    show_traceback=False):
    """把识别结果按原谱行输出成「字符序列」文本：控制台逐行打印 + 落一份 .txt。

    每行形如 `L01: ( 0 1 2 5 1 2 6· | 0 1 2 6 5 3 2 5 | ...`，用的是 textscore
    的语法（`^`高音、`v`低音、`_`减时线、`·`附点、`-`延时、`|`/`||`/`:||`/`||:` 小节线），
    所以这份文本**可以再喂回 convert.py**。用途：不用看渲染图，就能逐行对原谱核对
    识别结果（未识别的音显示为 `?`）。

    分行方式与渲染图一致：有 row 用 row（OCR / 文本谱），否则每 measures_per_line
    个小节一行（MusicXML / 网站 JSON）。
    """
    from . import textscore
    try:
        text = textscore.to_text(measures, measures_per_line=measures_per_line)
    except Exception as e:  # noqa: BLE001
        _warn_exc("逐行字符序列生成失败", e, show_traceback)
        return None
    if not text:
        return None
    print(f"[字符序列] {title or base}（每行 = 原谱一行）")
    for row in text.splitlines():
        print(f"  {row}")
    path = os.path.join(outdir, base + "_jianpu.txt")
    with open(path, "w", encoding="utf-8") as file:
        file.write(text + "\n")
    print(f"[ok] 逐行字符序列：{path}")
    return path


def main(argv=None):
    # 控制台编码兜底：Windows 控制台默认 GBK，`print()` 里只要出现一个 GBK 编不出的
    # 字符（例如箭头 `↳` U+21B3），**整个程序会抛 UnicodeEncodeError 崩在半路** ——
    # 2026-09-18 就踩了这个坑：只加了一行带箭头的诊断，结果 OCR 跑一半就死，
    # 日志里只剩 Traceback、什么诊断都没有。这里把 errors 设成 replace：
    # 编不出的字符显示成 `?`，但绝不因此崩。
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    args = _parse_args(argv if argv is not None else sys.argv[1:])
    os.makedirs(args.outdir, exist_ok=True)

    in_path = args.input
    base = os.path.splitext(os.path.basename(in_path))[0]
    input_type = _detect_input_type(in_path)

    meter = args.meter or "4/4"
    title = args.title or base
    if args.key:
        key_mark = args.key if str(args.key).startswith("1=") else f"1={args.key}"
    else:
        key_mark = "1=C"

    if input_type == "site_json":
        if args.scan_only:
            raise ValueError("--scan-only 仅用于图片输入")
        from . import jianpu_json
        metadata, measures = jianpu_json.parse(in_path)
        title = args.title or metadata["title"] or base
        key_mark = key_mark if args.key else metadata["key"]
        meter = metadata["meter"]
    elif input_type == "jianpu_text":
        if args.scan_only:
            raise ValueError("--scan-only 仅用于图片输入")
        from . import textscore
        metadata, measures = textscore.load(
            in_path, key_mark=key_mark, meter=meter, title=title)
        title = metadata["title"]
        key_mark = metadata["key"]
    elif input_type == "review_json":
        if args.scan_only:
            raise ValueError("审核 JSON 已是第一阶段产物，不能再使用 --scan-only")
        from . import review_io
        metadata, measures = review_io.load_review(in_path)
        if base.endswith("_review"):
            base = base[:-7]
        title = args.title or metadata["title"]
        if args.key:
            key_mark = args.key if str(args.key).startswith("1=") else f"1={args.key}"
        else:
            key_mark = metadata["key"]
        meter = metadata["meter"]
    elif input_type == "image":
        from . import ocr_jianpu
        debug_dir = args.ocr_debug_dir or (args.outdir if args.scan_only else None)
        # 调号 + 曲名都已知（`--key`/`--title`，run_all 从 piano_config.json 传）时跳过页眉 OCR：
        # 读出来也不会用 —— 下面两个分支都带 `if not args.key` / `if not args.title`，
        # 而 `_LAST_KEY_MARK`/`_LAST_TITLE` 也不落进 review JSON。实测省 4.4~9.8s/张。
        measures = ocr_jianpu.ocr_to_stream(
            in_path, debug_dir=debug_dir, measure_mode=args.measure_mode,
            header_ocr=not (args.key and args.title))
        # 页眉若读到调号就当作默认（`--key` 优先级更高；读不到保持 1=C 并提示手填）
        if not args.key:
            if ocr_jianpu._LAST_KEY_MARK:
                key_mark = ocr_jianpu._LAST_KEY_MARK
                print(f"[key] 页眉自动读到调号 {key_mark}（可用 --key 覆盖）")
            else:
                print("[key] 没从页眉读到调号，暂用 1=C —— 请在 piano_config.json 里填 \"key\"，或用 --key 指定")
        # 页眉若读到曲名就当作默认（`--title` 优先级更高；读不到回退文件名并提示手填）
        if not args.title:
            if ocr_jianpu._LAST_TITLE:
                title = ocr_jianpu._LAST_TITLE
                print(f"[title] 自动读到标题: {title}")
            else:
                print("[title] 没从页眉读到标题，暂用文件名 —— 请在 piano_config.json 里填 \"title\"，或用 --title 指定")
        if not args.no_overlay:
            # 紧挨着 OCR 调用：`overlay_letters` 用的底图是刚识别时留下的那张缩放图。
            from . import overlay_letters
            overlay_path = os.path.join(args.outdir, base + "_overlay.png")
            drawn, rest, unknown, ok = overlay_letters.render(measures, overlay_path)
            if ok:
                print(f"[ok] 原位替换图：{overlay_path}"
                      f"（盖了 {drawn} 个数字；休止 {rest} 个保持原样；"
                      f"没认出来/疑似误判 {unknown} 处已画红框）")
            else:
                print(f"[warn] 原位替换图写入失败：{overlay_path}")
        # 审核 JSON：**默认就出**（用户 2026-09-22 要求）——「发现某个音认错了、改一处再重跑」
        # 全靠它；而它必须由 OCR 那一趟生成（每个音的位置/时值/减时线/连音弧等），
        # 事后无法从 png/txt 反推（`.txt` 含 `?` 时还喂不回去）。
        from . import review_io
        review_path = os.path.join(args.outdir, base + "_review.json")
        review_io.save_review(
            review_path, measures, title=title, key_mark=key_mark,
            meter=meter, source_image=in_path)
        print(f"[ok] 审核 JSON：{review_path}")
        # 图片数字已经是首调唱名；调号由 --key 提供，默认 1=C。
        if args.scan_only:
            overlay_path = os.path.join(debug_dir, "ocr_overlay.png")
            if os.path.exists(overlay_path):
                print(f"[scan] 叠加图：{overlay_path}")
            _dump_char_text(measures, args.outdir, base, title=title,
                            show_traceback=args.traceback)
            _report_low_conf(measures)
            print("[next] 对照叠加图修改 JSON 后，再将 JSON 作为 convert.py 输入。")
            return 0
    elif input_type == "mxl":
        raise NotImplementedError(
            ".mxl 压缩格式暂不支持，请先用 MuseScore 等导出为未压缩的 .musicxml/.xml")
    else:
        if args.scan_only:
            raise ValueError("--scan-only 仅用于图片输入")
        score = musicxml_reader.parse(in_path, part_index=args.part)
        measures = solfege.build_stream(score, tonic_octave=args.tonic_octave)
        if args.key:
            key_mark = args.key if str(args.key).startswith("1=") else f"1={args.key}"
        else:
            key_mark = solfege.key_mark_from_fifths(score["key_fifths"])
        title = args.title or score["title"] or base
        meter = f"{score['beats']}/{score['beat_type']}"

    if args.meter:
        # `--meter` 覆盖（与 `--key` 同一套逻辑）：MusicXML / 网站 JSON 自带拍号，
        # 但用户可以强制指定（图片输入本来就只有这一个来源）。
        meter = args.meter

    # 逐行「字符序列」文本：控制台打印 + 落一份 <base>_jianpu.txt（可再喂回本工具）
    _dump_char_text(measures, args.outdir, base, title=title,
                    measures_per_line=args.render_measures_per_line,
                    show_traceback=args.traceback)
    # 可疑音清单（渲染图里同样会加黄底）
    _report_low_conf(measures)

    lines = letterscore.measures_to_lines(measures, measures_per_line=args.measures_per_line)
    xml_str = letterscore.to_xml(title, key_mark, meter, lines, mapping="mapping2")

    xml_path = os.path.join(args.outdir, base + "_letter.xml")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    render_paths = []
    render_failures = []
    if not args.no_render:
        try:
            from . import jianpu_render
            jianpu_path = os.path.join(args.outdir, base + "_jianpu.png")
            jianpu_render.render(measures, jianpu_path, title=title, key_mark=key_mark,
                                 meter=meter, measures_per_line=args.render_measures_per_line)
            render_paths.append(("简谱图片", jianpu_path))
        except ImportError:
            print("[warn] 未安装 matplotlib，跳过图片渲染（XML 已生成）")
        except Exception as e:  # noqa: BLE001
            render_failures.append("简谱图片")
            _warn_exc("简谱图片渲染失败（这是真出错，不是「未渲染」）", e, args.traceback)
        if args.letter_png:      # 默认不出（用户 2026-09-22 要求）；其余产物不受影响
            try:
                from . import letter_render
                letter_path = os.path.join(args.outdir, base + "_letter.png")
                letter_render.render(measures, letter_path, title=title, key_mark=key_mark,
                                     meter=meter,
                                     measures_per_line=args.render_measures_per_line)
                render_paths.append(("字母谱图片", letter_path))
            except ImportError:
                # matplotlib 缺失时上面那段已经报过一次，这里不重复报同一原因
                pass
            except Exception as e:  # noqa: BLE001
                render_failures.append("字母谱图片")
                _warn_exc("字母谱图片渲染失败（这是真出错，不是「未渲染」）", e, args.traceback)

    print(f"[ok] 字母谱 XML：{xml_path}")
    if render_paths:
        for label, path in render_paths:
            print(f"[ok] {label}：{path}")
    elif args.no_render:
        print("[skip] 未渲染图片（--no-render）")
    if render_failures:
        print(f"[warn] 本次有 {len(render_failures)} 处图片渲染失败："
              + "、".join(render_failures)
              + "（XML 不受影响；加 --traceback 可看完整调用栈）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
