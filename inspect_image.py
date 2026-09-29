# -*- coding: utf-8 -*-
"""
诊断脚本：检测图片真实格式 + PaddleOCR 文本识别（纯文本输出，便于贴回对话）。

兼容 PaddleOCR 2.x 与 3.x：
  - 2.x: PaddleOCR(use_angle_cls=True) + ocr(path, cls=True)
  - 3.x: PaddleOCR(use_textline_orientation=True) + predict(path)

用法：
  python inspect_image.py 1.jpg
  python inspect_image.py 2.png
"""

import os
import sys


def _create_ocr(lang="ch"):
    import os
    # paddle 推理库（C++ 层）对中文路径兼容性差，把模型缓存改到纯英文路径，
    # 规避 C:\Users\陈立文\.paddlex 下读不到配置文件的报错。
    for _k in ("PADDLE_PDX_CACHE_HOME", "PADDLX_CACHE_HOME", "PADDLEX_CACHE_HOME"):
        os.environ.setdefault(_k, "D:\\paddlex_cache")
    # 跳过模型源连通性检测（HuggingFace/ModelScope 在国内常不通，直接走百度 BOS）
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    print(f"[info] 模型缓存目录 = {os.environ.get('PADDLE_PDX_CACHE_HOME')}")

    from paddleocr import PaddleOCR
    # 优先禁用文档预处理（方向分类/去扭曲），绕开 CPU 上 doc_ori 模型初始化失败，
    # 也让简谱直接走「检测 + 识别」。
    attempts = [
        dict(lang=lang,
             use_doc_orientation_classify=False,
             use_doc_unwarping=False,
             use_textline_orientation=False),
        dict(lang=lang),
    ]
    last_err = None
    for kw in attempts:
        try:
            return PaddleOCR(**kw)
        except (ValueError, TypeError) as e:
            last_err = e
    raise last_err


def _run_rapidocr(path):
    """RapidOCR（onnxruntime 跑 PaddleOCR 模型）——绕开 paddlepaddle/paddlex 的坑。

    未安装时返回 None，让调用方退回 paddleocr。
    """
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return None
    try:
        engine = RapidOCR()
        out = engine(path)
        # 3.x 返回 (result, elapse)；2.x 返回 result
        result = out[0] if isinstance(out, tuple) else out
        if not result:
            return []
        tokens = []
        for item in result:
            if not (isinstance(item, (list, tuple)) and len(item) >= 2):
                continue
            box, text = item[0], item[1]
            score = item[2] if len(item) >= 3 else 1.0
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            tokens.append({
                "cx": (min(xs) + max(xs)) / 2.0,
                "cy": (min(ys) + max(ys)) / 2.0,
                "text": str(text),
                "conf": float(score),
            })
        return tokens
    except Exception as e:
        print(f"[RapidOCR 异常] {e}")
        return None


def _run_ocr(ocr, path):
    if hasattr(ocr, "predict"):
        return ocr.predict(path)
    return ocr.ocr(path, cls=True)


def _normalize_ocr_result(raw):
    """兼容 PaddleOCR 2.x / 3.x 返回结构，输出 [{cx, cy, text, conf}]。"""
    tokens = []

    def add(box, text, conf):
        if not text:
            return
        if box is not None and len(box) > 0:
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            cx = (min(xs) + max(xs)) / 2.0
            cy = (min(ys) + max(ys)) / 2.0
        else:
            cx = cy = 0.0
        tokens.append({"cx": cx, "cy": cy, "text": str(text), "conf": float(conf)})

    def handle_page(page):
        if isinstance(page, dict):
            texts = page.get("rec_texts") or page.get("texts") or []
            scores = page.get("rec_scores") or page.get("scores") or []
            boxes = (page.get("rec_polys") or page.get("rec_boxes")
                     or page.get("dt_polys") or page.get("boxes") or [])
            for i, text in enumerate(texts):
                box = boxes[i] if i < len(boxes) else None
                conf = scores[i] if i < len(scores) else 1.0
                add(box, text, conf)
            return
        if isinstance(page, (list, tuple)):
            for item in page:
                if isinstance(item, dict):
                    handle_page(item)
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    box, rest = item[0], item[1]
                    if isinstance(rest, (list, tuple)) and len(rest) >= 2:
                        add(box, rest[0], rest[1])
                    elif isinstance(rest, str):
                        add(box, rest, 1.0)

    if isinstance(raw, dict):
        handle_page(raw)
    elif isinstance(raw, (list, tuple)):
        for page in raw:
            handle_page(page)
    return tokens


def main(path):
    print("=" * 60)
    print(f"文件: {path}")

    try:
        size = os.path.getsize(path)
        print(f"大小: {size} 字节（{size/1024:.1f} KB）")
    except OSError as e:
        print(f"读取大小失败: {e}")
        return 1

    try:
        with open(path, "rb") as f:
            head = f.read(32)
        print(f"文件头 hex: {head.hex()}")
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            print("格式判断: PNG")
        elif head[:2] == b"\xff\xd8":
            print("格式判断: JPEG")
        elif head[:4] == b"RIFF" and head[8:12] == b"WEBP":
            print("格式判断: WEBP")
        elif head[:4] == b"GIF8":
            print("格式判断: GIF")
        elif head[4:8] == b"ftyp":
            print(f"格式判断: ISO-BMFF/HEIF（brand={head[8:12].decode(errors='ignore')}）")
        elif head[:2] == b"BM":
            print("格式判断: BMP")
        else:
            print("格式判断: 未知")
    except OSError as e:
        print(f"读取文件头失败: {e}")

    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print("OpenCV 读取: 失败")
        else:
            h, w = img.shape[:2]
            print(f"OpenCV 读取: 成功，宽 {w} x 高 {h} 像素")
    except ImportError:
        print("OpenCV: 未安装")
    except Exception as e:
        print(f"OpenCV 异常: {e}")

    try:
        from PIL import Image
        with Image.open(path) as im:
            print(f"PIL: 格式={im.format}，模式={im.mode}，尺寸={im.size} 像素")
    except ImportError:
        print("PIL: 未安装")
    except Exception as e:
        print(f"PIL 失败: {e}")

    print("\n=== OCR 识别结果（按 y 排序，空行表示 y 跳变 >35px）===")
    tokens = _run_rapidocr(path)
    if tokens is None:
        # RapidOCR 未安装/失败 → 退回 paddleocr
        try:
            from paddleocr import PaddleOCR  # noqa: F401
        except ImportError:
            print("OCR 未安装：请安装 rapidocr-onnxruntime（推荐）或 paddleocr")
            return 0
        try:
            ocr = _create_ocr("ch")
            raw = _run_ocr(ocr, path)
            tokens = _normalize_ocr_result(raw)
        except Exception:
            import traceback
            traceback.print_exc()
            return 0

    if not tokens:
        print("（未识别到任何文本）")
        return 0

    tokens.sort(key=lambda t: (t["cy"], t["cx"]))
    prev_cy = None
    for t in tokens:
        if prev_cy is not None and t["cy"] - prev_cy > 35:
            print()
        prev_cy = t["cy"]
        print(f"  y={t['cy']:7.1f}  x={t['cx']:7.1f}  conf={t['conf']:.2f}  text={t['text']!r}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python inspect_image.py <图片路径>")
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1]))
