# -*- coding: utf-8 -*-
"""把 slides/ 下的 12 页幻灯片按页序合成一个 PDF。

用法：
    python3 make_pdf.py
输出：
    AllGather选算法_10分钟分享.pdf
"""
import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SLIDES = os.path.join(HERE, "slides")
OUT = os.path.join(HERE, "AllGather选算法_10分钟分享.pdf")

# 按页序显式列出，不能用 sorted()：字符串排序会把 S10/S11/S12 排到 S1 前面
ORDER = [
    "S1_封面", "S2_全景", "S3_管线", "S4_代价函数", "S6_util阶梯",
    "S7_方法", "S8_corner_case", "S9_不是孤例", "S10_跳变带",
]

# 附录：完整图表集里挑出的补充页。APPENDIX_POS 是插入位置（0-based），2 = 放在 S2 之后、S3 之前
APPENDIX = ["15_算法全景矩阵"]  # 老/新路径算法全景（由 02 的拓扑覆盖 + 14 的 35 个名字合并）
APPENDIX_POS = 2

FONT_PATH = "/System/Library/Fonts/Hiragino Sans GB.ttc"
FOOTER_COLOR = "#9AA3AE"


def stamp_footer(page, page_no, total):
    """给附录页补上与幻灯片一致的页脚，避免页码跳号。"""
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(page)
    try:
        font = ImageFont.truetype(FONT_PATH, 21)
    except OSError:
        return
    draw.text((round(page.width * 0.945), round(page.height * 0.968)),
              f"AllGather Selector 研讨  ·  {page_no}/{total}",
              font=font, fill=FOOTER_COLOR, anchor="rs")


def normalize(path, page_size):
    """把非 16:9 的图按页宽缩放，居中贴到幻灯片同尺寸的白底上，保证 PDF 页尺寸统一。"""
    page_w, page_h = page_size
    im = Image.open(path).convert("RGB")
    scale = page_w / im.width
    new_w, new_h = page_w, round(im.height * scale)
    if new_h > page_h:  # 太高就改按页高缩放
        scale = page_h / im.height
        new_w, new_h = round(im.width * scale), page_h
    im = im.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (page_w, page_h), "white")
    canvas.paste(im, ((page_w - new_w) // 2, (page_h - new_h) // 2))
    return canvas


def main():
    paths = [os.path.join(SLIDES, name + ".png") for name in ORDER]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("缺少幻灯片，请先运行 make_slides.py：\n  " + "\n  ".join(missing))

    pages = [Image.open(p).convert("RGB") for p in paths]
    sizes = {im.size for im in pages}
    if len(sizes) != 1:
        print("警告：页面尺寸不一致", sizes)
    page_size = pages[0].size
    total = len(ORDER) + len(APPENDIX)

    for i, name in enumerate(APPENDIX):
        path = os.path.join(HERE, name + ".png")
        if not os.path.exists(path):
            print(f"警告：缺少附录图 {path}，已跳过")
            continue
        pos = APPENDIX_POS + i
        pg = normalize(path, page_size)
        stamp_footer(pg, pos + 1, total)
        pages.insert(pos, pg)

    pages[0].save(OUT, save_all=True, append_images=pages[1:], resolution=150.0, quality=95)
    print(f"saved {OUT}  pages={len(pages)}  size={os.path.getsize(OUT) / 1024 / 1024:.2f}MB")


if __name__ == "__main__":
    main()
