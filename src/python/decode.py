# coding: UTF-8

import sys
import glob
import re
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


TARGETDIR = Path("data/graycode_pattern")
CAPTUREDDIR = Path("data/captured")

BLACKTHR = 50
WHITETHR = 5


def print_usage() -> None:
    print(
        "Usage : python decode.py "
        "<projector image height> <projector image width> "
        "[graycode step(default=1)]"
    )
    print()


def numerical_sort(text: str, re_num: re.Pattern[str]) -> int:
    """キャプチャファイル名中の連番でソートするためのキーを返す。"""
    return int(re_num.split(text)[-2])


def load_images(pattern: str) -> List[np.ndarray]:
    """指定パターンの画像をグレースケールで読み込む。"""
    filenames = sorted(glob.glob(pattern))
    return [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in filenames]


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv

    if len(argv) not in (3, 4):
        print_usage()
        return

    try:
        proj_height = int(argv[1])
        proj_width = int(argv[2])
        step = int(argv[3]) if len(argv) == 4 else 1
    except ValueError:
        print("height, width, step は整数で指定してください。")
        print_usage()
        return

    gc_width = ((proj_width - 1) // step) + 1
    gc_height = ((proj_height - 1) // step) + 1

    graycode = cv2.structured_light.GrayCodePattern.create(gc_width, gc_height)
    graycode.setBlackThreshold(BLACKTHR)
    graycode.setWhiteThreshold(WHITETHR)

    re_num = re.compile(r"(\d+)")

    filenames = sorted(
        glob.glob(str(CAPTUREDDIR / "capture_*.png")),
        key=lambda t: numerical_sort(t, re_num),
    )

    expected_num = graycode.getNumberOfPatternImages() + 2
    if len(filenames) != expected_num:
        print(f"Number of images is not right (right number is {expected_num})")
        return

    imgs: List[np.ndarray] = [
        cv2.imread(fname, cv2.IMREAD_GRAYSCALE) for fname in filenames
    ]
    black = imgs.pop()
    white = imgs.pop()
    cam_height, cam_width = white.shape[:2]
    print("camera image size :", white.shape)
    print()

    viz_c2p = np.zeros((cam_height, cam_width, 3), np.uint8)

    # 差分画像をNumPyで一括計算して「候補ピクセル」のマスクを作成
    diff = white.astype(np.int16) - black.astype(np.int16)
    valid_mask = diff > BLACKTHR

    # 有効画素のインデックスをまとめて取得
    ys, xs = np.where(valid_mask)

    # valid maskを保存
    cv2.imwrite("valid_mask.png", (valid_mask * 255).astype(np.uint8))

    c2p_list: List[Tuple[Tuple[int, int], Tuple[float, float]]] = []

    # OpenCV の GrayCodePattern は現状ピクセル単位でしか問い合わせられないため
    # getProjPixel 呼び出し自体はループになるが、Python 側での条件分岐等は最小限にする
    for x, y in zip(xs, ys):
        err, proj_pix = graycode.getProjPixel(imgs, x, y)
        if not err:
            # プロジェクタ座標をステップサイズ分拡大して中心にオフセットをかける
            # これにより，得られたプロジェクタ座標（ブロック）の中心を指すようになる
            fixed_pix = (
                step * (proj_pix[0] + 0.5),
                step * (proj_pix[1] + 0.5),
            )

            viz_c2p[y, x, :] = [
                fixed_pix[0] % (np.iinfo(viz_c2p.dtype).max + 1),
                fixed_pix[1] % (np.iinfo(viz_c2p.dtype).max + 1),
                np.iinfo(viz_c2p.dtype).max // 2,
            ]
            c2p_list.append(((x, y), fixed_pix))

    print("=== Result ===")
    print(f"Decoded c2p correspondences : {len(c2p_list)}")
    cv2.imwrite("visualize_c2p.png", viz_c2p)
    print("Visualized image : './visualize_c2p.png'")
    with open("result_c2p.csv", "w", encoding="utf-8") as f:
        f.write("cam_x, cam_y, proj_x, proj_y\n")
        for (cam_x, cam_y), (proj_x, proj_y) in c2p_list:
            f.write(f"{cam_x}, {cam_y}, {proj_x}, {proj_y}\n")

    # NumPy形式でも保存
    np.save("result_c2p.npy", np.array(c2p_list, dtype=object))

    print("NumPy array : './result_c2p.npy'")
    print("output : './result_c2p.csv'")
    print()


if __name__ == "__main__":
    main()
