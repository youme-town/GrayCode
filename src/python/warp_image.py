import sys
import numpy as np
import cv2
from typing import List, Tuple


def warp_image(
    src_img: np.ndarray, map: List[Tuple[Tuple[float, float], Tuple[float, float]]]
) -> np.ndarray:
    H, W = src_img.shape[:2]
    map_x = np.array([[p[0] for p, _ in map]], dtype=np.float32).reshape(H, W)
    map_y = np.array([[p[1] for p, _ in map]], dtype=np.float32).reshape(H, W)

    warped_img = cv2.remap(
        src_img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return warped_img


def print_usage() -> None:
    print("Usage : python warp_image.py <input image> <output image> <map file>")
    print()


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv

    if len(argv) != 4:
        print_usage()
        return

    input_image_path = argv[1]
    output_image_path = argv[2]
    map_file_path = argv[3]

    # 入力画像の読み込み
    src_img = cv2.imread(input_image_path)

    # マップデータの読み込み
    map_data = np.load(map_file_path, allow_pickle=True)
    map_list: List[Tuple[Tuple[float, float], Tuple[float, float]]] = map_data.tolist()

    # 画像のワープ
    warped_img = warp_image(src_img, map_list)

    # 出力画像の保存
    cv2.imwrite(output_image_path, warped_img)


if __name__ == "__main__":
    main()
