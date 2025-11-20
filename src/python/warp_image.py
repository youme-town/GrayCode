import sys
import numpy as np
import cv2
from typing import List, Tuple


def warp_image(
    src_img: np.ndarray,
    map: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    dst_rect: Tuple[int, int, int, int] | None = None,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0),
) -> np.ndarray:
    # map の最後の要素から画像サイズを取得
    MAP_H = int(max(p[1] for p, _ in map)) + 1
    MAP_W = int(max(p[0] for p, _ in map)) + 1
    if abs(src_img.shape[0] - MAP_H) > 1.0:
        print(
            f"Warning: Source image height {src_img.shape[0]} "
            f"does not match map height {MAP_H}."
        )
    if abs(src_img.shape[1] - MAP_W) > 1.0:
        print(
            f"Warning: Source image width {src_img.shape[1]} "
            f"does not match map width {MAP_W}."
        )

    H, W = src_img.shape[:2]
    map_x = np.array([[p[0] for p, _ in map]], dtype=np.float32).reshape(H, W)
    map_y = np.array([[p[1] for p, _ in map]], dtype=np.float32).reshape(H, W)

    warped_img = cv2.remap(
        src_img,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=borderMode,
        borderValue=borderValue,
    )

    if dst_rect is not None:
        x, y, w, h = dst_rect
        warped_img = warped_img[y : y + h, x : x + w]

    return warped_img


def inverse_warp_image(
    src_img: np.ndarray,
    map: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    dst_rect: Tuple[int, int, int, int] | None = None,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0),
) -> np.ndarray:
    # mapの後ろのタプルの最大値から画像サイズを取得
    MAP_H = int(max(p[1] for _, p in map)) + 1
    MAP_W = int(max(p[0] for _, p in map)) + 1
    if abs(src_img.shape[0] - MAP_H) > 1.0:
        print(
            f"Warning: Source image height {src_img.shape[0]} "
            f"does not match map height {MAP_H}."
        )
    if abs(src_img.shape[1] - MAP_W) > 1.0:
        print(
            f"Warning: Source image width {src_img.shape[1]} "
            f"does not match map width {MAP_W}."
        )

    H, W = src_img.shape[:2]
    map_x = np.array([[p[0] for _, p in map]], dtype=np.float32).reshape(H, W)
    map_y = np.array([[p[1] for _, p in map]], dtype=np.float32).reshape(H, W)

    warped_img = cv2.remap(
        src_img,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=borderMode,
        borderValue=borderValue,
    )

    if dst_rect is not None:
        x, y, w, h = dst_rect
        warped_img = warped_img[y : y + h, x : x + w]

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
    warped_img = inverse_warp_image(
        src_img,
        map_list,
        dst_rect=(1920 // 2 - 500 // 2, 1080 // 2 - 500 // 2, 500, 500),
    )

    # 出力画像の保存
    cv2.imwrite(output_image_path, warped_img)


if __name__ == "__main__":
    main()
