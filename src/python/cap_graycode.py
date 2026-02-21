import cv2
import numpy as np
import glob
import sys
from pathlib import Path
from typing import List

from edsdk.camera_controller import CameraController

from .config import get_config


def open_cam() -> None:
    pass


def close_cam() -> None:
    pass


def capture() -> np.ndarray:
    cam_cfg = get_config().camera
    with CameraController(register_property_events=False) as camera:
        camera.set_properties(
            av=cam_cfg.av,
            tv=cam_cfg.tv,
            iso=cam_cfg.iso,
            image_quality=cam_cfg.image_quality,
        )
        imgs = camera.capture_numpy()
        img = imgs[0]
    return img


def print_usage() -> None:
    print("Usage : python cap_graycode.py <window position x> <window position y>")
    print()


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv

    if len(argv) != 3:
        print_usage()
        return

    try:
        window_pos_x = int(argv[1])
        window_pos_y = int(argv[2])
    except ValueError:
        print("height, width は整数で指定してください。")
        print_usage()
        return
    cfg = get_config()
    target_dir = Path(cfg.paths.pattern_dir)
    capture_dir = Path(cfg.paths.captured_dir)
    wait_ms = cfg.camera.wait_key_ms

    cam_height = 0
    cam_width = 0
    graycode_imgs: List[np.ndarray] = []
    # グレイコードをファイルから参照
    for idx, fname in enumerate(sorted(glob.glob(str(target_dir / "pattern_*.png")))):
        print(f"Loading pattern image: {fname}")
        pat_img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        if cam_height == 0 and cam_width == 0:
            cam_height, cam_width = pat_img.shape
        graycode_imgs.append(pat_img)

    cv2.namedWindow("Pattern", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Pattern", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow("Pattern", window_pos_x, window_pos_y)

    open_cam()

    # キャプチャディレクトリ作成
    capture_dir.mkdir(parents=True, exist_ok=True)

    for i, pat in enumerate(graycode_imgs):
        print(f"Displaying pattern image {i:02d}...")
        cv2.imshow("Pattern", pat)
        cv2.waitKey(wait_ms)
        captured_img = capture()
        captured_img_gray = cv2.cvtColor(captured_img, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(f"{capture_dir}/capture_{i:02d}.png", captured_img_gray)
        print(f"Captured and saved image: capture_{i:02d}.png")

    cv2.destroyAllWindows()
    close_cam()

    print("All patterns have been captured and saved.")

    print()
    print("=== Next step ===")
    print(
        "Run 'python decode.py <projector image height> <projector image width>' to decode the captured images."
    )
    print()


if __name__ == "__main__":
    main()
