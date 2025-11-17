import os
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
from scipy.interpolate import Rbf


def load_c2p_numpy(
    filename: str,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """NumPy形式で保存されたカメラ座標とプロジェクタ座標の対応表を読み込む。"""
    c2p_array = np.load(filename)  # shape: (N, 4)
    c2p_list: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for row in c2p_array:
        cam_x, cam_y, proj_x, proj_y = row
        c2p_list.append(((cam_x, cam_y), (proj_x, proj_y)))
    return c2p_list


def interpolate_c2p(
    cam_height: int,
    cam_width: int,
    c2p_list: List[Tuple[Tuple[float, float], Tuple[float, float]]],
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    known_cam_x = np.array([cam[0] for cam, _ in c2p_list], dtype=np.float32)
    known_cam_y = np.array([cam[1] for cam, _ in c2p_list], dtype=np.float32)
    known_proj_x = np.array([proj[0] for _, proj in c2p_list], dtype=np.float32)
    known_proj_y = np.array([proj[1] for _, proj in c2p_list], dtype=np.float32)

    if (max(known_cam_x) > cam_width - 1) or (max(known_cam_y) > cam_height - 1):
        print("Warning: Some known camera coordinates are out of bounds.")

    # call RBF interpolation
    rbf_proj_x = Rbf(
        known_cam_x, known_cam_y, known_proj_x, function="thin_plate", smooth=0
    )
    rbf_proj_y = Rbf(
        known_cam_x, known_cam_y, known_proj_y, function="thin_plate", smooth=0
    )

    # Create a grid of camera coordinates
    grid_cam_x, grid_cam_y = np.meshgrid(
        np.arange(cam_width, dtype=np.float32),
        np.arange(cam_height, dtype=np.float32),
        indexing="xy",
    )
    grid_proj_x = rbf_proj_x(grid_cam_x.ravel(), grid_cam_y.ravel()).reshape(
        cam_height, cam_width
    )
    grid_proj_y = rbf_proj_y(grid_cam_x.ravel(), grid_cam_y.ravel()).reshape(
        cam_height, cam_width
    )
    grid_proj_x = grid_proj_x.astype(np.float32)
    grid_proj_y = grid_proj_y.astype(np.float32)

    ret_c2p_list: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    # Collect interpolated correspondences
    cam_coords = np.stack([grid_cam_x, grid_cam_y], axis=-1).reshape(-1, 2)
    proj_coords = np.stack([grid_proj_x, grid_proj_y], axis=-1).reshape(-1, 2)

    c2p_array = np.concatenate([cam_coords, proj_coords], axis=1)  # shape: (N, 4)

    ret_c2p_list = [
        ((float(cx), float(cy)), (float(px), float(py))) for cx, cy, px, py in c2p_array
    ]

    return ret_c2p_list


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv

    if len(argv) != 4:
        print(
            "Usage : python compensate_c2p.py <c2p_numpy_filename> <cam_height> <cam_width>"
        )
        print()
        return

    try:
        cam_height = int(argv[2])
        cam_width = int(argv[3])
        c2p_numpy_filename = str(argv[1])
    except ValueError:
        print("cam_height, cam_width は整数で指定してください。")
        print(
            "Usage : python compensate_c2p.py <c2p_numpy_filename> <cam_height> <cam_width>"
        )
        print()
        return
    c2p_list = load_c2p_numpy(c2p_numpy_filename)
    print(
        f"Loaded {len(c2p_list)} camera-to-projector correspondences from '{c2p_numpy_filename}'"
    )
    c2p_list_interp = interpolate_c2p(cam_height, cam_width, c2p_list)

    c2p_array_interp = np.array(
        [
            [cam_x, cam_y, proj_x, proj_y]
            for (cam_x, cam_y), (proj_x, proj_y) in c2p_list_interp
        ],
        dtype=np.float32,
    )

    out_filename = os.path.splitext(c2p_numpy_filename)[0] + "_compensated.npy"
    np.save(out_filename, c2p_array_interp)
    print(f"Saved compensated correspondences to '{out_filename}'")

    with open("result_c2p_compensated.csv", "w", encoding="utf-8") as f:
        f.write("cam_x, cam_y, proj_x, proj_y\n")
        for (cam_x, cam_y), (proj_x, proj_y) in c2p_list_interp:
            f.write(f"{cam_x}, {cam_y}, {proj_x}, {proj_y}\n")
    print("output : './result_c2p_compensated.csv'")
    print()

    if __name__ == "__main__":
        main()
