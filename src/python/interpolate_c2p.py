# coding: utf-8

import os
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
from scipy.interpolate import Rbf, griddata
import cv2


def load_c2p_numpy(
    map_file_path: str,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    map_data = np.load(map_file_path, allow_pickle=True)
    map_list_raw = map_data.tolist()

    if not isinstance(map_list_raw, list):
        raise TypeError("map_list must be a list")

    for i, item in enumerate(map_list_raw[:10]):  # 先頭だけチェック
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not isinstance(item[0], (list, tuple))
            or len(item[0]) != 2
            or not isinstance(item[1], (list, tuple))
            or len(item[1]) != 2
        ):
            raise TypeError(f"map_list[{i}] must be [[x,y],[u,v]]")

    # [[x,y],[u,v]] → ((x,y),(u,v)) に変換
    map_list: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [
        ((float(item[0][0]), float(item[0][1])), (float(item[1][0]), float(item[1][1])))
        for item in map_list_raw
    ]

    return map_list


def interpolate_c2p_list(
    cam_height: int,
    cam_width: int,
    c2p_list: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    interp_method: str = "cubic",
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    known_cam_x = np.array([cam[0] for cam, _ in c2p_list], dtype=np.float32)
    known_cam_y = np.array([cam[1] for cam, _ in c2p_list], dtype=np.float32)
    known_proj_x = np.array([proj[0] for _, proj in c2p_list], dtype=np.float32)
    known_proj_y = np.array([proj[1] for _, proj in c2p_list], dtype=np.float32)

    if (max(known_cam_x) > cam_width - 1) or (max(known_cam_y) > cam_height - 1):
        print("Warning: Some known camera coordinates are out of bounds.")

    # Create a grid of camera coordinates
    grid_cam_x, grid_cam_y = np.meshgrid(
        np.arange(cam_width, dtype=np.float32),
        np.arange(cam_height, dtype=np.float32),
        indexing="xy",
    )

    grid_proj_x = griddata(
        points=(known_cam_x, known_cam_y),
        values=known_proj_x,
        xi=(grid_cam_x, grid_cam_y),
        method=interp_method,
    )
    grid_proj_y = griddata(
        points=(known_cam_x, known_cam_y),
        values=known_proj_y,
        xi=(grid_cam_x, grid_cam_y),
        method=interp_method,
    )

    grid_proj_x = grid_proj_x.astype(np.float32)
    grid_proj_y = grid_proj_y.astype(np.float32)

    ret_c2p_list: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    # Collect interpolated correspondences
    cam_coords = np.stack([grid_cam_x, grid_cam_y], axis=-1).reshape(-1, 2)
    proj_coords = np.stack([grid_proj_x, grid_proj_y], axis=-1).reshape(-1, 2)

    c2p_array = np.concatenate([cam_coords, proj_coords], axis=1)  # shape: (N, 4)

    # c2p_array: shape (N, 4), dtype float32 想定
    cam_coords = c2p_array[:, :2]  # (N, 2)
    proj_coords = c2p_array[:, 2:]  # (N, 2)

    # Python の list[ ((cx, cy), (px, py)), ... ] が欲しければ
    ret_c2p_list = [
        ((float(cx), float(cy)), (float(px), float(py)))
        for (cx, cy), (px, py) in zip(cam_coords, proj_coords)
    ]

    return ret_c2p_list


def create_vis_image(
    cam_height: int,
    cam_width: int,
    c2p_list_interp: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    dtype: np.dtype = np.dtype(np.uint8),
) -> np.ndarray:
    # list → ndarray: shape (N, 4)
    arr = np.array(
        [
            [cam_x, cam_y, proj_x, proj_y]
            for (cam_x, cam_y), (proj_x, proj_y) in c2p_list_interp
        ],
        dtype=np.float32,
    )

    cam_x = arr[:, 0]
    cam_y = arr[:, 1]
    proj_x = arr[:, 2]
    proj_y = arr[:, 3]

    ix = np.rint(cam_x).astype(np.int32)  # round
    iy = np.rint(cam_y).astype(np.int32)

    # 有効な点だけをマスク
    valid = (
        (0 <= ix)
        & (ix < cam_width)
        & (0 <= iy)
        & (iy < cam_height)
        & ~np.isnan(proj_x)
        & ~np.isnan(proj_y)
    )

    ix_v = ix[valid]
    iy_v = iy[valid]
    proj_x_v = proj_x[valid]
    proj_y_v = proj_y[valid]

    vis_image = np.zeros((cam_height, cam_width, 3), dtype=dtype)

    # 色をベクトル化して計算
    r = (proj_x_v).astype(np.int32) % (np.iinfo(dtype).max + 1)
    g = (proj_y_v).astype(np.int32) % (np.iinfo(dtype).max + 1)
    b = 128 * np.ones_like(r, dtype=np.int32)

    vis_image[iy_v, ix_v, 0] = r.astype(dtype)
    vis_image[iy_v, ix_v, 1] = g.astype(dtype)
    vis_image[iy_v, ix_v, 2] = b.astype(dtype)

    return vis_image


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv

    if len(argv) != 4:
        print(
            "Usage : python interpolate_c2p.py <c2p_numpy_filename> <cam_height> <cam_width>"
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
            "Usage : python interpolate_c2p.py <c2p_numpy_filename> <cam_height> <cam_width>"
        )
        print()
        return

    c2p_list: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    try:
        c2p_list = load_c2p_numpy(c2p_numpy_filename)
    except Exception as e:
        print(f"Error loading c2p numpy file: {e}")
        return

    print(
        f"Loaded {len(c2p_list)} camera-to-projector correspondences from '{c2p_numpy_filename}'"
    )
    c2p_list_interp = interpolate_c2p_list(cam_height, cam_width, c2p_list)

    # create image for visualization
    vis_image = create_vis_image(
        cam_height, cam_width, c2p_list_interp, dtype=np.dtype(np.uint8)
    )
    vis_filename = os.path.splitext(c2p_numpy_filename)[0] + "_compensated_vis.png"
    cv2.imwrite(vis_filename, vis_image)
    print(f"Saved visualization image to '{vis_filename}'")

    out_filename = os.path.splitext(c2p_numpy_filename)[0] + "_compensated.npy"
    np.save(out_filename, np.array(c2p_list_interp, dtype=object))
    print(f"Saved compensated correspondences to '{out_filename}'")

    with open("result_c2p_compensated.csv", "w", encoding="utf-8") as f:
        f.write("cam_x, cam_y, proj_x, proj_y\n")
        for (cam_x, cam_y), (proj_x, proj_y) in c2p_list_interp:
            f.write(f"{cam_x}, {cam_y}, {proj_x}, {proj_y}\n")
    print("output : './result_c2p_compensated.csv'")
    print()


if __name__ == "__main__":
    main()
