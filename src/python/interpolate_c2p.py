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
    method: str = "telea",  # "telea" or "ns" (Navier-Stokes)
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Fill missing correspondences using image inpainting.
    Suitable when most pixels already have correspondences.
    """
    # Initialize maps with NaN
    proj_x_map = np.full((cam_height, cam_width), np.nan, dtype=np.float32)
    proj_y_map = np.full((cam_height, cam_width), np.nan, dtype=np.float32)

    # Fill known correspondences
    for (cam_x, cam_y), (proj_x, proj_y) in c2p_list:
        ix, iy = int(round(cam_x)), int(round(cam_y))
        if 0 <= ix < cam_width and 0 <= iy < cam_height:
            proj_x_map[iy, ix] = proj_x
            proj_y_map[iy, ix] = proj_y

    # Create mask for unknown pixels
    mask = np.isnan(proj_x_map) | np.isnan(proj_y_map)
    mask_uint8 = (mask.astype(np.uint8)) * 255

    # For inpainting, we need valid values everywhere first
    # Temporarily fill NaN with 0 (will be overwritten by inpainting)
    proj_x_map_filled = np.nan_to_num(proj_x_map, nan=0.0)
    proj_y_map_filled = np.nan_to_num(proj_y_map, nan=0.0)

    # Normalize to 0-1 range for better inpainting
    px_min, px_max = np.nanmin(proj_x_map), np.nanmax(proj_x_map)
    py_min, py_max = np.nanmin(proj_y_map), np.nanmax(proj_y_map)

    proj_x_norm = (proj_x_map_filled - px_min) / (px_max - px_min + 1e-8)
    proj_y_norm = (proj_y_map_filled - py_min) / (py_max - py_min + 1e-8)

    # Convert to uint16 for higher precision inpainting
    proj_x_uint16 = (proj_x_norm * 65535).astype(np.uint16)
    proj_y_uint16 = (proj_y_norm * 65535).astype(np.uint16)

    # Inpainting
    inpaint_flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    inpaint_radius = 5

    proj_x_inpainted = cv2.inpaint(
        proj_x_uint16, mask_uint8, inpaint_radius, inpaint_flag
    )
    proj_y_inpainted = cv2.inpaint(
        proj_y_uint16, mask_uint8, inpaint_radius, inpaint_flag
    )

    # Convert back to original scale
    grid_proj_x = (proj_x_inpainted.astype(np.float32) / 65535) * (
        px_max - px_min
    ) + px_min
    grid_proj_y = (proj_y_inpainted.astype(np.float32) / 65535) * (
        py_max - py_min
    ) + py_min

    # Restore original known values exactly (avoid any floating point errors)
    grid_proj_x[~mask] = proj_x_map[~mask]
    grid_proj_y[~mask] = proj_y_map[~mask]

    # Convert to list format
    grid_cam_x, grid_cam_y = np.meshgrid(
        np.arange(cam_width, dtype=np.float32),
        np.arange(cam_height, dtype=np.float32),
        indexing="xy",
    )

    cam_coords = np.stack([grid_cam_x, grid_cam_y], axis=-1).reshape(-1, 2)
    proj_coords = np.stack([grid_proj_x, grid_proj_y], axis=-1).reshape(-1, 2)

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
