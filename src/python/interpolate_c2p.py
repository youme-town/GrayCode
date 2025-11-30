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


from scipy import sparse
from scipy.sparse.linalg import spsolve


def interpolate_c2p_list(
    cam_height: int,
    cam_width: int,
    c2p_list: List[Tuple[Tuple[float, float], Tuple[float, float]]],
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Fill missing correspondences using direct Laplacian solve.
    Much faster than iterative method.
    """
    # Initialize maps with NaN
    proj_x_map = np.full((cam_height, cam_width), np.nan, dtype=np.float64)
    proj_y_map = np.full((cam_height, cam_width), np.nan, dtype=np.float64)

    # Fill known correspondences
    for (cam_x, cam_y), (proj_x, proj_y) in c2p_list:
        ix, iy = int(cam_x), int(cam_y)
        if 0 <= ix < cam_width and 0 <= iy < cam_height:
            proj_x_map[iy, ix] = proj_x
            proj_y_map[iy, ix] = proj_y

    # Solve for each channel
    proj_x_filled = _laplacian_fill(proj_x_map)
    proj_y_filled = _laplacian_fill(proj_y_map)

    # Build output
    ret_c2p_list: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for iy in range(cam_height):
        for ix in range(cam_width):
            ret_c2p_list.append(
                (
                    (float(ix), float(iy)),
                    (float(proj_x_filled[iy, ix]), float(proj_y_filled[iy, ix])),
                )
            )

    return ret_c2p_list


def _laplacian_fill(data: np.ndarray) -> np.ndarray:
    """
    Fill NaN regions by solving Laplace equation.
    Known pixels act as Dirichlet boundary conditions.
    """
    h, w = data.shape
    n_pixels = h * w

    mask = np.isnan(data)
    if not np.any(mask):
        return data.copy()

    known_mask = ~mask

    # Pixel index mapping
    def idx(y, x):
        return y * w + x

    # Build sparse Laplacian matrix
    row, col, val = [], [], []
    rhs = np.zeros(n_pixels, dtype=np.float64)

    for iy in range(h):
        for ix in range(w):
            i = idx(iy, ix)

            if known_mask[iy, ix]:
                # Known pixel: identity equation
                row.append(i)
                col.append(i)
                val.append(1.0)
                rhs[i] = data[iy, ix]
            else:
                # Unknown pixel: Laplacian equation
                # sum of neighbors = 4 * center (or fewer at boundaries)
                neighbors = []
                if iy > 0:
                    neighbors.append((iy - 1, ix))
                if iy < h - 1:
                    neighbors.append((iy + 1, ix))
                if ix > 0:
                    neighbors.append((iy, ix - 1))
                if ix < w - 1:
                    neighbors.append((iy, ix + 1))

                n_neighbors = len(neighbors)
                row.append(i)
                col.append(i)
                val.append(float(n_neighbors))

                for ny, nx in neighbors:
                    row.append(i)
                    col.append(idx(ny, nx))
                    val.append(-1.0)

                rhs[i] = 0.0

    A = sparse.csr_matrix((val, (row, col)), shape=(n_pixels, n_pixels))
    solution = spsolve(A, rhs)

    result = solution.reshape(h, w)
    # Ensure known values are exactly preserved
    result[known_mask] = data[known_mask]

    return result


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
