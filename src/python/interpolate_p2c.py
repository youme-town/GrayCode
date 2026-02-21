# coding: utf-8

import os
import sys

import cv2
import numpy as np

try:
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .config import get_config, reload_config, split_cli_config_path


def load_p2c_numpy_array(map_file_path: str) -> np.ndarray:
    """P2C の npy を読み込んで (N, 4) float32 配列で返す。

    各行は [proj_x, proj_y, cam_x, cam_y]。
    同一プロジェクタ座標に複数のカメラ座標が対応する場合は全て展開される。

    対応する入力形式:
      - decode.py 互換: 0-d object array wrapping dict {(px,py): [(cx,cy), ...]}
      - (N, 4) 数値配列: [proj_x, proj_y, cam_x, cam_y]
    """
    map_data = np.load(map_file_path, allow_pickle=True)

    # (N, 4) 数値配列
    if isinstance(map_data, np.ndarray) and map_data.dtype != object:
        if map_data.ndim == 2 and map_data.shape[1] == 4:
            return map_data.astype(np.float32, copy=False)

    # decode.py 互換: 0-d object array wrapping dict
    if isinstance(map_data, np.ndarray) and map_data.dtype == object:
        item = map_data.item() if map_data.ndim == 0 else map_data
        if isinstance(item, dict):
            rows = []
            for (proj_x, proj_y), cam_list in item.items():
                for cam_x, cam_y in cam_list:
                    rows.append(
                        [float(proj_x), float(proj_y), float(cam_x), float(cam_y)]
                    )
            if not rows:
                raise ValueError("P2C dict is empty.")
            return np.array(rows, dtype=np.float32)

    raise TypeError("Unsupported P2C numpy format")


def interpolate_p2c_delaunay(
    proj_height: int,
    proj_width: int,
    p2c_arr: np.ndarray,
) -> np.ndarray:
    """ドロネー三角形分割による線形補間で全プロジェクタ画素のカメラ座標を求める。

    Args:
        proj_height: プロジェクタ画像の高さ
        proj_width:  プロジェクタ画像の幅
        p2c_arr:     (N, 4) float32 配列 [proj_x, proj_y, cam_x, cam_y]

    Returns:
        (proj_height * proj_width, 4) float32 配列
        各行: [proj_x, proj_y, cam_x, cam_y]
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for Delaunay interpolation.")

    print("Running Delaunay Interpolation (P2C)...")

    if not (
        isinstance(p2c_arr, np.ndarray)
        and p2c_arr.ndim == 2
        and p2c_arr.shape[1] == 4
    ):
        raise TypeError("p2c_arr must be a NumPy array with shape (N, 4)")

    # 1. NaN 除去
    valid_mask = ~np.isnan(p2c_arr).any(axis=1)
    valid_data = p2c_arr[valid_mask]

    if len(valid_data) < 4:
        raise ValueError("Not enough points for Delaunay triangulation.")

    # 入力点: プロジェクタ座標, 値: カメラ座標
    points = valid_data[:, 0:2]  # proj_x, proj_y
    values = valid_data[:, 2:4]  # cam_x, cam_y

    print(f"  - Input points: {len(points)} correspondences")

    # 2. 補間グリッドの作成 (プロジェクタ画像の全画素)
    grid_y, grid_x = np.mgrid[0:proj_height, 0:proj_width]
    query_points = np.stack(
        (grid_x.ravel().astype(np.float32), grid_y.ravel().astype(np.float32)), axis=1
    )

    # 3. ドロネー分割による線形補間
    print("  - Building Delaunay triangulation and interpolating (Linear)...")
    lin_interp = LinearNDInterpolator(points, values)
    interpolated_values = lin_interp(query_points)  # (N_pixels, 2)

    # 4. 凸包外部を最近傍で埋める
    nan_mask = np.isnan(interpolated_values[:, 0])
    if np.any(nan_mask):
        print(
            f"  - Filling {np.sum(nan_mask)} outside points with Nearest Neighbor..."
        )
        near_interp = NearestNDInterpolator(points, values)
        interpolated_values[nan_mask] = near_interp(query_points[nan_mask])

    # 5. 結果の整形
    out = np.empty((proj_height * proj_width, 4), dtype=np.float32)
    out[:, 0] = query_points[:, 0]  # proj_x
    out[:, 1] = query_points[:, 1]  # proj_y
    out[:, 2] = interpolated_values[:, 0].astype(np.float32)  # cam_x
    out[:, 3] = interpolated_values[:, 1].astype(np.float32)  # cam_y

    return out


def create_vis_image_p2c(
    proj_height: int,
    proj_width: int,
    p2c_interp: np.ndarray,
    dtype: np.dtype = np.dtype(np.uint8),
) -> np.ndarray:
    """P2C 補間結果を可視化する画像を生成する。

    プロジェクタ画像の各画素を、対応するカメラ座標で色付けする。
    """
    if not (
        isinstance(p2c_interp, np.ndarray)
        and p2c_interp.ndim == 2
        and p2c_interp.shape[1] == 4
    ):
        raise TypeError("p2c_interp must be a NumPy array with shape (N, 4)")

    arr = p2c_interp.astype(np.float32, copy=False)

    proj_x = arr[:, 0]
    proj_y = arr[:, 1]
    cam_x = arr[:, 2]
    cam_y = arr[:, 3]

    ix = np.rint(proj_x).astype(np.int32)
    iy = np.rint(proj_y).astype(np.int32)

    valid = (
        (0 <= ix)
        & (ix < proj_width)
        & (0 <= iy)
        & (iy < proj_height)
        & ~np.isnan(cam_x)
        & ~np.isnan(cam_y)
    )

    ix_v = ix[valid]
    iy_v = iy[valid]
    cam_x_v = cam_x[valid]
    cam_y_v = cam_y[valid]

    vis_image = np.zeros((proj_height, proj_width, 3), dtype=dtype)

    r = cam_x_v.astype(np.int32) % (np.iinfo(dtype).max + 1)
    g = cam_y_v.astype(np.int32) % (np.iinfo(dtype).max + 1)
    b = 128 * np.ones_like(r, dtype=np.int32)

    vis_image[iy_v, ix_v, 0] = r.astype(dtype)
    vis_image[iy_v, ix_v, 1] = g.astype(dtype)
    vis_image[iy_v, ix_v, 2] = b.astype(dtype)

    return vis_image


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv
    try:
        argv, config_path = split_cli_config_path(argv)
    except ValueError as e:
        print(e)
        print(
            "Usage : python interpolate_p2c.py <p2c_numpy_filename> <proj_height> <proj_width> [--config <config.toml>]"
        )
        print()
        return

    if config_path is not None:
        reload_config(config_path)

    if len(argv) < 4:
        print(
            "Usage : python interpolate_p2c.py "
            "<p2c_numpy_filename> <proj_height> <proj_width> [--config <config.toml>]"
        )
        print()
        return

    try:
        p2c_numpy_filename = str(argv[1])
        proj_height = int(argv[2])
        proj_width = int(argv[3])
    except ValueError:
        print("proj_height, proj_width は整数で指定してください。")
        print()
        return

    try:
        p2c_arr = load_p2c_numpy_array(p2c_numpy_filename)
    except Exception as e:
        print(f"Error loading P2C numpy file: {e}")
        return

    print(
        f"Loaded {len(p2c_arr)} projector-to-camera correspondences "
        f"from '{p2c_numpy_filename}'"
    )
    print(f"Target projector size: {proj_width}x{proj_height}")

    p2c_interp = interpolate_p2c_delaunay(proj_height, proj_width, p2c_arr)

    # 可視化画像
    vis_image = create_vis_image_p2c(
        proj_height, proj_width, p2c_interp, dtype=np.dtype(np.uint8)
    )
    vis_filename = (
        os.path.splitext(p2c_numpy_filename)[0] + "_compensated_delaunay_vis.png"
    )
    cv2.imwrite(vis_filename, vis_image)
    print(f"Saved visualization image to '{vis_filename}'")

    # npy 保存 (N, 4) float32
    out_filename = (
        os.path.splitext(p2c_numpy_filename)[0] + "_compensated_delaunay.npy"
    )
    np.save(out_filename, p2c_interp)
    print(f"Saved compensated correspondences to '{out_filename}'")

    # CSV 保存
    csv_filename = "result_p2c_compensated_delaunay.csv"
    precision = get_config().interpolate_p2c.csv_precision
    with open(csv_filename, "w", encoding="utf-8") as f:
        f.write("proj_x, proj_y, cam_x, cam_y\n")
        for row in p2c_interp:
            f.write(
                f"{row[0]:.{precision}f}, {row[1]:.{precision}f}, "
                f"{row[2]:.{precision}f}, {row[3]:.{precision}f}\n"
            )

    print(f"output : './{csv_filename}'")
    print()


if __name__ == "__main__":
    main()
