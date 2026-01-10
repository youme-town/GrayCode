# coding: utf-8

import os
import sys

import cv2
import numpy as np


def load_c2p_numpy(
    map_file_path: str,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """互換API: c2pのnpyを読み込んで従来形式のリストを返す。

    戻り値: List[((cam_x, cam_y), (proj_x, proj_y))]
    """
    arr = load_c2p_numpy_array(map_file_path)
    return [
        ((float(cam_x), float(cam_y)), (float(proj_x), float(proj_y)))
        for cam_x, cam_y, proj_x, proj_y in arr
    ]


def load_c2p_numpy_array(map_file_path: str) -> np.ndarray:
    """内部用: c2pのnpyを読み込んで (N,4) float32 配列で返す。

    互換入力:
      - decode.py 互換: dtype=object の [[x,y],[u,v]] / ndarray要素
      - 数値配列: (N,4) [cam_x,cam_y,proj_x,proj_y]
      - 参考: (H,W,2) [proj_x,proj_y]
    """
    map_data = np.load(map_file_path, allow_pickle=True)

    # 既に (N,4) の数値配列
    if isinstance(map_data, np.ndarray) and map_data.dtype != object:
        if map_data.ndim == 2 and map_data.shape[1] == 4:
            return map_data.astype(np.float32, copy=False)

        # (H,W,2) の密マップ
        if map_data.ndim == 3 and map_data.shape[2] == 2:
            h, w, _ = map_data.shape
            out = np.empty((h * w, 4), dtype=np.float32)
            x_coords = np.arange(w, dtype=np.float32)
            for y in range(h):
                row = slice(y * w, (y + 1) * w)
                out[row, 0] = x_coords
                out[row, 1] = np.float32(y)
                out[row, 2] = map_data[y, :, 0].astype(np.float32, copy=False)
                out[row, 3] = map_data[y, :, 1].astype(np.float32, copy=False)
            return out

    # decode.py 互換: dtype=object の配列
    if not (
        isinstance(map_data, np.ndarray)
        and map_data.dtype == object
        and map_data.ndim >= 1
    ):
        raise TypeError("Unsupported c2p numpy format")

    def _as_seq(x):
        # object配列の要素が np.ndarray で入ってくる場合がある
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    n = int(map_data.shape[0])
    out = np.empty((n, 4), dtype=np.float32)

    # 先頭だけチェック
    for i in range(min(10, n)):
        item = _as_seq(map_data[i])
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not isinstance(_as_seq(item[0]), (list, tuple))
            or len(item[0]) != 2
            or not isinstance(_as_seq(item[1]), (list, tuple))
            or len(item[1]) != 2
        ):
            raise TypeError(f"map_list[{i}] must be [[x,y],[u,v]]")

    for i in range(n):
        cam_xy, proj_uv = _as_seq(map_data[i])
        cam_xy = _as_seq(cam_xy)
        proj_uv = _as_seq(proj_uv)
        out[i, 0] = float(cam_xy[0])
        out[i, 1] = float(cam_xy[1])
        out[i, 2] = float(proj_uv[0])
        out[i, 3] = float(proj_uv[1])
    return out


def interpolate_c2p_array(
    cam_height: int,
    cam_width: int,
    c2p_list: np.ndarray,
) -> np.ndarray:
    """
    Fill missing correspondences using cv2.inpaint (float32).
    """
    # Initialize maps with NaN
    proj_x_map = np.full((cam_height, cam_width), np.nan, dtype=np.float32)
    proj_y_map = np.full((cam_height, cam_width), np.nan, dtype=np.float32)

    # Fill known correspondences (vectorized)
    if not (
        isinstance(c2p_list, np.ndarray)
        and c2p_list.ndim == 2
        and c2p_list.shape[1] == 4
    ):
        raise TypeError("c2p_list must be a NumPy array with shape (N,4)")

    cam_x = c2p_list[:, 0]
    cam_y = c2p_list[:, 1]
    proj_x = c2p_list[:, 2]
    proj_y = c2p_list[:, 3]

    ix = cam_x.astype(np.int32)
    iy = cam_y.astype(np.int32)
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
    proj_x_map[iy_v, ix_v] = proj_x[valid].astype(np.float32, copy=False)
    proj_y_map[iy_v, ix_v] = proj_y[valid].astype(np.float32, copy=False)

    # Solve for each channel
    proj_x_filled = _inpaint_fill_float32(
        proj_x_map, radius=3.0, method=cv2.INPAINT_TELEA
    )
    proj_y_filled = _inpaint_fill_float32(
        proj_y_map, radius=3.0, method=cv2.INPAINT_TELEA
    )

    # Build output as (N,4) float32 to avoid huge Python object overhead
    out = np.empty((cam_height * cam_width, 4), dtype=np.float32)
    x_coords = np.arange(cam_width, dtype=np.float32)
    for y in range(cam_height):
        row = slice(y * cam_width, (y + 1) * cam_width)
        out[row, 0] = x_coords
        out[row, 1] = np.float32(y)
        out[row, 2] = proj_x_filled[y, :].astype(np.float32, copy=False)
        out[row, 3] = proj_y_filled[y, :].astype(np.float32, copy=False)

    return out


def interpolate_c2p_list(
    cam_height: int,
    cam_width: int,
    c2p_list: list[tuple[tuple[float, float], tuple[float, float]]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """互換API: 従来形式のリストを受け、従来形式のリストを返す。"""

    # まず (N,4) に正規化して計算
    arr = np.empty((len(c2p_list), 4), dtype=np.float32)
    for i, ((cam_x, cam_y), (proj_x, proj_y)) in enumerate(c2p_list):
        arr[i, 0] = float(cam_x)
        arr[i, 1] = float(cam_y)
        arr[i, 2] = float(proj_x)
        arr[i, 3] = float(proj_y)

    out = interpolate_c2p_array(cam_height, cam_width, arr)
    return [
        ((float(cam_x), float(cam_y)), (float(proj_x), float(proj_y)))
        for cam_x, cam_y, proj_x, proj_y in out
    ]


def _inpaint_fill_float32(
    data: np.ndarray,
    radius: float,
    method: int,
) -> np.ndarray:
    """NaN を穴として cv2.inpaint で埋める（1ch float32）。

    OpenCV(4.11.0) では 32-bit float 1ch がサポートされるため、uint8 量子化は行わない。
    """

    if data.ndim != 2:
        raise TypeError("data must be a 2D array")

    src = data.astype(np.float32, copy=True)
    mask = np.isnan(src)
    if not np.any(mask):
        return src

    # inpaintのマスクは 0/255 の uint8
    mask_u8 = mask.astype(np.uint8) * 255

    # NaNは仮に0で埋めておく（maskで穴扱いになる）
    src[mask] = 0.0

    # float32 のまま inpaint
    dst = cv2.inpaint(src, mask_u8, float(radius), int(method))

    # 既知値は厳密に保持
    dst[~mask] = data[~mask].astype(np.float32, copy=False)
    return dst


def create_vis_image(
    cam_height: int,
    cam_width: int,
    c2p_list_interp: np.ndarray,
    dtype: np.dtype = np.dtype(np.uint8),
) -> np.ndarray:
    if not (
        isinstance(c2p_list_interp, np.ndarray)
        and c2p_list_interp.ndim == 2
        and c2p_list_interp.shape[1] == 4
    ):
        raise TypeError("c2p_list_interp must be a NumPy array with shape (N,4)")

    arr = c2p_list_interp.astype(np.float32, copy=False)

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

    try:
        # 内部計算は省メモリな (N,4) 配列で扱う
        c2p_arr = load_c2p_numpy_array(c2p_numpy_filename)
    except Exception as e:
        print(f"Error loading c2p numpy file: {e}")
        return

    print(
        f"Loaded {len(c2p_arr)} camera-to-projector correspondences from '{c2p_numpy_filename}'"
    )
    c2p_list_interp = interpolate_c2p_array(cam_height, cam_width, c2p_arr)

    # create image for visualization
    vis_image = create_vis_image(
        cam_height, cam_width, c2p_list_interp, dtype=np.dtype(np.uint8)
    )
    vis_filename = os.path.splitext(c2p_numpy_filename)[0] + "_compensated_vis.png"
    cv2.imwrite(vis_filename, vis_image)
    print(f"Saved visualization image to '{vis_filename}'")

    out_filename = os.path.splitext(c2p_numpy_filename)[0] + "_compensated.npy"
    # 外部互換性のため従来形式 dtype=object の (N,2,2) で保存
    n = cam_height * cam_width
    legacy = np.empty((n, 2, 2), dtype=object)
    legacy[:, 0, 0] = c2p_list_interp[:, 0].astype(np.float64, copy=False)
    legacy[:, 0, 1] = c2p_list_interp[:, 1].astype(np.float64, copy=False)
    legacy[:, 1, 0] = c2p_list_interp[:, 2].astype(np.float64, copy=False)
    legacy[:, 1, 1] = c2p_list_interp[:, 3].astype(np.float64, copy=False)
    np.save(out_filename, legacy)
    print(
        f"Saved compensated correspondences to '{out_filename}' (legacy object format)"
    )

    with open("result_c2p_compensated.csv", "w", encoding="utf-8") as f:
        f.write("cam_x, cam_y, proj_x, proj_y\n")
        for cam_x, cam_y, proj_x, proj_y in c2p_list_interp:
            f.write(
                f"{float(cam_x)}, {float(cam_y)}, {float(proj_x)}, {float(proj_y)}\n"
            )
    print("output : './result_c2p_compensated.csv'")
    print()


if __name__ == "__main__":
    main()
