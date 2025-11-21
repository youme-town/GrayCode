import sys
import numpy as np
import cv2
from typing import List, Tuple, Literal, Optional
from enum import Enum
from .interpolate_c2p import load_c2p_numpy


class AggregationMethod(Enum):
    """複数のピクセルが同じ位置にマップされる場合の集約方法"""

    MEAN = "mean"  # 平均
    MEDIAN = "median"  # 中央値
    MAX = "max"  # 最大値
    MIN = "min"  # 最小値
    FIRST = "first"  # 最初の値
    LAST = "last"  # 最後の値


class InpaintMethod(Enum):
    """穴埋め補完の方法"""

    NONE = "none"  # 補完なし
    TELEA = "telea"  # Telea のアルゴリズム
    NS = "ns"  # Navier-Stokes ベースのアルゴリズム


class PixelMapWarper:
    """
    画素対応マップを使用して画像をワーピングするクラス

    マップ形式: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    各要素: ((x, y), (u, v)) where (x,y)は元画像の座標、(u,v)は変換先の座標

    主な機能:
    - 画像とマップの解像度が異なる場合に対応
    - 順変換(forward warping)と逆変換(backward warping)の両方をサポート
    - 複数ピクセルが同じ位置にマップされる場合の集約方法を選択可能
    - 穴埋め補完機能
    """

    def __init__(
        self,
        pixel_map: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    ):
        """
        Args:
            pixel_map: 画素対応マップ ((x,y), (u,v)) のリスト
        """
        self.pixel_map = pixel_map
        self._cache_map_bounds()

    def _cache_map_bounds(self):
        """マップの範囲をキャッシュ"""
        if not self.pixel_map:
            self.src_bounds = (0, 0, 0, 0)
            self.dst_bounds = (0, 0, 0, 0)
            return

        src_xs = [p[0][0] for p in self.pixel_map]
        src_ys = [p[0][1] for p in self.pixel_map]
        dst_xs = [p[1][0] for p in self.pixel_map]
        dst_ys = [p[1][1] for p in self.pixel_map]

        self.src_bounds = (min(src_xs), min(src_ys), max(src_xs), max(src_ys))
        self.dst_bounds = (min(dst_xs), min(dst_ys), max(dst_xs), max(dst_ys))

    def forward_warp(
        self,
        src_img: np.ndarray,
        dst_size: Optional[Tuple[int, int]] = None,
        src_offset: Tuple[int, int] = (0, 0),
        aggregation: AggregationMethod = AggregationMethod.MEAN,
        inpaint: InpaintMethod = InpaintMethod.NONE,
        inpaint_radius: int = 3,
        crop_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """
        順変換: (x,y) -> (u,v) のマッピングで画像を変換

        Args:
            src_img: 入力画像
            dst_size: 出力画像サイズ (width, height)。Noneの場合はマップから自動計算
            src_offset: 入力画像のオフセット (offset_x, offset_y)
            aggregation: 複数ピクセルが同じ位置にマップされる場合の集約方法
            inpaint: 穴埋め補完の方法
            inpaint_radius: 穴埋め補完の半径
            crop_rect: 最終的にトリミングする矩形 (x, y, width, height)

        Returns:
            変換後の画像
        """
        if dst_size is None:
            # マップから出力サイズを計算
            dst_width = int(self.dst_bounds[2]) + 1
            dst_height = int(self.dst_bounds[3]) + 1
        else:
            dst_width, dst_height = dst_size

        # 出力画像の初期化
        if len(src_img.shape) == 3:
            dst_img = np.zeros(
                (dst_height, dst_width, src_img.shape[2]), dtype=np.float64
            )
            count_img = np.zeros(
                (dst_height, dst_width, src_img.shape[2]), dtype=np.float64
            )
        else:
            dst_img = np.zeros((dst_height, dst_width), dtype=np.float64)
            count_img = np.zeros((dst_height, dst_width), dtype=np.float64)

        # 集約方法に応じた処理
        if aggregation == AggregationMethod.MEAN:
            # 平均を取るために加算とカウント
            for (src_x, src_y), (dst_x, dst_y) in self.pixel_map:
                # NaN値をスキップ
                if (
                    np.isnan(src_x)
                    or np.isnan(src_y)
                    or np.isnan(dst_x)
                    or np.isnan(dst_y)
                ):
                    continue

                # 元画像の座標を計算（ピクセル中心座標系を考慮）
                # マップ座標0.5 = ピクセル0の中心 = 画像インデックス0
                img_x = int(np.floor(src_x)) - src_offset[0]
                img_y = int(np.floor(src_y)) - src_offset[1]

                # 範囲チェック
                if (
                    img_x < 0
                    or img_x >= src_img.shape[1]
                    or img_y < 0
                    or img_y >= src_img.shape[0]
                ):
                    continue

                # 出力座標を計算（ピクセル中心座標系を考慮）
                out_x = int(np.floor(dst_x))
                out_y = int(np.floor(dst_y))

                if out_x < 0 or out_x >= dst_width or out_y < 0 or out_y >= dst_height:
                    continue

                # ピクセル値を加算
                dst_img[out_y, out_x] += src_img[img_y, img_x].astype(np.float64)
                count_img[out_y, out_x] += 1

            # 平均を計算
            mask = count_img > 0
            dst_img[mask] /= count_img[mask]

        elif aggregation in [AggregationMethod.MAX, AggregationMethod.MIN]:
            # 最大値または最小値
            if aggregation == AggregationMethod.MAX:
                dst_img.fill(-np.inf)
            else:
                dst_img.fill(np.inf)

            for (src_x, src_y), (dst_x, dst_y) in self.pixel_map:
                # NaN値をスキップ
                if (
                    np.isnan(src_x)
                    or np.isnan(src_y)
                    or np.isnan(dst_x)
                    or np.isnan(dst_y)
                ):
                    continue

                # 元画像の座標を計算（ピクセル中心座標系を考慮）
                img_x = int(np.floor(src_x)) - src_offset[0]
                img_y = int(np.floor(src_y)) - src_offset[1]

                if (
                    img_x < 0
                    or img_x >= src_img.shape[1]
                    or img_y < 0
                    or img_y >= src_img.shape[0]
                ):
                    continue

                # 出力座標を計算（ピクセル中心座標系を考慮）
                out_x = int(np.floor(dst_x))
                out_y = int(np.floor(dst_y))

                if out_x < 0 or out_x >= dst_width or out_y < 0 or out_y >= dst_height:
                    continue

                pixel_value = src_img[img_y, img_x].astype(np.float64)
                count_img[out_y, out_x] += 1

                if aggregation == AggregationMethod.MAX:
                    dst_img[out_y, out_x] = np.maximum(
                        dst_img[out_y, out_x], pixel_value
                    )
                else:
                    dst_img[out_y, out_x] = np.minimum(
                        dst_img[out_y, out_x], pixel_value
                    )

            # 無限大の値を0に戻す
            mask = count_img == 0
            dst_img[mask] = 0

        elif aggregation == AggregationMethod.MEDIAN:
            # 中央値を取るためにリストを保持
            if len(src_img.shape) == 3:
                pixel_lists = [
                    [[] for _ in range(dst_width)] for _ in range(dst_height)
                ]
            else:
                pixel_lists = [
                    [[] for _ in range(dst_width)] for _ in range(dst_height)
                ]

            for (src_x, src_y), (dst_x, dst_y) in self.pixel_map:
                # NaN値をスキップ
                if (
                    np.isnan(src_x)
                    or np.isnan(src_y)
                    or np.isnan(dst_x)
                    or np.isnan(dst_y)
                ):
                    continue

                # 元画像の座標を計算（ピクセル中心座標系を考慮）
                img_x = int(np.floor(src_x)) - src_offset[0]
                img_y = int(np.floor(src_y)) - src_offset[1]

                if (
                    img_x < 0
                    or img_x >= src_img.shape[1]
                    or img_y < 0
                    or img_y >= src_img.shape[0]
                ):
                    continue

                # 出力座標を計算（ピクセル中心座標系を考慮）
                out_x = int(np.floor(dst_x))
                out_y = int(np.floor(dst_y))

                if out_x < 0 or out_x >= dst_width or out_y < 0 or out_y >= dst_height:
                    continue

                pixel_lists[out_y][out_x].append(
                    src_img[img_y, img_x].astype(np.float64)
                )

            # 中央値を計算
            for y in range(dst_height):
                for x in range(dst_width):
                    if pixel_lists[y][x]:
                        dst_img[y, x] = np.median(pixel_lists[y][x], axis=0)
                        count_img[y, x] = len(pixel_lists[y][x])

        elif aggregation in [AggregationMethod.FIRST, AggregationMethod.LAST]:
            # 最初または最後の値
            for (src_x, src_y), (dst_x, dst_y) in self.pixel_map:
                # NaN値をスキップ
                if (
                    np.isnan(src_x)
                    or np.isnan(src_y)
                    or np.isnan(dst_x)
                    or np.isnan(dst_y)
                ):
                    continue

                # 元画像の座標を計算（ピクセル中心座標系を考慮）
                img_x = int(np.floor(src_x)) - src_offset[0]
                img_y = int(np.floor(src_y)) - src_offset[1]

                if (
                    img_x < 0
                    or img_x >= src_img.shape[1]
                    or img_y < 0
                    or img_y >= src_img.shape[0]
                ):
                    continue

                # 出力座標を計算（ピクセル中心座標系を考慮）
                out_x = int(np.floor(dst_x))
                out_y = int(np.floor(dst_y))

                if out_x < 0 or out_x >= dst_width or out_y < 0 or out_y >= dst_height:
                    continue

                # FIRSTの場合は未設定の場合のみ、LASTの場合は常に上書き
                if (
                    aggregation == AggregationMethod.FIRST
                    and count_img[out_y, out_x] > 0
                ):
                    continue

                dst_img[out_y, out_x] = src_img[img_y, img_x].astype(np.float64)
                count_img[out_y, out_x] += 1

        # 穴埋め補完
        if inpaint != InpaintMethod.NONE:
            dst_img = self._apply_inpaint(dst_img, count_img, inpaint, inpaint_radius)

        # uint8に変換
        dst_img = np.clip(dst_img, 0, 255).astype(np.uint8)

        # トリミング
        if crop_rect is not None:
            x, y, w, h = crop_rect
            dst_img = dst_img[y : y + h, x : x + w]

        return dst_img

    def backward_warp(
        self,
        src_img: np.ndarray,
        dst_size: Optional[Tuple[int, int]] = None,
        src_offset: Tuple[int, int] = (0, 0),
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_CONSTANT,
        border_value: Tuple[int, int, int] = (0, 0, 0),
        crop_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """
        逆変換: (u,v) -> (x,y) のマッピングで画像を変換（cv2.remapを使用）

        Args:
            src_img: 入力画像
            dst_size: 出力画像サイズ (width, height)。Noneの場合はマップから自動計算
            src_offset: 入力画像のオフセット (offset_x, offset_y)
            interpolation: 補間方法
            border_mode: 境界モード
            border_value: 境界値
            crop_rect: 最終的にトリミングする矩形 (x, y, width, height)

        Returns:
            変換後の画像
        """
        if dst_size is None:
            # マップから出力サイズを計算
            dst_width = int(self.src_bounds[2]) + 1
            dst_height = int(self.src_bounds[3]) + 1
        else:
            dst_width, dst_height = dst_size

        # リマップ用のマップを作成
        # pixel_mapの要素を入れ替える: ((u,v), (x,y)) として扱う
        map_dict = {}
        for (src_x, src_y), (dst_x, dst_y) in self.pixel_map:
            # NaN値をスキップ
            if np.isnan(src_x) or np.isnan(src_y) or np.isnan(dst_x) or np.isnan(dst_y):
                continue

            # 逆変換なので、dst -> src へのマッピングを作成（ピクセル中心座標系を考慮）
            map_key = (int(np.floor(src_x)), int(np.floor(src_y)))
            map_dict[map_key] = (dst_x, dst_y)

        # マップ配列を作成
        map_x = np.zeros((dst_height, dst_width), dtype=np.float32)
        map_y = np.zeros((dst_height, dst_width), dtype=np.float32)

        for y in range(dst_height):
            for x in range(dst_width):
                if (x, y) in map_dict:
                    dst_x, dst_y = map_dict[(x, y)]
                    map_x[y, x] = dst_x - src_offset[0]
                    map_y[y, x] = dst_y - src_offset[1]
                else:
                    # マッピングが存在しない場合は-1（境界値を使用）
                    map_x[y, x] = -1
                    map_y[y, x] = -1

        # cv2.remapで変換
        dst_img = cv2.remap(
            src_img,
            map_x,
            map_y,
            interpolation=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )

        # トリミング
        if crop_rect is not None:
            x, y, w, h = crop_rect
            dst_img = dst_img[y : y + h, x : x + w]

        return dst_img

    def _apply_inpaint(
        self,
        img: np.ndarray,
        count_img: np.ndarray,
        method: InpaintMethod,
        radius: int,
    ) -> np.ndarray:
        """
        穴埋め補完を適用

        Args:
            img: 画像
            count_img: ピクセルのカウント（0のピクセルが穴）
            method: 補完方法
            radius: 補完半径

        Returns:
            補完後の画像
        """
        if method == InpaintMethod.NONE:
            return img

        # マスクを作成（穴の部分を255）
        if len(count_img.shape) == 3:
            mask = (count_img[:, :, 0] == 0).astype(np.uint8) * 255
        else:
            mask = (count_img == 0).astype(np.uint8) * 255

        # 画像をuint8に変換
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)

        # cv2.inpaintを使用
        if method == InpaintMethod.TELEA:
            inpaint_method = cv2.INPAINT_TELEA
        else:  # InpaintMethod.NS
            inpaint_method = cv2.INPAINT_NS

        inpainted = cv2.inpaint(img_uint8, mask, radius, inpaint_method)

        return inpainted.astype(np.float64)


def main() -> None:
    # マップデータの読み込み
    pixel_map = load_c2p_numpy("result_c2p_compensated.npy")

    # 画像の読み込み
    src_img = cv2.imread("input.png")
    if src_img is None:
        print("Error: failed to load image 'input.png'")
        return

    # Warperの作成
    warper = PixelMapWarper(pixel_map)

    # example offset for cropping
    OFFSET_X = 1920 // 2 - 500 // 2
    OFFSET_Y = 1080 // 2 - 500 // 2

    # 順変換（forward warping）- 複数ピクセルが同じ位置にマップされる場合は平均
    warped_img = warper.forward_warp(
        src_img,
        dst_size=(1920, 1080),  # 出力サイズを指定
        src_offset=(0, 0),  # 入力画像のオフセット
        aggregation=AggregationMethod.MEDIAN,  # 中央値を使用
        inpaint=InpaintMethod.TELEA,  # 穴埋め補完を使用
        inpaint_radius=1,  # 補完半径
        crop_rect=(OFFSET_X, OFFSET_Y, 500, 500),  # トリミング
    )

    # 逆変換（backward warping）- cv2.remapを使用
    inv_warped_img = warper.backward_warp(
        src_img,
        #        dst_size=(1920, 1080),
        src_offset=(0, 0),
        interpolation=cv2.INTER_LINEAR,
        # crop_rect=(OFFSET_X, OFFSET_Y, 500, 500),
    )

    # 保存
    cv2.imwrite("warped.jpg", warped_img)
    cv2.imwrite("inv_warped.jpg", inv_warped_img)


if __name__ == "__main__":
    main()


"""
# 異なる集約方法の使用例
```python
# 最大値を使用
warped_max = warper.forward_warp(
    src_img,
    aggregation=AggregationMethod.MAX
)

# 中央値を使用
warped_median = warper.forward_warp(
    src_img,
    aggregation=AggregationMethod.MEDIAN
)

# 最初の値を使用
warped_first = warper.forward_warp(
    src_img,
    aggregation=AggregationMethod.FIRST
)
```

# 画像とマップの解像度が異なる場合
```python
# 例: マップが1920x1080だが、画像が640x480の場合
# src_offsetで画像がマップのどの位置に対応するかを指定
warped_img = warper.forward_warp(
    small_img,
    dst_size=(1920, 1080),
    src_offset=(640, 300),  # 画像をマップの(640, 300)の位置に配置
    aggregation=AggregationMethod.MEAN
)
```

# 後方互換性のための従来の関数
```python
from warp_image import warp_image, inverse_warp_image

# 従来のwarp_image関数（cv2.remapベース）
warped = warp_image(
    src_img,
    pixel_map,
    dst_rect=(100, 100, 500, 500)
)

# 従来のinverse_warp_image関数（cv2.remapベース）
inv_warped = inverse_warp_image(
    src_img,
    pixel_map,
    dst_rect=(100, 100, 500, 500)
)
```
"""


# def print_usage() -> None:
#     print("Usage : python warp_image.py <input image> <output image> <map file>")
#     print()


# def main(argv: list[str] | None = None) -> None:
#     if argv is None:
#         argv = sys.argv

#     if len(argv) != 4:
#         print_usage()
#         return

#     input_image_path = argv[1]
#     output_image_path = argv[2]
#     map_file_path = argv[3]

#     # 入力画像の読み込み
#     src_img = cv2.imread(input_image_path)

#     # マップデータの読み込み
#     map_data = np.load(map_file_path, allow_pickle=True)
#     map_list: List[Tuple[Tuple[float, float], Tuple[float, float]]] = map_data.tolist()

#     # 画像のワープ
#     warped_img = inverse_warp_image(
#         src_img,
#         map_list,
#         dst_rect=(1920 // 2 - 500 // 2, 1080 // 2 - 500 // 2, 500, 500),
#     )

#     # 出力画像の保存
#     cv2.imwrite(output_image_path, warped_img)


# if __name__ == "__main__":
#     main()
