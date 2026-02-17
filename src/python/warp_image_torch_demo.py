"""Demo script for PixelMapWarperTorch: forward and backward warping comparison."""

import torch
import cv2

from src.python.warp_image_torch import (
    PixelMapWarperTorch,
    MapType,
    InpaintMethod,
)

from src.python.interpolate_p2c import load_p2c_numpy_array


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    src_img = cv2.imread("captured_rgb_img_3.png")
    dst_size = (1920, 1080)
    src_img_tensor = (
        torch.from_numpy(src_img).permute(2, 0, 1).float().to(device)
    )  # C,H,W

    p2c_array = load_p2c_numpy_array("result_p2c.npy")
    warper = PixelMapWarperTorch(p2c_array, device=device, map_type=MapType.P2C)
    print(f"Warper: {warper}")

    # Backward warp: プロジェクタ画像(UV) → カメラ画像(XY)
    print("\n=== Backward Warp ===")
    print("Backward warping (UV -> XY)...")
    result = warper.forward_warp(
        src_img_tensor,
        dst_size=dst_size,
        inpaint=InpaintMethod.CONV,
        inpaint_iter=5,
    )
    assert isinstance(result, torch.Tensor)

    cv2.imwrite(
        "backward_warp.png",
        result.permute(1, 2, 0).byte().cpu().numpy(),
    )

    print("\nSaved files:")
    print("  backward_warp.png - Backward warp result (UV -> XY)")


if __name__ == "__main__":
    main()
