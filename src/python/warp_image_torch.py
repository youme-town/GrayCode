"""
Pixel Map Warper (PyTorch Version)

A GPU-accelerated image warping module using pixel correspondence maps.
Supports both forward warping (splatting) and backward warping (sampling).

Coordinate Systems:
    - XY: Source coordinate system (pixel_map's src points)
    - UV: Destination coordinate system (pixel_map's dst points)

Usage:
    >>> warper = PixelMapWarperTorch(pixel_map, device="cuda")
    >>> # Forward warp: XY image -> UV image
    >>> uv_img = warper.forward_warp(xy_img, dst_size=(200, 200))
    >>> # Backward warp: UV image -> XY image
    >>> xy_img = warper.backward_warp(uv_img, dst_size=(100, 100))

Author: Your Name
License: MIT
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Union
from enum import Enum


class AggregationMethod(Enum):
    """Aggregation method when multiple pixels map to the same location."""

    MEAN = "mean"  # Average of all values
    MAX = "max"  # Maximum value
    MIN = "min"  # Minimum value
    LAST = "last"  # Last value (overwrite)


class InpaintMethod(Enum):
    """Inpainting method for filling holes."""

    NONE = "none"  # No inpainting
    CONV = "conv"  # Convolution-based inpainting (GPU compatible)


class PaddingMode(Enum):
    """Padding mode for out-of-bounds sampling."""

    ZEROS = "zeros"  # Fill with zeros
    BORDER = "border"  # Clamp to border values
    REFLECTION = "reflection"  # Reflect at boundaries


class PixelMapWarperTorch:
    """
    Warps images using pixel correspondence map (PyTorch version).

    The pixel map defines correspondences from XY coordinate system to UV coordinate system:
        (x, y) -> (u, v)

    Forward warp transforms an XY image to UV space (splatting).
    Backward warp transforms a UV image to XY space (sampling).

    Attributes:
        map_tensor: Pixel correspondence map as tensor (N, 4) [x, y, u, v]
        xy_bounds: Bounds in XY space (min_x, min_y, max_x, max_y)
        uv_bounds: Bounds in UV space (min_u, min_v, max_u, max_v)
        device: Computation device ('cpu' or 'cuda')
    """

    def __init__(
        self,
        pixel_map: Union[
            List[Tuple[Tuple[float, float], Tuple[float, float]]],
            np.ndarray,
            torch.Tensor,
        ],
        device: Optional[str] = None,
    ):
        """
        Initialize the warper with a pixel correspondence map.

        Args:
            pixel_map: Pixel correspondence map in one of the following formats:
                - List of tuples: [((src_x, src_y), (dst_x, dst_y)), ...]
                - NumPy array: shape (N, 4) with columns [src_x, src_y, dst_x, dst_y]
                - PyTorch tensor: shape (N, 4) with columns [src_x, src_y, dst_x, dst_y]
            device: Computation device. If None, uses CUDA if available.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Convert input to tensor
        if isinstance(pixel_map, list):
            flat_data = [[sx, sy, dx, dy] for (sx, sy), (dx, dy) in pixel_map]
            self.map_tensor = torch.tensor(
                flat_data, dtype=torch.float32, device=self.device
            )
        elif isinstance(pixel_map, np.ndarray):
            self.map_tensor = torch.from_numpy(pixel_map).float().to(self.device)
            if self.map_tensor.ndim == 3 and self.map_tensor.shape[1:] == (2, 2):
                self.map_tensor = self.map_tensor.view(-1, 4)
        elif isinstance(pixel_map, torch.Tensor):
            self.map_tensor = pixel_map.float().to(self.device)
        else:
            raise TypeError(f"Unsupported pixel_map type: {type(pixel_map)}")

        # Remove NaN entries
        mask = ~torch.isnan(self.map_tensor).any(dim=1)
        self.map_tensor = self.map_tensor[mask]

        self._cache_bounds()

    def _cache_bounds(self) -> None:
        """Cache the coordinate bounds of the map."""
        if self.map_tensor.numel() == 0:
            self.xy_bounds = (0.0, 0.0, 0.0, 0.0)
            self.uv_bounds = (0.0, 0.0, 0.0, 0.0)
            return

        x, y, u, v = self.map_tensor.T
        self.xy_bounds = (
            x.min().item(),
            y.min().item(),
            x.max().item(),
            y.max().item(),
        )
        self.uv_bounds = (
            u.min().item(),
            v.min().item(),
            u.max().item(),
            v.max().item(),
        )

    def forward_warp(
        self,
        src_img: torch.Tensor,
        dst_size: Optional[Tuple[int, int]] = None,
        src_offset: Tuple[int, int] = (0, 0),
        aggregation: AggregationMethod = AggregationMethod.MEAN,
        inpaint: InpaintMethod = InpaintMethod.NONE,
        inpaint_iter: int = 3,
        crop_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> torch.Tensor:
        """
        Forward warp (Splatting): XY image -> UV image.

        Transforms an image from XY coordinate system to UV coordinate system
        by "splatting" source pixels to their destination locations.

        Args:
            src_img: Source image in XY coordinates. Shape: (C, H, W) or (B, C, H, W)
            dst_size: Output size as (width, height). If None, auto-calculated from map.
            src_offset: Offset for source coordinates (x_offset, y_offset).
                       Use when src_img is a crop of the full XY image.
            aggregation: Method for handling overlapping pixels.
            inpaint: Method for filling holes in the output.
            inpaint_iter: Number of inpainting iterations.
            crop_rect: Crop region (x, y, width, height) in UV space.

        Returns:
            Warped image in UV coordinates. Shape matches input batch format.
        """
        # Normalize to (B, C, H, W)
        is_batch = src_img.ndim == 4
        if not is_batch:
            src_img = src_img.unsqueeze(0)

        B, C, H, W = src_img.shape
        src_img = src_img.to(self.device).float()

        # Calculate output size
        if dst_size is None:
            dst_w = int(self.uv_bounds[2]) + 1
            dst_h = int(self.uv_bounds[3]) + 1
        else:
            dst_w, dst_h = dst_size

        # Calculate coordinates
        src_x = torch.floor(self.map_tensor[:, 0]).long() - src_offset[0]
        src_y = torch.floor(self.map_tensor[:, 1]).long() - src_offset[1]
        dst_x = torch.floor(self.map_tensor[:, 2]).long()
        dst_y = torch.floor(self.map_tensor[:, 3]).long()

        # Filter valid coordinates
        valid_mask = (
            (src_x >= 0)
            & (src_x < W)
            & (src_y >= 0)
            & (src_y < H)
            & (dst_x >= 0)
            & (dst_x < dst_w)
            & (dst_y >= 0)
            & (dst_y < dst_h)
        )

        s_x, s_y = src_x[valid_mask], src_y[valid_mask]
        d_x, d_y = dst_x[valid_mask], dst_y[valid_mask]
        dst_indices = d_y * dst_w + d_x
        pixel_values = src_img[:, :, s_y, s_x]

        # Create output buffers
        out_img = torch.zeros(
            (B, C, dst_h * dst_w), device=self.device, dtype=torch.float32
        )
        count_img = torch.zeros(
            (1, 1, dst_h * dst_w), device=self.device, dtype=torch.float32
        )

        # Aggregate pixels
        if aggregation == AggregationMethod.MEAN:
            out_img.index_add_(2, dst_indices, pixel_values)
            ones = torch.ones_like(dst_indices, dtype=torch.float32).view(1, 1, -1)
            count_img.index_add_(2, dst_indices, ones)
            mask = count_img > 0
            out_img = torch.where(mask, out_img / (count_img + 1e-8), out_img)

        elif aggregation == AggregationMethod.MAX:
            out_img.fill_(-1e9)
            out_img.scatter_reduce_(
                2,
                dst_indices.unsqueeze(0).unsqueeze(0).expand(B, C, -1),
                pixel_values,
                reduce="amax",
                include_self=False,
            )
            out_img[out_img == -1e9] = 0
            ones = torch.ones_like(dst_indices, dtype=torch.float32).view(1, 1, -1)
            count_img.index_add_(2, dst_indices, ones)

        elif aggregation == AggregationMethod.MIN:
            out_img.fill_(1e9)
            out_img.scatter_reduce_(
                2,
                dst_indices.unsqueeze(0).unsqueeze(0).expand(B, C, -1),
                pixel_values,
                reduce="amin",
                include_self=False,
            )
            out_img[out_img == 1e9] = 0
            ones = torch.ones_like(dst_indices, dtype=torch.float32).view(1, 1, -1)
            count_img.index_add_(2, dst_indices, ones)

        elif aggregation == AggregationMethod.LAST:
            out_img.scatter_(
                2,
                dst_indices.unsqueeze(0).unsqueeze(0).expand(B, C, -1),
                pixel_values,
            )
            ones = torch.ones_like(dst_indices, dtype=torch.float32).view(1, 1, -1)
            count_img.index_add_(2, dst_indices, ones)

        # Reshape
        out_img = out_img.view(B, C, dst_h, dst_w)
        count_img = count_img.view(1, 1, dst_h, dst_w)

        # Inpainting
        if inpaint == InpaintMethod.CONV:
            out_img = self._apply_inpaint_conv(out_img, count_img, inpaint_iter)

        # Crop
        if crop_rect is not None:
            cx, cy, cw, ch = crop_rect
            out_img = out_img[:, :, cy : cy + ch, cx : cx + cw]

        return out_img if is_batch else out_img.squeeze(0)

    def backward_warp(
        self,
        uv_img: torch.Tensor,
        dst_size: Optional[Tuple[int, int]] = None,
        src_rect: Optional[Tuple[int, int, int, int]] = None,
        mode: str = "bilinear",
        padding_mode: PaddingMode = PaddingMode.ZEROS,
        inpaint: InpaintMethod = InpaintMethod.NONE,
        inpaint_iter: int = 5,
        return_mask: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Backward warp (Sampling): UV image -> XY image.

        Transforms an image from UV coordinate system to XY coordinate system
        by sampling from the UV image at mapped locations.

        Args:
            uv_img: Source image in UV coordinates. Shape: (C, H, W) or (B, C, H, W)
            dst_size: Output size as (width, height) in XY space.
                     If None, auto-calculated from map bounds.
            src_rect: Region in UV space that uv_img covers, as (u, v, width, height).
                     If None, assumes uv_img starts at UV coordinate (0, 0).
            mode: Interpolation mode ('bilinear' or 'nearest').
            padding_mode: How to handle pixels with no correspondence.
            inpaint: Inpainting method for grid holes.
            inpaint_iter: Number of inpainting iterations.
            return_mask: If True, also return a mask of valid pixels.

        Returns:
            Warped image in XY coordinates.
            If return_mask=True, returns tuple (image, mask).
        """
        # Normalize to (B, C, H, W)
        is_batch = uv_img.ndim == 4
        if not is_batch:
            uv_img = uv_img.unsqueeze(0)

        B, C, H_uv, W_uv = uv_img.shape
        uv_img = uv_img.to(self.device).float()

        # Parse src_rect
        if src_rect is not None:
            uv_offset_x, uv_offset_y, uv_w, uv_h = src_rect
        else:
            uv_offset_x, uv_offset_y = 0, 0
            uv_w, uv_h = W_uv, H_uv

        # Determine output size
        if dst_size is None:
            dst_w = int(self.xy_bounds[2] - self.xy_bounds[0]) + 1
            dst_h = int(self.xy_bounds[3] - self.xy_bounds[1]) + 1
            xy_offset_x = int(self.xy_bounds[0])
            xy_offset_y = int(self.xy_bounds[1])
        else:
            dst_w, dst_h = dst_size
            xy_offset_x, xy_offset_y = 0, 0

        # Build sampling grid
        x_coords, y_coords, u_coords, v_coords = self.map_tensor.T
        x_int = torch.floor(x_coords).long() - xy_offset_x
        y_int = torch.floor(y_coords).long() - xy_offset_y

        # Filter valid coordinates
        valid = (x_int >= 0) & (x_int < dst_w) & (y_int >= 0) & (y_int < dst_h)
        if src_rect is not None:
            uv_valid = (
                (u_coords >= uv_offset_x)
                & (u_coords < uv_offset_x + uv_w)
                & (v_coords >= uv_offset_y)
                & (v_coords < uv_offset_y + uv_h)
            )
            valid = valid & uv_valid

        x_valid, y_valid = x_int[valid], y_int[valid]
        u_valid, v_valid = u_coords[valid], v_coords[valid]

        # Create grid buffers
        grid_uv = torch.zeros(
            (1, 2, dst_h, dst_w), device=self.device, dtype=torch.float32
        )
        grid_count = torch.zeros(
            (1, 1, dst_h, dst_w), device=self.device, dtype=torch.float32
        )

        # Scatter UV coordinates
        flat_indices = y_valid * dst_w + x_valid
        if len(flat_indices) > 0:
            uv_stack = torch.stack([u_valid, v_valid], dim=0)
            grid_flat = grid_uv.view(1, 2, -1)
            count_flat = grid_count.view(1, 1, -1)

            grid_flat.index_add_(2, flat_indices, uv_stack.unsqueeze(0))
            ones = torch.ones(1, 1, len(flat_indices), device=self.device)
            count_flat.index_add_(2, flat_indices, ones)

            grid_uv = grid_flat.view(1, 2, dst_h, dst_w)
            grid_count = count_flat.view(1, 1, dst_h, dst_w)

        # Average overlapping points and mark invalid pixels
        valid_mask = grid_count > 0
        grid_uv = torch.where(valid_mask, grid_uv / (grid_count + 1e-8), grid_uv)

        INVALID_COORD = -1e6
        grid_uv = torch.where(
            valid_mask, grid_uv, torch.full_like(grid_uv, INVALID_COORD)
        )

        # Inpaint grid
        if inpaint != InpaintMethod.NONE and inpaint_iter > 0:
            original_valid = valid_mask.clone()
            grid_uv_for_inpaint = torch.where(
                valid_mask, grid_uv, torch.zeros_like(grid_uv)
            )
            grid_uv_inpainted = self._apply_inpaint_conv(
                grid_uv_for_inpaint, grid_count, inpaint_iter
            )
            inpaint_filled = grid_uv_inpainted.abs().sum(dim=1, keepdim=True) > 1e-6
            grid_uv = torch.where(inpaint_filled, grid_uv_inpainted, grid_uv)
            valid_mask_after = original_valid | inpaint_filled
            grid_uv = torch.where(
                valid_mask_after, grid_uv, torch.full_like(grid_uv, INVALID_COORD)
            )
        else:
            valid_mask_after = valid_mask

        # Convert to pixel coordinates
        sample_x = grid_uv[:, 0:1, :, :] - uv_offset_x
        sample_y = grid_uv[:, 1:2, :, :] - uv_offset_y

        # Filter out-of-bounds
        in_bounds = (
            (sample_x >= 0) & (sample_x < W_uv) & (sample_y >= 0) & (sample_y < H_uv)
        )
        valid_mask_after = valid_mask_after & in_bounds
        sample_x = torch.where(
            in_bounds, sample_x, torch.full_like(sample_x, INVALID_COORD)
        )
        sample_y = torch.where(
            in_bounds, sample_y, torch.full_like(sample_y, INVALID_COORD)
        )

        # Normalize to [-1, 1]
        norm_x = 2.0 * sample_x / max(W_uv - 1, 1) - 1.0
        norm_y = 2.0 * sample_y / max(H_uv - 1, 1) - 1.0
        grid = torch.cat([norm_x, norm_y], dim=1).permute(0, 2, 3, 1)
        grid_batch = grid.expand(B, -1, -1, -1)

        # Sample
        out_img = F.grid_sample(
            uv_img,
            grid_batch,
            mode=mode,
            padding_mode=padding_mode.value,
            align_corners=True,
        )

        # Apply mask for ZEROS mode
        if padding_mode == PaddingMode.ZEROS and inpaint == InpaintMethod.NONE:
            out_img = out_img * valid_mask.expand(B, C, -1, -1).float()

        # Remove batch dim
        if not is_batch:
            out_img = out_img.squeeze(0)
            valid_mask_after = valid_mask_after.squeeze(0)

        if return_mask:
            mask_out = (
                valid_mask_after.squeeze(1)
                if valid_mask_after.shape[1] == 1
                else valid_mask_after
            )
            return out_img, mask_out
        return out_img

    def _apply_inpaint_conv(
        self,
        img: torch.Tensor,
        count_img: torch.Tensor,
        iterations: int = 3,
    ) -> torch.Tensor:
        """
        GPU-based inpainting using convolution.

        Fills empty regions with weighted average of neighboring valid pixels.

        Args:
            img: Image tensor (B, C, H, W)
            count_img: Count tensor indicating valid pixels (B, 1, H, W)
            iterations: Number of inpainting iterations

        Returns:
            Inpainted image
        """
        mask = (count_img == 0).float()
        C = img.shape[1]

        # 3x3 kernel (excluding center)
        kernel = torch.tensor(
            [[1, 1, 1], [1, 0, 1], [1, 1, 1]], device=self.device, dtype=torch.float32
        )
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)

        weight_kernel = torch.tensor(
            [[1, 1, 1], [1, 0, 1], [1, 1, 1]], device=self.device, dtype=torch.float32
        ).view(1, 1, 3, 3)

        current_img = img.clone()
        current_valid = (count_img > 0).float()

        for _ in range(iterations):
            is_hole = mask > 0.5

            # Weighted average of neighbors
            neighbor_sum = F.conv2d(
                current_img * current_valid.expand(-1, C, -1, -1),
                kernel,
                padding=1,
                groups=C,
            )
            neighbor_weight = F.conv2d(current_valid, weight_kernel, padding=1).clamp(
                min=1e-8
            )
            neighbor_avg = neighbor_sum / neighbor_weight

            # Update holes
            current_img = torch.where(is_hole, neighbor_avg, current_img)
            current_valid = torch.where(
                is_hole & (neighbor_weight > 1e-6),
                torch.ones_like(current_valid),
                current_valid,
            )

            # Erode mask
            mask = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)

        return current_img

    def get_bounds(self) -> dict:
        """
        Get the coordinate bounds of the pixel map.

        Returns:
            Dictionary with 'xy' and 'uv' bounds as (min_x, min_y, max_x, max_y).
        """
        return {
            "xy": self.xy_bounds,
            "uv": self.uv_bounds,
        }

    def __len__(self) -> int:
        """Return the number of pixel correspondences."""
        return len(self.map_tensor)

    def __repr__(self) -> str:
        return (
            f"PixelMapWarperTorch("
            f"n_points={len(self)}, "
            f"xy_bounds={self.xy_bounds}, "
            f"uv_bounds={self.uv_bounds}, "
            f"device='{self.device}')"
        )


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------


def main():
    """Example demonstrating forward and backward warping."""
    import cv2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create test data
    src_h, src_w = 100, 100
    dst_h, dst_w = 200, 200

    # Source image: color gradient
    src_img_np = np.zeros((src_h, src_w, 3), dtype=np.uint8)
    for y in range(src_h):
        for x in range(src_w):
            src_img_np[y, x] = [x * 2, y * 2, (x + y)]
    src_img_tensor = torch.from_numpy(src_img_np).permute(2, 0, 1)

    # Create map: scale + translation
    # src(x, y) -> dst(x * 1.8 + 20, y * 1.8 + 20)
    scale = 1.8
    map_list = [
        ((x, y), (x * scale + 20, y * scale + 20))
        for y in range(src_h)
        for x in range(src_w)
    ]

    # Initialize warper
    warper = PixelMapWarperTorch(map_list, device=device)
    print(f"Warper: {warper}")

    # Forward warp: XY -> UV
    print("\nForward warping (XY -> UV)...")
    out_forward = warper.forward_warp(
        src_img_tensor,
        dst_size=(dst_w, dst_h),
        aggregation=AggregationMethod.MEAN,
        inpaint=InpaintMethod.CONV,
        inpaint_iter=3,
    )

    # Backward warp: UV -> XY
    print("Backward warping (UV -> XY)...")
    out_backward, mask = warper.backward_warp(
        out_forward,
        dst_size=(src_w, src_h),
        padding_mode=PaddingMode.ZEROS,
        inpaint=InpaintMethod.CONV,
        inpaint_iter=5,
        return_mask=True,
    )

    # Save results
    cv2.imwrite("forward_warp.png", out_forward.permute(1, 2, 0).byte().cpu().numpy())
    cv2.imwrite("backward_warp.png", out_backward.permute(1, 2, 0).byte().cpu().numpy())
    cv2.imwrite(
        "backward_mask.png", (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
    )

    print("\nSaved: forward_warp.png, backward_warp.png, backward_mask.png")

    # Example with src_rect
    print("\nBackward warping with src_rect...")
    crop_u, crop_v, crop_w, crop_h = 50, 50, 100, 100
    uv_cropped = out_forward[:, crop_v : crop_v + crop_h, crop_u : crop_u + crop_w]

    out_cropped = warper.backward_warp(
        uv_cropped,
        dst_size=(src_w, src_h),
        src_rect=(crop_u, crop_v, crop_w, crop_h),
        padding_mode=PaddingMode.ZEROS,
        inpaint=InpaintMethod.CONV,
        inpaint_iter=5,
    )

    cv2.imwrite(
        "backward_cropped.png", out_cropped.permute(1, 2, 0).byte().cpu().numpy()
    )
    print("Saved: backward_cropped.png")


if __name__ == "__main__":
    main()
