#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Automatic Couinaud-like liver segmentation module.

Input:
- pv_mask: numpy array, shape (D, H, W), portal vein mask
- liver_mask: numpy array, shape (D, H, W), liver mask
- spacing: tuple (x, y, z) in mm

Output:
{
    "segment_map": np.ndarray,      # shape (D, H, W), voxel value in [1..8]
    "segment_volumes": {
        "segment_1": float,
        ...
        "segment_8": float
    },
    "segment_centroids": {
        "segment_1": (x, y, z),
        ...
        "segment_8": (x, y, z)
    }
}
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import ndimage


class LiverSegmenter:
    """Portal-vein guided Couinaud-like partition on a liver mask."""

    # code = lr*4 + ap*2 + si
    # lr: 0=left, 1=right
    # ap: 0=posterior, 1=anterior
    # si: 0=inferior, 1=superior
    _CODE_TO_SEGMENT = np.array([3, 1, 4, 2, 6, 7, 5, 8], dtype=np.uint8)

    def _to_bool_3d(self, mask: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(mask)
        if arr.ndim != 3:
            raise ValueError(f"{name} must be a 3D array, got ndim={arr.ndim}")
        return arr.astype(bool)

    def _normalize_spacing(self, spacing: Tuple[float, float, float]) -> Tuple[float, float, float]:
        try:
            sx = float(spacing[0])
            sy = float(spacing[1])
            sz = float(spacing[2])
            return (
                sx if sx > 0 else 1.0,
                sy if sy > 0 else 1.0,
                sz if sz > 0 else 1.0,
            )
        except Exception:
            return (1.0, 1.0, 1.0)

    def _build_empty_result(self, shape: Tuple[int, int, int]) -> Dict[str, object]:
        segment_map = np.zeros(shape, dtype=np.uint8)
        segment_volumes = {f"segment_{i}": 0.0 for i in range(1, 9)}
        segment_centroids = {f"segment_{i}": (float("nan"), float("nan"), float("nan")) for i in range(1, 9)}
        return {
            "segment_map": segment_map,
            "segment_volumes": segment_volumes,
            "segment_centroids": segment_centroids,
        }

    def _liver_bbox(self, liver_mask: np.ndarray, pad: int = 2) -> Tuple[slice, slice, slice]:
        coords = np.argwhere(liver_mask)
        zmin, ymin, xmin = coords.min(axis=0)
        zmax, ymax, xmax = coords.max(axis=0)
        z0 = max(0, int(zmin) - pad)
        y0 = max(0, int(ymin) - pad)
        x0 = max(0, int(xmin) - pad)
        z1 = min(liver_mask.shape[0], int(zmax) + pad + 1)
        y1 = min(liver_mask.shape[1], int(ymax) + pad + 1)
        x1 = min(liver_mask.shape[2], int(xmax) + pad + 1)
        return slice(z0, z1), slice(y0, y1), slice(x0, x1)

    def _liver_midpoint(self, liver_mask: np.ndarray) -> Tuple[float, float, float]:
        coords = np.argwhere(liver_mask)
        zmin, ymin, xmin = coords.min(axis=0).astype(float)
        zmax, ymax, xmax = coords.max(axis=0).astype(float)
        return (0.5 * (zmin + zmax), 0.5 * (ymin + ymax), 0.5 * (xmin + xmax))

    def _geometric_partition_roi(
        self,
        liver_roi: np.ndarray,
        roi_origin_zyx: Tuple[int, int, int],
        liver_mid_zyx: Tuple[float, float, float],
    ) -> np.ndarray:
        z0, y0, x0 = roi_origin_zyx
        z_mid, y_mid, x_mid = liver_mid_zyx

        z_idx, y_idx, x_idx = np.indices(liver_roi.shape)
        z_glb = z_idx + z0
        y_glb = y_idx + y0
        x_glb = x_idx + x0

        lr = (x_glb >= x_mid).astype(np.uint8)
        ap = (y_glb >= y_mid).astype(np.uint8)
        si = (z_glb < z_mid).astype(np.uint8)
        code = lr * 4 + ap * 2 + si

        seg = self._CODE_TO_SEGMENT[code]
        seg = seg.astype(np.uint8, copy=False)
        seg[~liver_roi] = 0
        return seg

    def _extract_pv_components(self, pv_roi: np.ndarray, liver_roi: np.ndarray) -> np.ndarray:
        # Prefer components in/near liver; include one-voxel liver dilation to keep hilum continuity.
        structure = ndimage.generate_binary_structure(3, 2)
        liver_near = ndimage.binary_dilation(liver_roi, structure=structure, iterations=1)
        pv_valid = pv_roi & liver_near

        if not np.any(pv_valid):
            return np.zeros_like(pv_roi, dtype=np.int32)

        labels, n_comp = ndimage.label(pv_valid, structure=structure)
        if n_comp <= 0:
            return np.zeros_like(pv_roi, dtype=np.int32)

        counts = np.bincount(labels.ravel())
        if counts.size <= 1:
            return np.zeros_like(pv_roi, dtype=np.int32)

        counts[0] = 0
        total = int(np.sum(counts))
        min_size = max(8, int(total * 0.001))
        keep = np.where(counts >= min_size)[0]
        keep = keep[keep > 0]

        # Always keep at least the largest component.
        if keep.size == 0:
            keep = np.array([int(np.argmax(counts))], dtype=np.int32)

        # Keep a bounded number of components to avoid noisy micro-seeds.
        if keep.size > 32:
            order = np.argsort(counts[keep])[::-1]
            keep = keep[order[:32]]

        remap = np.zeros(counts.shape[0], dtype=np.int32)
        remap[keep] = np.arange(1, keep.size + 1, dtype=np.int32)
        return remap[labels]

    def _map_branches_to_segments(
        self, branch_labels: np.ndarray, liver_mid_zyx: Tuple[float, float, float]
    ) -> Dict[int, int]:
        branch_ids = np.unique(branch_labels)
        branch_ids = branch_ids[branch_ids > 0]
        if branch_ids.size == 0:
            return {}

        z_mid, y_mid, x_mid = liver_mid_zyx
        ones = np.ones_like(branch_labels, dtype=np.uint8)
        centroids = ndimage.center_of_mass(ones, labels=branch_labels, index=branch_ids.tolist())

        branch_to_segment: Dict[int, int] = {}
        for bid, c in zip(branch_ids.tolist(), centroids):
            if c is None or any(np.isnan(v) for v in c):
                continue
            cz, cy, cx = float(c[0]), float(c[1]), float(c[2])
            lr = 1 if cx >= x_mid else 0
            ap = 1 if cy >= y_mid else 0
            si = 1 if cz < z_mid else 0
            code = lr * 4 + ap * 2 + si
            branch_to_segment[int(bid)] = int(self._CODE_TO_SEGMENT[code])

        return branch_to_segment

    def _voronoi_assign_segments(
        self,
        branch_labels: np.ndarray,
        branch_to_segment: Dict[int, int],
        liver_roi: np.ndarray,
        spacing_xyz: Tuple[float, float, float],
        geometric_fallback_roi: np.ndarray,
    ) -> np.ndarray:
        if not np.any(branch_labels > 0) or not branch_to_segment:
            return geometric_fallback_roi.copy()

        sx, sy, sz = spacing_xyz
        # distance_transform_edt expects axis order of the input array: (z, y, x)
        sampling_zyx = (sz, sy, sx)
        background = branch_labels == 0
        _, indices = ndimage.distance_transform_edt(background, sampling=sampling_zyx, return_indices=True)
        nearest_branch = branch_labels[tuple(indices)]

        max_branch_id = int(np.max(branch_labels))
        lookup = np.zeros(max_branch_id + 1, dtype=np.uint8)
        for bid, seg_id in branch_to_segment.items():
            if 0 < bid <= max_branch_id:
                lookup[bid] = np.uint8(seg_id)

        segment_roi = lookup[np.clip(nearest_branch, 0, max_branch_id)]
        segment_roi = segment_roi.astype(np.uint8, copy=False)
        segment_roi[~liver_roi] = 0

        # If any liver voxels are still unlabeled, fill with geometric partition.
        missing = liver_roi & (segment_roi == 0)
        if np.any(missing):
            segment_roi[missing] = geometric_fallback_roi[missing]
        return segment_roi

    def _compute_volumes(
        self, segment_map: np.ndarray, spacing_xyz: Tuple[float, float, float]
    ) -> Dict[str, float]:
        sx, sy, sz = spacing_xyz
        voxel_volume_ml = (sx * sy * sz) / 1000.0
        volumes: Dict[str, float] = {}
        for i in range(1, 9):
            volumes[f"segment_{i}"] = float(np.sum(segment_map == i)) * voxel_volume_ml
        return volumes

    def _compute_centroids(self, segment_map: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
        centroids: Dict[str, Tuple[float, float, float]] = {}
        for i in range(1, 9):
            coords = np.argwhere(segment_map == i)
            if coords.size == 0:
                centroids[f"segment_{i}"] = (float("nan"), float("nan"), float("nan"))
                continue
            cz, cy, cx = np.mean(coords, axis=0)
            centroids[f"segment_{i}"] = (float(cx), float(cy), float(cz))
        return centroids

    def segment(
        self,
        pv_mask: np.ndarray,
        liver_mask: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Dict[str, object]:
        pv = self._to_bool_3d(pv_mask, "pv_mask")
        liver = self._to_bool_3d(liver_mask, "liver_mask")
        if pv.shape != liver.shape:
            raise ValueError("pv_mask and liver_mask must have identical shape")

        spacing_xyz = self._normalize_spacing(spacing)
        if not np.any(liver):
            return self._build_empty_result(liver.shape)

        liver_mid_zyx = self._liver_midpoint(liver)
        z_sl, y_sl, x_sl = self._liver_bbox(liver, pad=2)
        roi_origin = (z_sl.start, y_sl.start, x_sl.start)

        liver_roi = liver[z_sl, y_sl, x_sl]
        pv_roi = pv[z_sl, y_sl, x_sl]

        geometric_roi = self._geometric_partition_roi(
            liver_roi=liver_roi,
            roi_origin_zyx=roi_origin,
            liver_mid_zyx=liver_mid_zyx,
        )

        branch_labels = self._extract_pv_components(pv_roi=pv_roi, liver_roi=liver_roi)
        branch_to_segment = self._map_branches_to_segments(
            branch_labels=branch_labels,
            liver_mid_zyx=(liver_mid_zyx[0] - roi_origin[0], liver_mid_zyx[1] - roi_origin[1], liver_mid_zyx[2] - roi_origin[2]),
        )
        # _map_branches_to_segments expects coordinates in the same reference as branch_labels (ROI space)
        # The conversion above keeps branch/liver comparison consistent.

        segment_roi = self._voronoi_assign_segments(
            branch_labels=branch_labels,
            branch_to_segment=branch_to_segment,
            liver_roi=liver_roi,
            spacing_xyz=spacing_xyz,
            geometric_fallback_roi=geometric_roi,
        )

        segment_map = np.zeros_like(liver, dtype=np.uint8)
        segment_map[z_sl, y_sl, x_sl] = segment_roi
        segment_map[~liver] = 0

        return {
            "segment_map": segment_map,
            "segment_volumes": self._compute_volumes(segment_map, spacing_xyz),
            "segment_centroids": self._compute_centroids(segment_map),
        }


def run_liver_segmentation(
    pv_mask: np.ndarray,
    liver_mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, object]:
    """Convenience wrapper for direct module-level usage."""
    return LiverSegmenter().segment(pv_mask=pv_mask, liver_mask=liver_mask, spacing=spacing)


__all__ = ["LiverSegmenter", "run_liver_segmentation"]
