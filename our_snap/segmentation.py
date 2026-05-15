#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seed based 3D vessel segmentation.

The public seed coordinate convention is (x, y, z).  The loaded image arrays
are stored as (Y, X, Z), so every array access must use image[y, x, z].
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.ndimage import sobel

try:
    from skimage.filters import frangi, sato
except Exception:
    frangi = None
    sato = None


class ActiveContourSegmentation:
    """Seed-constrained vessel segmentation used by the manual Segment 3D tool."""

    def __init__(self):
        self.image_data: Optional[np.ndarray] = None
        self.image_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
        self.seed_points: List[Tuple[int, int, int]] = []
        self.current_mask: Optional[np.ndarray] = None
        self.iteration_count = 0
        self.last_postprocess_info: Dict[str, float] = {}

        self.max_growth_radius_mm = self._env_float(
            "VESSELSEG_3D_MAX_GROWTH_RADIUS_MM", 45.0, 8.0, 180.0
        )
        self.seed_radius_mm = self._env_float(
            "VESSELSEG_3D_SEED_RADIUS_MM", 2.5, 0.8, 8.0
        )
        self.min_component_voxels = int(
            self._env_float("VESSELSEG_3D_MIN_COMPONENT_VOXELS", 18.0, 1.0, 2000.0)
        )
        self.max_vessel_radius_mm = self._env_float(
            "VESSELSEG_3D_MAX_VESSEL_RADIUS_MM", 5.0, 1.0, 18.0
        )
        self.seed_snap_radius_mm = self._env_float(
            "VESSELSEG_3D_SEED_SNAP_RADIUS_MM", 10.0, 0.0, 35.0
        )
        self.vesselness_percentile = self._env_float(
            "VESSELSEG_3D_VESSELNESS_PERCENTILE", 82.0, 50.0, 98.0
        )

    @staticmethod
    def _env_float(name: str, default: float, min_value: float, max_value: float) -> float:
        try:
            value = float(os.environ.get(name, default))
        except Exception:
            value = default
        return float(min(max(value, min_value), max_value))

    def set_image(self, image_data: np.ndarray, spacing: Optional[Tuple[float, float, float]] = None):
        """Set image data.  image_data shape is (Y, X, Z)."""
        self.image_data = image_data
        self.image_spacing = self._normalize_spacing(spacing)
        self.current_mask = np.zeros(image_data.shape, dtype=np.uint8)
        self.last_postprocess_info = {}

    def set_seed_points(self, seed_points: List[Tuple[int, int, int]]):
        """Set seed points in public voxel coordinates: (x, y, z)."""
        self.seed_points = self._valid_seed_points(seed_points)
        self._initialize_mask_from_seeds()

    @staticmethod
    def _normalize_spacing(spacing: Optional[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        if spacing is None or len(spacing) < 3:
            return (1.0, 1.0, 1.0)
        values = []
        for value in spacing[:3]:
            try:
                parsed = float(value)
            except Exception:
                parsed = 1.0
            values.append(parsed if parsed > 0 else 1.0)
        return (values[0], values[1], values[2])

    def _axis_spacing_yxz(self) -> Tuple[float, float, float]:
        """Return spacing in array-axis order (Y, X, Z)."""
        sx, sy, sz = self.image_spacing
        return (sy, sx, sz)

    def _valid_seed_points(
        self, seed_points: Optional[List[Tuple[int, int, int]]] = None
    ) -> List[Tuple[int, int, int]]:
        if self.image_data is None or not seed_points:
            return []
        h, w, d = self.image_data.shape
        valid: List[Tuple[int, int, int]] = []
        for seed in seed_points:
            try:
                x, y, z = int(seed[0]), int(seed[1]), int(seed[2])
            except Exception:
                continue
            if 0 <= x < w and 0 <= y < h and 0 <= z < d:
                valid.append((x, y, z))
        return list(dict.fromkeys(valid))

    def _seed_indices_yxz(self) -> List[Tuple[int, int, int]]:
        return [(y, x, z) for x, y, z in self.seed_points]

    def _ball_offsets(self, radius_mm: float) -> np.ndarray:
        sy, sx, sz = self._axis_spacing_yxz()
        ry = max(1, int(np.ceil(radius_mm / max(sy, 1e-6))))
        rx = max(1, int(np.ceil(radius_mm / max(sx, 1e-6))))
        rz = max(1, int(np.ceil(radius_mm / max(sz, 1e-6))))
        yy, xx, zz = np.ogrid[-ry : ry + 1, -rx : rx + 1, -rz : rz + 1]
        dist = (yy * sy) ** 2 + (xx * sx) ** 2 + (zz * sz) ** 2
        return np.argwhere(dist <= radius_mm * radius_mm) - np.array([ry, rx, rz])

    def _paint_seed_ball_yxz(
        self,
        target: np.ndarray,
        centers_yxz: List[Tuple[int, int, int]],
        radius_mm: float,
    ):
        offsets = self._ball_offsets(radius_mm)
        h, w, d = target.shape
        for cy, cx, cz in centers_yxz:
            coords = offsets + np.array([cy, cx, cz])
            inside = (
                (coords[:, 0] >= 0)
                & (coords[:, 0] < h)
                & (coords[:, 1] >= 0)
                & (coords[:, 1] < w)
                & (coords[:, 2] >= 0)
                & (coords[:, 2] < d)
            )
            coords = coords[inside]
            target[coords[:, 0], coords[:, 1], coords[:, 2]] = True

    def _paint_seed_ball(self, target: np.ndarray, radius_mm: float):
        self._paint_seed_ball_yxz(target, self._seed_indices_yxz(), radius_mm)

    def _initialize_mask_from_seeds(self):
        if self.image_data is None:
            return
        seed_core = np.zeros(self.image_data.shape, dtype=bool)
        if self.seed_points:
            self._paint_seed_ball(seed_core, self.seed_radius_mm)
        self.current_mask = seed_core.astype(np.uint8)

    def _estimate_intensity_window(self, seed_core: np.ndarray) -> Tuple[float, float, float]:
        image = np.asarray(self.image_data, dtype=np.float32)
        finite = np.isfinite(image)
        center_values = []
        for y, x, z in self._seed_indices_yxz():
            value = image[y, x, z]
            if np.isfinite(value):
                center_values.append(float(value))
        center_values = np.asarray(center_values, dtype=np.float32)

        local_values = image[seed_core & finite]
        if center_values.size == 0 and local_values.size == 0:
            values = image[finite]
            if values.size == 0:
                return -np.inf, np.inf, 0.0
            low, high = np.percentile(values, [65.0, 99.5])
            return float(low), float(high), float(np.median(values))

        if center_values.size > 0:
            median = float(np.median(center_values))
            if local_values.size > 0:
                local_median = float(np.median(local_values))
                if median > local_median + 20.0:
                    local_values = local_values[local_values >= max(median - 80.0, local_median)]
                seed_values = local_values if local_values.size >= center_values.size else center_values
            else:
                seed_values = center_values
        else:
            seed_values = local_values
            median = float(np.median(seed_values))

        p10, p90 = np.percentile(seed_values, [10.0, 90.0])
        mad = float(np.median(np.abs(seed_values - median)))
        robust_sigma = max(1.4826 * mad, float(p90 - p10) / 2.0, 12.0)

        low = median - max(35.0, 2.8 * robust_sigma)
        high = median + max(70.0, 4.0 * robust_sigma)

        all_values = image[finite]
        if all_values.size:
            global_p1, global_p99 = np.percentile(all_values, [1.0, 99.0])
            ct_like = global_p1 < -100.0 or global_p99 > 180.0 or median > 60.0
            if ct_like and median > 60.0:
                low = max(low, median - 80.0, 45.0)
                high = min(max(high, median + 120.0), 520.0)

        if high <= low:
            low = median - 40.0
            high = median + 80.0
        return float(low), float(high), median

    def _seed_roi_slices(self, radius_mm: float) -> Tuple[slice, slice, slice]:
        if self.image_data is None or not self.seed_points:
            return (slice(0, 0), slice(0, 0), slice(0, 0))

        h, w, d = self.image_data.shape
        sy, sx, sz = self._axis_spacing_yxz()
        pad = np.array(
            [
                max(2, int(np.ceil(radius_mm / max(sy, 1e-6)))),
                max(2, int(np.ceil(radius_mm / max(sx, 1e-6)))),
                max(1, int(np.ceil(radius_mm / max(sz, 1e-6)))),
            ],
            dtype=int,
        )
        centers = np.asarray(self._seed_indices_yxz(), dtype=int)
        mins = np.maximum(centers.min(axis=0) - pad, 0)
        maxs = np.minimum(centers.max(axis=0) + pad + 1, np.array([h, w, d], dtype=int))
        return (
            slice(int(mins[0]), int(maxs[0])),
            slice(int(mins[1]), int(maxs[1])),
            slice(int(mins[2]), int(maxs[2])),
        )

    def _compute_vesselness_roi(self, roi: np.ndarray) -> np.ndarray:
        arr = np.asarray(roi, dtype=np.float32)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return np.zeros_like(arr, dtype=np.float32)

        values = arr[finite]
        lo, hi = np.percentile(values, [1.0, 99.5])
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
        norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        norm[~finite] = 0.0

        if frangi is not None or sato is not None:
            responses = []
            sigmas = (0.8, 1.2, 1.8, 2.5, 3.2)
            if frangi is not None:
                try:
                    responses.append(
                        frangi(norm, sigmas=sigmas, black_ridges=False).astype(np.float32)
                    )
                except Exception as e:
                    print(f"frangi vesselness failed, using fallback: {e}")
            if sato is not None:
                try:
                    responses.append(
                        sato(norm, sigmas=sigmas, black_ridges=False).astype(np.float32)
                    )
                except Exception as e:
                    print(f"sato vesselness failed, using fallback: {e}")
            if responses:
                vesselness = np.maximum.reduce(responses)
                vesselness = np.nan_to_num(vesselness, nan=0.0, posinf=0.0, neginf=0.0)
                positive = vesselness[vesselness > 0]
                if positive.size > 0:
                    scale = float(np.percentile(positive, 99.5))
                    if scale > 0:
                        vesselness = np.clip(vesselness / scale, 0.0, 1.0)
                return vesselness.astype(np.float32)

        # Fallback: bright ridges with strong local contrast.
        smooth = ndimage.gaussian_filter(norm, sigma=1.0)
        local_mean = ndimage.uniform_filter(smooth, size=7)
        ridge = np.clip(smooth - local_mean, 0.0, 1.0)
        positive = ridge[ridge > 0]
        if positive.size > 0:
            ridge = np.clip(ridge / max(float(np.percentile(positive, 99.0)), 1e-6), 0.0, 1.0)
        return ridge.astype(np.float32)

    def _remove_small_components(self, mask: np.ndarray, min_voxels: int = 3) -> np.ndarray:
        if mask is None or not np.any(mask):
            return np.zeros_like(mask, dtype=bool)
        labeled, n_comp = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
        if n_comp == 0:
            return np.zeros_like(mask, dtype=bool)
        sizes = np.bincount(labeled.ravel())
        keep = np.where(sizes >= int(min_voxels))[0]
        keep = keep[keep != 0]
        if keep.size == 0:
            return np.zeros_like(mask, dtype=bool)
        return np.isin(labeled, keep)

    def _snap_seeds_to_centerline(
        self,
        centerline_roi: np.ndarray,
        roi_slices: Tuple[slice, slice, slice],
    ) -> List[Tuple[int, int, int]]:
        if not np.any(centerline_roi):
            return []

        sy, sx, sz = self._axis_spacing_yxz()
        dist, indices = ndimage.distance_transform_edt(
            ~centerline_roi,
            sampling=(sy, sx, sz),
            return_indices=True,
        )
        snapped: List[Tuple[int, int, int]] = []
        y0, x0, z0 = roi_slices[0].start, roi_slices[1].start, roi_slices[2].start
        rh, rw, rd = centerline_roi.shape
        for y, x, z in self._seed_indices_yxz():
            ly, lx, lz = int(y - y0), int(x - x0), int(z - z0)
            if not (0 <= ly < rh and 0 <= lx < rw and 0 <= lz < rd):
                continue
            nearest_mm = float(dist[ly, lx, lz])
            if nearest_mm <= self.seed_snap_radius_mm:
                ny = int(indices[0, ly, lx, lz] + y0)
                nx = int(indices[1, ly, lx, lz] + x0)
                nz = int(indices[2, ly, lx, lz] + z0)
                snapped.append((ny, nx, nz))

        return list(dict.fromkeys(snapped))

    def _candidate_intensity_mask(
        self,
        roi: np.ndarray,
        vesselness: np.ndarray,
        seed_core_roi: np.ndarray,
    ) -> Tuple[np.ndarray, float, float, float]:
        arr = np.asarray(roi, dtype=np.float32)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return np.zeros_like(seed_core_roi, dtype=bool), -np.inf, np.inf, 0.0

        values = arr[finite]
        base_low = float(np.percentile(values, 45.0))
        base_high = float(np.percentile(values, 99.7))

        center_values = []
        for y, x, z in self._seed_indices_yxz():
            value = self.image_data[y, x, z]
            if np.isfinite(value):
                center_values.append(float(value))
        seed_median = float(np.median(center_values)) if center_values else float(np.median(values))

        if seed_median > float(np.percentile(values, 60.0)):
            low = max(base_low, seed_median - 90.0)
            high = min(max(base_high, seed_median + 160.0), seed_median + 260.0)
        else:
            vessel_values = arr[(vesselness >= 0.25) & finite]
            if vessel_values.size >= 8:
                v_med = float(np.median(vessel_values))
                low = max(base_low, v_med - 110.0)
                high = max(base_high, v_med + 160.0)
                seed_median = v_med
            else:
                low = base_low
                high = base_high

        if high <= low:
            high = low + 1.0
        mask = finite & (arr >= low) & (arr <= high)
        mask |= seed_core_roi
        return mask, float(low), float(high), float(seed_median)

    def _build_constraints(self) -> Dict[str, np.ndarray]:
        if self.image_data is None:
            raise RuntimeError("image data is not set")

        image = np.asarray(self.image_data, dtype=np.float32)
        original_seed_core = np.zeros(image.shape, dtype=bool)
        self._paint_seed_ball(original_seed_core, self.seed_radius_mm)
        if not np.any(original_seed_core):
            return {
                "seed_core": original_seed_core,
                "allowed": np.zeros_like(original_seed_core),
                "connected": np.zeros_like(original_seed_core),
                "radius_mask": np.zeros_like(original_seed_core),
                "centerline": np.zeros_like(original_seed_core),
            }

        roi_slices = self._seed_roi_slices(self.max_growth_radius_mm + self.seed_snap_radius_mm)
        roi = image[roi_slices]
        vesselness_roi = self._compute_vesselness_roi(roi)
        original_seed_core_roi = original_seed_core[roi_slices]
        intensity_roi, low, high, seed_median = self._candidate_intensity_mask(
            roi,
            vesselness_roi,
            original_seed_core_roi,
        )

        positive = vesselness_roi[vesselness_roi > 0]
        if positive.size > 0:
            vessel_thr = max(0.12, float(np.percentile(positive, self.vesselness_percentile)))
        else:
            vessel_thr = 1.0
        centerline_roi = (vesselness_roi >= vessel_thr) & intensity_roi
        centerline_roi = self._remove_small_components(centerline_roi, min_voxels=3)

        snapped_seed_points = self._snap_seeds_to_centerline(centerline_roi, roi_slices)
        seed_core = np.zeros(image.shape, dtype=bool)
        if snapped_seed_points:
            self._paint_seed_ball_yxz(seed_core, snapped_seed_points, min(self.seed_radius_mm, 1.4))
        else:
            self.last_postprocess_info = {
                "intensity_low": float(low),
                "intensity_high": float(high),
                "seed_median": float(seed_median),
                "vesselness_threshold": float(vessel_thr),
                "snapped_seed_count": 0.0,
            }
            return {
                "seed_core": seed_core,
                "allowed": np.zeros_like(seed_core),
                "connected": np.zeros_like(seed_core),
                "radius_mask": np.zeros_like(seed_core),
                "centerline": np.zeros_like(seed_core),
            }

        centerline = np.zeros_like(seed_core, dtype=bool)
        centerline[roi_slices] = centerline_roi

        sy, sx, sz = self._axis_spacing_yxz()
        distance_mm = ndimage.distance_transform_edt(~seed_core, sampling=(sy, sx, sz))
        radius_mask = distance_mm <= float(self.max_growth_radius_mm)

        center_dist_mm = ndimage.distance_transform_edt(~centerline, sampling=(sy, sx, sz))
        vessel_radius_mask = center_dist_mm <= float(self.max_vessel_radius_mm)

        intensity_mask = np.zeros_like(seed_core, dtype=bool)
        intensity_mask[roi_slices] = intensity_roi
        allowed = intensity_mask & vessel_radius_mask & radius_mask
        allowed |= seed_core
        structure = ndimage.generate_binary_structure(3, 1)
        connected = ndimage.binary_propagation(seed_core, structure=structure, mask=allowed)

        self.last_postprocess_info = {
            "intensity_low": float(low),
            "intensity_high": float(high),
            "seed_median": float(seed_median),
            "max_growth_radius_mm": float(self.max_growth_radius_mm),
            "max_vessel_radius_mm": float(self.max_vessel_radius_mm),
            "vesselness_threshold": float(vessel_thr),
            "snapped_seed_count": float(len(snapped_seed_points)),
            "centerline_voxels": float(np.count_nonzero(centerline)),
            "allowed_voxels": float(np.count_nonzero(allowed)),
            "connected_voxels": float(np.count_nonzero(connected)),
        }
        return {
            "seed_core": seed_core,
            "allowed": allowed,
            "connected": connected,
            "radius_mask": radius_mask,
            "centerline": centerline,
        }

    def _keep_seed_connected_components(self, mask: np.ndarray, seed_core: np.ndarray) -> np.ndarray:
        if mask is None or not np.any(mask):
            return np.zeros_like(seed_core, dtype=bool)
        structure = ndimage.generate_binary_structure(3, 1)
        labeled, n_comp = ndimage.label(mask, structure=structure)
        if n_comp == 0:
            return np.zeros_like(seed_core, dtype=bool)
        seed_labels = np.unique(labeled[seed_core])
        seed_labels = seed_labels[seed_labels > 0]
        if seed_labels.size == 0:
            return np.zeros_like(seed_core, dtype=bool)

        sizes = np.bincount(labeled.ravel())
        keep_labels = []
        for cid in seed_labels:
            if int(sizes[int(cid)]) >= self.min_component_voxels:
                keep_labels.append(int(cid))
        if not keep_labels:
            keep_labels = [int(seed_labels[np.argmax(sizes[seed_labels])])]
        return np.isin(labeled, keep_labels)

    def _postprocess_vessel_mask(
        self,
        raw_mask: np.ndarray,
        constraints: Dict[str, np.ndarray],
        context: str,
    ) -> np.ndarray:
        """Apply vessel-specific constraints without dilation or closing."""
        seed_core = constraints["seed_core"]
        connected = constraints["connected"]
        allowed = constraints["allowed"]
        if not np.any(seed_core):
            return np.zeros_like(raw_mask, dtype=np.uint8)

        if not np.any(connected):
            final = seed_core.copy()
        else:
            raw = (raw_mask > 0) & allowed
            raw |= seed_core
            seed_limited = self._keep_seed_connected_components(raw, seed_core)

            connected_voxels = int(np.count_nonzero(connected))
            limited_voxels = int(np.count_nonzero(seed_limited))
            seed_voxels = int(np.count_nonzero(seed_core))

            min_safe = max(seed_voxels * 3, int(0.35 * max(connected_voxels, 1)))
            if limited_voxels < min_safe:
                final = connected
                self.last_postprocess_info[f"{context}_volume_guard"] = 1.0
            else:
                final = seed_limited & connected
                self.last_postprocess_info[f"{context}_volume_guard"] = 0.0

        final = self._keep_seed_connected_components(final, seed_core)
        final = self._filter_bulky_components(final, constraints.get("centerline"))
        seed_voxels = int(np.count_nonzero(seed_core))
        if int(np.count_nonzero(final)) <= max(self.min_component_voxels, seed_voxels * 2):
            final = np.zeros_like(raw_mask, dtype=bool)

        self.last_postprocess_info[f"{context}_final_voxels"] = float(np.count_nonzero(final))
        return final.astype(np.uint8)

    def _filter_bulky_components(
        self,
        mask: np.ndarray,
        centerline: Optional[np.ndarray],
    ) -> np.ndarray:
        if mask is None or not np.any(mask):
            return np.zeros_like(mask, dtype=bool)

        structure = ndimage.generate_binary_structure(3, 1)
        labeled, n_comp = ndimage.label(mask, structure=structure)
        if n_comp == 0:
            return np.zeros_like(mask, dtype=bool)

        sy, sx, sz = self._axis_spacing_yxz()
        centerline = centerline if centerline is not None else np.zeros_like(mask, dtype=bool)
        out = np.zeros_like(mask, dtype=bool)
        kept = 0
        rejected = 0
        sizes = np.bincount(labeled.ravel())
        for cid in range(1, n_comp + 1):
            size = int(sizes[cid]) if cid < sizes.size else 0
            if size <= 0:
                continue
            comp = labeled == cid
            if size < 80:
                out |= comp
                kept += 1
                continue

            coords = np.argwhere(comp).astype(np.float64)
            coords[:, 0] *= sy
            coords[:, 1] *= sx
            coords[:, 2] *= sz
            if coords.shape[0] < 4:
                out |= comp
                kept += 1
                continue

            centered = coords - coords.mean(axis=0, keepdims=True)
            cov = np.cov(centered, rowvar=False)
            eig = np.sort(np.linalg.eigvalsh(cov))
            eig = np.maximum(eig, 1e-6)
            elongation = float(np.sqrt(eig[-1] / eig[0]))
            line_fraction = float(np.count_nonzero(centerline & comp)) / float(max(size, 1))

            if elongation >= 2.0 or (elongation >= 1.45 and line_fraction >= 0.015):
                out |= comp
                kept += 1
            else:
                rejected += 1

        self.last_postprocess_info["bulky_components_kept"] = float(kept)
        self.last_postprocess_info["bulky_components_rejected"] = float(rejected)
        return out

    def _iterate_region_competition(
        self,
        iterations: int,
        smoothness: float,
        expansion_force: float,
        constraints: Dict[str, np.ndarray],
    ) -> np.ndarray:
        image = np.asarray(self.image_data, dtype=np.float32)
        seed_core = constraints["seed_core"]
        connected = constraints["connected"]
        allowed = constraints["allowed"]
        mask = np.maximum(self.current_mask.astype(np.float32), seed_core.astype(np.float32))

        if np.any(connected):
            mask = np.maximum(mask, connected.astype(np.float32) * 0.35)

        grad = np.sqrt(sum(np.square(np.gradient(image, axis=i)) for i in range(3)))
        grad_scale = float(np.percentile(grad[np.isfinite(grad)], 95.0)) if np.any(np.isfinite(grad)) else 1.0
        grad_scale = max(grad_scale, 1e-6)
        edge_stop = np.clip(1.0 - grad / grad_scale, 0.0, 1.0)

        for _ in range(max(1, int(iterations))):
            inside = mask > 0.5
            if not np.any(inside):
                inside = seed_core
            inside_values = image[inside]
            target = float(np.median(inside_values)) if inside_values.size else 0.0
            contrast = np.exp(-np.square(image - target) / (2.0 * max(25.0, abs(target) * 0.12) ** 2))
            region_force = (contrast - 0.45) * float(expansion_force)
            smooth_force = -float(smoothness) * 0.08 * ndimage.laplace(mask)
            total_force = region_force + smooth_force + 0.12 * edge_stop
            mask = np.clip(mask + 0.12 * total_force, 0.0, 1.0)

            binary = (mask > 0.5) & allowed
            binary |= seed_core
            binary = self._keep_seed_connected_components(binary, seed_core)
            mask = binary.astype(np.float32)

        return mask > 0.5

    def region_competition(
        self,
        iterations: int = 100,
        smoothness: float = 1.0,
        expansion_force: float = 1.0,
    ) -> np.ndarray:
        if self.image_data is None:
            return None
        if not self.seed_points:
            self.seed_points = self._valid_seed_points(self.seed_points)
        if not self.seed_points:
            return np.zeros_like(self.image_data, dtype=np.uint8)

        self._initialize_mask_from_seeds()
        constraints = self._build_constraints()
        if not np.any(constraints["connected"]):
            self.current_mask = np.zeros_like(self.image_data, dtype=np.uint8)
            return self.current_mask
        raw = constraints["connected"]
        self.current_mask = self._postprocess_vessel_mask(raw, constraints, context="region")
        self.iteration_count = iterations
        return self.current_mask

    def edge_based(
        self,
        iterations: int = 100,
        smoothness: float = 1.0,
        edge_weight: float = 1.0,
    ) -> np.ndarray:
        if self.image_data is None:
            return None
        if not self.seed_points:
            self.seed_points = self._valid_seed_points(self.seed_points)
        if not self.seed_points:
            return np.zeros_like(self.image_data, dtype=np.uint8)

        self._initialize_mask_from_seeds()
        constraints = self._build_constraints()
        if not np.any(constraints["connected"]):
            self.current_mask = np.zeros_like(self.image_data, dtype=np.uint8)
            return self.current_mask
        self.current_mask = self._postprocess_vessel_mask(
            constraints["connected"], constraints, context="edge"
        )
        self.iteration_count = iterations
        return self.current_mask

    def get_mask(self) -> Optional[np.ndarray]:
        return self.current_mask.copy() if self.current_mask is not None else None
