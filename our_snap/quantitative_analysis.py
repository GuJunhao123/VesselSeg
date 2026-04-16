#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quantitative analysis and risk assessment utilities."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

try:
    from quantification import VesselQuantifier
except Exception:
    VesselQuantifier = None


class QuantitativeAnalyzer:
    """High-level quantitative analyzer used by the UI/report pipeline."""

    def __init__(self):
        """Initialize analyzer."""
        self._vessel_quantifier = VesselQuantifier() if VesselQuantifier is not None else None

    def _normalize_spacing(self, voxel_size: Tuple[float, float, float]) -> Tuple[float, float, float]:
        try:
            sx, sy, sz = float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2])
            return (sx if sx > 0 else 1.0, sy if sy > 0 else 1.0, sz if sz > 0 else 1.0)
        except Exception:
            return (1.0, 1.0, 1.0)

    def _skeletonize_3d(self, mask: np.ndarray) -> np.ndarray:
        from skimage import morphology

        m = np.asarray(mask).astype(bool)
        if not np.any(m):
            return np.zeros_like(m, dtype=bool)
        try:
            return morphology.skeletonize_3d(m).astype(bool)
        except Exception:
            return morphology.skeletonize(m).astype(bool)

    def _compute_skeleton_length(self, skeleton: np.ndarray, voxel_size: Tuple[float, float, float]) -> float:
        """Sum skeleton edge lengths using 13 forward offsets to avoid double counting."""
        if not np.any(skeleton):
            return 0.0

        sx, sy, sz = self._normalize_spacing(voxel_size)
        offsets = [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, -1, 0),
            (1, 0, 1),
            (1, 0, -1),
            (0, 1, 1),
            (0, 1, -1),
            (1, 1, 1),
            (1, 1, -1),
            (1, -1, 1),
            (1, -1, -1),
        ]

        s = skeleton.astype(np.uint8)
        length = 0.0
        for dx, dy, dz in offsets:
            shifted = np.roll(s, shift=(dx, dy, dz), axis=(0, 1, 2))
            valid = np.ones_like(s, dtype=bool)
            if dx > 0:
                valid[:dx, :, :] = False
            elif dx < 0:
                valid[dx:, :, :] = False
            if dy > 0:
                valid[:, :dy, :] = False
            elif dy < 0:
                valid[:, dy:, :] = False
            if dz > 0:
                valid[:, :, :dz] = False
            elif dz < 0:
                valid[:, :, dz:] = False

            edges = (s > 0) & (shifted > 0) & valid
            edge_count = int(np.sum(edges))
            step_len = float(np.sqrt((dx * sx) ** 2 + (dy * sy) ** 2 + (dz * sz) ** 2))
            length += edge_count * step_len

        return float(length)

    def _compute_branch_and_junction(self, skeleton: np.ndarray) -> Tuple[int, int, np.ndarray]:
        if not np.any(skeleton):
            return 0, 0, np.zeros_like(skeleton, dtype=bool)

        structure = ndimage.generate_binary_structure(3, 2)
        neighbors = ndimage.convolve(skeleton.astype(np.uint8), structure, mode="constant")
        degree = neighbors - skeleton.astype(np.uint8)

        endpoint = skeleton & (degree == 1)
        junction = skeleton & (degree >= 3)

        endpoint_count = int(np.sum(endpoint))
        junction_count = int(np.sum(junction))
        _, num_components = ndimage.label(skeleton)

        branch_count = max(1, endpoint_count // 2 + junction_count)
        if num_components > 1:
            branch_count += int(num_components - 1)

        return int(branch_count), int(junction_count), junction

    def _compute_mean_branch_angle(
        self,
        skeleton: np.ndarray,
        junction_mask: np.ndarray,
        voxel_size: Tuple[float, float, float],
    ) -> float:
        if not np.any(junction_mask):
            return 0.0

        sx, sy, sz = self._normalize_spacing(voxel_size)
        offsets = [
            (i, j, k)
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            for k in (-1, 0, 1)
            if not (i == 0 and j == 0 and k == 0)
        ]

        angles = []
        coords = np.argwhere(junction_mask)
        shape = skeleton.shape
        for x, y, z in coords:
            vectors = []
            for dx, dy, dz in offsets:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2] and skeleton[nx, ny, nz]:
                    v = np.array([dx * sx, dy * sy, dz * sz], dtype=np.float64)
                    n = np.linalg.norm(v)
                    if n > 1e-8:
                        vectors.append(v / n)
            if len(vectors) >= 2:
                vectors = vectors[:3]
                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        cosv = float(np.clip(np.dot(vectors[i], vectors[j]), -1.0, 1.0))
                        angles.append(float(np.degrees(np.arccos(cosv))))

        return float(np.mean(angles)) if angles else 0.0

    def _compute_curvature(self, skeleton: np.ndarray, voxel_size: Tuple[float, float, float]) -> float:
        if not np.any(skeleton):
            return 0.0

        sx, sy, sz = self._normalize_spacing(voxel_size)
        structure = ndimage.generate_binary_structure(3, 2)
        neighbors = ndimage.convolve(skeleton.astype(np.uint8), structure, mode="constant")
        degree = neighbors - skeleton.astype(np.uint8)
        mid_points = np.argwhere(skeleton & (degree == 2))
        if mid_points.size == 0:
            return 0.0

        offsets = [
            (i, j, k)
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            for k in (-1, 0, 1)
            if not (i == 0 and j == 0 and k == 0)
        ]

        curvatures = []
        shape = skeleton.shape
        for x, y, z in mid_points:
            nbr = []
            for dx, dy, dz in offsets:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2] and skeleton[nx, ny, nz]:
                    nbr.append((dx, dy, dz))
            if len(nbr) >= 2:
                v1 = np.array([nbr[0][0] * sx, nbr[0][1] * sy, nbr[0][2] * sz], dtype=np.float64)
                v2 = np.array([nbr[1][0] * sx, nbr[1][1] * sy, nbr[1][2] * sz], dtype=np.float64)
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 1e-8 and n2 > 1e-8:
                    cosv = float(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))
                    ang = float(np.arccos(cosv))
                    seg_len = 0.5 * (n1 + n2)
                    if seg_len > 1e-8:
                        curvatures.append(ang / seg_len)

        return float(np.mean(curvatures)) if curvatures else 0.0

    def _estimate_surface_area(self, mask: np.ndarray, voxel_size: Tuple[float, float, float]) -> float:
        """Estimate surface area (mm^2) from exposed voxel faces."""
        if not np.any(mask):
            return 0.0

        sx, sy, sz = self._normalize_spacing(voxel_size)
        m = mask.astype(np.uint8)

        area = 0.0
        area += float(np.sum(np.abs(np.diff(m, axis=0)))) * sy * sz
        area += float(np.sum(np.abs(np.diff(m, axis=1)))) * sx * sz
        area += float(np.sum(np.abs(np.diff(m, axis=2)))) * sx * sy

        area += float(np.sum(m[0, :, :]) + np.sum(m[-1, :, :])) * sy * sz
        area += float(np.sum(m[:, 0, :]) + np.sum(m[:, -1, :])) * sx * sz
        area += float(np.sum(m[:, :, 0]) + np.sum(m[:, :, -1])) * sx * sy
        return float(area)

    def couinaud_segmentation(
        self,
        liver_mask: np.ndarray,
        portal_vein_mask: np.ndarray,
        hepatic_vein_mask: np.ndarray,
    ) -> np.ndarray:
        """Simple geometry+blood-supply based Couinaud-like partition."""
        segmentation = np.zeros_like(liver_mask, dtype=np.uint8)
        liver = np.asarray(liver_mask) > 0
        if not np.any(liver):
            return segmentation

        portal = np.asarray(portal_vein_mask) > 0
        hepatic = np.asarray(hepatic_vein_mask) > 0

        portal_distance = ndimage.distance_transform_edt(~portal) if np.any(portal) else np.full(liver.shape, np.inf)
        hepatic_distance = ndimage.distance_transform_edt(~hepatic) if np.any(hepatic) else np.full(liver.shape, np.inf)

        coords = np.argwhere(liver)
        zmin, xmin, ymin = coords.min(axis=0)
        zmax, xmax, ymax = coords.max(axis=0)
        z, x, y = np.mgrid[0 : liver.shape[0], 0 : liver.shape[1], 0 : liver.shape[2]]

        zn = (z - zmin) / max(float(zmax - zmin), 1.0)
        xn = (x - xmin) / max(float(xmax - xmin), 1.0)

        if np.any(portal) and np.any(hepatic):
            portal_dominant = portal_distance <= hepatic_distance
        elif np.any(portal):
            portal_dominant = np.ones_like(liver, dtype=bool)
        elif np.any(hepatic):
            portal_dominant = np.zeros_like(liver, dtype=bool)
        else:
            portal_dominant = xn < 0.5

        superior = liver & (zn < 0.33)
        middle = liver & (zn >= 0.33) & (zn < 0.66)
        inferior = liver & (zn >= 0.66)

        left_side = xn < 0.5
        right_side = ~left_side

        segmentation[superior & portal_dominant] = 8
        segmentation[superior & (~portal_dominant)] = 7

        segmentation[middle & left_side & portal_dominant] = 2
        segmentation[middle & left_side & (~portal_dominant)] = 3
        segmentation[middle & right_side & portal_dominant] = 5
        segmentation[middle & right_side & (~portal_dominant)] = 6

        segmentation[inferior & portal_dominant] = 1
        segmentation[inferior & (~portal_dominant)] = 4

        segmentation[liver & (segmentation == 0)] = 4
        return segmentation

    def calculate_volumes(self, segmentation: np.ndarray, voxel_size: tuple = (1.0, 1.0, 1.0)) -> dict:
        """Calculate per-segment and total liver volume (mL)."""
        sx, sy, sz = self._normalize_spacing(voxel_size)
        voxel_volume = sx * sy * sz / 1000.0
        volumes: Dict[str, float] = {}
        for i in range(1, 9):
            volumes[f"segment_{i}"] = float(np.sum(segmentation == i)) * voxel_volume
        volumes["total_liver"] = float(sum(volumes.values()))
        return volumes

    def _derive_structure_masks(
        self,
        vessel_union: np.ndarray,
        vessel_analysis: Optional[Dict[str, Any]],
        trunk_mask: Optional[np.ndarray] = None,
        small_mask: Optional[np.ndarray] = None,
        junc_mask: Optional[np.ndarray] = None,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build trunk/small/junction masks from model outputs or robust fallback."""
        trunk = (
            np.asarray(trunk_mask).astype(bool)
            if trunk_mask is not None and np.asarray(trunk_mask).shape == vessel_union.shape
            else np.zeros_like(vessel_union, dtype=bool)
        )
        small = (
            np.asarray(small_mask).astype(bool)
            if small_mask is not None and np.asarray(small_mask).shape == vessel_union.shape
            else np.zeros_like(vessel_union, dtype=bool)
        )
        junc = (
            np.asarray(junc_mask).astype(bool)
            if junc_mask is not None and np.asarray(junc_mask).shape == vessel_union.shape
            else np.zeros_like(vessel_union, dtype=bool)
        )

        if vessel_analysis:
            try:
                structure_map = vessel_analysis.get("structure_map")
                if structure_map is not None and np.asarray(structure_map).shape == vessel_union.shape:
                    sm = np.asarray(structure_map)
                    trunk |= sm == 1
                    small |= sm == 2
                    junc |= sm == 3

                for key in ("pv", "hv"):
                    item = vessel_analysis.get(key, {})
                    if not isinstance(item, dict):
                        continue
                    for map_name, target in (("trunk_map", "trunk"), ("small_map", "small"), ("junction_map", "junc")):
                        arr = item.get(map_name)
                        if arr is None:
                            continue
                        arr = np.asarray(arr).astype(bool)
                        if arr.shape != vessel_union.shape:
                            continue
                        if target == "trunk":
                            trunk |= arr
                        elif target == "small":
                            small |= arr
                        else:
                            junc |= arr
            except Exception:
                pass

        trunk &= vessel_union
        small &= vessel_union
        junc &= vessel_union

        if not np.any(trunk) and not np.any(small):
            if np.any(vessel_union):
                edt_mm = ndimage.distance_transform_edt(vessel_union, sampling=spacing)
                vals = edt_mm[vessel_union]
                if vals.size > 0:
                    thr = float(np.percentile(vals, 65.0))
                    trunk = vessel_union & (edt_mm >= thr)
                if not np.any(trunk):
                    trunk = vessel_union.copy()
            small = vessel_union & (~trunk)

        if not np.any(small):
            small = vessel_union & (~trunk)

        overlap = trunk & small
        if np.any(overlap):
            small[overlap] = False

        if not np.any(junc):
            skeleton = self._skeletonize_3d(vessel_union)
            if np.any(skeleton):
                structure = ndimage.generate_binary_structure(3, 2)
                neighbors = ndimage.convolve(skeleton.astype(np.uint8), structure, mode="constant")
                points = skeleton & (neighbors >= 4)
                junc = ndimage.binary_dilation(points, iterations=1) & vessel_union

        return trunk, small, junc

    def extract_vascular_metrics(
        self,
        portal_vein_mask: np.ndarray,
        hepatic_vein_mask: np.ndarray,
        voxel_size: tuple = (1.0, 1.0, 1.0),
        liver_mask: Optional[np.ndarray] = None,
        vessel_analysis: Optional[Dict] = None,
        trunk_mask: Optional[np.ndarray] = None,
        small_mask: Optional[np.ndarray] = None,
        junc_mask: Optional[np.ndarray] = None,
    ) -> dict:
        """Extract vascular metrics; includes template output + backward-compatible keys."""
        metrics: Dict[str, Any] = {}

        sx, sy, sz = self._normalize_spacing(voxel_size)
        spacing = (sx, sy, sz)
        voxel_volume_ml = (sx * sy * sz) / 1000.0

        portal = np.asarray(portal_vein_mask).astype(bool)
        hepatic = np.asarray(hepatic_vein_mask).astype(bool)
        vessel_union = portal | hepatic

        if vessel_union.ndim != 3:
            raise ValueError(f"vessel mask must be 3D, got ndim={vessel_union.ndim}")

        trunk, small, junc = self._derive_structure_masks(
            vessel_union=vessel_union,
            vessel_analysis=vessel_analysis,
            trunk_mask=trunk_mask,
            small_mask=small_mask,
            junc_mask=junc_mask,
            spacing=spacing,
        )

        if self._vessel_quantifier is None and VesselQuantifier is not None:
            self._vessel_quantifier = VesselQuantifier()

        if self._vessel_quantifier is not None:
            template = self._vessel_quantifier.quantify(
                hv_mask=hepatic,
                pv_mask=portal,
                trunk_mask=trunk,
                small_mask=small,
                junc_mask=junc,
                spacing=spacing,
            )
        else:
            # Basic fallback
            template = {
                "hv_volume_ml": float(np.sum(hepatic)) * voxel_volume_ml,
                "pv_volume_ml": float(np.sum(portal)) * voxel_volume_ml,
                "total_vessel_volume_ml": float(np.sum(vessel_union)) * voxel_volume_ml,
                "trunk_volume_ml": float(np.sum(trunk)) * voxel_volume_ml,
                "small_volume_ml": float(np.sum(small)) * voxel_volume_ml,
                "pv_hv_ratio": float(np.sum(portal) / max(float(np.sum(hepatic)), 1.0)),
                "trunk_small_ratio": float(np.sum(trunk) / max(float(np.sum(small)), 1.0)),
                "hv_length_mm": 0.0,
                "pv_length_mm": 0.0,
                "total_length_mm": 0.0,
                "branch_count": 0,
                "junction_count": int(np.sum(junc)),
                "max_branch_order": 0,
                "hv_avg_diameter_mm": 0.0,
                "pv_avg_diameter_mm": 0.0,
                "hv_max_diameter_mm": 0.0,
                "pv_max_diameter_mm": 0.0,
                "hv_min_diameter_mm": 0.0,
                "pv_min_diameter_mm": 0.0,
                "risk_level": "LOW",
                "risk_score": 0,
                "risk_details": [],
            }

        metrics.update(template)

        # Backward-compatible aliases used by existing UI/report rendering
        metrics["vascular_volume"] = float(template["total_vessel_volume_ml"])
        metrics["portal_volume"] = float(template["pv_volume_ml"])
        metrics["hepatic_volume"] = float(template["hv_volume_ml"])
        metrics["pv_hv_volume_ratio"] = float(template["pv_hv_ratio"])
        metrics["pv_hv_ratio_abnormal"] = bool((template["pv_hv_ratio"] < 0.5) or (template["pv_hv_ratio"] > 2.0))

        metrics["vascular_length"] = float(template["total_length_mm"])
        metrics["portal_length"] = float(template["pv_length_mm"])
        metrics["hepatic_length"] = float(template["hv_length_mm"])

        pv_vox = float(np.sum(portal))
        hv_vox = float(np.sum(hepatic))
        if pv_vox + hv_vox > 0:
            mean_d = (
                float(template["pv_avg_diameter_mm"]) * pv_vox
                + float(template["hv_avg_diameter_mm"]) * hv_vox
            ) / (pv_vox + hv_vox)
        else:
            mean_d = 0.0
        metrics["mean_diameter"] = float(mean_d)
        metrics["max_diameter"] = float(max(template["pv_max_diameter_mm"], template["hv_max_diameter_mm"]))

        min_candidates = [v for v in (float(template["pv_min_diameter_mm"]), float(template["hv_min_diameter_mm"])) if v > 0]
        metrics["min_diameter"] = float(min(min_candidates)) if min_candidates else 0.0

        metrics["portal_mean_diameter"] = float(template["pv_avg_diameter_mm"])
        metrics["hepatic_mean_diameter"] = float(template["hv_avg_diameter_mm"])

        skeleton = self._skeletonize_3d(vessel_union)
        portal_skeleton = self._skeletonize_3d(portal)
        hepatic_skeleton = self._skeletonize_3d(hepatic)

        portal_branch, _, _ = self._compute_branch_and_junction(portal_skeleton)
        hepatic_branch, _, _ = self._compute_branch_and_junction(hepatic_skeleton)
        metrics["portal_branch_count"] = int(portal_branch)
        metrics["hepatic_branch_count"] = int(hepatic_branch)

        if liver_mask is not None and np.any(liver_mask):
            liver_vox = float(np.sum(np.asarray(liver_mask) > 0))
            metrics["vascular_density"] = float(np.sum(vessel_union)) / max(liver_vox, 1.0)
            metrics["volume_fraction"] = (
                float(metrics["vascular_volume"]) / (liver_vox * voxel_volume_ml)
                if liver_vox > 0
                else 0.0
            )
        else:
            metrics["vascular_density"] = float(np.sum(vessel_union)) / float(max(vessel_union.size, 1))
            metrics["volume_fraction"] = (
                float(metrics["vascular_volume"]) / (float(vessel_union.size) * voxel_volume_ml)
                if vessel_union.size > 0
                else 0.0
            )

        metrics["surface_area"] = self._estimate_surface_area(vessel_union, spacing)
        metrics["mean_branch_angle"] = self._compute_mean_branch_angle(skeleton, junc, spacing)
        try:
            _, num_features = ndimage.label(vessel_union)
            metrics["connectivity"] = 1.0 / float(max(num_features, 1))
        except Exception:
            metrics["connectivity"] = 0.0
        metrics["curvature"] = self._compute_curvature(skeleton, spacing)

        micro_branch_voxels = float(np.sum(small))
        if vessel_analysis:
            try:
                pv = vessel_analysis.get("pv", {})
                hv = vessel_analysis.get("hv", {})
                micro_branch_voxels = float(
                    pv.get("level_counts", {}).get("level_5_plus", 0)
                    + hv.get("level_counts", {}).get("level_5_plus", 0)
                )
            except Exception:
                pass
        metrics["micro_branch_voxels"] = float(micro_branch_voxels)

        return metrics

    def assess_tumor_vessel_distance(
        self,
        tumor_mask: np.ndarray,
        portal_vein_mask: np.ndarray,
        hepatic_vein_mask: np.ndarray,
        voxel_size: tuple = (1.0, 1.0, 1.0),
    ) -> dict:
        """Evaluate tumor-to-vessel distance and a simple risk tier."""
        tumor = np.asarray(tumor_mask) > 0
        assessment = {
            "min_distance": float("inf"),
            "mean_distance": float("inf"),
            "risk_level": "NOT_EVALUATED",
            "risk_score": 0,
            "tumor_voxel_count": int(np.count_nonzero(tumor)),
            "assessable": False,
        }
        sx, sy, sz = self._normalize_spacing(voxel_size)

        vessel_mask = (np.asarray(portal_vein_mask) > 0) | (np.asarray(hepatic_vein_mask) > 0)

        if not np.any(tumor):
            assessment["reason"] = "no_tumor_label"
            return assessment
        if not np.any(vessel_mask):
            assessment["reason"] = "no_vessel_label"
            return assessment

        try:
            dist_map = ndimage.distance_transform_edt(~vessel_mask, sampling=(sx, sy, sz))
            tumor_dist = dist_map[tumor]
            if tumor_dist.size > 0:
                min_dist = float(np.min(tumor_dist))
                mean_dist = float(np.mean(tumor_dist))
            else:
                min_dist = float("inf")
                mean_dist = float("inf")

            assessment["min_distance"] = min_dist
            assessment["mean_distance"] = mean_dist

            if min_dist < 1.0:
                assessment["risk_level"] = "HIGH"
                assessment["risk_score"] = 3
            elif min_dist < 3.0:
                assessment["risk_level"] = "MEDIUM"
                assessment["risk_score"] = 2
            else:
                assessment["risk_level"] = "LOW"
                assessment["risk_score"] = 1
            assessment["assessable"] = True
        except Exception:
            assessment["reason"] = "exception"

        return assessment

    def calculate_remaining_liver_volume(
        self,
        liver_mask: np.ndarray,
        resection_mask: np.ndarray,
        voxel_size: tuple = (1.0, 1.0, 1.0),
    ) -> float:
        """Calculate remaining liver volume (mL)."""
        remaining_mask = np.asarray(liver_mask).astype(bool) & (~np.asarray(resection_mask).astype(bool))
        sx, sy, sz = self._normalize_spacing(voxel_size)
        voxel_volume = sx * sy * sz / 1000.0
        return float(np.sum(remaining_mask)) * voxel_volume

    def assess_surgical_risk(self, volumes: dict, vascular_metrics: dict, tumor_vessel_assessment: dict) -> dict:
        """Aggregate a coarse surgical risk score."""
        total_risk_score = int(tumor_vessel_assessment.get("risk_score", 0))

        total_liver = float(volumes.get("total_liver", 0.0))
        if total_liver < 1000:
            total_risk_score += 2
        elif total_liver < 1500:
            total_risk_score += 1

        vascular_volume = float(vascular_metrics.get("vascular_volume", 0.0))
        if vascular_volume < 10:
            total_risk_score += 1

        if total_risk_score >= 5:
            return {
                "overall_risk": "HIGH",
                "recommendation": "Conservative strategy or additional evaluation is recommended.",
            }
        if total_risk_score >= 3:
            return {
                "overall_risk": "MEDIUM",
                "recommendation": "Surgery may be considered with careful planning.",
            }
        return {
            "overall_risk": "LOW",
            "recommendation": "Surgical treatment appears feasible.",
        }

    def generate_report(
        self,
        liver_mask: np.ndarray,
        vessel_mask: np.ndarray,
        tumor_mask: np.ndarray,
        voxel_size: tuple = (1.0, 1.0, 1.0),
    ) -> dict:
        """Generate a complete quantitative report dictionary."""
        report: Dict[str, Any] = {}

        portal_vein_mask = (np.asarray(vessel_mask) == 1).astype(np.uint8)
        hepatic_vein_mask = (np.asarray(vessel_mask) == 2).astype(np.uint8)

        segmentation = self.couinaud_segmentation(liver_mask, portal_vein_mask, hepatic_vein_mask)
        report["segmentation"] = segmentation

        volumes = self.calculate_volumes(segmentation, voxel_size)
        report["volumes"] = volumes

        vascular_metrics = self.extract_vascular_metrics(
            portal_vein_mask,
            hepatic_vein_mask,
            voxel_size=voxel_size,
            liver_mask=liver_mask,
        )
        report["vascular_metrics"] = vascular_metrics

        tumor_vessel_assessment = self.assess_tumor_vessel_distance(
            tumor_mask,
            portal_vein_mask,
            hepatic_vein_mask,
            voxel_size,
        )
        report["tumor_vessel_assessment"] = tumor_vessel_assessment

        risk_assessment = self.assess_surgical_risk(
            volumes,
            vascular_metrics,
            tumor_vessel_assessment,
        )

        if float(volumes.get("total_liver", 0.0)) < 700:
            risk_assessment["overall_risk"] = "HIGH"
            risk_assessment["recommendation"] = "Additional liver reserve evaluation is recommended."

        report["risk_assessment"] = risk_assessment
        return report
