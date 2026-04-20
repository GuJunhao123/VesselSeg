#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Surgical cut simulation module.

Input:
- liver_mask: numpy array, shape (D, H, W), liver mask
- vessel_mask: numpy array, shape (D, H, W), vessel mask (HV+PV merged or labeled)
- cut_plane_params: {"point": (x, y, z), "normal": (nx, ny, nz)}
- spacing: tuple, (x, y, z) in mm

Output:
{
    "remaining_volume_ml": float,
    "resected_volume_ml": float,
    "resection_rate": float,  # %
    "cut_vessels": {
        "hv_count": int,
        "pv_count": int,
        "total_count": int,
        "critical_vessels": list
    },
    "risk_assessment": str,  # LOW / MEDIUM / HIGH
    "cut_plane_visualization": dict
}
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage


class SurgerySimulator:
    """Plane-based surgical resection simulator."""

    def __init__(self, min_branch_voxels: int = 10, critical_branch_ml: float = 2.0) -> None:
        self.min_branch_voxels = int(max(1, min_branch_voxels))
        self.critical_branch_ml = float(max(0.1, critical_branch_ml))

    def _normalize_spacing(self, spacing: Tuple[float, float, float]) -> Tuple[float, float, float]:
        try:
            sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
            return (
                sx if sx > 0 else 1.0,
                sy if sy > 0 else 1.0,
                sz if sz > 0 else 1.0,
            )
        except Exception:
            return (1.0, 1.0, 1.0)

    def _to_bool_3d(self, arr: np.ndarray, name: str) -> np.ndarray:
        a = np.asarray(arr)
        if a.ndim != 3:
            raise ValueError(f"{name} must be 3D, got ndim={a.ndim}")
        return a.astype(bool)

    def _validate_inputs(
        self,
        liver_mask: np.ndarray,
        vessel_mask: np.ndarray,
        cut_plane_params: Dict[str, Tuple[float, float, float]],
    ) -> None:
        liver = np.asarray(liver_mask)
        vessel = np.asarray(vessel_mask)
        if liver.ndim != 3 or vessel.ndim != 3:
            raise ValueError("liver_mask and vessel_mask must be 3D arrays")
        if liver.shape != vessel.shape:
            raise ValueError("liver_mask and vessel_mask must have identical shape")
        if not isinstance(cut_plane_params, dict):
            raise ValueError("cut_plane_params must be a dict")
        if "point" not in cut_plane_params or "normal" not in cut_plane_params:
            raise ValueError("cut_plane_params must contain 'point' and 'normal'")
        if len(cut_plane_params["point"]) != 3 or len(cut_plane_params["normal"]) != 3:
            raise ValueError("cut_plane_params['point'] and ['normal'] must be length-3")

    def _parse_vessel_masks(self, vessel_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Parse vessel mask into pv/hv/union.

        Supported formats:
        - labeled: 0 background, 1 PV, 2 HV
        - merged binary: 0/1
        """
        v = np.asarray(vessel_mask)
        uniq = np.unique(v)
        uniq_set = set(uniq.tolist())

        if uniq_set.issubset({0, 1, 2}):
            pv = v == 1
            hv = v == 2
            union = (pv | hv) if (np.any(pv) or np.any(hv)) else (v > 0)
            has_split = np.any(pv) and np.any(hv)
            return pv, hv, union, bool(has_split)

        # Generic fallback: all non-zero is vessel union, no PV/HV split available
        union = v > 0
        pv = np.zeros_like(union, dtype=bool)
        hv = np.zeros_like(union, dtype=bool)
        return pv, hv, union, False

    def _plane_point_mm(
        self,
        point: Tuple[float, float, float],
        shape_zyx: Tuple[int, int, int],
        spacing_xyz: Tuple[float, float, float],
        point_is_mm: bool = False,
    ) -> np.ndarray:
        """
        Accept point in either voxel-index coordinates (x,y,z) or physical mm.
        Auto-detect by range against array shape.
        """
        px, py, pz = float(point[0]), float(point[1]), float(point[2])
        if point_is_mm:
            return np.array([px, py, pz], dtype=np.float64)
        d, h, w = shape_zyx
        sx, sy, sz = spacing_xyz
        is_voxel_like = (0.0 <= px <= max(w - 1, 1)) and (0.0 <= py <= max(h - 1, 1)) and (0.0 <= pz <= max(d - 1, 1))
        if is_voxel_like:
            return np.array([px * sx, py * sy, pz * sz], dtype=np.float64)
        return np.array([px, py, pz], dtype=np.float64)

    def _prepare_plane(
        self,
        cut_plane_params: Dict[str, Tuple[float, float, float]],
        shape_zyx: Tuple[int, int, int],
        spacing_xyz: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        normal = np.asarray(cut_plane_params["normal"], dtype=np.float64)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            raise ValueError("cut_plane_params['normal'] norm is too small")
        normal = normal / norm
        point_mm = self._plane_point_mm(
            cut_plane_params["point"],
            shape_zyx,
            spacing_xyz,
            point_is_mm=bool(cut_plane_params.get("point_is_mm", False)),
        )
        return point_mm, normal

    def _signed_distances_mm(
        self, shape_zyx: Tuple[int, int, int], point_mm: np.ndarray, normal: np.ndarray, spacing_xyz: Tuple[float, float, float]
    ) -> np.ndarray:
        d, h, w = shape_zyx
        sx, sy, sz = spacing_xyz
        z, y, x = np.indices((d, h, w))
        x_mm = x.astype(np.float64) * sx
        y_mm = y.astype(np.float64) * sy
        z_mm = z.astype(np.float64) * sz
        return (normal[0] * (x_mm - point_mm[0]) + normal[1] * (y_mm - point_mm[1]) + normal[2] * (z_mm - point_mm[2]))

    def _count_cut_branches(
        self,
        vessel_binary: np.ndarray,
        signed_dist_mm: np.ndarray,
        voxel_ml: float,
        name_prefix: str,
        eps_mm: float,
    ) -> Tuple[int, List[Dict[str, object]]]:
        if not np.any(vessel_binary):
            return 0, []

        structure = ndimage.generate_binary_structure(3, 2)
        labels, n_comp = ndimage.label(vessel_binary.astype(bool), structure=structure)
        if n_comp <= 0:
            return 0, []

        cut_count = 0
        critical: List[Dict[str, object]] = []

        for cid in range(1, int(n_comp) + 1):
            comp = labels == cid
            vox = int(np.count_nonzero(comp))
            if vox < self.min_branch_voxels:
                continue
            sd = signed_dist_mm[comp]
            on_a = bool(np.any(sd <= -eps_mm))
            on_b = bool(np.any(sd >= eps_mm))
            if not (on_a and on_b):
                continue

            cut_count += 1
            comp_ml = float(vox) * voxel_ml
            if comp_ml >= self.critical_branch_ml:
                critical.append(
                    {
                        "name": f"{name_prefix}_branch_{cut_count}",
                        "component_id": int(cid),
                        "volume_ml": float(comp_ml),
                        "voxel_count": int(vox),
                    }
                )
        return int(cut_count), critical

    def _risk_level(
        self,
        remaining_volume_ml: float,
        resection_rate_pct: float,
        total_cut_vessels: int,
        critical_vessels: List[Dict[str, object]],
    ) -> str:
        score = 0
        if resection_rate_pct >= 70.0:
            score += 2
        elif resection_rate_pct >= 50.0:
            score += 1

        if remaining_volume_ml < 500.0:
            score += 2
        elif remaining_volume_ml < 800.0:
            score += 1

        if total_cut_vessels >= 5:
            score += 2
        elif total_cut_vessels >= 2:
            score += 1

        if len(critical_vessels) > 0:
            score += 2

        if score >= 5:
            return "HIGH"
        if score >= 3:
            return "MEDIUM"
        return "LOW"

    def simulate_cut(
        self,
        liver_mask: np.ndarray,
        vessel_mask: np.ndarray,
        cut_plane_params: Dict[str, Tuple[float, float, float]],
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Dict[str, object]:
        self._validate_inputs(liver_mask, vessel_mask, cut_plane_params)

        liver = self._to_bool_3d(liver_mask, "liver_mask")
        pv_mask, hv_mask, vessel_union, has_split_vessels = self._parse_vessel_masks(vessel_mask)
        spacing_xyz = self._normalize_spacing(spacing)
        point_mm, normal = self._prepare_plane(cut_plane_params, liver.shape, spacing_xyz)

        voxel_ml = (spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2]) / 1000.0
        signed = self._signed_distances_mm(liver.shape, point_mm, normal, spacing_xyz)

        liver_sd = signed[liver]
        keep = liver_sd <= 0.0
        remaining_voxels = int(np.count_nonzero(keep))
        total_liver_voxels = int(liver_sd.size)
        resected_voxels = int(total_liver_voxels - remaining_voxels)

        remaining_volume_ml = float(remaining_voxels) * voxel_ml
        resected_volume_ml = float(resected_voxels) * voxel_ml
        resection_rate = (100.0 * resected_voxels / total_liver_voxels) if total_liver_voxels > 0 else 0.0

        # Tolerance near plane (mm), reduce numerical flicker.
        eps_mm = 0.5 * min(spacing_xyz)

        if has_split_vessels:
            pv_count, pv_critical = self._count_cut_branches(
                pv_mask, signed, voxel_ml, "pv", eps_mm
            )
            hv_count, hv_critical = self._count_cut_branches(
                hv_mask, signed, voxel_ml, "hv", eps_mm
            )
            total_count = int(pv_count + hv_count)
            critical_vessels = pv_critical + hv_critical
        else:
            # No PV/HV split available from input mask.
            total_count, critical_vessels = self._count_cut_branches(
                vessel_union, signed, voxel_ml, "vessel", eps_mm
            )
            pv_count = 0
            hv_count = 0

        risk = self._risk_level(
            remaining_volume_ml=remaining_volume_ml,
            resection_rate_pct=float(resection_rate),
            total_cut_vessels=int(total_count),
            critical_vessels=critical_vessels,
        )

        # Plane equation: a*x + b*y + c*z + d = 0 (in mm space).
        a, b, c = float(normal[0]), float(normal[1]), float(normal[2])
        d = float(-(a * point_mm[0] + b * point_mm[1] + c * point_mm[2]))
        depth, height, width = liver.shape
        sx, sy, sz = spacing_xyz

        vis = {
            "point_mm": (float(point_mm[0]), float(point_mm[1]), float(point_mm[2])),
            "normal": (a, b, c),
            "plane_equation": {"a": a, "b": b, "c": c, "d": d},
            "keep_side_rule": "dot(normal, (X - point_mm)) <= 0",
            "volume_bounds_mm": {
                "x": (0.0, float(max(width - 1, 0) * sx)),
                "y": (0.0, float(max(height - 1, 0) * sy)),
                "z": (0.0, float(max(depth - 1, 0) * sz)),
            },
            "input_point": tuple(float(v) for v in cut_plane_params["point"]),
            "input_normal": tuple(float(v) for v in cut_plane_params["normal"]),
            "input_point_interpreted_as": (
                "mm" if bool(cut_plane_params.get("point_is_mm", False)) else "voxel_index_if_in_range_else_mm"
            ),
        }

        return {
            "remaining_volume_ml": float(remaining_volume_ml),
            "resected_volume_ml": float(resected_volume_ml),
            "resection_rate": float(resection_rate),
            "cut_vessels": {
                "hv_count": int(hv_count),
                "pv_count": int(pv_count),
                "total_count": int(total_count),
                "critical_vessels": critical_vessels,
            },
            "risk_assessment": str(risk),
            "cut_plane_visualization": vis,
        }


def run_surgery_simulation(
    liver_mask: np.ndarray,
    vessel_mask: np.ndarray,
    cut_plane_params: Dict[str, Tuple[float, float, float]],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, object]:
    """Convenience function for module-level usage."""
    return SurgerySimulator().simulate_cut(
        liver_mask=liver_mask,
        vessel_mask=vessel_mask,
        cut_plane_params=cut_plane_params,
        spacing=spacing,
    )


__all__ = ["SurgerySimulator", "run_surgery_simulation"]
