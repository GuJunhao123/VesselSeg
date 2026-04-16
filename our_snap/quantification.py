#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vessel quantification module.

Expected inputs:
- hv_mask, pv_mask, trunk_mask, small_mask, junc_mask: 3D numpy array
- spacing: (x, y, z) in mm

Expected output keys follow the user-provided template exactly.
"""

from collections import deque
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage


class VesselQuantifier:
    """Quantify vessel morphology/topology and compute risk score."""

    def __init__(self) -> None:
        self._offsets_26 = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]
        # Forward offsets for length summation (avoid double-counting edges)
        self._offsets_forward_13 = [
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

    def _to_bool(self, mask: np.ndarray) -> np.ndarray:
        if mask is None:
            return np.zeros((0, 0, 0), dtype=bool)
        arr = np.asarray(mask)
        if arr.ndim != 3:
            raise ValueError(f"mask must be 3D, got ndim={arr.ndim}")
        return arr.astype(bool)

    def _skeletonize_3d(self, mask: np.ndarray) -> np.ndarray:
        from skimage import morphology

        m = mask.astype(bool)
        if not np.any(m):
            return np.zeros_like(m, dtype=bool)
        try:
            return morphology.skeletonize_3d(m).astype(bool)
        except Exception:
            # Fallback path for environments without skeletonize_3d
            return morphology.skeletonize(m).astype(bool)

    def _skeleton_length_mm(self, skeleton: np.ndarray, spacing: Tuple[float, float, float]) -> float:
        if skeleton is None or not np.any(skeleton):
            return 0.0

        sx, sy, sz = self._normalize_spacing(spacing)
        s = skeleton.astype(np.uint8)
        length = 0.0

        for dx, dy, dz in self._offsets_forward_13:
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
            step = float(np.sqrt((dx * sx) ** 2 + (dy * sy) ** 2 + (dz * sz) ** 2))
            length += edge_count * step

        return float(length)

    def _diameter_stats_mm(
        self, mask: np.ndarray, spacing: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        if mask is None or not np.any(mask):
            return 0.0, 0.0, 0.0
        edt = ndimage.distance_transform_edt(mask.astype(bool), sampling=self._normalize_spacing(spacing))
        diam = edt[mask.astype(bool)] * 2.0
        if diam.size == 0:
            return 0.0, 0.0, 0.0
        return float(np.mean(diam)), float(np.max(diam)), float(np.min(diam))

    def _compute_degree(self, skeleton: np.ndarray) -> np.ndarray:
        if skeleton is None or not np.any(skeleton):
            return np.zeros_like(skeleton, dtype=np.int16)
        structure = ndimage.generate_binary_structure(3, 2)
        nbr = ndimage.convolve(skeleton.astype(np.uint8), structure, mode="constant", cval=0)
        return (nbr - skeleton.astype(np.uint8)).astype(np.int16)

    def _in_bounds(self, coord: Tuple[int, int, int], shape: Tuple[int, int, int]) -> bool:
        x, y, z = coord
        return 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]

    def _neighbors_in_skeleton(
        self, coord: Tuple[int, int, int], skeleton: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        shape = skeleton.shape
        x, y, z = coord
        out: List[Tuple[int, int, int]] = []
        for dx, dy, dz in self._offsets_26:
            c2 = (x + dx, y + dy, z + dz)
            if self._in_bounds(c2, shape) and skeleton[c2]:
                out.append(c2)
        return out

    def _build_skeleton_graph(
        self, skeleton: np.ndarray, junc_hint: np.ndarray
    ) -> Tuple[np.ndarray, Dict[int, List[int]], int]:
        if skeleton is None or not np.any(skeleton):
            return np.zeros((0, 3), dtype=np.int32), {}, 0

        deg = self._compute_degree(skeleton)
        node_mask = skeleton & ((deg != 2) | junc_hint.astype(bool))

        # Pure loop (all degree=2) fallback: create one node per connected component
        if not np.any(node_mask):
            comp, n_comp = ndimage.label(skeleton, structure=ndimage.generate_binary_structure(3, 2))
            if n_comp > 0:
                for cid in range(1, n_comp + 1):
                    pts = np.argwhere(comp == cid)
                    if pts.size > 0:
                        p = tuple(int(v) for v in pts[0])
                        node_mask[p] = True

        node_coords = np.argwhere(node_mask).astype(np.int32)
        if node_coords.size == 0:
            return np.zeros((0, 3), dtype=np.int32), {}, 0

        node_id_map = np.full(skeleton.shape, -1, dtype=np.int32)
        for i, p in enumerate(node_coords):
            node_id_map[tuple(int(v) for v in p)] = int(i)

        adjacency: Dict[int, set] = {i: set() for i in range(len(node_coords))}
        edges = set()
        max_steps = int(np.count_nonzero(skeleton)) + 5

        for i, p in enumerate(node_coords):
            src = (int(p[0]), int(p[1]), int(p[2]))
            for nbr in self._neighbors_in_skeleton(src, skeleton):
                prev = src
                cur = nbr
                steps = 0

                while steps < max_steps:
                    steps += 1
                    nid = int(node_id_map[cur])
                    if nid >= 0:
                        if nid != i:
                            a, b = (i, nid) if i < nid else (nid, i)
                            if (a, b) not in edges:
                                edges.add((a, b))
                                adjacency[a].add(b)
                                adjacency[b].add(a)
                        break

                    cands = [c for c in self._neighbors_in_skeleton(cur, skeleton) if c != prev]
                    if not cands:
                        break
                    prev, cur = cur, cands[0]

        branch_count = len(edges)
        if branch_count == 0 and np.any(skeleton):
            # Keep branch_count meaningful on degenerate graph
            _, n_comp = ndimage.label(skeleton, structure=ndimage.generate_binary_structure(3, 2))
            branch_count = max(1, int(n_comp))

        adjacency_list: Dict[int, List[int]] = {k: sorted(list(v)) for k, v in adjacency.items()}
        return node_coords, adjacency_list, int(branch_count)

    def _max_branch_order(
        self,
        node_coords: np.ndarray,
        adjacency: Dict[int, List[int]],
        node_mask: np.ndarray,
        trunk_mask: np.ndarray,
    ) -> int:
        if node_coords.size == 0:
            return 0

        roots: List[int] = []
        if trunk_mask is not None and np.any(trunk_mask):
            for i, p in enumerate(node_coords):
                if trunk_mask[tuple(int(v) for v in p)]:
                    roots.append(i)

        if not roots:
            # Fallback: choose highest-degree nodes as roots
            max_deg = -1
            for i in range(len(node_coords)):
                d = len(adjacency.get(i, []))
                if d > max_deg:
                    max_deg = d
                    roots = [i]
                elif d == max_deg:
                    roots.append(i)

        if not roots:
            return 0

        depth: Dict[int, int] = {}
        q = deque()
        for r in roots:
            if r not in depth:
                depth[r] = 1
                q.append(r)

        while q:
            u = q.popleft()
            du = depth[u]
            for v in adjacency.get(u, []):
                if v not in depth:
                    depth[v] = du + 1
                    q.append(v)

        # Disconnected components fallback
        for i in range(len(node_coords)):
            if i in depth:
                continue
            depth[i] = 1
            q.append(i)
            while q:
                u = q.popleft()
                du = depth[u]
                for v in adjacency.get(u, []):
                    if v not in depth:
                        depth[v] = du + 1
                        q.append(v)

        return int(max(depth.values())) if depth else 0

    def quantify(
        self,
        hv_mask: np.ndarray,
        pv_mask: np.ndarray,
        trunk_mask: np.ndarray,
        small_mask: np.ndarray,
        junc_mask: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Dict[str, object]:
        hv = self._to_bool(hv_mask)
        pv = self._to_bool(pv_mask)
        trunk = self._to_bool(trunk_mask)
        small = self._to_bool(small_mask)
        junc = self._to_bool(junc_mask)

        if hv.shape != pv.shape:
            raise ValueError("hv_mask and pv_mask must have the same shape")
        if trunk.shape != hv.shape or small.shape != hv.shape or junc.shape != hv.shape:
            raise ValueError("trunk_mask/small_mask/junc_mask must match hv/pv shape")

        sx, sy, sz = self._normalize_spacing(spacing)
        voxel_mm3 = sx * sy * sz
        voxel_ml = voxel_mm3 / 1000.0

        vessel_union = hv | pv

        # 1) Volumes
        hv_volume_ml = float(np.sum(hv)) * voxel_ml
        pv_volume_ml = float(np.sum(pv)) * voxel_ml
        total_vessel_volume_ml = float(np.sum(vessel_union)) * voxel_ml
        trunk_volume_ml = float(np.sum(trunk)) * voxel_ml
        small_volume_ml = float(np.sum(small)) * voxel_ml

        # 2) Ratios
        pv_hv_ratio = float(pv_volume_ml / hv_volume_ml) if hv_volume_ml > 1e-8 else (float("inf") if pv_volume_ml > 0 else 0.0)
        trunk_small_ratio = float(trunk_volume_ml / small_volume_ml) if small_volume_ml > 1e-8 else (float("inf") if trunk_volume_ml > 0 else 0.0)

        # 3) Lengths (skeleton + diagonal-aware edge summation)
        hv_skeleton = self._skeletonize_3d(hv)
        pv_skeleton = self._skeletonize_3d(pv)
        union_skeleton = self._skeletonize_3d(vessel_union)

        hv_length_mm = self._skeleton_length_mm(hv_skeleton, (sx, sy, sz))
        pv_length_mm = self._skeleton_length_mm(pv_skeleton, (sx, sy, sz))
        total_length_mm = self._skeleton_length_mm(union_skeleton, (sx, sy, sz))

        # 4) Topology
        deg_union = self._compute_degree(union_skeleton)
        junc_from_skel = union_skeleton & (deg_union >= 3)
        junc_effective = junc if np.any(junc) else junc_from_skel
        junction_count = int(np.sum(junc_effective))

        node_coords, adjacency, branch_count = self._build_skeleton_graph(union_skeleton, junc_effective)
        node_mask = np.zeros_like(union_skeleton, dtype=bool)
        if node_coords.size > 0:
            node_mask[tuple(node_coords.T)] = True
        trunk_skeleton = self._skeletonize_3d(trunk & vessel_union)
        max_branch_order = self._max_branch_order(node_coords, adjacency, node_mask, trunk_skeleton)

        # 5) Diameters
        hv_avg_diameter_mm, hv_max_diameter_mm, hv_min_diameter_mm = self._diameter_stats_mm(hv, (sx, sy, sz))
        pv_avg_diameter_mm, pv_max_diameter_mm, pv_min_diameter_mm = self._diameter_stats_mm(pv, (sx, sy, sz))

        # 6) Risk scoring by strict template rules
        risk_score = 0
        risk_details: List[str] = []

        if (pv_hv_ratio < 0.5) or (pv_hv_ratio > 2.0):
            risk_score += 1
            risk_details.append("PV/HV volume ratio out of range (<0.5 or >2.0)")

        if trunk_small_ratio < 0.3:
            risk_score += 1
            risk_details.append("Trunk/Small branch volume ratio is low (<0.3)")

        if junction_count < 10:
            risk_score += 1
            risk_details.append("Junction count is low (<10)")

        if (hv_avg_diameter_mm < 1.5) or (hv_avg_diameter_mm > 5.0):
            risk_score += 1
            risk_details.append("HV average diameter out of range (<1.5 mm or >5.0 mm)")

        if (pv_avg_diameter_mm < 2.0) or (pv_avg_diameter_mm > 6.0):
            risk_score += 1
            risk_details.append("PV average diameter out of range (<2.0 mm or >6.0 mm)")

        if risk_score >= 3:
            risk_level = "HIGH"
        elif risk_score >= 1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            # Volume metrics (mL)
            "hv_volume_ml": float(hv_volume_ml),
            "pv_volume_ml": float(pv_volume_ml),
            "total_vessel_volume_ml": float(total_vessel_volume_ml),
            "trunk_volume_ml": float(trunk_volume_ml),
            "small_volume_ml": float(small_volume_ml),

            # Ratio metrics
            "pv_hv_ratio": float(pv_hv_ratio),
            "trunk_small_ratio": float(trunk_small_ratio),

            # Length metrics (mm)
            "hv_length_mm": float(hv_length_mm),
            "pv_length_mm": float(pv_length_mm),
            "total_length_mm": float(total_length_mm),

            # Topology metrics
            "branch_count": int(branch_count),
            "junction_count": int(junction_count),
            "max_branch_order": int(max_branch_order),

            # Diameter metrics (mm)
            "hv_avg_diameter_mm": float(hv_avg_diameter_mm),
            "pv_avg_diameter_mm": float(pv_avg_diameter_mm),
            "hv_max_diameter_mm": float(hv_max_diameter_mm),
            "pv_max_diameter_mm": float(pv_max_diameter_mm),
            "hv_min_diameter_mm": float(hv_min_diameter_mm),
            "pv_min_diameter_mm": float(pv_min_diameter_mm),

            # Risk assessment
            "risk_level": risk_level,
            "risk_score": int(risk_score),
            "risk_details": risk_details,
        }

