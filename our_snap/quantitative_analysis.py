#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quantitative analysis and risk assessment utilities."""

from itertools import permutations, product
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

try:
    import torch
    from monai.metrics import DiceMetric, HausdorffDistanceMetric
except Exception:
    torch = None
    DiceMetric = None
    HausdorffDistanceMetric = None

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

    def _binary_surface(self, mask: np.ndarray) -> np.ndarray:
        """Return 1-voxel-thick binary surface."""
        m = np.asarray(mask).astype(bool)
        if not np.any(m):
            return np.zeros_like(m, dtype=bool)
        structure = ndimage.generate_binary_structure(3, 1)
        eroded = ndimage.binary_erosion(m, structure=structure, iterations=1, border_value=0)
        surface = m & (~eroded)
        if not np.any(surface):
            # Keep degenerate tiny masks evaluable.
            surface = m.copy()
        return surface

    def _to_binary_onehot_tensors(
        self, pred_mask: np.ndarray, ref_mask: np.ndarray
    ) -> Optional[Tuple["torch.Tensor", "torch.Tensor"]]:
        if torch is None:
            return None
        pred = np.asarray(pred_mask).astype(bool)
        ref = np.asarray(ref_mask).astype(bool)
        pred_fg = torch.from_numpy(pred.astype(np.float32))
        ref_fg = torch.from_numpy(ref.astype(np.float32))
        pred_oh = torch.stack([1.0 - pred_fg, pred_fg], dim=0).unsqueeze(0)
        ref_oh = torch.stack([1.0 - ref_fg, ref_fg], dim=0).unsqueeze(0)
        return pred_oh, ref_oh

    def _monai_binary_dice(self, pred_mask: np.ndarray, ref_mask: np.ndarray) -> Optional[float]:
        if DiceMetric is None:
            return None
        pred = np.asarray(pred_mask).astype(bool)
        ref = np.asarray(ref_mask).astype(bool)
        if not np.any(pred) and not np.any(ref):
            return 1.0
        tensors = self._to_binary_onehot_tensors(pred, ref)
        if tensors is None:
            return None
        pred_oh, ref_oh = tensors
        metric = DiceMetric(include_background=False, reduction="mean_batch")
        metric(y_pred=pred_oh, y=ref_oh)
        value = metric.aggregate()
        metric.reset()
        try:
            dice = float(value.detach().cpu().flatten()[0].item())
        except Exception:
            dice = float(value) if value is not None else float("nan")
        if not np.isfinite(dice):
            return 0.0
        return float(np.clip(dice, 0.0, 1.0))

    def _monai_binary_hd95(
        self,
        pred_mask: np.ndarray,
        ref_mask: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> Optional[float]:
        if HausdorffDistanceMetric is None:
            return None
        pred = np.asarray(pred_mask).astype(bool)
        ref = np.asarray(ref_mask).astype(bool)
        if not np.any(pred) and not np.any(ref):
            return 0.0
        if (not np.any(pred)) != (not np.any(ref)):
            return float("inf")
        tensors = self._to_binary_onehot_tensors(pred, ref)
        if tensors is None:
            return None
        pred_oh, ref_oh = tensors
        metric = HausdorffDistanceMetric(
            include_background=False,
            reduction="mean_batch",
            percentile=95,
        )
        metric(y_pred=pred_oh, y=ref_oh, spacing=spacing)
        value = metric.aggregate()
        metric.reset()
        try:
            hd95 = float(value.detach().cpu().flatten()[0].item())
        except Exception:
            hd95 = float(value) if value is not None else float("nan")
        if np.isnan(hd95):
            return float("inf")
        if hd95 < 0:
            return 0.0
        return float(hd95)

    def _overlap_stats(self, pred_mask: np.ndarray, ref_mask: np.ndarray) -> Dict[str, float]:
        pred = np.asarray(pred_mask).astype(bool)
        ref = np.asarray(ref_mask).astype(bool)
        pred_voxels = int(np.sum(pred))
        ref_voxels = int(np.sum(ref))
        intersection = int(np.sum(pred & ref))
        union = pred_voxels + ref_voxels - intersection

        if pred_voxels == 0 and ref_voxels == 0:
            dice = 1.0
            jaccard = 1.0
        else:
            dice = (2.0 * intersection) / max(float(pred_voxels + ref_voxels), 1.0)
            jaccard = float(intersection) / max(float(union), 1.0)

        monai_dice = self._monai_binary_dice(pred, ref)
        if monai_dice is not None:
            dice = monai_dice

        return {
            "pred_voxels": float(pred_voxels),
            "ref_voxels": float(ref_voxels),
            "intersection_voxels": float(intersection),
            "union_voxels": float(union),
            "dice": float(dice),
            "jaccard": float(jaccard),
        }

    def _surface_distance_stats(
        self,
        pred_mask: np.ndarray,
        ref_mask: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> Dict[str, float]:
        pred = np.asarray(pred_mask).astype(bool)
        ref = np.asarray(ref_mask).astype(bool)

        if not np.any(pred) and not np.any(ref):
            return {
                "asd_mm": 0.0,
                "hd95_mm": 0.0,
                "mean_pred_to_ref_mm": 0.0,
                "mean_ref_to_pred_mm": 0.0,
            }
        if (not np.any(pred)) != (not np.any(ref)):
            inf = float("inf")
            return {
                "asd_mm": inf,
                "hd95_mm": inf,
                "mean_pred_to_ref_mm": inf,
                "mean_ref_to_pred_mm": inf,
            }

        pred_surface = self._binary_surface(pred)
        ref_surface = self._binary_surface(ref)
        if not np.any(pred_surface) or not np.any(ref_surface):
            inf = float("inf")
            return {
                "asd_mm": inf,
                "hd95_mm": inf,
                "mean_pred_to_ref_mm": inf,
                "mean_ref_to_pred_mm": inf,
            }

        ref_distance = ndimage.distance_transform_edt(~ref_surface, sampling=spacing)
        pred_distance = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)

        pred_to_ref = ref_distance[pred_surface]
        ref_to_pred = pred_distance[ref_surface]
        if pred_to_ref.size == 0 or ref_to_pred.size == 0:
            inf = float("inf")
            return {
                "asd_mm": inf,
                "hd95_mm": inf,
                "mean_pred_to_ref_mm": inf,
                "mean_ref_to_pred_mm": inf,
            }

        # Standard symmetric HD95:
        # HD95 = max( percentile95(pred->ref), percentile95(ref->pred) )
        hd95_pred_to_ref = float(np.percentile(pred_to_ref, 95))
        hd95_ref_to_pred = float(np.percentile(ref_to_pred, 95))
        hd95_mm = max(hd95_pred_to_ref, hd95_ref_to_pred)
        monai_hd95_mm = self._monai_binary_hd95(pred, ref, spacing)
        if monai_hd95_mm is not None:
            hd95_mm = float(monai_hd95_mm)
        merged = np.concatenate([pred_to_ref, ref_to_pred], axis=0)
        return {
            "asd_mm": float(np.mean(merged)),
            "hd95_mm": float(hd95_mm),
            "hd95_monai_mm": float(hd95_mm),
            "hd95_pred_to_ref_mm": float(hd95_pred_to_ref),
            "hd95_ref_to_pred_mm": float(hd95_ref_to_pred),
            "mean_pred_to_ref_mm": float(np.mean(pred_to_ref)),
            "mean_ref_to_pred_mm": float(np.mean(ref_to_pred)),
        }

    def _accuracy_percent_from_metrics(self, dice: float, jaccard: float, asd_mm: float, hd95_mm: float) -> float:
        """
        Paper-style matching degree (0-100).

        The referenced vessel-segmentation paper reports Dice/DSC as the main
        percentage accuracy metric and reports distance metrics such as HD95
        separately. Therefore the final "matching degree" should be the Dice
        percentage, while Jaccard/ASD/HD95 remain independent quality indicators.
        """
        dice_score = float(np.clip(dice, 0.0, 1.0))
        return float(dice_score * 100.0)

    def _swap_pv_hv_labels(self, mask: np.ndarray) -> np.ndarray:
        """Swap label ids 1<->2 while keeping other labels unchanged."""
        arr = np.asarray(mask)
        out = np.array(arr, copy=True)
        m1 = arr == 1
        m2 = arr == 2
        out[m1] = 2
        out[m2] = 1
        return out

    def _apply_axis_transform(
        self,
        arr: np.ndarray,
        perm: Tuple[int, int, int],
        flips: Tuple[bool, bool, bool],
    ) -> np.ndarray:
        out = np.transpose(arr, perm) if perm != (0, 1, 2) else arr
        for axis, need_flip in enumerate(flips):
            if need_flip:
                out = np.flip(out, axis=axis)
        return out

    def _diagnose_axis_or_label_mismatch(
        self,
        ai: np.ndarray,
        doctor: np.ndarray,
        label_ids: Sequence[int],
        baseline_union_dice: float,
    ) -> Dict[str, Any]:
        """
        Try axis permutations/flips and optional label 1<->2 swap.
        Used only as diagnostics to determine whether poor match is likely caused
        by orientation/index-space mismatch rather than model prediction quality.
        """
        ai_arr = np.asarray(ai)
        doctor_arr = np.asarray(doctor)
        if ai_arr.ndim != 3 or doctor_arr.ndim != 3:
            return {"performed": False, "reason": "non_3d_input"}
        if ai_arr.shape != doctor_arr.shape:
            return {
                "performed": False,
                "reason": "shape_mismatch",
                "ai_shape": tuple(int(v) for v in ai_arr.shape),
                "doctor_shape": tuple(int(v) for v in doctor_arr.shape),
            }

        # Only run expensive diagnosis when baseline overlap is suspiciously low.
        baseline_union_dice = float(np.clip(baseline_union_dice, 0.0, 1.0))
        if baseline_union_dice >= 0.80:
            return {
                "performed": False,
                "reason": "baseline_good_enough",
                "baseline_union_dice": baseline_union_dice,
            }

        ref_union = doctor_arr > 0
        valid_label_ids = [int(v) for v in label_ids if int(v) > 0]

        candidates = []
        seen = set()
        for perm in permutations((0, 1, 2)):
            permuted_shape = tuple(int(ai_arr.shape[p]) for p in perm)
            if permuted_shape != tuple(int(v) for v in doctor_arr.shape):
                continue
            for flips in product((False, True), repeat=3):
                key = (perm, flips)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((perm, flips))

        if not candidates:
            return {
                "performed": False,
                "reason": "no_valid_axis_candidates",
                "baseline_union_dice": baseline_union_dice,
            }

        best = {
            "union_dice": baseline_union_dice,
            "union_jaccard": float("nan"),
            "label_mean_dice": baseline_union_dice,
            "perm": (0, 1, 2),
            "flips": (False, False, False),
            "swap_labels_1_2": False,
        }

        tested = 0
        for swap_labels in (False, True):
            base = self._swap_pv_hv_labels(ai_arr) if swap_labels else ai_arr
            for perm, flips in candidates:
                tested += 1
                transformed = self._apply_axis_transform(base, perm=perm, flips=flips)
                if transformed.shape != doctor_arr.shape:
                    continue

                union_stats = self._overlap_stats(transformed > 0, ref_union)
                union_dice = float(union_stats["dice"])
                union_jaccard = float(union_stats["jaccard"])

                label_dices = []
                for lid in valid_label_ids:
                    pred_l = transformed == lid
                    ref_l = doctor_arr == lid
                    if np.any(pred_l) or np.any(ref_l):
                        label_dices.append(float(self._overlap_stats(pred_l, ref_l)["dice"]))
                label_mean_dice = float(np.mean(label_dices)) if label_dices else union_dice

                better = False
                if union_dice > float(best["union_dice"]) + 1e-8:
                    better = True
                elif abs(union_dice - float(best["union_dice"])) <= 1e-8 and label_mean_dice > float(best["label_mean_dice"]) + 1e-8:
                    better = True

                if better:
                    best = {
                        "union_dice": union_dice,
                        "union_jaccard": union_jaccard,
                        "label_mean_dice": label_mean_dice,
                        "perm": tuple(int(v) for v in perm),
                        "flips": tuple(bool(v) for v in flips),
                        "swap_labels_1_2": bool(swap_labels),
                    }

        improvement = float(best["union_dice"]) - baseline_union_dice
        non_identity_transform = (
            best["perm"] != (0, 1, 2)
            or any(best["flips"])
            or bool(best["swap_labels_1_2"])
        )
        likely_misaligned = bool(
            non_identity_transform
            and improvement >= 0.15
            and float(best["union_dice"]) >= 0.45
        )

        return {
            "performed": True,
            "tested_candidates": int(tested),
            "baseline_union_dice": baseline_union_dice,
            "best_union_dice": float(best["union_dice"]),
            "best_union_jaccard": float(best["union_jaccard"]),
            "best_label_mean_dice": float(best["label_mean_dice"]),
            "improvement_dice": float(improvement),
            "best_transform": {
                "perm": list(best["perm"]),
                "flip_axes": [int(i) for i, f in enumerate(best["flips"]) if f],
                "swap_labels_1_2": bool(best["swap_labels_1_2"]),
            },
            "likely_misaligned": likely_misaligned,
            "note": (
                "Likely axis/flip/label-id mismatch"
                if likely_misaligned
                else "No strong evidence of orientation mismatch"
            ),
        }

    def evaluate_segmentation_accuracy(
        self,
        ai_label: np.ndarray,
        doctor_label: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        labels: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """
        Compare AI segmentation against doctor labels with Dice/Jaccard/ASD/HD95.

        Returns per-label metrics, merged-vessel metrics, and an aggregated
        accuracy percentage score in [0, 100].
        """
        ai = np.asarray(ai_label)
        doctor = np.asarray(doctor_label)
        if ai.ndim != 3 or doctor.ndim != 3:
            raise ValueError(f"ai/doctor label must be 3D, got {ai.ndim} and {doctor.ndim}")
        if ai.shape != doctor.shape:
            raise ValueError(f"shape mismatch: ai={ai.shape}, doctor={doctor.shape}")

        sx, sy, sz = self._normalize_spacing(spacing)
        spacing_xyz_mm = (sx, sy, sz)
        # Distance maps use ndarray axis order. This project stores labels as (H, W, D),
        # while spacing input is (x, y, z), so we map to (y, x, z) for EDT sampling.
        distance_sampling_mm = (sy, sx, sz)
        available_nonzero_labels = sorted(
            [int(v) for v in np.unique(np.concatenate([ai.ravel(), doctor.ravel()])) if int(v) > 0]
        )
        requested_label_ids = [int(v) for v in labels] if labels is not None else [1, 2]
        requested_label_ids = [v for v in requested_label_ids if v > 0]
        requested_label_ids = list(dict.fromkeys(requested_label_ids))
        label_ids = [v for v in requested_label_ids if v in available_nonzero_labels]
        fallback_note = ""
        if not label_ids and available_nonzero_labels:
            label_ids = list(available_nonzero_labels)
            fallback_note = (
                f"requested labels {requested_label_ids} not found in current case, "
                "fallback to all non-zero labels."
            )
        compare_all_nonzero = len(label_ids) == 0
        label_name_map = {1: "pv", 2: "hv"}

        per_label: Dict[str, Dict[str, Any]] = {}
        label_scores: list = []

        for label_id in label_ids:
            ai_mask = ai == label_id
            doctor_mask = doctor == label_id

            overlap = self._overlap_stats(ai_mask, doctor_mask)
            surface_distance = self._surface_distance_stats(ai_mask, doctor_mask, distance_sampling_mm)
            accuracy_percent = self._accuracy_percent_from_metrics(
                overlap["dice"],
                overlap["jaccard"],
                surface_distance["asd_mm"],
                surface_distance["hd95_mm"],
            )
            label_scores.append(float(accuracy_percent))

            key = f"label_{label_id}_{label_name_map.get(label_id, 'class')}"
            per_label[key] = {
                "label_id": int(label_id),
                "dice": float(overlap["dice"]),
                "jaccard": float(overlap["jaccard"]),
                "asd_mm": float(surface_distance["asd_mm"]),
                "hd95_mm": float(surface_distance["hd95_mm"]),
                "mean_ai_to_doctor_surface_mm": float(surface_distance["mean_pred_to_ref_mm"]),
                "mean_doctor_to_ai_surface_mm": float(surface_distance["mean_ref_to_pred_mm"]),
                "ai_voxels": int(overlap["pred_voxels"]),
                "doctor_voxels": int(overlap["ref_voxels"]),
                "intersection_voxels": int(overlap["intersection_voxels"]),
                "matching_degree_percent": float(accuracy_percent),
                "accuracy_percent": float(accuracy_percent),
            }

        if compare_all_nonzero:
            ai_union = ai > 0
            doctor_union = doctor > 0
        else:
            ai_union = np.isin(ai, label_ids)
            doctor_union = np.isin(doctor, label_ids)
        union_overlap = self._overlap_stats(ai_union, doctor_union)
        union_distance = self._surface_distance_stats(ai_union, doctor_union, distance_sampling_mm)
        union_accuracy_percent = self._accuracy_percent_from_metrics(
            union_overlap["dice"],
            union_overlap["jaccard"],
            union_distance["asd_mm"],
            union_distance["hd95_mm"],
        )
        alignment_diagnosis = self._diagnose_axis_or_label_mismatch(
            ai=ai,
            doctor=doctor,
            label_ids=label_ids if label_ids else requested_label_ids,
            baseline_union_dice=float(union_overlap["dice"]),
        )

        # Paper-style final matching degree: overall Dice/DSC percentage.
        # Per-label percentages are still returned for debugging, but the single
        # final score should match the reported overall Dice value.
        final_matching_percent = float(union_accuracy_percent)

        return {
            "status": "ok",
            "requested_labels": requested_label_ids,
            "labels_evaluated": label_ids,
            "compare_scope": "all_nonzero_labels" if compare_all_nonzero else "selected_labels",
            "fallback_note": fallback_note,
            "spacing_mm": spacing_xyz_mm,
            "distance_sampling_mm": distance_sampling_mm,
            "per_label": per_label,
            "overall": {
                "dice": float(union_overlap["dice"]),
                "jaccard": float(union_overlap["jaccard"]),
                "asd_mm": float(union_distance["asd_mm"]),
                "hd95_mm": float(union_distance["hd95_mm"]),
                "ai_voxels": int(union_overlap["pred_voxels"]),
                "doctor_voxels": int(union_overlap["ref_voxels"]),
                "intersection_voxels": int(union_overlap["intersection_voxels"]),
                "matching_degree_percent": float(final_matching_percent),
                "accuracy_percent": float(final_matching_percent),
                "union_matching_percent": float(union_accuracy_percent),
            },
            "final_matching_degree_percent": float(final_matching_percent),
            "macro_matching_degree_percent": float(np.mean(label_scores) if label_scores else final_matching_percent),
            "macro_accuracy_percent": float(np.mean(label_scores) if label_scores else final_matching_percent),
            "score_definition": {
                "formula": "Matching Degree (%) = Dice Similarity Coefficient (DSC) * 100",
                "range": "0-100%",
                "meaning": "参考论文将 Dice/DSC 作为主要分割精度百分比；Jaccard、ASD、HD95 作为独立指标显示，不再混入任意阈值加权公式",
            },
            "alignment_diagnosis": alignment_diagnosis,
        }

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
