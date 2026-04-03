#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量化分析与风险评估模块
功能：
1. 基于门静脉供血区自动划分肝脏8个功能区段（Couinaud分段）
2. 计算各段体积
3. 提供11项量化指标
4. 自动评估肿瘤-血管距离及手术风险
"""

import numpy as np
from scipy import ndimage
import nibabel as nib
import os
from typing import Dict, Tuple, Optional

class QuantitativeAnalyzer:
    """量化分析器"""
    
    def __init__(self):
        """初始化"""
        pass

    def _normalize_spacing(self, voxel_size: Tuple[float, float, float]) -> Tuple[float, float, float]:
        try:
            sx, sy, sz = float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2])
            return (sx if sx > 0 else 1.0, sy if sy > 0 else 1.0, sz if sz > 0 else 1.0)
        except Exception:
            return (1.0, 1.0, 1.0)

    def _skeletonize_3d(self, mask: np.ndarray) -> np.ndarray:
        from skimage import morphology
        try:
            return morphology.skeletonize_3d(mask.astype(bool)).astype(bool)
        except Exception:
            return morphology.skeletonize(mask.astype(bool)).astype(bool)

    def _compute_skeleton_length(self, skeleton: np.ndarray, voxel_size: Tuple[float, float, float]) -> float:
        """
        通过13个前向邻居累计边长，避免重复计数。
        """
        if not np.any(skeleton):
            return 0.0
        sx, sy, sz = self._normalize_spacing(voxel_size)
        # 13个前向偏移
        offsets = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1),
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        ]
        length = 0.0
        s = skeleton.astype(np.uint8)
        for dx, dy, dz in offsets:
            shifted = np.roll(s, shift=(dx, dy, dz), axis=(0, 1, 2))
            # 边界处roll会环绕，需裁剪掉无效部分
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
            step_len = np.sqrt((dx * sx) ** 2 + (dy * sy) ** 2 + (dz * sz) ** 2)
            length += edge_count * step_len
        return float(length)

    def _compute_branch_and_junction(self, skeleton: np.ndarray) -> Tuple[int, int, np.ndarray]:
        if not np.any(skeleton):
            return 0, 0, np.zeros_like(skeleton, dtype=bool)
        structure = ndimage.generate_binary_structure(3, 2)  # 26连通
        neighbors = ndimage.convolve(skeleton.astype(np.uint8), structure, mode='constant')
        degree = neighbors - skeleton.astype(np.uint8)  # 去掉自身
        endpoint = skeleton & (degree == 1)
        junction = skeleton & (degree >= 3)

        endpoint_count = int(np.sum(endpoint))
        junction_count = int(np.sum(junction))
        _, num_components = ndimage.label(skeleton)

        # 基于端点/分叉点估计分支数：
        # - 直线（2端点）=> 1分支
        # - Y形（3端点+1分叉）=> 3分支
        branch_count = max(1, endpoint_count // 2 + junction_count)
        if num_components > 1:
            branch_count += int(num_components - 1)

        return branch_count, junction_count, junction

    def _compute_mean_branch_angle(self, skeleton: np.ndarray, junction_mask: np.ndarray, voxel_size: Tuple[float, float, float]) -> float:
        if not np.any(junction_mask):
            return 0.0
        sx, sy, sz = self._normalize_spacing(voxel_size)
        offsets = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if not (i == 0 and j == 0 and k == 0)]
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
                # 取前3个向量做两两角度
                vectors = vectors[:3]
                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        cosv = float(np.clip(np.dot(vectors[i], vectors[j]), -1.0, 1.0))
                        ang = float(np.degrees(np.arccos(cosv)))
                        angles.append(ang)
        return float(np.mean(angles)) if angles else 0.0

    def _compute_curvature(self, skeleton: np.ndarray, voxel_size: Tuple[float, float, float]) -> float:
        if not np.any(skeleton):
            return 0.0
        sx, sy, sz = self._normalize_spacing(voxel_size)
        structure = ndimage.generate_binary_structure(3, 2)
        neighbors = ndimage.convolve(skeleton.astype(np.uint8), structure, mode='constant')
        degree = neighbors - skeleton.astype(np.uint8)
        mid_points = np.argwhere(skeleton & (degree == 2))
        if mid_points.size == 0:
            return 0.0

        offsets = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if not (i == 0 and j == 0 and k == 0)]
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
                    ang = np.arccos(cosv)  # 弧度
                    seg_len = 0.5 * (n1 + n2)
                    if seg_len > 1e-8:
                        curvatures.append(float(ang / seg_len))
        return float(np.mean(curvatures)) if curvatures else 0.0

    def _estimate_surface_area(self, mask: np.ndarray, voxel_size: Tuple[float, float, float]) -> float:
        """
        基于暴露体素面的面积估计表面积(mm²)。
        """
        if not np.any(mask):
            return 0.0
        sx, sy, sz = self._normalize_spacing(voxel_size)
        m = mask.astype(np.uint8)
        area = 0.0
        # x方向面
        diff_x = np.abs(np.diff(m, axis=0))
        area += float(np.sum(diff_x)) * sy * sz
        # y方向面
        diff_y = np.abs(np.diff(m, axis=1))
        area += float(np.sum(diff_y)) * sx * sz
        # z方向面
        diff_z = np.abs(np.diff(m, axis=2))
        area += float(np.sum(diff_z)) * sx * sy
        # 边界外表面
        area += float(np.sum(m[0, :, :]) + np.sum(m[-1, :, :])) * sy * sz
        area += float(np.sum(m[:, 0, :]) + np.sum(m[:, -1, :])) * sx * sz
        area += float(np.sum(m[:, :, 0]) + np.sum(m[:, :, -1])) * sx * sy
        return area
    
    def couinaud_segmentation(self, liver_mask: np.ndarray, portal_vein_mask: np.ndarray, hepatic_vein_mask: np.ndarray) -> np.ndarray:
        """
        基于门静脉和肝静脉进行Couinaud分段
        
        Args:
            liver_mask: 肝脏掩码
            portal_vein_mask: 门静脉掩码
            hepatic_vein_mask: 肝静脉掩码
            
        Returns:
            肝脏分段掩码，标签1-8对应Couinaud分段
        """
        segmentation = np.zeros_like(liver_mask, dtype=np.uint8)
        liver = liver_mask > 0
        if not np.any(liver):
            return segmentation

        portal = portal_vein_mask > 0
        hepatic = hepatic_vein_mask > 0

        # 计算到血管的距离场（mm维度相同，仅用于相对比较）
        portal_distance = ndimage.distance_transform_edt(~portal) if np.any(portal) else np.full(liver.shape, np.inf)
        hepatic_distance = ndimage.distance_transform_edt(~hepatic) if np.any(hepatic) else np.full(liver.shape, np.inf)

        # 在肝脏包围盒内做归一化坐标，降低全图空白对分段的影响
        coords = np.argwhere(liver)
        zmin, xmin, ymin = coords.min(axis=0)
        zmax, xmax, ymax = coords.max(axis=0)
        z, x, y = np.mgrid[0:liver.shape[0], 0:liver.shape[1], 0:liver.shape[2]]

        zn = (z - zmin) / max(float(zmax - zmin), 1.0)
        xn = (x - xmin) / max(float(xmax - xmin), 1.0)

        # 血供主导方向（更接近门静脉为True）
        if np.any(portal) and np.any(hepatic):
            portal_dominant = portal_distance <= hepatic_distance
        elif np.any(portal):
            portal_dominant = np.ones_like(liver, dtype=bool)
        elif np.any(hepatic):
            portal_dominant = np.zeros_like(liver, dtype=bool)
        else:
            # 双血管都缺失时退化为纯几何划分
            portal_dominant = xn < 0.5

        superior = liver & (zn < 0.33)
        middle = liver & (zn >= 0.33) & (zn < 0.66)
        inferior = liver & (zn >= 0.66)

        left_side = xn < 0.5
        right_side = ~left_side

        # 上部：7/8段（按血供主导）
        segmentation[superior & portal_dominant] = 8
        segmentation[superior & (~portal_dominant)] = 7

        # 中部：2/3/5/6段（按左右+血供）
        segmentation[middle & left_side & portal_dominant] = 2
        segmentation[middle & left_side & (~portal_dominant)] = 3
        segmentation[middle & right_side & portal_dominant] = 5
        segmentation[middle & right_side & (~portal_dominant)] = 6

        # 下部：1/4段（按血供主导）
        segmentation[inferior & portal_dominant] = 1
        segmentation[inferior & (~portal_dominant)] = 4

        # 兜底：任何未赋值的肝体素归入4段
        segmentation[liver & (segmentation == 0)] = 4
        return segmentation
    
    def calculate_volumes(self, segmentation: np.ndarray, voxel_size: tuple = (1.0, 1.0, 1.0)) -> dict:
        """
        计算各段体积
        
        Args:
            segmentation: 肝脏分段掩码
            voxel_size: 体素大小 (x, y, z)，单位mm
            
        Returns:
            各段体积字典，单位ml
        """
        volumes = {}
        voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2] / 1000  # 转换为ml
        
        for i in range(1, 9):
            segment_volume = np.sum(segmentation == i) * voxel_volume
            volumes[f"segment_{i}"] = segment_volume
        
        # 计算总体积
        volumes["total_liver"] = np.sum([v for v in volumes.values()])
        
        return volumes
    
    def extract_vascular_metrics(
        self,
        portal_vein_mask: np.ndarray,
        hepatic_vein_mask: np.ndarray,
        voxel_size: tuple = (1.0, 1.0, 1.0),
        liver_mask: Optional[np.ndarray] = None,
        vessel_analysis: Optional[Dict] = None,
    ) -> dict:
        """
        提取血管量化指标
        
        Args:
            portal_vein_mask: 门静脉掩码
            hepatic_vein_mask: 肝静脉掩码
            voxel_size: 体素大小 (x, y, z)，单位mm
            
        Returns:
            血管量化指标字典
        """
        metrics = {}

        sx, sy, sz = self._normalize_spacing(voxel_size)
        voxel_volume_ml = (sx * sy * sz) / 1000.0

        portal = portal_vein_mask.astype(bool)
        hepatic = hepatic_vein_mask.astype(bool)
        vessel_mask = portal | hepatic

        # 体积类
        total_volume_ml = float(np.sum(vessel_mask)) * voxel_volume_ml
        metrics["vascular_volume"] = total_volume_ml
        metrics["portal_volume"] = float(np.sum(portal)) * voxel_volume_ml
        metrics["hepatic_volume"] = float(np.sum(hepatic)) * voxel_volume_ml

        # 门静脉/肝静脉比例与异常提示
        pv_hv_ratio = metrics["portal_volume"] / max(metrics["hepatic_volume"], 1e-8)
        metrics["pv_hv_volume_ratio"] = float(pv_hv_ratio)
        metrics["pv_hv_ratio_abnormal"] = bool((pv_hv_ratio < 0.5) or (pv_hv_ratio > 2.0))

        # 骨架与长度
        skeleton = self._skeletonize_3d(vessel_mask)
        vessel_length_mm = self._compute_skeleton_length(skeleton, (sx, sy, sz))
        metrics["vascular_length"] = vessel_length_mm

        # 分血管统计：长度、直径、分支
        portal_skeleton = self._skeletonize_3d(portal)
        hepatic_skeleton = self._skeletonize_3d(hepatic)
        metrics["portal_length"] = self._compute_skeleton_length(portal_skeleton, (sx, sy, sz))
        metrics["hepatic_length"] = self._compute_skeleton_length(hepatic_skeleton, (sx, sy, sz))

        # 直径
        try:
            edt_mm = ndimage.distance_transform_edt(vessel_mask, sampling=(sx, sy, sz))
            vessel_r = edt_mm[vessel_mask]
            mean_diameter = float(2.0 * np.mean(vessel_r)) if vessel_r.size > 0 else 0.0
            max_diameter = float(2.0 * np.max(vessel_r)) if vessel_r.size > 0 else 0.0
        except Exception:
            mean_diameter = 0.0
            max_diameter = 0.0
        metrics["mean_diameter"] = mean_diameter
        metrics["max_diameter"] = max_diameter
        if np.any(portal):
            portal_edt = ndimage.distance_transform_edt(portal, sampling=(sx, sy, sz))
            metrics["portal_mean_diameter"] = float(2.0 * np.mean(portal_edt[portal]))
        else:
            metrics["portal_mean_diameter"] = 0.0
        if np.any(hepatic):
            hepatic_edt = ndimage.distance_transform_edt(hepatic, sampling=(sx, sy, sz))
            metrics["hepatic_mean_diameter"] = float(2.0 * np.mean(hepatic_edt[hepatic]))
        else:
            metrics["hepatic_mean_diameter"] = 0.0

        # 分支/分叉
        branch_count, junction_count, junction_mask = self._compute_branch_and_junction(skeleton)
        metrics["branch_count"] = branch_count
        metrics["junction_count"] = junction_count
        portal_branch, _, _ = self._compute_branch_and_junction(portal_skeleton)
        hepatic_branch, _, _ = self._compute_branch_and_junction(hepatic_skeleton)
        metrics["portal_branch_count"] = portal_branch
        metrics["hepatic_branch_count"] = hepatic_branch

        # 密度与体积分数
        if liver_mask is not None and np.any(liver_mask):
            liver_vox = float(np.sum(liver_mask > 0))
            metrics["vascular_density"] = float(np.sum(vessel_mask)) / liver_vox
            metrics["volume_fraction"] = (
                total_volume_ml / (liver_vox * voxel_volume_ml) if liver_vox > 0 else 0.0
            )
        else:
            metrics["vascular_density"] = float(np.sum(vessel_mask)) / float(vessel_mask.size)
            metrics["volume_fraction"] = (
                total_volume_ml / (float(vessel_mask.size) * voxel_volume_ml)
                if vessel_mask.size > 0 else 0.0
            )

        # 表面积
        metrics["surface_area"] = self._estimate_surface_area(vessel_mask, (sx, sy, sz))

        # 平均分支角度
        metrics["mean_branch_angle"] = self._compute_mean_branch_angle(
            skeleton, junction_mask, (sx, sy, sz)
        )

        # 连通性
        try:
            _, num_features = ndimage.label(vessel_mask)
            metrics["connectivity"] = 1.0 / float(max(num_features, 1))
        except Exception:
            metrics["connectivity"] = 0.0

        # 弯曲度
        metrics["curvature"] = self._compute_curvature(skeleton, (sx, sy, sz))

        # 可选：引入AI结构识别结果
        if vessel_analysis:
            try:
                pv = vessel_analysis.get("pv", {})
                hv = vessel_analysis.get("hv", {})
                pv_trunk = float(pv.get("trunk_voxel_count", 0))
                hv_trunk = float(hv.get("trunk_voxel_count", 0))
                pv_small = float(pv.get("small_voxel_count", 0))
                hv_small = float(hv.get("small_voxel_count", 0))
                trunk = pv_trunk + hv_trunk
                small = pv_small + hv_small
                metrics["trunk_small_ratio"] = trunk / max(small, 1.0)
                metrics["micro_branch_voxels"] = float(
                    pv.get("level_counts", {}).get("level_5_plus", 0)
                    + hv.get("level_counts", {}).get("level_5_plus", 0)
                )
            except Exception:
                pass

        return metrics
    
    def assess_tumor_vessel_distance(self, tumor_mask: np.ndarray, portal_vein_mask: np.ndarray, hepatic_vein_mask: np.ndarray, voxel_size: tuple = (1.0, 1.0, 1.0)) -> dict:
        """
        评估肿瘤与血管的距离
        
        Args:
            tumor_mask: 肿瘤掩码
            portal_vein_mask: 门静脉掩码
            hepatic_vein_mask: 肝静脉掩码
            voxel_size: 体素大小 (x, y, z)，单位mm
            
        Returns:
            肿瘤-血管距离评估结果
        """
        assessment = {}
        sx, sy, sz = self._normalize_spacing(voxel_size)

        vessel_mask = (portal_vein_mask > 0) | (hepatic_vein_mask > 0)
        tumor = tumor_mask > 0

        if np.any(tumor) and np.any(vessel_mask):
            try:
                # 到“最近血管体素”的欧式距离场(mm)
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

                # 风险分级
                if min_dist < 1.0:
                    assessment["risk_level"] = "高风险"
                    assessment["risk_score"] = 3
                elif min_dist < 3.0:
                    assessment["risk_level"] = "中风险"
                    assessment["risk_score"] = 2
                else:
                    assessment["risk_level"] = "低风险"
                    assessment["risk_score"] = 1
            except Exception:
                assessment["min_distance"] = float("inf")
                assessment["mean_distance"] = float("inf")
                assessment["risk_level"] = "未知"
                assessment["risk_score"] = 0
        else:
            assessment["min_distance"] = float("inf")
            assessment["mean_distance"] = float("inf")
            assessment["risk_level"] = "未知"
            assessment["risk_score"] = 0

        return assessment
    
    def calculate_remaining_liver_volume(self, liver_mask: np.ndarray, resection_mask: np.ndarray, voxel_size: tuple = (1.0, 1.0, 1.0)) -> float:
        """
        计算剩余肝体积
        
        Args:
            liver_mask: 肝脏掩码
            resection_mask: 切除区域掩码
            voxel_size: 体素大小 (x, y, z)，单位mm
            
        Returns:
            剩余肝体积，单位ml
        """
        # 计算剩余肝脏掩码
        remaining_mask = liver_mask & (~resection_mask)
        
        # 计算体积
        voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2] / 1000  # 转换为ml
        remaining_volume = np.sum(remaining_mask) * voxel_volume
        
        return remaining_volume
    
    def assess_surgical_risk(self, volumes: dict, vascular_metrics: dict, tumor_vessel_assessment: dict) -> dict:
        """
        手术风险评估
        
        Args:
            volumes: 肝脏各段体积
            vascular_metrics: 血管量化指标
            tumor_vessel_assessment: 肿瘤-血管距离评估
            
        Returns:
            手术风险评估结果
        """
        risk = {}
        
        # 计算总体风险评分
        total_risk_score = 0
        
        # 基于肿瘤-血管距离评估风险
        risk_score = tumor_vessel_assessment.get("risk_score", 0)
        total_risk_score += risk_score
        
        # 基于肝脏体积评估风险
        total_liver = volumes.get("total_liver", 0)
        if total_liver < 1000:
            total_risk_score += 2
        elif total_liver < 1500:
            total_risk_score += 1
        
        # 基于血管指标评估风险
        vascular_volume = vascular_metrics.get("vascular_volume", 0)
        if vascular_volume < 10:
            total_risk_score += 1
        
        # 确定风险等级
        if total_risk_score >= 5:
            risk["overall_risk"] = "高风险"
            risk["recommendation"] = "建议保守治疗或进一步评估"
        elif total_risk_score >= 3:
            risk["overall_risk"] = "中风险"
            risk["recommendation"] = "可考虑手术，需谨慎评估"
        else:
            risk["overall_risk"] = "低风险"
            risk["recommendation"] = "适合手术治疗"
        
        return risk
    
    def generate_report(self, liver_mask: np.ndarray, vessel_mask: np.ndarray, tumor_mask: np.ndarray, voxel_size: tuple = (1.0, 1.0, 1.0)) -> dict:
        """
        生成完整的量化分析报告
        
        Args:
            liver_mask: 肝脏掩码
            vessel_mask: 血管掩码
            tumor_mask: 肿瘤掩码
            voxel_size: 体素大小 (x, y, z)，单位mm
            
        Returns:
            完整的分析报告
        """
        report = {}
        
        # 1. 肝脏分段
        portal_vein_mask = (vessel_mask == 1).astype(np.uint8)
        hepatic_vein_mask = (vessel_mask == 2).astype(np.uint8)
        segmentation = self.couinaud_segmentation(liver_mask, portal_vein_mask, hepatic_vein_mask)
        report["segmentation"] = segmentation
        
        # 2. 体积计算
        volumes = self.calculate_volumes(segmentation, voxel_size)
        report["volumes"] = volumes
        
        # 3. 血管量化指标
        vascular_metrics = self.extract_vascular_metrics(portal_vein_mask, hepatic_vein_mask, voxel_size)
        report["vascular_metrics"] = vascular_metrics
        
        # 4. 肿瘤-血管距离评估
        tumor_vessel_assessment = self.assess_tumor_vessel_distance(
            tumor_mask,
            portal_vein_mask,
            hepatic_vein_mask,
            voxel_size
        )
        report["tumor_vessel_assessment"] = tumor_vessel_assessment
        
        # 5. 手术风险评估
        risk_assessment = {
            "overall_risk": "低风险",
            "recommendation": "手术可行"
        }
        
        # 根据指标评估风险
        if tumor_vessel_assessment["risk_score"] >= 2:
            risk_assessment["overall_risk"] = "中风险"
            risk_assessment["recommendation"] = "建议谨慎手术"
        
        if tumor_vessel_assessment["risk_score"] == 3:
            risk_assessment["overall_risk"] = "高风险"
            risk_assessment["recommendation"] = "建议非手术治疗"
        
        # 检查剩余肝体积
        if volumes["total_liver"] < 700:  # 假设标准肝体积
            risk_assessment["overall_risk"] = "高风险"
            risk_assessment["recommendation"] = "建议肝储备功能评估"
        
        report["risk_assessment"] = risk_assessment
        
        return report
