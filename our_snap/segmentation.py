#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分割算法模块
功能：实现主动轮廓（Snake）算法，包括Region Competition和Edge-Based两种方法
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel
from typing import Tuple, List, Optional

class ActiveContourSegmentation:
    """主动轮廓分割算法 - 完全自研实现"""
    
    def __init__(self):
        """初始化"""
        self.image_data = None
        self.seed_points = []  # 种子点列表 [(x, y, z), ...]
        self.current_mask = None  # 当前分割掩码
        self.iteration_count = 0
        
    def set_image(self, image_data: np.ndarray):
        """设置图像数据"""
        self.image_data = image_data
        self.current_mask = np.zeros(image_data.shape, dtype=np.uint8)
    
    def set_seed_points(self, seed_points: List[Tuple[int, int, int]]):
        """设置种子点"""
        self.seed_points = seed_points
        # 从种子点初始化掩码
        self._initialize_mask_from_seeds()
    
    def _initialize_mask_from_seeds(self):
        """从种子点初始化掩码"""
        if self.image_data is None or len(self.seed_points) == 0:
            return
        
        self.current_mask = np.zeros(self.image_data.shape, dtype=np.uint8)
        
        # 在每个种子点周围创建初始区域
        for seed in self.seed_points:
            x, y, z = seed
            # 创建球形初始区域（半径5像素）
            radius = 5
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        if dx*dx + dy*dy + dz*dz <= radius*radius:
                            nx, ny, nz = x + dx, y + dy, z + dz
                            if (0 <= nx < self.image_data.shape[0] and
                                0 <= ny < self.image_data.shape[1] and
                                0 <= nz < self.image_data.shape[2]):
                                self.current_mask[nx, ny, nz] = 1
    
    def region_competition(self, 
                          iterations: int = 100,
                          smoothness: float = 1.0,
                          expansion_force: float = 1.0) -> np.ndarray:
        """
        Region Competition主动轮廓算法
        
        Args:
            iterations: 迭代次数
            smoothness: 平滑度参数（越大越平滑）
            expansion_force: 扩张力（正数扩张，负数收缩）
            
        Returns:
            分割掩码 (H, W, D)，uint8类型
        """
        if self.image_data is None:
            return None
        
        mask = self.current_mask.copy().astype(np.float32)
        if not np.any(mask > 0.5):
            self._initialize_mask_from_seeds()
            mask = self.current_mask.copy().astype(np.float32)
            if not np.any(mask > 0.5):
                return np.zeros_like(self.image_data, dtype=np.uint8)

        image = self.image_data.astype(np.float32)
        # 图像梯度对固定图像是恒定的，提前计算可减少耗时
        grad_x = np.gradient(image, axis=0)
        grad_y = np.gradient(image, axis=1)
        grad_z = np.gradient(image, axis=2)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        global_mean = float(np.mean(image))
        
        for i in range(iterations):
            # 计算区域内的平均灰度值
            inside_region = mask > 0.5
            outside_region = ~inside_region
            if not np.any(inside_region):
                # 避免空区域导致NaN，回退到初始化并停止
                self._initialize_mask_from_seeds()
                mask = self.current_mask.copy().astype(np.float32)
                break
            if not np.any(outside_region):
                outside_mean = global_mean
            else:
                outside_mean = float(np.mean(image[outside_region]))
            inside_mean = float(np.mean(image[inside_region]))
            
            # 计算区域竞争力
            region_force = np.zeros_like(mask)
            region_force[inside_region] = (image[inside_region] - inside_mean) * expansion_force
            region_force[outside_region] = (outside_mean - image[outside_region]) * expansion_force
            
            # 平滑项（拉普拉斯算子）
            laplacian = ndimage.laplace(mask)
            smooth_force = -smoothness * laplacian
            
            # 边缘项（梯度）
            edge_force = -0.1 * grad_magnitude
            
            # 总力
            total_force = region_force + smooth_force + edge_force
            
            # 更新掩码
            mask += 0.1 * total_force
            mask = np.clip(mask, 0, 1)
            
            # 二值化
            mask_binary = (mask > 0.5).astype(np.float32)
            
            # 形态学操作保持连通性
            mask_binary = ndimage.binary_closing(mask_binary, structure=np.ones((3, 3, 3)))
            mask_binary = ndimage.binary_opening(mask_binary, structure=np.ones((3, 3, 3)))
            if not np.any(mask_binary):
                # 若演化到全空，回退到初始种子区域，避免“有种子但无结果”
                self._initialize_mask_from_seeds()
                mask_binary = self.current_mask > 0
            
            mask = mask_binary.astype(np.float32)
        
        self.current_mask = (mask > 0.5).astype(np.uint8)
        self.iteration_count = iterations
        return self.current_mask
    
    def edge_based(self,
                   iterations: int = 100,
                   smoothness: float = 1.0,
                   edge_weight: float = 1.0) -> np.ndarray:
        """
        Edge-Based主动轮廓算法（基于边缘检测）
        
        Args:
            iterations: 迭代次数
            smoothness: 平滑度参数
            edge_weight: 边缘权重
            
        Returns:
            分割掩码 (H, W, D)，uint8类型
        """
        if self.image_data is None:
            return None
        
        mask = self.current_mask.copy().astype(np.float32)
        if not np.any(mask > 0.5):
            self._initialize_mask_from_seeds()
            mask = self.current_mask.copy().astype(np.float32)
            if not np.any(mask > 0.5):
                return np.zeros_like(self.image_data, dtype=np.uint8)
        
        # 计算边缘图（使用Sobel算子）
        edges = np.zeros_like(self.image_data, dtype=np.float32)
        for z in range(self.image_data.shape[2]):
            slice_img = self.image_data[:, :, z]
            edge_x = sobel(slice_img, axis=0)
            edge_y = sobel(slice_img, axis=1)
            edges[:, :, z] = np.sqrt(edge_x**2 + edge_y**2)
        
        # 归一化边缘图
        edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
        
        for i in range(iterations):
            # 平滑项
            laplacian = ndimage.laplace(mask)
            smooth_force = -smoothness * laplacian
            
            # 边缘吸引项（向边缘靠近）
            edge_force = edge_weight * (1.0 - edges)  # 边缘处值小，吸引力大
            
            # 总力
            total_force = smooth_force + edge_force
            
            # 更新掩码
            mask += 0.1 * total_force
            mask = np.clip(mask, 0, 1)
            
            # 二值化
            mask_binary = (mask > 0.5).astype(np.float32)
            
            # 形态学操作
            mask_binary = ndimage.binary_closing(mask_binary, structure=np.ones((3, 3, 3)))
            mask_binary = ndimage.binary_opening(mask_binary, structure=np.ones((3, 3, 3)))
            if not np.any(mask_binary):
                self._initialize_mask_from_seeds()
                mask_binary = self.current_mask > 0
            
            mask = mask_binary.astype(np.float32)
        
        self.current_mask = (mask > 0.5).astype(np.uint8)
        self.iteration_count = iterations
        return self.current_mask
    
    def get_mask(self) -> np.ndarray:
        """获取当前分割掩码"""
        return self.current_mask.copy() if self.current_mask is not None else None

