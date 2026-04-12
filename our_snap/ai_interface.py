#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI自动分割接口模块
功能：支持ONNX模型和PyTorch模型推理，实现AI自动分割
"""

import numpy as np
import onnxruntime as ort
from typing import Optional, Tuple, Dict, Any, Mapping
import os
import sys

# 添加模型所在目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

# 尝试导入 PyTorch，用于支持 pre4.py 中的模型
try:
    import torch
    # 确保导入路径正确
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
    from pre4 import mobileuvit, mobileuvit_l
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
        CropForegroundd, SpatialPadd, ScaleIntensityRanged, NormalizeIntensityd, 
        EnsureTyped
    )
    from monai.inferers import SlidingWindowInferer
    import nibabel as nib
    from nilearn.image import resample_to_img
    PYTORCH_AVAILABLE = True
except ImportError as e:
    print(f"PyTorch 相关库导入失败: {e}")
    PYTORCH_AVAILABLE = False

class AISegmentationInterface:
    """AI分割接口 - 完全自研实现"""
    
    def __init__(self):
        """初始化"""
        self.model_session = None  # ONNX 模型会话
        self.pytorch_model = None  # PyTorch 模型
        self.model_type = None  # 'onnx' 或 'pytorch'
        self.model_input_shape = None
        self.model_output_shape = None
        self.is_model_loaded = False
        
        # MobileUViT 模型参数
        self.model_variant = 'mobileuvit_l'  # 'mobileuvit' 或 'mobileuvit_l'
        self.decomposition_method = 'percentile_50'  # 血管分解方法
        # 保存最近一次血管结构分析结果
        self.last_vessel_analysis: Dict[str, Any] = {}
    
    def load_model(self, model_path: str) -> bool:
        """
        加载模型（支持ONNX和PyTorch格式）
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            成功返回True，失败返回False
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 根据文件扩展名判断模型类型
            ext = os.path.splitext(model_path)[1].lower()
            
            if ext == '.onnx':
                # 加载 ONNX 模型
                self.model_session = ort.InferenceSession(
                    model_path,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']  # 优先使用GPU
                )
                
                # 获取输入输出信息
                input_info = self.model_session.get_inputs()[0]
                output_info = self.model_session.get_outputs()[0]
                
                self.model_input_shape = input_info.shape
                self.model_output_shape = output_info.shape
                self.model_type = 'onnx'
                
                print(f"ONNX 模型加载成功: {model_path}")
                print(f"输入形状: {self.model_input_shape}")
                print(f"输出形状: {self.model_output_shape}")
                
            elif ext in ['.pth', '.pt']:
                if not PYTORCH_AVAILABLE:
                    raise RuntimeError("PyTorch 不可用，无法加载.pth模型文件")
                
                # 加载 PyTorch 模型
                try:
                    if self.model_variant == 'mobileuvit':
                        self.pytorch_model = mobileuvit(inch=1)
                        print("使用 mobileuvit 小型模型")
                    else:
                        self.pytorch_model = mobileuvit_l(inch=1)
                        print("使用 mobileuvit_l 大型模型")
                except Exception as e:
                    print(f"模型实例化失败: {e}")
                    raise RuntimeError(f"无法创建模型变体 {self.model_variant}: {e}")
                
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                print(f"使用设备: {device}")
                
                try:
                    state_dict = torch.load(model_path, map_location=device)
                    
                    # 智能加载权重，处理模型变体之间的差异
                    if self.model_variant == 'mobileuvit':
                        # 小型模型：过滤掉不匹配的权重
                        print("为小型模型调整权重...")
                        
                        # 构建新的state_dict，只包含匹配的权重
                        new_state_dict = {}
                        for key, value in state_dict.items():
                            # 跳过transformer_bottleneck.3（小型模型没有第4层）
                            if 'transformer_bottleneck.3' in key:
                                print(f"跳过不匹配的权重: {key}")
                                continue
                            
                            # 处理patch_embeddings.stem.0的大小不匹配
                            if key == 'patch_embeddings.stem.0.weight':
                                # 从32通道裁剪到16通道
                                if value.shape[0] == 32 and self.pytorch_model.patch_embeddings.stem[0].weight.shape[0] == 16:
                                    print("调整patch_embeddings.stem.0.weight从32通道到16通道")
                                    new_value = value[:16, :, :, :, :]
                                    new_state_dict[key] = new_value
                                    continue
                            elif key == 'patch_embeddings.stem.0.bias':
                                # 从32通道裁剪到16通道
                                if value.shape[0] == 32 and self.pytorch_model.patch_embeddings.stem[0].bias.shape[0] == 16:
                                    print("调整patch_embeddings.stem.0.bias从32通道到16通道")
                                    new_value = value[:16]
                                    new_state_dict[key] = new_value
                                    continue
                            
                            # 其他权重，如果形状匹配则保留
                            if key in self.pytorch_model.state_dict():
                                if value.shape == self.pytorch_model.state_dict()[key].shape:
                                    new_state_dict[key] = value
                                    print(f"加载权重: {key}")
                                else:
                                    print(f"跳过形状不匹配的权重: {key}")
                            else:
                                print(f"跳过不存在的权重: {key}")
                        
                        # 加载调整后的权重
                        self.pytorch_model.load_state_dict(new_state_dict, strict=False)
                        print("小型模型权重加载成功（部分权重可能被跳过）")
                    else:
                        # 大型模型：直接加载
                        self.pytorch_model.load_state_dict(state_dict)
                        print("大型模型权重加载成功")
                except Exception as e:
                    print(f"模型权重加载失败: {e}")
                    raise RuntimeError(f"无法加载模型权重: {e}")
                
                self.pytorch_model.to(device)
                self.pytorch_model.eval()
                
                self.model_input_shape = (1, 1, 96, 96, 96)  # 与 pred4.py 一致
                self.model_output_shape = (1, 3, 96, 96, 96)  # 与 pred4.py 一致
                self.model_type = 'pytorch'
                
                print(f"PyTorch 模型加载成功: {model_path}")
                print(f"模型变体: {self.model_variant}")
                print(f"血管分解方法: {self.decomposition_method}")
                print(f"输入形状: {self.model_input_shape}")
                print(f"输出形状: {self.model_output_shape}")
                
            else:
                raise ValueError(f"不支持的模型格式: {ext}")
            
            self.is_model_loaded = True
            return True
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.is_model_loaded = False
            return False
    
    def set_model_variant(self, variant: str):
        """设置MobileUViT模型变体
        
        Args:
            variant: 'mobileuvit' 或 'mobileuvit_l'
        """
        self.model_variant = variant
        print(f"模型变体已设置为: {variant}")
    
    def set_decomposition_method(self, method: str):
        """设置血管分解方法
        
        Args:
            method: 'adaptive', 'fixed_1.0', 'fixed_1.5', 'fixed_2.0', 'fixed_2.5',
                    'median', 'percentile_25', 'percentile_50', 'percentile_75'
        """
        self.decomposition_method = method
        print(f"血管分解方法已设置为: {method}")
    
    def get_model_variants(self):
        """获取可用的模型变体列表"""
        return ['mobileuvit', 'mobileuvit_l']
    
    def get_decomposition_methods(self):
        """获取可用的血管分解方法列表"""
        return [
            'adaptive',
            'fixed_1.0',
            'fixed_1.5', 
            'fixed_2.0',
            'fixed_2.5',
            'median',
            'percentile_25',
            'percentile_50',
            'percentile_75'
        ]

    def _skeletonize_3d_mask(self, vessel_mask: np.ndarray) -> np.ndarray:
        """3D骨架化，兼容不同scikit-image版本。"""
        from skimage import morphology

        try:
            # 新版本优先
            skeleton = morphology.skeletonize_3d(vessel_mask)
            return skeleton.astype(bool)
        except Exception:
            # 兼容回退
            skeleton = morphology.skeletonize(vessel_mask)
            return skeleton.astype(bool)

    def _compute_level_maps(
        self,
        vessel_mask: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Dict[str, np.ndarray]:
        """
        基于距离变换计算血管分级、主干/细分支、分叉点。
        返回:
            level_map: 0背景, 1~5分别对应5个等级
            trunk_map: 主干区域
            small_map: 细分支区域
            junction_map: 分叉点邻域
            skeleton: 骨架
            radius_mm_map: 半径(mm)估计
        """
        from scipy import ndimage

        vessel = vessel_mask.astype(bool)
        if not np.any(vessel):
            zero = np.zeros_like(vessel, dtype=bool)
            return {
                "level_map": np.zeros_like(vessel, dtype=np.uint8),
                "trunk_map": zero,
                "small_map": zero,
                "junction_map": zero,
                "skeleton": zero,
                "radius_mm_map": np.zeros_like(vessel, dtype=np.float32),
            }

        sx, sy, sz = spacing
        edt_mm = ndimage.distance_transform_edt(vessel, sampling=(sx, sy, sz))
        skeleton = self._skeletonize_3d_mask(vessel)

        if not np.any(skeleton):
            # 没有骨架时兜底：全部视作small
            return {
                "level_map": np.where(vessel, 5, 0).astype(np.uint8),
                "trunk_map": np.zeros_like(vessel, dtype=bool),
                "small_map": vessel,
                "junction_map": np.zeros_like(vessel, dtype=bool),
                "skeleton": skeleton,
                "radius_mm_map": edt_mm.astype(np.float32),
            }

        # 把骨架半径传播到管腔体素
        _, idx = ndimage.distance_transform_edt(~skeleton, return_indices=True)
        propagated_r = edt_mm[tuple(idx)]

        # 5级分层（按直径）
        # 1级: 直径 > 3.0mm   => r >= 1.5
        # 2级: 2.0~3.0mm      => 1.0 <= r < 1.5
        # 3级: 1.0~2.0mm      => 0.5 <= r < 1.0
        # 4级: 0.5~1.0mm      => 0.25 <= r < 0.5
        # 5级: <0.5mm         => r < 0.25
        level1 = (propagated_r >= 1.5) & vessel
        level2 = (propagated_r >= 1.0) & (propagated_r < 1.5) & vessel
        level3 = (propagated_r >= 0.5) & (propagated_r < 1.0) & vessel
        level4 = (propagated_r >= 0.25) & (propagated_r < 0.5) & vessel
        level5 = (propagated_r < 0.25) & vessel

        level_map = np.zeros_like(vessel, dtype=np.uint8)
        level_map[level1] = 1
        level_map[level2] = 2
        level_map[level3] = 3
        level_map[level4] = 4
        level_map[level5] = 5

        # 分叉点检测：骨架邻居>=3
        structure = ndimage.generate_binary_structure(3, 2)  # 26连通
        neighbors = ndimage.convolve(skeleton.astype(np.uint8), structure, mode='constant')
        # neighbors包含中心点自身，因此阈值>=4等价于至少3个邻居
        junction_points = skeleton & (neighbors >= 4)
        junction_map = ndimage.binary_dilation(junction_points, iterations=1) & vessel

        # 主干定义：1级+2级；细分支定义：其余血管
        trunk_map = (level1 | level2) & vessel
        small_map = vessel & (~trunk_map)

        return {
            "level_map": level_map,
            "trunk_map": trunk_map,
            "small_map": small_map,
            "junction_map": junction_map,
            "skeleton": skeleton,
            "radius_mm_map": propagated_r.astype(np.float32),
        }

    def analyze_vessel_structure(
        self,
        mask: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Dict[str, Any]:
        """
        对PV/HV做结构识别（主干/细支/分叉点/5级分层）并缓存结果。
        """
        result: Dict[str, Any] = {
            "spacing": spacing,
            "pv": {},
            "hv": {},
            "structure_map": np.zeros_like(mask, dtype=np.uint8),  # 0背景 1主干 2细支 3分叉
            "level_map": np.zeros_like(mask, dtype=np.uint8),      # 0背景 1..5分级
        }

        vessel_defs = {1: "pv", 2: "hv"}
        for label_id, name in vessel_defs.items():
            vessel_mask = (mask == label_id)
            info = self._compute_level_maps(vessel_mask, spacing=spacing)

            # 统计
            level_map = info["level_map"]
            level_counts = {
                "level_1": int(np.sum(level_map == 1)),
                "level_2": int(np.sum(level_map == 2)),
                "level_3": int(np.sum(level_map == 3)),
                "level_4": int(np.sum(level_map == 4)),
                "level_5_plus": int(np.sum(level_map == 5)),
            }

            result[name] = {
                "mask": vessel_mask.astype(np.uint8),
                "trunk_map": info["trunk_map"].astype(np.uint8),
                "small_map": info["small_map"].astype(np.uint8),
                "junction_map": info["junction_map"].astype(np.uint8),
                "skeleton": info["skeleton"].astype(np.uint8),
                "level_map": level_map.astype(np.uint8),
                "radius_mm_map": info["radius_mm_map"],
                "voxel_count": int(np.sum(vessel_mask)),
                "trunk_voxel_count": int(np.sum(info["trunk_map"])),
                "small_voxel_count": int(np.sum(info["small_map"])),
                "junction_voxel_count": int(np.sum(info["junction_map"])),
                "level_counts": level_counts,
            }

            # 融合到全局图（若重叠，分叉优先）
            result["level_map"] = np.maximum(result["level_map"], level_map.astype(np.uint8))
            result["structure_map"][info["trunk_map"]] = 1
            result["structure_map"][info["small_map"]] = np.maximum(
                result["structure_map"][info["small_map"]], 2
            )
            result["structure_map"][info["junction_map"]] = 3

        self.last_vessel_analysis = result
        return result

    def get_last_vessel_analysis(self) -> Dict[str, Any]:
        """获取最近一次血管结构分析结果。"""
        return self.last_vessel_analysis if self.last_vessel_analysis else {}

    def _normalize_spacing_tuple(
        self,
        spacing: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """归一化spacing，确保为正值。"""
        try:
            sx = float(spacing[0])
            sy = float(spacing[1])
            sz = float(spacing[2])
            sx = sx if sx > 0 else 1.0
            sy = sy if sy > 0 else 1.0
            sz = sz if sz > 0 else 1.0
            return sx, sy, sz
        except Exception:
            return 1.0, 1.0, 1.0

    def _prepare_nifti_payload(
        self,
        image_data: np.ndarray,
        spacing: Tuple[float, float, float],
        metadata: Optional[Mapping[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        将内存图像准备为NIfTI写盘所需的数据与仿射。
        返回:
            nii_data: 写入NIfTI的数据
            affine: 4x4仿射
            swapped_xy: 是否做过x/y轴互换（用于结果还原）
        """
        nii_data = np.asarray(image_data, dtype=np.float32)
        swapped_xy = False

        # DICOM/NRRD(SITK)路径在本项目内采用(H, W, D)=(Y, X, Z)。
        # 写入NIfTI前转为(X, Y, Z)，确保与affine语义一致。
        if isinstance(metadata, Mapping) and "sitk_image" in metadata and nii_data.ndim == 3:
            nii_data = np.transpose(nii_data, (1, 0, 2))
            swapped_xy = True

        affine = None
        if isinstance(metadata, Mapping):
            meta_affine = metadata.get("affine")
            if meta_affine is not None:
                candidate = np.asarray(meta_affine, dtype=np.float64)
                if candidate.shape == (4, 4) and np.all(np.isfinite(candidate)):
                    affine = candidate

        if affine is None:
            sx, sy, sz = self._normalize_spacing_tuple(spacing)
            affine = np.diag([sx, sy, sz, 1.0]).astype(np.float64)

        return nii_data, affine, swapped_xy

    def _restore_internal_axis_order(self, mask: np.ndarray, swapped_xy: bool) -> np.ndarray:
        """将NIfTI空间结果还原为项目内部轴顺序。"""
        if swapped_xy and mask.ndim == 3:
            return np.transpose(mask, (1, 0, 2))
        return mask

    def _fragmentation_metrics(self, mask: np.ndarray) -> Tuple[int, float]:
        """
        计算碎片化指标：
        - 连通域个数（联合PV/HV）
        - 最大连通域占比
        """
        from scipy import ndimage

        vessel = mask > 0
        total = int(np.count_nonzero(vessel))
        if total == 0:
            return 0, 0.0

        structure = ndimage.generate_binary_structure(3, 2)
        comp, n_comp = ndimage.label(vessel, structure=structure)
        if n_comp == 0:
            return 0, 0.0

        sizes = np.bincount(comp.ravel())
        sizes[0] = 0
        largest = int(sizes.max()) if sizes.size > 1 else 0
        largest_ratio = (largest / float(total)) if total > 0 else 0.0
        return int(n_comp), float(largest_ratio)

    def _cleanup_fragmented_vessels(self, mask: np.ndarray) -> np.ndarray:
        """
        针对明显碎片化的AI结果做连通性约束：
        1) 先在联合血管上保留主要连通团块
        2) 再按标签保留主连通结构，去除远离主干的散点伪阳性
        """
        from scipy import ndimage

        if mask is None or mask.ndim != 3:
            return mask

        if not np.any(mask > 0):
            return mask

        structure = ndimage.generate_binary_structure(3, 2)
        cleaned = np.copy(mask).astype(np.uint8)

        # 先轻度闭运算，连接近邻碎块，避免主血管被切成很多段
        vessel_union = ndimage.binary_closing(cleaned > 0, structure=np.ones((3, 3, 3)))
        union_comp, union_n = ndimage.label(vessel_union, structure=structure)
        if union_n > 0:
            union_sizes = np.bincount(union_comp.ravel())
            union_sizes[0] = 0
            sorted_ids = np.argsort(union_sizes)[::-1]
            largest = int(union_sizes[sorted_ids[0]]) if sorted_ids.size > 0 else 0

            keep_union_ids = []
            for cid in sorted_ids:
                if cid == 0:
                    continue
                size = int(union_sizes[cid])
                if size <= 0:
                    continue
                # 保留最大的主体；若第二大也足够大则一起保留，防止误删另一条主静脉树
                if not keep_union_ids:
                    keep_union_ids.append(int(cid))
                    continue
                if len(keep_union_ids) < 2 and size >= max(1500, int(0.22 * largest)):
                    keep_union_ids.append(int(cid))
                    continue
                break

            if keep_union_ids:
                union_keep = np.isin(union_comp, keep_union_ids)
                cleaned[~union_keep] = 0

        # 分标签精修：去除明显“薄片型”小连通域，只保留主连通成分
        refined = np.zeros_like(cleaned, dtype=np.uint8)
        for label in (1, 2):
            binary = cleaned == label
            if not np.any(binary):
                continue

            comp, n_comp = ndimage.label(binary, structure=structure)
            if n_comp == 0:
                continue

            sizes = np.bincount(comp.ravel())
            sizes[0] = 0
            ids = np.argsort(sizes)[::-1]
            largest = int(sizes[ids[0]]) if ids.size > 0 else 0
            keep_ids = []

            for cid in ids:
                if cid == 0:
                    continue
                size = int(sizes[cid])
                if size <= 0:
                    continue
                cmask = comp == cid
                coords = np.argwhere(cmask)
                if coords.size == 0:
                    continue
                span = coords.max(axis=0) - coords.min(axis=0) + 1
                min_span = int(span.min())
                max_span = int(span.max())

                # 过滤“单层薄片”类伪阳性：很扁且体素数不大
                if min_span <= 2 and max_span >= 18 and size < max(4000, int(0.05 * largest)):
                    continue

                if not keep_ids:
                    keep_ids.append(int(cid))
                    continue

                # 额外保留次级主分支（足够大）
                if len(keep_ids) < 2 and size >= max(500, int(0.2 * largest)):
                    keep_ids.append(int(cid))
                    continue

            if keep_ids:
                refined[np.isin(comp, keep_ids)] = label

        # 若过度清理导致空结果，则回退到清理前
        if np.any(refined > 0):
            cleaned = refined

        return cleaned

    def _keep_largest_component_per_label(self, mask: np.ndarray) -> np.ndarray:
        """保守模式：每个标签仅保留最大连通域。"""
        from scipy import ndimage

        if mask is None or mask.ndim != 3:
            return mask

        out = np.zeros_like(mask, dtype=np.uint8)
        structure = ndimage.generate_binary_structure(3, 2)
        for label in (1, 2):
            comp, n_comp = ndimage.label(mask == label, structure=structure)
            if n_comp == 0:
                continue
            sizes = np.bincount(comp.ravel())
            sizes[0] = 0
            best_id = int(np.argmax(sizes))
            if best_id > 0 and sizes[best_id] > 0:
                out[comp == best_id] = label

        return out
    
    def preprocess_image(self, image_data: np.ndarray) -> np.ndarray:
        """
        预处理图像数据
        
        Args:
            image_data: 输入图像 (H, W, D)
            
        Returns:
            预处理后的图像，形状为模型输入要求
        """
        # 归一化到0-1
        img_min = image_data.min()
        img_max = image_data.max()
        if img_max > img_min:
            img_normalized = (image_data - img_min) / (img_max - img_min)
        else:
            img_normalized = image_data.astype(np.float32)
        
        # 根据模型输入形状调整
        # 假设模型输入为 (1, 1, H, W, D) 或 (1, H, W, D)
        if len(self.model_input_shape) == 5:
            # 5D输入：(batch, channel, H, W, D)
            img_processed = img_normalized[np.newaxis, np.newaxis, :, :, :]
        elif len(self.model_input_shape) == 4:
            # 4D输入：(batch, H, W, D)
            img_processed = img_normalized[np.newaxis, :, :, :]
        else:
            # 其他情况，尝试自动适配
            img_processed = img_normalized
        
        # 转换为float32
        img_processed = img_processed.astype(np.float32)
        
        return img_processed
    
    def postprocess_mask(
        self,
        model_output: np.ndarray,
        original_shape: Tuple[int, int, int],
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """
        后处理模型输出
        
        Args:
            model_output: 模型输出
            original_shape: 原始图像形状 (H, W, D)
            
        Returns:
            后处理后的掩码 (H, W, D)，uint8类型
            标签定义：
            0: 背景
            1: 门静脉
            2: 肝静脉
        """
        # 获取输出数据
        if isinstance(model_output, list):
            output_data = model_output[0]
        else:
            output_data = model_output

        # 移除batch维度（仅当首维为1时）
        while output_data.ndim > 3 and output_data.shape[0] == 1:
            output_data = np.squeeze(output_data, axis=0)

        # 处理多通道输出（3通道：背景、门静脉、肝静脉）
        if output_data.ndim == 4:
            if output_data.shape[0] == 3:
                # C,H,W,D -> H,W,D
                output_data = np.argmax(output_data, axis=0)
            elif output_data.shape[-1] == 3:
                # H,W,D,C -> H,W,D
                output_data = np.argmax(output_data, axis=-1)

        # 确保是uint8类型
        mask = output_data.astype(np.uint8)

        # 调整尺寸（如果需要），使用最近邻保持多标签
        if mask.shape != original_shape:
            try:
                from scipy.ndimage import zoom
                zoom_factors = [float(orig) / float(cur) for orig, cur in zip(original_shape, mask.shape)]
                mask = zoom(mask.astype(np.float32), zoom_factors, order=0, mode="nearest")
                mask = np.rint(mask).astype(np.uint8)
            except Exception as e:
                print(f"掩码尺寸回采样失败: {e}")

        # 基础去噪：去掉每类血管极小孤立块，降低假阳性
        try:
            from scipy import ndimage
            clean_mask = np.zeros_like(mask, dtype=np.uint8)
            for label in (1, 2):
                comp, n_comp = ndimage.label(mask == label)
                if n_comp == 0:
                    continue
                sizes = np.bincount(comp.ravel())
                # 背景索引0不算；阈值自适应，最低8体素
                min_keep = max(8, int(0.00001 * mask.size))
                keep_ids = np.where(sizes >= min_keep)[0]
                keep_ids = keep_ids[keep_ids != 0]
                if keep_ids.size > 0:
                    clean_mask[np.isin(comp, keep_ids)] = label
            # 若清洗后全空，回退
            if np.any(clean_mask > 0):
                mask = clean_mask
        except Exception as e:
            print(f"后处理去噪失败，继续使用原始掩码: {e}")

        # 碎片化检测与连通性清理（解决3D中大量离散薄片的问题）
        try:
            n_comp, largest_ratio = self._fragmentation_metrics(mask)
            print(f"分割连通性评估: components={n_comp}, largest_ratio={largest_ratio:.3f}")
            if n_comp >= 14 or largest_ratio < 0.42:
                print("检测到明显碎片化，启用主连通血管清理...")
                cleaned_mask = self._cleanup_fragmented_vessels(mask)
                if np.any(cleaned_mask > 0):
                    mask = cleaned_mask
                    n2, r2 = self._fragmentation_metrics(mask)
                    print(f"清理后连通性: components={n2}, largest_ratio={r2:.3f}")
                    if n2 >= 10 or r2 < 0.50:
                        print("仍然存在碎片，启用最大连通域保守模式...")
                        conservative = self._keep_largest_component_per_label(mask)
                        if np.any(conservative > 0):
                            mask = conservative
                            n3, r3 = self._fragmentation_metrics(mask)
                            print(f"保守模式后连通性: components={n3}, largest_ratio={r3:.3f}")
        except Exception as e:
            print(f"连通性清理失败，继续使用原始掩码: {e}")

        # 结构识别：PV/HV 主干/细分支/分叉/5级分层
        try:
            analysis = self.analyze_vessel_structure(mask, spacing=spacing)

            # 分解策略控制“显示保留级别”
            # 默认尽量保留细分支，满足5级以上识别需求
            keep_max_level = 5
            m = self.decomposition_method or "adaptive"
            if m in ("fixed_2.5", "percentile_75"):
                keep_max_level = 2
            elif m in ("fixed_2.0",):
                keep_max_level = 3
            elif m in ("fixed_1.5",):
                keep_max_level = 4
            # adaptive / percentile_25 / percentile_50 / median / fixed_1.0 -> 保留到5级

            if keep_max_level < 5:
                filtered = np.zeros_like(mask, dtype=np.uint8)
                for label, name in ((1, "pv"), (2, "hv")):
                    level_map = analysis.get(name, {}).get("level_map")
                    if level_map is None:
                        continue
                    keep = (level_map > 0) & (level_map <= keep_max_level)
                    filtered[keep] = label
                # 避免过度裁剪导致空结果
                if np.any(filtered > 0):
                    mask = filtered
                    # 重新分析一次，保证缓存和最终mask一致
                    self.analyze_vessel_structure(mask, spacing=spacing)

            print(f"血管结构识别完成: 非零体素={np.count_nonzero(mask)}")
            print(f"  PV体素: {np.sum(mask == 1)}")
            print(f"  HV体素: {np.sum(mask == 2)}")
        except Exception as e:
            self.last_vessel_analysis = {}
            print(f"血管结构识别失败: {e}")

        return mask
    
    def segment(
        self,
        image_data: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional[np.ndarray]:
        """
        执行AI分割
        
        Args:
            image_data: 输入图像 (H, W, D)
            spacing: 体素间距
            metadata: 影像元数据（优先使用其中affine）
            
        Returns:
            分割掩码 (H, W, D)，uint8类型，失败返回None
        """
        if not self.is_model_loaded:
            raise RuntimeError("模型未加载，请先调用load_model()")
        
        try:
            original_shape = tuple(int(v) for v in image_data.shape)
            
            if self.model_type == 'onnx':
                # ONNX 模型推理
                if self.model_session is None:
                    raise RuntimeError("ONNX 模型会话未初始化")
                
                # 预处理
                input_data = self.preprocess_image(image_data)
                
                # 获取输入输出名称
                input_name = self.model_session.get_inputs()[0].name
                output_name = self.model_session.get_outputs()[0].name
                
                # 推理
                outputs = self.model_session.run([output_name], {input_name: input_data})
                
                # 后处理
                mask = self.postprocess_mask(outputs[0], original_shape, spacing=spacing)
                
            elif self.model_type == 'pytorch' and PYTORCH_AVAILABLE:
                # PyTorch 模型推理（与 pred4.py 逻辑一致）
                if self.pytorch_model is None:
                    raise RuntimeError("PyTorch 模型未初始化")
                
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                
                # 构建临时 NIfTI 文件路径
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tmp:
                    tmp_path = tmp.name
                
                # 保存临时 NIfTI 文件
                nifti_data, nifti_affine, swapped_xy = self._prepare_nifti_payload(
                    image_data=image_data,
                    spacing=spacing,
                    metadata=metadata,
                )
                tmp_nii = nib.Nifti1Image(nifti_data, affine=nifti_affine)
                nib.save(tmp_nii, tmp_path)
                
                try:
                    # 定义变换（与 pred4.py 一致）
                    PATCH_SIZE = (96, 96, 96)
                    
                    # 空间变换
                    spatial_ops = [
                        LoadImaged(keys=["image"]),
                        EnsureChannelFirstd(keys=["image"]),
                        Orientationd(keys=["image"], axcodes="LPI"),
                        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=('bilinear')),
                        # 无GT标签时使用更稳健的体部阈值，避免ROI过碎导致伪阳性扩散
                        CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > -200),
                        SpatialPadd(keys=["image"], spatial_size=PATCH_SIZE),
                    ]
                    
                    print(f"使用参数: 模型变体={self.model_variant}, 血管分解方法={self.decomposition_method}")
                    
                    # 推理专用 Transform
                    inf_transforms = Compose(spatial_ops + [
                        ScaleIntensityRanged(keys="image", a_min=0, a_max=400, b_min=0.0, b_max=1.0, clip=True),
                        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                        EnsureTyped(keys=["image"]),
                    ])
                    
                    # 执行变换
                    data_inf = inf_transforms({"image": tmp_path})
                    input_tensor = data_inf["image"].unsqueeze(0).to(device)
                    
                    # 滑窗推理
                    inferer = SlidingWindowInferer(roi_size=PATCH_SIZE, sw_batch_size=1, overlap=0.6, mode="gaussian")
                    
                    with torch.no_grad():
                        output = inferer(input_tensor, self.pytorch_model)
                        # 兼容Tensor/tuple输出，统一提取全局多分类logits
                        logits = output[0] if isinstance(output, (tuple, list)) else output
                        if not isinstance(logits, torch.Tensor):
                            raise RuntimeError(f"模型输出类型异常: {type(logits)}")
                        if logits.ndim == 4:
                            logits = logits.unsqueeze(0)
                        if logits.ndim != 5:
                            raise RuntimeError(f"模型输出维度异常: {tuple(logits.shape)}")
                        pred_label = torch.argmax(logits, dim=1).cpu().numpy()  # [B, H, W, D]
                    
                    # 还原到原始空间
                    orig_nii = nib.load(tmp_path)
                    pred_nii_cropped = nib.Nifti1Image(
                        pred_label[0].astype(np.uint8), 
                        data_inf["image"].meta["affine"]
                    )
                    
                    final_pred_nii = resample_to_img(
                        pred_nii_cropped, 
                        orig_nii, 
                        interpolation='nearest'
                    )
                    
                    mask = final_pred_nii.get_fdata().astype(np.uint8)
                    mask = self._restore_internal_axis_order(mask, swapped_xy)
                    # PyTorch路径也做统一结构识别与策略后处理
                    mask = self.postprocess_mask(mask, original_shape, spacing=spacing)
                    
                finally:
                    # 清理临时文件
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
            
            return mask
            
        except Exception as e:
            print(f"AI分割失败: {e}")
            return None
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.is_model_loaded
