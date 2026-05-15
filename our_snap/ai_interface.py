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
from contextlib import nullcontext

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
        self.last_loaded_model_path: Optional[str] = None
        self.last_loaded_model_mtime: Optional[float] = None
        self.last_loaded_variant: Optional[str] = None
        self.last_loaded_type: Optional[str] = None
        
        # MobileUViT 模型参数
        self.model_variant = 'mobileuvit_l'  # 'mobileuvit' 或 'mobileuvit_l'
        self.decomposition_method = 'percentile_50'  # 血管分解方法
        # 严格碎片清理：每个标签仅保留最大连通域（默认关闭，避免误删主体）
        strict_flag = os.environ.get("VESSELSEG_STRICT_FRAGMENT_PRUNE", "0")
        self.strict_fragment_prune = str(strict_flag).strip().lower() not in ("0", "false", "off", "no")
        # 重清理（连通域形态学强裁剪）默认关闭，避免把主体裁成大块
        heavy_flag = os.environ.get("VESSELSEG_HEAVY_CLEANUP", "0")
        self.heavy_cleanup = str(heavy_flag).strip().lower() in ("1", "true", "on", "yes")
        # 速度-质量预设：
        # - quality(默认): 精度优先，碎片更少
        # - fast: 速度优先
        speed_profile = str(os.environ.get("VESSELSEG_SPEED_PROFILE", "quality")).strip().lower()
        self.speed_profile = speed_profile if speed_profile in ("quality", "fast") else "quality"
        postprocess_profile = str(os.environ.get("VESSELSEG_POSTPROCESS_PROFILE", "")).strip().lower()
        if not postprocess_profile:
            postprocess_profile = "balanced"
        self.postprocess_profile = postprocess_profile if postprocess_profile in ("clean", "balanced", "recall") else "balanced"
        if self.postprocess_profile == "clean":
            self.cleanup_min_retain_ratio = 0.40
            self.final_min_retain_ratio = 0.60
            self.prune_min_retain_ratio = 0.0
            self.primary_max_components = 3
            self.primary_min_ratio = 0.12
            self.primary_min_voxels = 120
            self.auto_volume_weight = 0.20
        elif self.postprocess_profile == "balanced":
            self.cleanup_min_retain_ratio = 0.45
            self.final_min_retain_ratio = 0.55
            self.prune_min_retain_ratio = 0.60
            self.primary_max_components = 3
            self.primary_min_ratio = 0.10
            self.primary_min_voxels = 120
            self.auto_volume_weight = 0.20
        else:
            self.cleanup_min_retain_ratio = 0.78
            self.final_min_retain_ratio = 0.88
            self.prune_min_retain_ratio = 0.94
            self.primary_max_components = 5
            self.primary_min_ratio = 0.06
            self.primary_min_voxels = 80
            self.auto_volume_weight = 0.45
        default_fragment_level = "conservative" if self.postprocess_profile == "recall" else "balanced"
        fragment_level = os.environ.get("VESSELSEG_FRAGMENT_CLEAN_LEVEL", default_fragment_level)
        self.fragment_clean_level = str(fragment_level).strip().lower()
        default_infer_mode = "stable" if self.speed_profile == "fast" else "auto"
        # PyTorch推理模式：auto(stable+legacy融合) / stable / legacy
        infer_mode = os.environ.get("VESSELSEG_PYTORCH_INFER_MODE", default_infer_mode)
        self.pytorch_infer_mode = str(infer_mode).strip().lower() if infer_mode is not None else default_infer_mode
        # 质量模式默认全图推理，避免 ROI 裁剪漏掉远端血管分支；fast 模式才默认启用 ROI。
        default_roi_focus = "1" if self.speed_profile == "fast" else "0"
        roi_flag = os.environ.get("VESSELSEG_ROI_FOCUS", default_roi_focus)
        self.roi_focus_enabled = str(roi_flag).strip().lower() not in ("0", "false", "off", "no")
        # 稳定路径默认启用1mm各向同性重采样，降低厚层扫描带来的条带碎片
        iso_flag = os.environ.get("VESSELSEG_STABLE_ISO", "1")
        self.stable_iso_enabled = str(iso_flag).strip().lower() not in ("0", "false", "off", "no")
        default_iso_max_vox = "18000000" if self.speed_profile == "fast" else "50000000"
        iso_max_vox = os.environ.get("VESSELSEG_STABLE_ISO_MAX_VOXELS", default_iso_max_vox)
        try:
            self.stable_iso_max_voxels = max(2_000_000, int(iso_max_vox))
        except Exception:
            self.stable_iso_max_voxels = 18_000_000 if self.speed_profile == "fast" else 50_000_000
        default_sw_overlap = "0.35" if self.speed_profile == "fast" else "0.6"
        sw_overlap = os.environ.get("VESSELSEG_SW_OVERLAP", default_sw_overlap)
        try:
            self.sw_overlap = float(sw_overlap)
        except Exception:
            self.sw_overlap = 0.35
        self.sw_overlap = float(min(0.8, max(0.05, self.sw_overlap)))
        sw_batch = os.environ.get("VESSELSEG_SW_BATCH", "")
        if str(sw_batch).strip():
            try:
                self.sw_batch_size = max(1, int(sw_batch))
            except Exception:
                self.sw_batch_size = 1
        else:
            self.sw_batch_size = 2 if torch.cuda.is_available() else 1
        amp_flag = os.environ.get("VESSELSEG_USE_AMP", "1")
        self.use_amp = str(amp_flag).strip().lower() not in ("0", "false", "off", "no")
        # 召回增强会把低置信度血管候选救回来；它能减少漏分，但也容易制造假阳性。
        # 默认只在显式 recall 模式启用，平衡/清理模式优先保证与医生标签的主体一致性。
        default_recall_boost = "1" if self.postprocess_profile == "recall" else "0"
        recall_flag = os.environ.get("VESSELSEG_RECALL_BOOST", default_recall_boost)
        self.recall_boost_enabled = str(recall_flag).strip().lower() not in ("0", "false", "off", "no")
        default_recall_profile = "aggressive" if self.postprocess_profile == "recall" else "balanced"
        recall_profile = str(os.environ.get("VESSELSEG_RECALL_PROFILE", default_recall_profile)).strip().lower()
        self.recall_profile = recall_profile if recall_profile in ("conservative", "balanced", "aggressive") else "balanced"
        recall_defaults = {
            "conservative": (0.40, -0.03),
            "balanced": (0.33, -0.10),
            "aggressive": (0.28, -0.18),
        }
        default_prob, default_margin = recall_defaults[self.recall_profile]
        try:
            self.recall_min_vessel_prob = float(os.environ.get("VESSELSEG_RECALL_MIN_PROB", default_prob))
        except Exception:
            self.recall_min_vessel_prob = default_prob
        try:
            self.recall_bg_margin = float(os.environ.get("VESSELSEG_RECALL_BG_MARGIN", default_margin))
        except Exception:
            self.recall_bg_margin = default_margin
        self.recall_min_vessel_prob = float(min(0.75, max(0.05, self.recall_min_vessel_prob)))
        self.recall_bg_margin = float(min(0.50, max(-0.60, self.recall_bg_margin)))
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
            model_path = os.path.abspath(model_path)
            model_mtime = float(os.path.getmtime(model_path))
            
            # 根据文件扩展名判断模型类型
            ext = os.path.splitext(model_path)[1].lower()
            current_type = "onnx" if ext == ".onnx" else ("pytorch" if ext in [".pth", ".pt"] else None)
            if current_type is None:
                raise ValueError(f"不支持的模型格式: {ext}")

            # 已加载同一模型/同一变体时直接复用，避免重复加载权重导致的卡顿
            if (
                self.is_model_loaded
                and self.last_loaded_model_path == model_path
                and self.last_loaded_model_mtime == model_mtime
                and self.last_loaded_type == current_type
                and (
                    current_type != "pytorch"
                    or self.last_loaded_variant == self.model_variant
                )
            ):
                print("模型已加载且未变化，复用现有模型")
                return True
             
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
                if device.type == "cuda":
                    try:
                        torch.backends.cudnn.benchmark = True
                    except Exception:
                        pass
                
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
            self.last_loaded_model_path = model_path
            self.last_loaded_model_mtime = model_mtime
            self.last_loaded_variant = self.model_variant if self.model_type == "pytorch" else None
            self.last_loaded_type = self.model_type
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
        old_variant = self.model_variant
        self.model_variant = variant
        if old_variant != variant and self.model_type == "pytorch":
            # 变体切换后需重新加载权重
            self.is_model_loaded = False
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

    def _cleanup_fragmented_vessels(
        self,
        mask: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
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
        sx, sy, sz = self._normalize_spacing_tuple(spacing)
        cleaned = np.copy(mask).astype(np.uint8)

        # 先轻度闭运算，连接近邻碎块，避免主血管被切成很多段
        vessel_union = ndimage.binary_closing(cleaned > 0, structure=np.ones((3, 3, 3)))
        union_comp, union_n = ndimage.label(vessel_union, structure=structure)
        if union_n > 0:
            union_sizes = np.bincount(union_comp.ravel())
            union_sizes[0] = 0
            sorted_ids = np.argsort(union_sizes)[::-1]
            largest = int(union_sizes[sorted_ids[0]]) if sorted_ids.size > 0 else 0
            if self.postprocess_profile == "recall":
                max_union_components = max(3, self.primary_max_components)
                union_min_voxels = 300
                union_min_ratio = 0.06
                union_near_mm = 70.0
                union_span_factor = 0.85
            elif self.postprocess_profile == "balanced":
                max_union_components = max(2, self.primary_max_components)
                union_min_voxels = 800
                union_min_ratio = 0.14
                union_near_mm = 55.0
                union_span_factor = 0.70
            else:
                max_union_components = 2
                union_min_voxels = 1500
                union_min_ratio = 0.28
                union_near_mm = 45.0
                union_span_factor = 0.60

            keep_union_ids = []
            first_centroid_mm = None
            first_span_mm = None
            for cid in sorted_ids:
                if cid == 0:
                    continue
                size = int(union_sizes[cid])
                if size <= 0:
                    continue
                # 保留最大的主体；若第二大也足够大则一起保留，防止误删另一条主静脉树
                if not keep_union_ids:
                    keep_union_ids.append(int(cid))
                    coords = np.argwhere(union_comp == cid)
                    if coords.size > 0:
                        centroid = coords.mean(axis=0)
                        first_centroid_mm = np.array([centroid[0] * sx, centroid[1] * sy, centroid[2] * sz], dtype=np.float64)
                        span = coords.max(axis=0) - coords.min(axis=0) + 1
                        first_span_mm = np.array([span[0] * sx, span[1] * sy, span[2] * sz], dtype=np.float64)
                    continue

                # 第二大连通域需足够大且与主体空间邻近，避免保留远处整块假阳性
                if (
                    len(keep_union_ids) < max_union_components
                    and size >= max(union_min_voxels, int(union_min_ratio * largest))
                ):
                    near_main = False
                    coords = np.argwhere(union_comp == cid)
                    if coords.size > 0 and first_centroid_mm is not None and first_span_mm is not None:
                        centroid = coords.mean(axis=0)
                        centroid_mm = np.array([centroid[0] * sx, centroid[1] * sy, centroid[2] * sz], dtype=np.float64)
                        dist_mm = float(np.linalg.norm(centroid_mm - first_centroid_mm))
                        ref_mm = float(np.linalg.norm(first_span_mm))
                        max_allow_mm = max(union_near_mm, union_span_factor * ref_mm)
                        near_main = dist_mm <= max_allow_mm
                    if near_main:
                        keep_union_ids.append(int(cid))
                        continue
                    if self.postprocess_profile != "clean":
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
                thin_limit = max(4000, int(0.05 * largest))
                if self.postprocess_profile == "recall":
                    thin_limit = max(250, int(0.015 * largest))
                elif self.postprocess_profile == "balanced":
                    thin_limit = max(1200, int(0.03 * largest))
                if min_span <= 2 and max_span >= 18 and size < thin_limit:
                    continue

                if not keep_ids:
                    keep_ids.append(int(cid))
                    continue

                # 额外保留次级主分支（足够大）
                if (
                    len(keep_ids) < self.primary_max_components
                    and size >= max(self.primary_min_voxels, int(self.primary_min_ratio * largest))
                ):
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

    def _keep_primary_components_per_label(
        self,
        mask: np.ndarray,
        max_components: int = 2,
        min_ratio: float = 0.20,
        min_voxels: int = 140,
    ) -> np.ndarray:
        """每个标签保留主连通域及少量足够大的次级主分支。"""
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
            ids = np.argsort(sizes)[::-1]
            if ids.size == 0:
                continue

            largest = int(sizes[ids[0]]) if ids[0] != 0 else 0
            keep_ids = []
            for cid in ids:
                if cid == 0:
                    continue
                size = int(sizes[cid])
                if size <= 0:
                    continue

                if not keep_ids:
                    keep_ids.append(int(cid))
                    continue

                if (
                    len(keep_ids) < max_components
                    and size >= max(min_voxels, int(min_ratio * max(largest, 1)))
                ):
                    keep_ids.append(int(cid))

            if keep_ids:
                out[np.isin(comp, keep_ids)] = label

        return out

    def _keep_primary_union_components(
        self,
        mask: np.ndarray,
        max_components: int = 1,
        min_ratio: float = 0.12,
        min_voxels: int = 500,
    ) -> np.ndarray:
        """在PV/HV联合血管上保留主体连通树，用于去掉远端假阳性碎片。"""
        from scipy import ndimage

        if mask is None or mask.ndim != 3:
            return mask

        vessel = mask > 0
        if not np.any(vessel):
            return mask

        structure = ndimage.generate_binary_structure(3, 2)
        comp, n_comp = ndimage.label(vessel, structure=structure)
        if n_comp <= max(1, int(max_components)):
            return mask.astype(np.uint8)

        sizes = np.bincount(comp.ravel())
        sizes[0] = 0
        ids = np.argsort(sizes)[::-1]
        largest = int(sizes[ids[0]]) if ids.size > 0 and ids[0] != 0 else 0
        if largest <= 0:
            return mask.astype(np.uint8)

        keep_ids = []
        for cid in ids:
            if cid == 0:
                continue
            size = int(sizes[cid])
            if size <= 0:
                continue

            if not keep_ids:
                keep_ids.append(int(cid))
                continue

            if (
                len(keep_ids) < max(1, int(max_components))
                and size >= max(int(min_voxels), int(float(min_ratio) * largest))
            ):
                keep_ids.append(int(cid))

        if not keep_ids:
            return mask.astype(np.uint8)

        out = np.zeros_like(mask, dtype=np.uint8)
        keep = np.isin(comp, keep_ids)
        out[keep] = mask[keep]
        return out

    def _retains_enough_label_volume(
        self,
        before_mask: np.ndarray,
        after_mask: np.ndarray,
        min_ratio: float,
        min_voxels: int = 250,
        context: str = "后处理",
    ) -> bool:
        """保护真实小分支，避免清理步骤把标签体素裁得过多。"""
        if min_ratio <= 0.0:
            return True
        if before_mask is None or after_mask is None:
            return True

        for label in (1, 2):
            before = int(np.sum(before_mask == label))
            after = int(np.sum(after_mask == label))
            if before < int(min_voxels):
                continue
            if after < int(min_ratio * before):
                print(
                    f"{context}触发召回保护: label={label}, "
                    f"before={before}, after={after}, "
                    f"retain={after / max(before, 1):.3f}, "
                    f"min={min_ratio:.2f}"
                )
                return False
        return True

    def _prune_far_small_fragments(
        self,
        mask: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """
        删除“远离主体”的孤立碎片，尽量不影响主体及近邻分支。
        规则采用分级阈值：
        - 距离很近：保留（可能是被切断的真实分支）
        - 距离较远且较小：删除
        - 距离很远且中等体积：删除
        并保留“非常大”的次级组件，避免误删被断开的主干段。
        """
        from scipy import ndimage

        if mask is None or mask.ndim != 3:
            return mask

        sx, sy, sz = self._normalize_spacing_tuple(spacing)
        spacing_mm = (sx, sy, sz)
        structure = ndimage.generate_binary_structure(3, 2)
        out = np.copy(mask).astype(np.uint8)
        level = str(getattr(self, "fragment_clean_level", "balanced")).strip().lower()
        is_aggressive = level in ("aggressive", "high", "2")
        is_conservative = level in ("conservative", "low", "0")

        for label in (1, 2):
            comp, n_comp = ndimage.label(out == label, structure=structure)
            if n_comp <= 1:
                continue

            sizes = np.bincount(comp.ravel())
            sizes[0] = 0
            main_id = int(np.argmax(sizes))
            if main_id <= 0:
                continue

            main_mask = comp == main_id
            coords = np.argwhere(main_mask)
            if coords.size == 0:
                continue
            main_size = int(sizes[main_id])
            main_centroid = coords.mean(axis=0).astype(np.float64)
            span = coords.max(axis=0) - coords.min(axis=0) + 1
            main_extent_mm = float(np.linalg.norm(np.array([span[0] * sx, span[1] * sy, span[2] * sz], dtype=np.float64)))

            # 预计算到主干的物理距离
            dist_to_main = ndimage.distance_transform_edt(~main_mask, sampling=spacing_mm)

            # 根据模式调整阈值：默认 balanced
            if is_conservative:
                near_keep_mm = 16.0
                soft_far_mm = min(65.0, max(30.0, 0.33 * main_extent_mm))
                hard_far_mm = min(86.0, max(46.0, 0.48 * main_extent_mm))
                small_drop_ratio = 0.30
                medium_drop_ratio = 0.50
            elif is_aggressive:
                near_keep_mm = 10.0
                soft_far_mm = min(46.0, max(18.0, 0.22 * main_extent_mm))
                hard_far_mm = min(62.0, max(28.0, 0.30 * main_extent_mm))
                small_drop_ratio = 0.42
                medium_drop_ratio = 0.72
            else:
                near_keep_mm = 12.0
                soft_far_mm = min(56.0, max(22.0, 0.26 * main_extent_mm))
                hard_far_mm = min(72.0, max(34.0, 0.36 * main_extent_mm))
                small_drop_ratio = 0.35
                medium_drop_ratio = 0.64

            # 即使距离远，也保留“非常大”的次级组件，减少误删风险
            very_large_keep_ratio = 0.80

            keep_ids = {main_id}
            ids = np.where(sizes > 0)[0]
            for cid in ids:
                if cid == 0 or cid == main_id:
                    continue
                size = int(sizes[cid])
                cmask = comp == cid
                if not np.any(cmask):
                    continue

                min_dist_mm = float(dist_to_main[cmask].min())
                size_ratio = float(size) / float(max(main_size, 1))
                if size_ratio >= very_large_keep_ratio:
                    keep_ids.add(int(cid))
                    continue

                ccoords = np.argwhere(cmask)
                if ccoords.size > 0:
                    centroid = ccoords.mean(axis=0).astype(np.float64)
                    centroid_dist_mm = float(
                        np.linalg.norm((centroid - main_centroid) * np.array([sx, sy, sz], dtype=np.float64))
                    )
                else:
                    centroid_dist_mm = float(min_dist_mm)

                # 临近主体的组件保留（可能是被切断的真实分支）
                if min_dist_mm <= near_keep_mm:
                    keep_ids.add(int(cid))
                    continue

                should_drop = False
                # 很远 + 中等或更小：删除
                if min_dist_mm > hard_far_mm and size_ratio < medium_drop_ratio:
                    should_drop = True
                # 较远 + 较小：删除
                elif min_dist_mm > soft_far_mm and size_ratio < small_drop_ratio:
                    should_drop = True
                # 补充规则：质心也明显远离且组件不大，删除
                elif (
                    min_dist_mm > 0.8 * soft_far_mm
                    and centroid_dist_mm > max(1.3 * soft_far_mm, 0.55 * main_extent_mm)
                    and size_ratio < 0.55
                ):
                    should_drop = True

                if should_drop:
                    continue
                keep_ids.add(int(cid))

            out[(out == label) & (~np.isin(comp, list(keep_ids)))] = 0

        return out

    def _estimate_focus_roi(
        self,
        image_data: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Tuple[slice, slice, slice]:
        """
        基于强度和空间先验估计肝脏血管ROI，减少全腹部误检。
        返回(H, W, D)上的切片范围。
        """
        from scipy import ndimage

        arr = np.asarray(image_data, dtype=np.float32)
        if arr.ndim != 3:
            return (slice(0, arr.shape[0]), slice(0, arr.shape[1]), slice(0, arr.shape[2]))

        h, w, d = arr.shape
        sx, sy, sz = self._normalize_spacing_tuple(spacing)

        # 体部掩码 + 对比增强血管候选
        body = arr > -300.0
        vessel_like = (arr >= 70.0) & (arr <= 360.0) & body
        vessel_like = ndimage.binary_opening(vessel_like, structure=np.ones((3, 3, 1), dtype=bool))
        vessel_like = ndimage.binary_closing(vessel_like, structure=np.ones((5, 5, 3), dtype=bool))

        structure = ndimage.generate_binary_structure(3, 1)
        comp, n_comp = ndimage.label(vessel_like, structure=structure)
        if n_comp == 0:
            return (slice(0, h), slice(0, w), slice(0, d))

        sizes = np.bincount(comp.ravel())
        sizes[0] = 0
        ids = np.where(sizes > 0)[0]
        if ids.size == 0:
            return (slice(0, h), slice(0, w), slice(0, d))

        center = np.array([h / 2.0, w / 2.0, d / 2.0], dtype=np.float64)
        shape_vec = np.array([max(1, h), max(1, w), max(1, d)], dtype=np.float64)

        best_id = int(ids[0])
        best_score = -1.0
        for cid in ids:
            coords = np.argwhere(comp == cid)
            if coords.size == 0:
                continue
            centroid = coords.mean(axis=0)
            norm_dist = float(np.linalg.norm((centroid - center) / shape_vec))
            size = int(sizes[cid])
            score = float(size) * (1.35 - min(norm_dist, 1.0))
            if score > best_score:
                best_score = score
                best_id = int(cid)

        coords = np.argwhere(comp == best_id)
        if coords.size == 0:
            return (slice(0, h), slice(0, w), slice(0, d))

        mins = coords.min(axis=0).astype(int)
        maxs = coords.max(axis=0).astype(int) + 1

        # 物理边界扩张（约45mm），并保证最小体积范围
        pad = np.array([
            max(12, int(45.0 / max(sx, 1e-6))),
            max(12, int(45.0 / max(sy, 1e-6))),
            max(8, int(45.0 / max(sz, 1e-6))),
        ], dtype=int)
        min_size = np.array([120, 120, 72], dtype=int)

        starts = mins - pad
        ends = maxs + pad
        for i in range(3):
            cur = int(ends[i] - starts[i])
            if cur < int(min_size[i]):
                need = int(min_size[i] - cur)
                starts[i] -= need // 2
                ends[i] += need - need // 2

        starts = np.maximum(starts, 0)
        ends = np.minimum(ends, np.array([h, w, d], dtype=int))

        if np.any(ends <= starts):
            return (slice(0, h), slice(0, w), slice(0, d))

        return (
            slice(int(starts[0]), int(ends[0])),
            slice(int(starts[1]), int(ends[1])),
            slice(int(starts[2]), int(ends[2])),
        )

    def _resample_volume(
        self,
        volume: np.ndarray,
        src_spacing: Tuple[float, float, float],
        dst_spacing: Tuple[float, float, float],
        order: int = 1,
    ) -> np.ndarray:
        """按spacing做体数据重采样。"""
        from scipy.ndimage import zoom

        sx, sy, sz = self._normalize_spacing_tuple(src_spacing)
        tx, ty, tz = self._normalize_spacing_tuple(dst_spacing)
        zoom_factors = [sx / tx, sy / ty, sz / tz]
        return zoom(volume, zoom_factors, order=order, mode="nearest")
    
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

    def _extract_logits(self, output: Any) -> "torch.Tensor":
        """兼容Tensor/tuple输出，统一提取logits张量。"""
        logits = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError(f"模型输出类型异常: {type(logits)}")
        if logits.ndim == 4:
            logits = logits.unsqueeze(0)
        if logits.ndim != 5:
            raise RuntimeError(f"模型输出维度异常: {tuple(logits.shape)}")
        return logits

    def _logits_to_label(self, logits: "torch.Tensor") -> "torch.Tensor":
        """把 logits 转为标签；可选召回增强，减少血管漏分。"""
        probs = torch.softmax(logits.float(), dim=1)
        pred_label = torch.argmax(probs, dim=1)
        if (
            not self.recall_boost_enabled
            or probs.shape[1] < 3
            or self.recall_min_vessel_prob <= 0.0
        ):
            return pred_label

        vessel_probs = probs[:, 1:3]
        vessel_prob, vessel_idx = torch.max(vessel_probs, dim=1)
        bg_prob = probs[:, 0]
        rescue_mask = (
            (pred_label == 0)
            & (vessel_prob >= float(self.recall_min_vessel_prob))
            & ((vessel_prob - bg_prob) >= float(self.recall_bg_margin))
        )
        rescued = int(rescue_mask.sum().item())
        if rescued <= 0:
            return pred_label

        boosted = pred_label.clone()
        boosted[rescue_mask] = vessel_idx[rescue_mask] + 1
        print(
            f"recall boost: rescued={rescued}, "
            f"profile={self.recall_profile}, "
            f"min_prob={self.recall_min_vessel_prob:.2f}, "
            f"bg_margin={self.recall_bg_margin:.2f}"
        )
        return boosted

    def _run_pytorch_infer_once(
        self,
        inferer: "SlidingWindowInferer",
        input_tensor: "torch.Tensor",
        device: "torch.device",
        amp_enabled: bool,
    ) -> np.ndarray:
        """执行一次推理，返回argmax后的标签体。"""
        with torch.no_grad():
            amp_ctx = (
                torch.cuda.amp.autocast(enabled=True)
                if (device.type == "cuda" and amp_enabled)
                else nullcontext()
            )
            with amp_ctx:
                output = inferer(input_tensor, self.pytorch_model)
                logits = self._extract_logits(output)
                finite_ok = bool(torch.isfinite(logits).all().item())
                if not finite_ok:
                    raise RuntimeError("logits含NaN/Inf")
                pred_label = self._logits_to_label(logits).to(dtype=torch.uint8).cpu().numpy()
        return pred_label

    def _infer_pred_with_guard(
        self,
        inferer: "SlidingWindowInferer",
        input_tensor: "torch.Tensor",
        device: "torch.device",
        path_name: str,
    ) -> np.ndarray:
        """
        推理保护：
        - 优先使用当前配置（CUDA+AMP）；
        - 若输出全背景或数值异常，自动回退到FP32重算一次。
        """
        prefer_amp = bool(device.type == "cuda" and self.use_amp)
        attempts = []
        if prefer_amp:
            attempts.append(("cuda_amp", True))
        attempts.append(("fp32", False))

        last_pred = None
        last_exc = None
        for idx, (tag, amp_enabled) in enumerate(attempts):
            try:
                pred_label = self._run_pytorch_infer_once(
                    inferer=inferer,
                    input_tensor=input_tensor,
                    device=device,
                    amp_enabled=amp_enabled,
                )
            except Exception as e:
                last_exc = e
                print(f"{path_name}推理({tag})失败: {e}")
                continue

            fg_vox = int(np.count_nonzero(pred_label))
            print(f"{path_name}推理({tag})前景体素={fg_vox}")
            last_pred = pred_label
            if fg_vox > 0:
                return pred_label

            # 当前尝试为全背景且仍有回退路径时，继续下一次尝试
            if idx < len(attempts) - 1:
                print(f"{path_name}推理({tag})全背景，自动回退重试")

        if last_pred is not None:
            return last_pred
        if last_exc is not None:
            raise RuntimeError(f"{path_name}推理失败: {last_exc}")
        raise RuntimeError(f"{path_name}推理失败: 未获得有效输出")

    def _prepare_stable_infer_input(
        self,
        image_data: np.ndarray,
        metadata: Optional[Mapping[str, Any]],
    ) -> Tuple[np.ndarray, bool]:
        """
        稳定推理输入准备：
        1) 仅做必要轴向对齐（SITK来源做X/Y互换）；
        2) 在内存中完成强度预处理，避免NIfTI写盘+重采样带来的跨平台差异。
        """
        arr = np.asarray(image_data, dtype=np.float32)
        swapped_xy = False

        # 项目内部DICOM/NRRD为(Y, X, Z)；模型训练习惯为(X, Y, Z)
        if isinstance(metadata, Mapping) and "sitk_image" in metadata and arr.ndim == 3:
            arr = np.transpose(arr, (1, 0, 2))
            swapped_xy = True

        arr = np.nan_to_num(arr, nan=-1024.0, posinf=4000.0, neginf=-1024.0)
        arr = np.clip(arr, 0.0, 400.0)
        arr = arr / 400.0

        # 与NormalizeIntensityd(nonzero=True)保持一致：仅对非零体素做z-score
        nonzero = arr > 0
        if np.any(nonzero):
            values = arr[nonzero]
            mean_v = float(values.mean())
            std_v = float(values.std())
            if std_v > 1e-6:
                arr[nonzero] = (values - mean_v) / std_v
            else:
                arr[nonzero] = values - mean_v

        return arr.astype(np.float32), swapped_xy

    def _segment_pytorch_stable(
        self,
        image_data: np.ndarray,
        original_shape: Tuple[int, int, int],
        spacing: Tuple[float, float, float],
        metadata: Optional[Mapping[str, Any]] = None,
        apply_postprocess: bool = True,
    ) -> np.ndarray:
        """稳定推理链路：内存直推，不走NIfTI回采样。"""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        patch_size = (96, 96, 96)
        inferer = SlidingWindowInferer(
            roi_size=patch_size,
            sw_batch_size=self.sw_batch_size,
            overlap=self.sw_overlap,
            mode="gaussian",
        )

        roi_slices = (
            self._estimate_focus_roi(image_data, spacing=spacing)
            if self.roi_focus_enabled
            else (slice(0, original_shape[0]), slice(0, original_shape[1]), slice(0, original_shape[2]))
        )
        roi_shape = tuple(int(s.stop - s.start) for s in roi_slices)
        print(f"stable ROI focus={self.roi_focus_enabled}, roi_shape={roi_shape}, full_shape={original_shape}")
        image_roi = np.asarray(image_data[roi_slices], dtype=np.float32)
        input_arr, swapped_xy = self._prepare_stable_infer_input(image_roi, metadata)

        # 先重采样到1mm再推理，减少厚层数据导致的层间断裂/条带假阳性
        src_spacing = self._normalize_spacing_tuple(spacing)
        infer_arr = input_arr
        infer_spacing = src_spacing
        if self.stable_iso_enabled:
            try:
                voxel_count = int(np.prod(np.array(input_arr.shape, dtype=np.int64)))
                # 粗略限制，避免极端数据重采样后爆内存
                if voxel_count <= self.stable_iso_max_voxels:
                    infer_arr = self._resample_volume(
                        input_arr,
                        src_spacing=src_spacing,
                        dst_spacing=(1.0, 1.0, 1.0),
                        order=1,
                    ).astype(np.float32)
                    infer_spacing = (1.0, 1.0, 1.0)
            except Exception as e:
                print(f"stable路径1mm重采样失败，回退原spacing推理: {e}")
                infer_arr = input_arr
                infer_spacing = src_spacing

        input_tensor = torch.from_numpy(infer_arr).unsqueeze(0).unsqueeze(0).to(device)

        pred_label = self._infer_pred_with_guard(
            inferer=inferer,
            input_tensor=input_tensor,
            device=device,
            path_name="stable",
        )

        roi_mask = pred_label[0].astype(np.uint8)

        # 从推理空间回到ROI空间
        if tuple(roi_mask.shape) != tuple(input_arr.shape):
            try:
                if infer_spacing != src_spacing:
                    roi_mask = self._resample_volume(
                        roi_mask.astype(np.float32),
                        src_spacing=infer_spacing,
                        dst_spacing=src_spacing,
                        order=0,
                    )
                    roi_mask = np.rint(roi_mask).astype(np.uint8)
                else:
                    from scipy.ndimage import zoom
                    zoom_factors = [
                        float(input_arr.shape[i]) / float(max(1, roi_mask.shape[i]))
                        for i in range(3)
                    ]
                    roi_mask = zoom(roi_mask.astype(np.float32), zoom_factors, order=0, mode="nearest")
                    roi_mask = np.rint(roi_mask).astype(np.uint8)
            except Exception as e:
                print(f"stable路径回采样到ROI失败: {e}")

        roi_mask = self._restore_internal_axis_order(roi_mask, swapped_xy)

        # 将ROI结果还原到全尺寸体数据
        mask = np.zeros(original_shape, dtype=np.uint8)
        if mask[roi_slices].shape == roi_mask.shape:
            mask[roi_slices] = roi_mask
        else:
            # 兜底：尺寸异常时直接回退全尺寸推理结果
            from scipy.ndimage import zoom
            zoom_factors = [float(dst) / float(src) for dst, src in zip(mask[roi_slices].shape, roi_mask.shape)]
            resized = zoom(roi_mask.astype(np.float32), zoom_factors, order=0, mode="nearest")
            mask[roi_slices] = np.rint(resized).astype(np.uint8)

        if apply_postprocess:
            mask = self.postprocess_mask(mask, original_shape, spacing=spacing)
        return mask

    def _segment_pytorch_legacy(
        self,
        image_data: np.ndarray,
        original_shape: Tuple[int, int, int],
        spacing: Tuple[float, float, float],
        metadata: Optional[Mapping[str, Any]] = None,
        apply_postprocess: bool = True,
    ) -> np.ndarray:
        """旧链路：NIfTI临时写盘+方向/间距变换+回采样。"""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            tmp_path = tmp.name

        nifti_data, nifti_affine, swapped_xy = self._prepare_nifti_payload(
            image_data=image_data,
            spacing=spacing,
            metadata=metadata,
        )
        tmp_nii = nib.Nifti1Image(nifti_data, affine=nifti_affine)
        nib.save(tmp_nii, tmp_path)

        try:
            patch_size = (96, 96, 96)
            spatial_ops = [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="LPI"),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > -200),
                SpatialPadd(keys=["image"], spatial_size=patch_size),
            ]

            inf_transforms = Compose(spatial_ops + [
                ScaleIntensityRanged(keys="image", a_min=0, a_max=400, b_min=0.0, b_max=1.0, clip=True),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                EnsureTyped(keys=["image"]),
            ])

            data_inf = inf_transforms({"image": tmp_path})
            input_tensor = data_inf["image"].unsqueeze(0).to(device)

            inferer = SlidingWindowInferer(
                roi_size=patch_size,
                sw_batch_size=self.sw_batch_size,
                overlap=self.sw_overlap,
                mode="gaussian",
            )

            pred_label = self._infer_pred_with_guard(
                inferer=inferer,
                input_tensor=input_tensor,
                device=device,
                path_name="legacy",
            )

            orig_nii = nib.load(tmp_path)
            pred_nii_cropped = nib.Nifti1Image(
                pred_label[0].astype(np.uint8),
                data_inf["image"].meta["affine"],
            )

            final_pred_nii = resample_to_img(
                pred_nii_cropped,
                orig_nii,
                interpolation="nearest",
            )

            mask = final_pred_nii.get_fdata().astype(np.uint8)
            mask = self._restore_internal_axis_order(mask, swapped_xy)
            if apply_postprocess:
                mask = self.postprocess_mask(mask, original_shape, spacing=spacing)
            return mask
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _fuse_pytorch_masks(
        self,
        stable_mask: Optional[np.ndarray],
        legacy_mask: Optional[np.ndarray],
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Optional[np.ndarray]:
        """
        融合stable/legacy两条推理路径，尽量保留主体血管树并抑制离散碎片。
        """
        from scipy import ndimage

        if stable_mask is None and legacy_mask is None:
            return None
        if stable_mask is None:
            return legacy_mask.astype(np.uint8)
        if legacy_mask is None:
            return stable_mask.astype(np.uint8)

        if stable_mask.shape != legacy_mask.shape:
            # 形状不一致时优先稳定路径
            return stable_mask.astype(np.uint8)

        sx, sy, sz = self._normalize_spacing_tuple(spacing)
        spacing_mm = (sx, sy, sz)
        structure = ndimage.generate_binary_structure(3, 2)

        keep_maps: Dict[int, np.ndarray] = {}
        for label in (1, 2):
            a = stable_mask == label
            b = legacy_mask == label
            union = a | b
            if not np.any(union):
                keep_maps[label] = np.zeros_like(union, dtype=bool)
                continue

            inter = a & b
            core = inter if np.any(inter) else (a if np.count_nonzero(a) >= np.count_nonzero(b) else b)
            core_dil = ndimage.binary_dilation(core, structure=np.ones((3, 3, 3)), iterations=2)

            comp, n_comp = ndimage.label(union, structure=structure)
            sizes = np.bincount(comp.ravel()) if n_comp > 0 else np.array([0])
            if sizes.size > 0:
                sizes[0] = 0
            largest = int(sizes.max()) if sizes.size > 1 else int(np.count_nonzero(union))

            dist_to_core = (
                ndimage.distance_transform_edt(~core, sampling=spacing_mm)
                if np.any(core)
                else None
            )

            keep = np.zeros_like(union, dtype=bool)
            if n_comp > 0:
                for cid in np.argsort(sizes)[::-1]:
                    if cid == 0:
                        continue
                    size = int(sizes[cid])
                    if size <= 0:
                        continue
                    cmask = comp == cid

                    if np.any(core_dil & cmask):
                        keep |= cmask
                        continue

                    # 次级大分支：放宽保留，避免 stable/legacy 融合时误删真实远端分支。
                    if size >= max(160, int(0.05 * max(largest, 1))):
                        if dist_to_core is None:
                            keep |= cmask
                        else:
                            min_dist = float(dist_to_core[cmask].min()) if np.any(cmask) else 1e9
                            if min_dist <= max(35.0, max(spacing_mm) * 12.0):
                                keep |= cmask
                            elif size >= max(600, int(0.18 * max(largest, 1))):
                                keep |= cmask

            if not np.any(keep):
                best_id = int(np.argmax(sizes)) if sizes.size > 1 else 0
                if best_id > 0:
                    keep = comp == best_id
                if not np.any(keep):
                    keep = core

            keep_maps[label] = keep | core

        keep_pv = keep_maps.get(1, np.zeros_like(stable_mask, dtype=bool))
        keep_hv = keep_maps.get(2, np.zeros_like(stable_mask, dtype=bool))

        conflict = keep_pv & keep_hv
        if np.any(conflict):
            stable_pref = stable_mask[conflict]
            legacy_pref = legacy_mask[conflict]
            pref = np.where(stable_pref > 0, stable_pref, legacy_pref)
            keep_pv[conflict] = pref == 1
            keep_hv[conflict] = pref == 2

        fused = np.zeros_like(stable_mask, dtype=np.uint8)
        fused[keep_pv] = 1
        fused[keep_hv] = 2
        return fused

    def _evaluate_mask_quality(self, mask: np.ndarray) -> Dict[str, float]:
        """评估掩码连通性质量，用于auto模式候选结果选择。"""
        n_comp, largest_ratio = self._fragmentation_metrics(mask)
        voxels = int(np.count_nonzero(mask > 0))
        score = float(largest_ratio) - 0.02 * max(0, int(n_comp) - 8)
        return {
            "score": float(score),
            "components": int(n_comp),
            "largest_ratio": float(largest_ratio),
            "voxels": float(voxels),
        }
    
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
                output_data = np.argmax(output_data, axis=0)
            elif output_data.shape[-1] == 3:
                output_data = np.argmax(output_data, axis=-1)

        mask = output_data.astype(np.uint8)
        mask = np.where((mask == 1) | (mask == 2), mask, 0).astype(np.uint8)
        base_mask = mask.copy()

        # 调整尺寸（如果需要），使用最近邻保持多标签
        if mask.shape != original_shape:
            try:
                from scipy.ndimage import zoom
                zoom_factors = [float(orig) / float(cur) for orig, cur in zip(original_shape, mask.shape)]
                mask = zoom(mask.astype(np.float32), zoom_factors, order=0, mode="nearest")
                mask = np.rint(mask).astype(np.uint8)
                base_mask = mask.copy()
            except Exception as e:
                print(f"掩码尺寸回采样失败: {e}")

        # 保守去噪：仅去除极小孤立碎片，不做默认膨胀/闭操作。
        try:
            from scipy import ndimage
            structure = ndimage.generate_binary_structure(3, 2)
            clean_mask = np.zeros_like(mask, dtype=np.uint8)

            for label in (1, 2):
                comp, n_comp = ndimage.label(mask == label, structure=structure)
                if n_comp == 0:
                    continue

                sizes = np.bincount(comp.ravel())
                sizes[0] = 0
                min_keep = max(10, int(0.00001 * mask.size))
                keep_ids = np.where(sizes >= min_keep)[0]
                keep_ids = keep_ids[keep_ids != 0]

                # 保底：若阈值导致该类全空，至少保留最大连通域
                if keep_ids.size == 0:
                    best_id = int(np.argmax(sizes))
                    if best_id > 0 and sizes[best_id] > 0:
                        keep_ids = np.array([best_id], dtype=np.int64)

                if keep_ids.size > 0:
                    clean_mask[np.isin(comp, keep_ids)] = label

            if np.any(clean_mask > 0):
                mask = clean_mask
        except Exception as e:
            print(f"后处理去噪失败，继续使用当前掩码: {e}")

        # 自动碎片化收敛：明显碎片化时，优先保留“主血管树”形态（带误删保护）
        try:
            q_before = self._evaluate_mask_quality(mask)
            n_comp = int(q_before["components"])
            largest_ratio = float(q_before["largest_ratio"])

            if n_comp >= 10 or largest_ratio < 0.55:
                candidates = []
                severe_fragmented = n_comp >= 10 or largest_ratio < 0.45

                main_tree_mask = self._keep_primary_union_components(
                    mask,
                    max_components=1 if self.postprocess_profile != "recall" else 2,
                    min_ratio=0.16 if self.postprocess_profile == "clean" else 0.12,
                    min_voxels=600 if self.postprocess_profile == "clean" else 450,
                )
                if np.any(main_tree_mask > 0):
                    candidates.append(("main_tree", main_tree_mask))

                cleaned_mask = self._cleanup_fragmented_vessels(mask, spacing=spacing)
                if np.any(cleaned_mask > 0):
                    candidates.append(("cleaned", cleaned_mask))

                primary_mask = self._keep_primary_components_per_label(
                    cleaned_mask if np.any(cleaned_mask > 0) else mask,
                    max_components=self.primary_max_components,
                    min_ratio=self.primary_min_ratio,
                    min_voxels=self.primary_min_voxels,
                )
                if np.any(primary_mask > 0):
                    candidates.append(("primary", primary_mask))

                best_name = None
                best_mask = None
                before_vox = max(float(q_before["voxels"]), 1.0)
                best_score = float(q_before["score"]) + self.auto_volume_weight

                for name, cand in candidates:
                    min_retain = self.cleanup_min_retain_ratio
                    if severe_fragmented and name in ("main_tree", "cleaned", "primary"):
                        # 严重碎片化时，很多被删除体素本身就是假阳性；此时不能用过高的
                        # 体素保留率阻止清理，否则会把远端假阳性原样留下。
                        min_retain = min(min_retain, 0.45 if self.postprocess_profile != "recall" else 0.65)
                    if not self._retains_enough_label_volume(
                        mask,
                        cand,
                        min_retain,
                        context=f"自动去碎片候选{name}",
                    ):
                        continue

                    q = self._evaluate_mask_quality(cand)
                    retain_bonus = min(1.0, float(q["voxels"]) / before_vox)
                    score = float(q["score"]) + self.auto_volume_weight * retain_bonus
                    if score > best_score:
                        best_score = score
                        best_name = name
                        best_mask = cand

                if best_mask is not None:
                    mask = best_mask
                    q_after = self._evaluate_mask_quality(mask)
                    print(
                        f"自动去碎片采用{best_name}: "
                        f"components={int(q_after['components'])}, "
                        f"largest_ratio={q_after['largest_ratio']:.3f}"
                    )
        except Exception as e:
            print(f"自动去碎片失败，继续使用当前掩码: {e}")

        # 重清理默认关闭，仅在显式开启时进一步执行
        if self.heavy_cleanup:
            try:
                n_comp, largest_ratio = self._fragmentation_metrics(mask)
                print(f"分割连通性评估: components={n_comp}, largest_ratio={largest_ratio:.3f}")
                if n_comp >= 14 or largest_ratio < 0.45:
                    cleaned_mask = self._cleanup_fragmented_vessels(mask, spacing=spacing)
                    if np.any(cleaned_mask > 0):
                        mask = cleaned_mask
            except Exception as e:
                print(f"重清理失败，继续使用当前掩码: {e}")

        # 严格模式保持可选，并加误删保护
        if self.strict_fragment_prune:
            try:
                strict_mask = self._keep_largest_component_per_label(mask)
                if np.any(strict_mask > 0):
                    accept_strict = True
                    for label in (1, 2):
                        before = int(np.sum(mask == label))
                        after = int(np.sum(strict_mask == label))
                        if before >= 400 and after < int(0.30 * before):
                            accept_strict = False
                            break

                    if accept_strict:
                        mask = strict_mask
                    else:
                        print("严格模式触发误删保护，已回退到非严格结果")
            except Exception as e:
                print(f"严格碎片清理失败，继续使用当前掩码: {e}")

        # 最终形态收敛：若仍明显碎片化，强制收敛到主连通血管树（医生标注风格）
        try:
            q_final = self._evaluate_mask_quality(mask)
            if int(q_final["components"]) >= 8 or float(q_final["largest_ratio"]) < 0.55:
                primary_final = self._keep_primary_components_per_label(
                    mask,
                    max_components=self.primary_max_components,
                    min_ratio=self.primary_min_ratio,
                    min_voxels=self.primary_min_voxels,
                )
                if np.any(primary_final > 0):
                    if self._retains_enough_label_volume(
                        mask,
                        primary_final,
                        self.final_min_retain_ratio,
                        context="最终形态收敛",
                    ):
                        mask = primary_final
                        q2 = self._evaluate_mask_quality(mask)
                        print(
                            f"最终形态收敛后: components={int(q2['components'])}, "
                            f"largest_ratio={q2['largest_ratio']:.3f}"
                        )
        except Exception as e:
            print(f"最终形态收敛失败，继续使用当前掩码: {e}")

        # 最终细修：去掉远离主体的“小碎片”
        try:
            pruned = self._prune_far_small_fragments(mask, spacing=spacing)
            if np.any(pruned > 0):
                before = self._evaluate_mask_quality(mask)
                after = self._evaluate_mask_quality(pruned)
                # 仅在连通性不变差时接受
                if (
                    after["score"] >= before["score"] - 0.02
                    and self._retains_enough_label_volume(
                        mask,
                        pruned,
                        self.prune_min_retain_ratio,
                        context="远离碎片清理",
                    )
                ):
                    mask = pruned
                    print(
                        f"远离碎片清理后: components={int(after['components'])}, "
                        f"largest_ratio={after['largest_ratio']:.3f}"
                    )
        except Exception as e:
            print(f"远离碎片清理失败，继续使用当前掩码: {e}")

        # 误删保护：若后处理把非空预测裁成全空，则回退到基础掩码
        if not np.any(mask > 0) and np.any(base_mask > 0):
            print("后处理触发误删保护：结果全空，回退到基础掩码")
            mask = base_mask.copy()

        # 结构识别缓存
        try:
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
                # PyTorch 模型推理
                if self.pytorch_model is None:
                    raise RuntimeError("PyTorch 模型未初始化")
                infer_mode = self.pytorch_infer_mode
                print(
                    f"使用参数: 模型变体={self.model_variant}, "
                    f"血管分解方法={self.decomposition_method}, 推理模式={infer_mode}, "
                    f"速度预设={self.speed_profile}, 后处理={self.postprocess_profile}, "
                    f"召回={self.recall_profile}"
                )

                if infer_mode == "legacy":
                    mask = self._segment_pytorch_legacy(
                        image_data=image_data,
                        original_shape=original_shape,
                        spacing=spacing,
                        metadata=metadata,
                    )
                elif infer_mode == "stable":
                    mask = self._segment_pytorch_stable(
                        image_data=image_data,
                        original_shape=original_shape,
                        spacing=spacing,
                        metadata=metadata,
                    )
                else:
                    # auto模式：双路径推理并融合，再统一后处理
                    stable_raw = None
                    legacy_raw = None

                    try:
                        stable_raw = self._segment_pytorch_stable(
                            image_data=image_data,
                            original_shape=original_shape,
                            spacing=spacing,
                            metadata=metadata,
                            apply_postprocess=False,
                        )
                    except Exception as e:
                        print(f"stable路径失败: {e}")

                    try:
                        legacy_raw = self._segment_pytorch_legacy(
                            image_data=image_data,
                            original_shape=original_shape,
                            spacing=spacing,
                            metadata=metadata,
                            apply_postprocess=False,
                        )
                    except Exception as e:
                        print(f"legacy路径失败: {e}")

                    if stable_raw is None and legacy_raw is None:
                        raise RuntimeError("stable与legacy两条推理路径均失败")

                    fused_raw = self._fuse_pytorch_masks(
                        stable_mask=stable_raw,
                        legacy_mask=legacy_raw,
                        spacing=spacing,
                    )
                    if fused_raw is None:
                        fused_raw = stable_raw if stable_raw is not None else legacy_raw

                    # 对候选结果分别后处理并评分，自动选择更接近“连贯主体血管树”的结果
                    candidate_raw = {
                        "fused": fused_raw,
                        "stable": stable_raw,
                        "legacy": legacy_raw,
                    }
                    candidate_proc: Dict[str, np.ndarray] = {}
                    candidate_quality: Dict[str, Dict[str, float]] = {}
                    max_vox = 1.0

                    for name, raw_mask in candidate_raw.items():
                        if raw_mask is None:
                            continue
                        proc_mask = self.postprocess_mask(raw_mask, original_shape, spacing=spacing)
                        candidate_proc[name] = proc_mask
                        quality = self._evaluate_mask_quality(proc_mask)
                        candidate_quality[name] = quality
                        max_vox = max(max_vox, quality["voxels"])

                    if not candidate_proc:
                        raise RuntimeError("候选结果后处理后为空")

                    best_name = None
                    best_score = -1e9
                    for name, quality in candidate_quality.items():
                        # 质量模式更偏向完整体素量，避免自动选择“过瘦主干”。
                        score = quality["score"] + self.auto_volume_weight * (quality["voxels"] / max_vox)
                        if score > best_score:
                            best_score = score
                            best_name = name

                    if best_name is None:
                        best_name = "fused" if "fused" in candidate_proc else list(candidate_proc.keys())[0]

                    q = candidate_quality[best_name]
                    print(
                        f"auto选择结果: {best_name}, "
                        f"components={int(q['components'])}, "
                        f"largest_ratio={q['largest_ratio']:.3f}, "
                        f"voxels={int(q['voxels'])}"
                    )
                    mask = candidate_proc[best_name]
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
            
            return mask
            
        except Exception as e:
            print(f"AI分割失败: {e}")
            return None
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.is_model_loaded
