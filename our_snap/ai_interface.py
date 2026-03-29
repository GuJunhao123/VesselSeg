#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI自动分割接口模块
功能：支持ONNX模型和PyTorch模型推理，实现AI自动分割
"""

import numpy as np
import onnxruntime as ort
from typing import Optional, Tuple
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
                    providers=['CPUExecutionProvider', 'CUDAExecutionProvider']  # 优先使用GPU
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
    
    def postprocess_mask(self, model_output: np.ndarray, original_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        后处理模型输出
        
        Args:
            model_output: 模型输出
            original_shape: 原始图像形状 (H, W, D)
            
        Returns:
            后处理后的掩码 (H, W, D)，uint8类型
        """
        # 获取输出数据
        if isinstance(model_output, list):
            output_data = model_output[0]
        else:
            output_data = model_output
        
        # 移除batch和channel维度
        while len(output_data.shape) > 3:
            output_data = np.squeeze(output_data, axis=0)
        
        # 如果输出是概率图，进行二值化
        if output_data.dtype == np.float32 or output_data.dtype == np.float64:
            if output_data.max() <= 1.0:
                # 概率图，阈值0.5
                mask = (output_data > 0.5).astype(np.uint8)
            else:
                # 可能是logits，使用sigmoid
                mask = (1.0 / (1.0 + np.exp(-output_data)) > 0.5).astype(np.uint8)
        else:
            mask = output_data.astype(np.uint8)
        
        # 应用血管分解方法进行后处理
        if self.decomposition_method and mask.max() > 0:
            try:
                # 导入必要的库
                from skimage import morphology
                from scipy import ndimage
                from skimage.filters import threshold_otsu
                
                print(f"开始应用血管分解方法: {self.decomposition_method}")
                
                # 对每个标签进行分解处理
                processed_mask = np.zeros_like(mask, dtype=np.uint8)
                
                for label in range(1, mask.max() + 1):
                    vessel_mask = (mask == label).astype(bool)
                    
                    if not np.any(vessel_mask):
                        continue
                    
                    # 提取骨架
                    skeleton = morphology.skeletonize(vessel_mask)
                    
                    if not np.any(skeleton):
                        # 无法提取骨架，整体视为small
                        processed_mask[vessel_mask] = label
                        continue
                    
                    # 计算距离变换
                    edt = ndimage.distance_transform_edt(vessel_mask)
                    radii = edt[skeleton]
                    
                    if radii.size == 0:
                        processed_mask[vessel_mask] = label
                        continue
                    
                    # 根据分解方法选择阈值
                    if self.decomposition_method.startswith("fixed_"):
                        fixed_val = float(self.decomposition_method.split("_")[1])
                        th = fixed_val
                        print(f"  使用固定阈值: {th}")
                    elif self.decomposition_method == "adaptive":
                        if np.unique(radii).size > 1:
                            th = threshold_otsu(radii)
                        else:
                            th = radii.max() + 1e-5
                        print(f"  使用自适应阈值: {th:.3f}")
                    elif self.decomposition_method == "median":
                        th = np.median(radii)
                        print(f"  使用中位数阈值: {th:.3f}")
                    elif self.decomposition_method.startswith("percentile_"):
                        percentile_val = int(self.decomposition_method.split("_")[1])
                        th = np.percentile(radii, percentile_val)
                        print(f"  使用{percentile_val}%百分位数阈值: {th:.3f}")
                    else:
                        # 默认使用adaptive
                        if np.unique(radii).size > 1:
                            th = threshold_otsu(radii)
                        else:
                            th = radii.max() + 1e-5
                        print(f"  使用默认自适应阈值: {th:.3f}")
                    
                    # 计算trunk、small和junction
                    _, idx = ndimage.distance_transform_edt(~skeleton, return_indices=True)
                    prop_r = edt[tuple(idx)]
                    
                    trunk = (prop_r >= th) & vessel_mask
                    non_trunk = vessel_mask & (~trunk)
                    
                    if not np.any(trunk):
                        # 没有主干，全部视为small
                        processed_mask[vessel_mask] = label
                        print(f"  标签{label}: 无主干，全部为细血管")
                        continue
                    
                    trunk_dil = ndimage.binary_dilation(trunk, iterations=2)
                    junction = trunk_dil & non_trunk
                    small = non_trunk & (~junction)
                    
                    # 根据分解方法应用不同的后处理策略
                    if self.decomposition_method == "adaptive":
                        # 自适应：保留所有部分
                        processed_mask[trunk] = label
                        processed_mask[small] = label
                        processed_mask[junction] = label
                        print(f"  标签{label}: 自适应模式 - 保留主干{np.sum(trunk)}，细血管{np.sum(small)}，连接点{np.sum(junction)}")
                    elif self.decomposition_method.startswith("fixed_"):
                        # 固定阈值：只保留主干
                        processed_mask[trunk] = label
                        # 可选：根据阈值大小决定是否保留细血管
                        if float(self.decomposition_method.split("_")[1]) >= 2.0:
                            # 高阈值：只保留主干
                            print(f"  标签{label}: 高阈值模式 - 只保留主干{np.sum(trunk)}")
                        else:
                            # 低阈值：保留主干和部分细血管
                            processed_mask[small] = label
                            print(f"  标签{label}: 低阈值模式 - 保留主干{np.sum(trunk)}和细血管{np.sum(small)}")
                    elif self.decomposition_method == "median":
                        # 中位数：平衡保留
                        processed_mask[trunk] = label
                        processed_mask[small] = label
                        print(f"  标签{label}: 中位数模式 - 保留主干{np.sum(trunk)}和细血管{np.sum(small)}")
                    elif self.decomposition_method.startswith("percentile_"):
                        percentile_val = int(self.decomposition_method.split("_")[1])
                        if percentile_val >= 50:
                            # 高百分位数：更保守，只保留主干
                            processed_mask[trunk] = label
                            print(f"  标签{label}: 高百分位数{percentile_val}% - 只保留主干{np.sum(trunk)}")
                        else:
                            # 低百分位数：更激进，保留更多
                            processed_mask[trunk] = label
                            processed_mask[small] = label
                            print(f"  标签{label}: 低百分位数{percentile_val}% - 保留主干{np.sum(trunk)}和细血管{np.sum(small)}")
                
                mask = processed_mask
                print(f"血管分解后处理完成，最终非零像素: {np.count_nonzero(mask)}")
            except Exception as e:
                print(f"血管分解后处理失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 调整尺寸（如果需要）
        if mask.shape != original_shape:
            from scipy.ndimage import zoom
            zoom_factors = [orig / mask for orig, mask in zip(original_shape, mask.shape)]
            mask = zoom(mask, zoom_factors, order=1)
            mask = (mask > 0.5).astype(np.uint8)
        
        return mask
    
    def segment(self, image_data: np.ndarray) -> Optional[np.ndarray]:
        """
        执行AI分割
        
        Args:
            image_data: 输入图像 (H, W, D)
            
        Returns:
            分割掩码 (H, W, D)，uint8类型，失败返回None
        """
        if not self.is_model_loaded:
            raise RuntimeError("模型未加载，请先调用load_model()")
        
        try:
            original_shape = image_data.shape
            
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
                mask = self.postprocess_mask(outputs[0], original_shape)
                
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
                tmp_nii = nib.Nifti1Image(image_data, affine=np.eye(4))
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
                        # 由于没有标签，使用图像本身作为前景
                        CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > 0),
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
                        pred_label = torch.argmax(output[0], dim=1).cpu().numpy()  # [1, H, W, D]
                    
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

