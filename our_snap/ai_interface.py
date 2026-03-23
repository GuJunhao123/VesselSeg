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
sys.path.append('d:\\Code_Program\\program\\VesselSeg\\demo')

# 尝试导入 PyTorch，用于支持 pred4.py 中的模型
try:
    import torch
    from model1 import mobileuvit_l
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
                
            elif ext in ['.pth', '.pt'] and PYTORCH_AVAILABLE:
                # 加载 PyTorch 模型
                self.pytorch_model = mobileuvit_l(inch=1)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.pytorch_model.load_state_dict(torch.load(model_path, map_location=device))
                self.pytorch_model.to(device)
                self.pytorch_model.eval()
                
                self.model_input_shape = (1, 1, 96, 96, 96)  # 与 pred4.py 一致
                self.model_output_shape = (1, 3, 96, 96, 96)  # 与 pred4.py 一致
                self.model_type = 'pytorch'
                
                print(f"PyTorch 模型加载成功: {model_path}")
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

