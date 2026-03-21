#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI自动分割接口模块
功能：支持ONNX模型推理，实现AI自动分割
"""

import numpy as np
import onnxruntime as ort
from typing import Optional, Tuple
import os

class AISegmentationInterface:
    """AI分割接口 - 完全自研实现"""
    
    def __init__(self):
        """初始化"""
        self.model_session = None
        self.model_input_shape = None
        self.model_output_shape = None
        self.is_model_loaded = False
    
    def load_model(self, model_path: str) -> bool:
        """
        加载ONNX模型
        
        Args:
            model_path: ONNX模型文件路径
            
        Returns:
            成功返回True，失败返回False
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 创建ONNX Runtime会话
            self.model_session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider', 'CUDAExecutionProvider']  # 优先使用GPU
            )
            
            # 获取输入输出信息
            input_info = self.model_session.get_inputs()[0]
            output_info = self.model_session.get_outputs()[0]
            
            self.model_input_shape = input_info.shape
            self.model_output_shape = output_info.shape
            
            self.is_model_loaded = True
            print(f"模型加载成功: {model_path}")
            print(f"输入形状: {self.model_input_shape}")
            print(f"输出形状: {self.model_output_shape}")
            
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
        if not self.is_model_loaded or self.model_session is None:
            raise RuntimeError("模型未加载，请先调用load_model()")
        
        try:
            original_shape = image_data.shape
            
            # 预处理
            input_data = self.preprocess_image(image_data)
            
            # 获取输入输出名称
            input_name = self.model_session.get_inputs()[0].name
            output_name = self.model_session.get_outputs()[0].name
            
            # 推理
            outputs = self.model_session.run([output_name], {input_name: input_data})
            
            # 后处理
            mask = self.postprocess_mask(outputs[0], original_shape)
            
            return mask
            
        except Exception as e:
            print(f"AI分割失败: {e}")
            return None
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.is_model_loaded

