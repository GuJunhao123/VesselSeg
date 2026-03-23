#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型转换脚本
功能：将 PyTorch 模型转换为 ONNX 格式
"""

import torch
import os
import sys

# 添加模型所在目录到路径
sys.path.append('d:\\Code_Program\\program\\VesselSeg\\demo')

from model1 import mobileuvit_l


def convert_model():
    """转换 PyTorch 模型为 ONNX 格式"""
    # 模型路径
    model_path = "d:\\Code_Program\\program\\VesselSeg\\resource\\best_model.pth"
    output_path = "d:\\Code_Program\\program\\VesselSeg\\resource\\best_model.onnx"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在: {model_path}")
        return False
    
    try:
        # 加载模型
        print("加载 PyTorch 模型...")
        model = mobileuvit_l(inch=1)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # 创建示例输入
        print("创建示例输入...")
        dummy_input = torch.randn(1, 1, 96, 96, 96)  # (batch, channel, H, W, D)
        
        # 转换为 ONNX
        print("转换为 ONNX 格式...")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        print(f"模型转换成功！")
        print(f"ONNX 模型保存到: {output_path}")
        return True
        
    except Exception as e:
        print(f"转换失败: {e}")
        return False


if __name__ == "__main__":
    convert_model()
