#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试数据生成器
功能：生成简单的3D测试图像，用于测试软件功能
"""

import numpy as np
import nibabel as nib
from pathlib import Path

def generate_test_image(output_path="test_image.nii.gz", size=(128, 128, 64)):
    """
    生成测试用的3D医学影像
    
    Args:
        output_path: 输出文件路径
        size: 图像尺寸 (H, W, D)
    """
    print(f"正在生成测试图像: {size}")
    
    # 创建基础图像（随机噪声 + 结构）
    data = np.random.rand(*size).astype(np.float32) * 500
    
    # 添加一些结构（模拟组织）
    center = np.array(size) // 2
    for z in range(size[2]):
        for y in range(size[1]):
            for x in range(size[0]):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
                if dist < min(size) * 0.3:
                    data[x, y, z] += 300  # 中心区域更亮
    
    # 添加偏移，模拟真实影像值范围
    data += 200
    
    # 创建仿射变换矩阵（标准方向，1mm间距）
    affine = np.eye(4)
    affine[0, 0] = 1.0  # x方向spacing (mm)
    affine[1, 1] = 1.0  # y方向spacing (mm)
    affine[2, 2] = 1.0  # z方向spacing (mm)
    affine[0, 3] = -size[0] / 2.0  # x原点
    affine[1, 3] = -size[1] / 2.0  # y原点
    affine[2, 3] = -size[2] / 2.0  # z原点
    
    # 创建NIfTI图像
    nii_img = nib.Nifti1Image(data, affine)
    
    # 保存
    nib.save(nii_img, output_path)
    
    print(f"✓ 测试图像已生成: {output_path}")
    print(f"  尺寸: {size}")
    print(f"  数值范围: [{data.min():.2f}, {data.max():.2f}]")
    print(f"  数据类型: {data.dtype}")
    
    return output_path

def generate_test_label(output_path="test_label.nii.gz", size=(128, 128, 64)):
    """
    生成测试用的标签掩码
    
    Args:
        output_path: 输出文件路径
        size: 图像尺寸 (H, W, D)
    """
    print(f"正在生成测试标签: {size}")
    
    # 创建简单的球形标签
    label_data = np.zeros(size, dtype=np.uint8)
    
    center = np.array(size) // 2
    radius = min(size) * 0.2
    
    for z in range(size[2]):
        for y in range(size[1]):
            for x in range(size[0]):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
                if dist < radius:
                    label_data[x, y, z] = 1  # 标签值1
    
    # 使用相同的仿射矩阵
    affine = np.eye(4)
    affine[0, 0] = 1.0
    affine[1, 1] = 1.0
    affine[2, 2] = 1.0
    affine[0, 3] = -size[0] / 2.0
    affine[1, 3] = -size[1] / 2.0
    affine[2, 3] = -size[2] / 2.0
    
    # 创建NIfTI图像
    nii_img = nib.Nifti1Image(label_data, affine)
    
    # 保存
    nib.save(nii_img, output_path)
    
    print(f"✓ 测试标签已生成: {output_path}")
    print(f"  标签像素数: {np.sum(label_data > 0)}")
    print(f"  标签占比: {np.sum(label_data > 0) / label_data.size * 100:.2f}%")
    
    return output_path

if __name__ == '__main__':
    print("=" * 50)
    print("测试数据生成器")
    print("=" * 50)
    
    # 生成测试图像
    image_path = generate_test_image("test_image.nii.gz")
    
    print()
    
    # 生成测试标签
    label_path = generate_test_label("test_label.nii.gz")
    
    print()
    print("=" * 50)
    print("生成完成！")
    print("=" * 50)
    print(f"测试图像: {image_path}")
    print(f"测试标签: {label_path}")
    print("\n现在可以使用这些文件测试软件功能：")
    print("1. 运行 python main.py")
    print("2. 点击'加载影像'，选择 test_image.nii.gz")
    print("3. 点击'加载标签'，选择 test_label.nii.gz")

