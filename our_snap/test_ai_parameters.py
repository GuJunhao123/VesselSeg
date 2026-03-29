#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试不同AI分割参数的效果"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))

from ai_interface import AISegmentationInterface

def test_different_parameters():
    print("=" * 60)
    print("测试不同AI分割参数的效果")
    print("=" * 60)
    
    try:
        ai = AISegmentationInterface()
        print("AI接口初始化成功")
        
        # 模型路径
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'best_model.pth')
        
        # 创建测试图像
        test_image = np.random.rand(64, 64, 64) * 255
        test_image = test_image.astype(np.float32)
        print(f"测试图像形状: {test_image.shape}")
        print(f"测试图像范围: [{test_image.min()}, {test_image.max()}]")
        
        # 测试不同的模型变体
        model_variants = ['mobileuvit', 'mobileuvit_l']
        
        # 测试不同的血管分解方法
        decomposition_methods = ['adaptive', 'percentile_50', 'fixed_2.0']
        
        for model_variant in model_variants:
            for method in decomposition_methods:
                print(f"\n{'-' * 60}")
                print(f"测试: 模型={model_variant}, 分解方法={method}")
                print(f"{'-' * 60}")
                
                try:
                    # 设置参数
                    ai.set_model_variant(model_variant)
                    ai.set_decomposition_method(method)
                    
                    # 加载模型
                    if not ai.load_model(model_path):
                        print("  模型加载失败")
                        continue
                    
                    # 执行分割
                    result = ai.segment(test_image)
                    
                    if result is not None:
                        print(f"  分割成功!")
                        print(f"  结果形状: {result.shape}")
                        print(f"  结果类型: {result.dtype}")
                        print(f"  非零像素数量: {np.count_nonzero(result)}")
                        print(f"  最大标签值: {result.max()}")
                    else:
                        print("  分割失败")
                        
                except Exception as e:
                    print(f"  错误: {e}")
                    import traceback
                    traceback.print_exc()
        
        print(f"\n{'=' * 60}")
        print("测试完成")
        print(f"{'=' * 60}")
        
    except Exception as e:
        print(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_different_parameters()
