#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试reportlab导入情况
"""

import sys
import os

# 确保当前目录在Python路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Python版本:", sys.version)
print("Python路径:", sys.path)

# 直接导入reportlab
try:
    import reportlab
    print("直接导入reportlab成功! 版本:", reportlab.__version__)
except ImportError as e:
    print(f"直接导入reportlab失败: {e}")

# 导入report_generator
try:
    from report_generator import ReportGenerator
    print("导入report_generator成功!")
    
    # 创建实例
    generator = ReportGenerator()
    print("ReportGenerator实例化成功!")
    print(f"reportlab可用: {getattr(generator, 'reportlab_available', False)}")
    
    # 测试生成报告
    test_data = {
        'patient_id': 'test',
        'exam_date': '2026-03-30',
        'segmentation_method': 'test',
        'model_variant': 'test'
    }
    
    test_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_report.pdf')
    result = generator.generate_report(test_data, test_output)
    print(f"生成报告结果: {result}")
    
    if os.path.exists(test_output):
        print(f"测试报告生成成功: {test_output}")
    else:
        print("测试报告生成失败")
        
except ImportError as e:
    print(f"导入report_generator失败: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"其他错误: {e}")
    import traceback
    traceback.print_exc()
