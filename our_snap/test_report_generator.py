#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试report_generator模块是否可以正常导入
"""

import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

# 尝试导入report_generator
try:
    from report_generator import ReportGenerator
    print("report_generator导入成功!")
    
    # 测试生成报告
    generator = ReportGenerator()
    print("ReportGenerator实例化成功!")
except ImportError as e:
    print(f"report_generator导入失败: {e}")
    import traceback
    traceback.print_exc()
