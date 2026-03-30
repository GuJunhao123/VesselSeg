#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试reportlab是否可以正常导入
"""

import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

# 尝试导入reportlab
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    print("reportlab导入成功!")
except ImportError as e:
    print(f"reportlab导入失败: {e}")
    import traceback
    traceback.print_exc()
