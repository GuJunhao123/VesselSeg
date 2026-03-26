#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本，用于找出程序崩溃的原因
"""

import sys
import traceback

print("Starting test script")
print(f"Python version: {sys.version}")

# 尝试单独导入各个模块，看看是哪个模块导致了崩溃
print("\nTesting imports...")

# 测试 PyQt5 导入
try:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    print("PyQt5 imported successfully")
except Exception as e:
    print(f"Error importing PyQt5: {e}")
    traceback.print_exc()
    sys.exit(1)

# 测试 numpy 导入
try:
    import numpy as np
    print("numpy imported successfully")
except Exception as e:
    print(f"Error importing numpy: {e}")
    traceback.print_exc()
    sys.exit(1)

# 测试 matplotlib 导入
try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    print("matplotlib imported successfully")
except Exception as e:
    print(f"Error importing matplotlib: {e}")
    traceback.print_exc()
    sys.exit(1)

# 测试自定义模块导入
try:
    from image_io import ImageIO
    print("image_io imported successfully")
except Exception as e:
    print(f"Error importing image_io: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from visualization import FourViewWidget
    print("visualization imported successfully")
except Exception as e:
    print(f"Error importing visualization: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from segmentation import ActiveContourSegmentation
    print("segmentation imported successfully")
except Exception as e:
    print(f"Error importing segmentation: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from ai_interface import AISegmentationInterface
    print("ai_interface imported successfully")
except Exception as e:
    print(f"Error importing ai_interface: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nAll imports successful")

# 设置高DPI支持
print("\nSetting up QApplication")
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

app = QApplication(sys.argv)
print("QApplication created successfully")

# 测试 MainWindow 导入和创建
print("\nTesting MainWindow")
try:
    from ui_layout import MainWindow
    print("MainWindow class imported successfully")
    
    print("Creating MainWindow instance...")
    window = MainWindow()
    print("MainWindow created successfully")
    
except Exception as e:
    print(f"Error with MainWindow: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nMainWindow created successfully")
print("Testing show()...")

try:
    window.show()
    print("MainWindow shown successfully")
except Exception as e:
    print(f"Error showing MainWindow: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nAll tests passed! Running event loop...")
sys.exit(app.exec_())
