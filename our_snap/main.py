#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
医学影像分割软件 - 主程序入口
功能：支持三视图可视化、手动标注、主动轮廓分割、AI自动分割
"""

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from ui_layout import MainWindow

def main():
    """主函数"""
    # 创建应用程序
    app = QApplication(sys.argv)
    app.setApplicationName("医学影像分割软件")
    app.setOrganizationName("MedicalImageSegmentation")
    
    # 设置高DPI支持
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 运行事件循环
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

