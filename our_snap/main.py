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

def _force_window_visible(app: QApplication, window: MainWindow) -> None:
    """确保主窗口在可视区域内，但不强制全屏尺寸。"""
    try:
        screen = app.primaryScreen()
        if screen is None:
            return

        avail = screen.availableGeometry()
        # 目标尺寸：优先使用当前窗口大小；仅在超出屏幕时收缩到屏幕的90%
        max_w = int(avail.width() * 0.90)
        max_h = int(avail.height() * 0.90)
        target_w = min(max(window.width(), 1000), max_w)
        target_h = min(max(window.height(), 700), max_h)

        geo = window.geometry()
        out_of_bounds = (
            geo.right() < avail.left()
            or geo.left() > avail.right()
            or geo.bottom() < avail.top()
            or geo.top() > avail.bottom()
        )

        if out_of_bounds:
            target_x = avail.x() + max(0, (avail.width() - target_w) // 2)
            target_y = avail.y() + max(0, (avail.height() - target_h) // 2)
            window.setGeometry(target_x, target_y, target_w, target_h)
        else:
            # 仅收缩过大的窗口，不改用户已有位置
            need_resize = geo.width() > max_w or geo.height() > max_h
            if need_resize:
                window.resize(target_w, target_h)
    except Exception as e:
        print(f"窗口位置修复失败: {e}")

def main():
    """主函数"""
    # 设置高DPI支持（必须在QApplication创建之前设置）
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # 创建应用程序
    app = QApplication(sys.argv)
    app.setApplicationName("医学影像分割软件")
    app.setOrganizationName("MedicalImageSegmentation")

    # 创建主窗口
    window = MainWindow()
    # 显式保留系统标题栏缩放能力
    window.setWindowFlags(window.windowFlags() | Qt.WindowMinMaxButtonsHint)
    window.setMaximumSize(16777215, 16777215)
    window.show()
    _force_window_visible(app, window)
    window.setWindowState(Qt.WindowNoState)
    window.showNormal()
    window.raise_()
    window.activateWindow()
    
    # 运行事件循环
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
