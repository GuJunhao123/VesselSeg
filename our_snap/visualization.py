#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化模块
功能：2×2布局视图（轴位/矢状位/冠状位/3D预览），十字线联动，方向标识
"""

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


class SliceViewer(QWidget):
    """单个切片视图"""
    
    slice_changed = pyqtSignal(int)
    crosshair_changed = pyqtSignal(float, float)
    mouse_clicked = pyqtSignal(float, float)
    mouse_dragged = pyqtSignal(float, float)
    window_changed = pyqtSignal(float, float)
    
    def __init__(self, view_type: str, parent=None):
        """
        初始化切片视图
        
        Args:
            view_type: 视图类型 ('axial', 'coronal', 'sagittal', '3d')
        """
        super().__init__(parent)
        self.view_type = view_type
        self.image_data = None
        self.label_data = None
        self.image_spacing = None
        self.current_slice = 0
        self.window_width = 400.0
        self.window_level = 0.0
        self.crosshair_x = 0.0
        self.crosshair_y = 0.0
        self.show_label = True
        self.show_crosshair = True
        self.label_opacity = 0.5
        self.label_colors = {1: (1.0, 0.0, 0.0, 1.0)}
        self.tool_mode = 'pointer'
        self.brush_size = 3
        self.active_label_id = 1
        
        # 方向标识
        self.direction_labels = self._get_direction_labels()
        
        self._init_ui()
    
    def _get_direction_labels(self) -> dict:
        """获取方向标识"""
        if self.view_type == 'axial':
            return {'top': 'P', 'bottom': 'A', 'left': 'R', 'right': 'L'}
        elif self.view_type == 'coronal':
            return {'top': 'I', 'bottom': 'S', 'left': 'A', 'right': 'P'}
        elif self.view_type == 'sagittal':
            return {'top': 'I', 'bottom': 'S', 'left': 'R', 'right': 'L'}
        else:
            return {}
    
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(1)
        
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(2, 2, 2, 2)
        title_layout.setSpacing(4)
        
        view_title = QLabel(self.view_type.upper())
        view_title.setFont(QFont("Arial", 8, QFont.Bold))
        title_layout.addWidget(view_title)
        
        title_layout.addStretch()
        
        if self.direction_labels:
            dir_label = QLabel(
                f"{self.direction_labels.get('left', '')} | {self.direction_labels.get('right', '')}"
            )
            dir_label.setFont(QFont("Arial", 7))
            title_layout.addWidget(dir_label)
        
        btn_zoom_fit = QPushButton("Fit")
        btn_zoom_fit.setMaximumWidth(35)
        btn_zoom_fit.setMaximumHeight(18)
        btn_zoom_fit.setFont(QFont("Arial", 7))
        btn_zoom_fit.clicked.connect(self._on_zoom_fit)
        title_layout.addWidget(btn_zoom_fit)
        
        layout.addLayout(title_layout)
        
        bg_color = 'white'
        self.figure = Figure(figsize=(4, 4), dpi=120, facecolor=bg_color)
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, facecolor=bg_color)
        self.ax.axis('off')
        
        self.zoom_factor = 1.0
        self.zoom_center_x = 0.0
        self.zoom_center_y = 0.0
        self.xlim_range = None
        self.ylim_range = None
        
        layout.addWidget(self.canvas, stretch=1)
        
        # 切片索引显示（如"24 of 47"）- 紧凑显示
        self.slice_label = QLabel("0 of 0")
        self.slice_label.setAlignment(Qt.AlignCenter)
        self.slice_label.setFont(QFont("Arial", 8))
        self.slice_label.setMaximumHeight(18)  # 限制高度
        layout.addWidget(self.slice_label)
        
        self.setLayout(layout)
        
        # 设置明显的边框
        self.setStyleSheet("""
            QWidget {
                border: 2px solid #333333;
                background-color: white;
            }
        """)
        
        # 设置最小尺寸
        self.setMinimumSize(300, 300)
        
        # 连接鼠标事件
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        
        self.mouse_pressed = False
        self.mouse_button = None  # 记录按下的按钮（1=左键，3=右键）
        self.last_mouse_pos = None
    
    def set_image_data(self, image_data: np.ndarray, spacing=None):
        """
        设置图像数据
        
        Args:
            image_data: 图像数据 (H, W, D)
            spacing: 像素间距 (spacing_x, spacing_y, spacing_z)，可选
        """
        self.image_data = image_data
        self.image_spacing = spacing
        if image_data is not None:
            # 初始化十字线位置到图像中心
            if self.view_type == 'axial':
                # 轴位视图转置后，坐标也需要调整
                self.crosshair_x = image_data.shape[0] / 2.0  # 转置后X对应原Y
                self.crosshair_y = image_data.shape[1] / 2.0  # 转置后Y对应原X
            elif self.view_type == 'coronal':
                self.crosshair_x = image_data.shape[1] / 2.0
                self.crosshair_y = image_data.shape[2] / 2.0
            elif self.view_type == 'sagittal':
                self.crosshair_x = image_data.shape[0] / 2.0
                self.crosshair_y = image_data.shape[2] / 2.0
            
            # 重置缩放和视图范围（使用物理坐标）
            self.zoom_factor = 1.0
            # 立即更新显示
            self._update_display()
            slice_img = self._get_slice_data()
            if slice_img is not None:
                h, w = slice_img.shape
                # 计算物理坐标范围
                xlim, ylim, _, _ = self._get_physical_range(h, w)
                self.xlim_range = xlim
                self.ylim_range = ylim
        
        self._update_slice_range()
        self._update_display()
    
    def set_label_data(self, label_data: np.ndarray):
        """设置标签数据"""
        self.label_data = label_data
        self._update_display()
    
    def set_label_color(self, label_value: int, color: tuple):
        """
        设置标签颜色
        
        Args:
            label_value: 标签值（1, 2, 3...）
            color: 颜色 (R, G, B, A) 或 (R, G, B)，值范围0-1
        """
        if len(color) == 3:
            color = (*color, self.label_opacity)
        self.label_colors[label_value] = color
        self._update_display()
    
    def set_crosshair(self, x: float, y: float):
        """设置十字线位置（物理坐标）"""
        if self.image_spacing is not None and self.image_data is not None:
            slice_img = self._get_slice_data()
            if slice_img is not None:
                if self.view_type == 'axial':
                    self.crosshair_x = x / self.image_spacing[1]
                    self.crosshair_y = y / self.image_spacing[0]
                elif self.view_type == 'coronal':
                    self.crosshair_x = x / self.image_spacing[1]
                    self.crosshair_y = y / self.image_spacing[2]
                elif self.view_type == 'sagittal':
                    self.crosshair_x = x / self.image_spacing[1]
                    self.crosshair_y = y / self.image_spacing[2]
                else:
                    self.crosshair_x = x
                    self.crosshair_y = y
            else:
                self.crosshair_x = x
                self.crosshair_y = y
        else:
            self.crosshair_x = x
            self.crosshair_y = y
        
        self._update_display()
        self.crosshair_changed.emit(self.crosshair_x, self.crosshair_y)
    
    def set_slice(self, slice_idx: int):
        """设置当前切片"""
        if self.image_data is None:
            return
        
        max_slice = self._get_max_slice()
        self.current_slice = max(0, min(slice_idx, max_slice - 1))
        self._update_display()
    
    def set_tool_mode(self, mode: str):
        """设置工具模式"""
        self.tool_mode = mode
        # 更新鼠标样式
        if mode == 'brush':
            self.canvas.setCursor(Qt.CrossCursor)
        elif mode == 'eraser':
            self.canvas.setCursor(Qt.CrossCursor)
        elif mode == 'zoom':
            self.canvas.setCursor(Qt.SizeAllCursor)
        elif mode == 'pan':
            self.canvas.setCursor(Qt.OpenHandCursor)
        else:  # pointer
            self.canvas.setCursor(Qt.ArrowCursor)
    
    def set_brush_size(self, size: int):
        """设置画笔大小"""
        self.brush_size = max(1, min(50, size))
    
    def set_active_label_id(self, label_id: int):
        """设置活动标签ID"""
        self.active_label_id = label_id
    
    def set_window(self, window_width: float, window_level: float):
        """设置窗宽窗位"""
        self.window_width = window_width
        self.window_level = window_level
        self._update_display()
    
    def _get_max_slice(self) -> int:
        """获取最大切片数"""
        if self.image_data is None:
            return 0
        
        if self.view_type == 'axial':
            return self.image_data.shape[2]  # Z 轴切片数
        elif self.view_type == 'coronal':
            return self.image_data.shape[1]  # X 轴切片数（coronal固定X）
        elif self.view_type == 'sagittal':
            return self.image_data.shape[0]  # Y 轴切片数（sagittal固定Y）
        return 0
    
    def _update_slice_range(self):
        """更新切片范围"""
        max_slice = self._get_max_slice()
        if max_slice > 0:
            self.current_slice = min(self.current_slice, max_slice - 1)
    
    def _get_physical_range(self, h: int, w: int):
        """计算物理坐标范围 (xlim_range, ylim_range, max_x, max_y)"""
        if self.image_spacing is not None:
            if self.view_type == 'axial':
                xlim = (0, w * self.image_spacing[1])
                ylim = (0, h * self.image_spacing[0])
                max_x = w * self.image_spacing[1]
                max_y = h * self.image_spacing[0]
            elif self.view_type == 'coronal':
                xlim = (0, w * self.image_spacing[1])
                ylim = (0, h * self.image_spacing[2])
                max_x = w * self.image_spacing[1]
                max_y = h * self.image_spacing[2]
            elif self.view_type == 'sagittal':
                xlim = (0, w * self.image_spacing[1])
                ylim = (0, h * self.image_spacing[2])
                max_x = w * self.image_spacing[1]
                max_y = h * self.image_spacing[2]
            else:
                xlim = (0, w)
                ylim = (0, h)
                max_x = w
                max_y = h
        else:
            xlim = (0, w)
            ylim = (0, h)
            max_x = w
            max_y = h
        return xlim, ylim, max_x, max_y
    
    def _get_slice_data(self) -> np.ndarray:
        """
        获取当前切片的图像数据
        注意：numpy数组索引为 (H, W, D)，对应医学影像的 (Y, X, Z)
        
        标准切片提取：
        - Axial（轴位）：从上往下看，显示X-Y平面，Z轴切片
          -> image_data[:, :, z] 得到 (Y, X) 的2D数组
        - Coronal（冠状位）：从前面看，显示X-Z平面，Y轴切片
          -> image_data[y, :, :] 得到 (X, Z) 的2D数组，需要转置为 (Z, X)
        - Sagittal（矢状位）：从侧面看，显示Y-Z平面，X轴切片
          -> image_data[:, x, :] 得到 (Y, Z) 的2D数组，需要转置为 (Z, Y)
        """
        if self.image_data is None:
            return None
        
        if self.view_type == 'axial':
            # 轴位：直接提取Z轴切片，显示X-Y平面
            # 轴位视图横着放（水平方向）
            # image_data[:, :, z] 得到 (Y, X)，需要转置为 (X, Y) 以横着显示
            return self.image_data[:, :, self.current_slice].T
        elif self.view_type == 'coronal':
            # 冠状位：提取Y轴切片，转置为 (Z, X)，然后上下翻转
            # 方向标识：上=I, 下=S, 左=A, 右=P（x轴对应Y方向，y轴对应Z方向）
            slice_data = self.image_data[self.current_slice, :, :].T
            return np.flipud(slice_data)  # 上下翻转
        elif self.view_type == 'sagittal':
            # 矢状位：提取X轴切片，显示Y-Z平面，然后上下翻转
            # image_data[:, x, :] 得到 (Y, Z)，转置为 (Z, Y) 以匹配显示方向
            slice_data = self.image_data[:, self.current_slice, :].T
            return np.flipud(slice_data)  # 上下翻转
        return None
    
    def _get_slice_label(self) -> np.ndarray:
        """获取当前切片的标签数据（与_get_slice_data保持一致）"""
        if self.label_data is None:
            return None
        
        if self.view_type == 'axial':
            # 轴位：与图像保持一致，需要转置
            return self.label_data[:, :, self.current_slice].T
        elif self.view_type == 'coronal':
            # 冠状位：与图像保持一致，需要上下翻转
            slice_label = self.label_data[self.current_slice, :, :].T
            return np.flipud(slice_label)
        elif self.view_type == 'sagittal':
            # 矢状位：与图像保持一致，需要上下翻转
            slice_label = self.label_data[:, self.current_slice, :].T
            return np.flipud(slice_label)
        return None
    
    def _update_display(self):
        """更新显示"""
        self.ax.clear()
        self.ax.axis('off')
        # 纯白色背景
        self.ax.set_facecolor('white')
        
        slice_img = self._get_slice_data()
        if slice_img is None:
            self.canvas.draw()
            return
        
        # 获取切片的实际尺寸
        h, w = slice_img.shape
        
        # 应用窗宽窗位
        img_min = self.window_level - self.window_width / 2
        img_max = self.window_level + self.window_width / 2
        
        # 归一化到0-255
        img_normalized = np.clip((slice_img - img_min) / (img_max - img_min) * 255, 0, 255).astype(np.uint8)
        
        # 计算aspect ratio以保持物理比例
        # 关键理解：matplotlib的aspect参数是 (y轴单位长度) / (x轴单位长度)
        # 如果图像有h行w列，y轴有h个单位，x轴有w个单位
        # 要让物理比例正确，需要：aspect = (y轴物理单位长度) / (x轴物理单位长度)
        # 即：aspect = (y轴spacing) / (x轴spacing)
        if self.image_spacing is not None:
            if self.view_type == 'axial':
                # Axial：x=原X，y=原Y
                aspect_ratio = self.image_spacing[1] / self.image_spacing[0]  # Y_spacing / X_spacing
            elif self.view_type == 'coronal':
                # Coronal：x=原Y，y=原Z
                aspect_ratio = self.image_spacing[2] / self.image_spacing[1]  # Z_spacing / Y_spacing
            elif self.view_type == 'sagittal':
                # Sagittal：x=原X，y=原Z
                aspect_ratio = self.image_spacing[2] / self.image_spacing[0]  # Z_spacing / X_spacing
            else:
                aspect_ratio = 1.0
        else:
            # 如果没有spacing信息，使用像素比例
            aspect_ratio = h / w
        
        # 使用extent来定义物理坐标范围
        # 注意：在matplotlib中，extent=[xmin, xmax, ymin, ymax]
        # 当origin='upper'时，ymin对应显示顶部，ymax对应显示底部
        # 所以如果图像数据的Y轴是从P到A（向下增加），那么ymin应该对应P，ymax应该对应A
        # 但为了保持方向标识正确，我们需要交换ymin和ymax
        if self.image_spacing is not None:
            if self.view_type == 'axial':
                # 轴位：X-Y平面，extent定义X和Y的物理范围
                # 上=P，下=A，所以ymin应该对应P（顶部），ymax应该对应A（底部）
                # 但origin='upper'时，ymin对应顶部，所以extent的y值需要反转
                extent_x = (0, w * self.image_spacing[1])  # Y方向物理范围
                extent_y = (h * self.image_spacing[0], 0)    # X方向物理范围（反转：顶部=0，底部=h*spacing）
                extent = [extent_x[0], extent_x[1], extent_y[0], extent_y[1]]
            elif self.view_type == 'coronal':
                # 冠状位：方向标识上=I, 下=S, 左=A, 右=P（x轴对应Y方向，y轴对应Z方向）
                extent_x = (0, w * self.image_spacing[1])  # Y方向（前后A-P）
                extent_y = (h * self.image_spacing[2], 0)  # Z方向（上下S-I，反转）
                extent = [extent_x[0], extent_x[1], extent_y[0], extent_y[1]]
            elif self.view_type == 'sagittal':
                # 矢状位：Z-Y平面，extent定义Z和Y的物理范围
                # 上=S，下=I，所以ymin应该对应S（顶部），ymax应该对应I（底部）
                extent_x = (0, w * self.image_spacing[1])  # Y方向物理范围
                extent_y = (h * self.image_spacing[2], 0)  # Z方向物理范围（反转：顶部=0，底部=h*spacing）
                extent = [extent_x[0], extent_x[1], extent_y[0], extent_y[1]]
            else:
                extent = [0, w, h, 0]  # 反转y轴
        else:
            extent = [0, w, h, 0]  # 反转y轴
        
        # 显示图像 - 使用extent定义物理坐标，aspect='equal'保持物理比例
        self.ax.imshow(img_normalized, cmap='gray', interpolation='antialiased', 
                      origin='upper', aspect='equal', extent=extent,
                      rasterized=False, filternorm=True)
        
        # 设置坐标轴范围（支持缩放，使用物理坐标）
        if self.xlim_range is None or self.ylim_range is None:
            # 初始化：显示完整图像（物理坐标）
            self.xlim_range = (extent[0], extent[1])
            # 注意：extent的y值已经反转，所以ylim_range也需要反转
            self.ylim_range = (extent[3], extent[2])  # 反转y轴范围
        
        self.ax.set_xlim(self.xlim_range[0], self.xlim_range[1])
        # 反转y轴，使顶部对应较小的y值（符合origin='upper'和方向标识）
        self.ax.set_ylim(self.ylim_range[1], self.ylim_range[0])
        
        # 叠加标签
        if self.show_label and self.label_data is not None:
            slice_label = self._get_slice_label()
            if slice_label is not None and np.any(slice_label > 0):
                label_colored = np.zeros((*slice_label.shape, 4), dtype=np.float32)
                
                for label_value in np.unique(slice_label):
                    if label_value > 0:
                        if label_value in self.label_colors:
                            color = self.label_colors[label_value]
                        else:
                            # 默认红色，50%透明度
                            color = (1.0, 0.0, 0.0, 0.5)  # 默认50%透明度
                        
                        mask = (slice_label == label_value)
                        label_colored[mask, 0] = color[0]
                        label_colored[mask, 1] = color[1]
                        label_colored[mask, 2] = color[2]
                        # 使用标签的透明度设置（50%默认）
                        label_colored[mask, 3] = color[3] if len(color) > 3 else self.label_opacity
                
                # 标签叠加也使用相同的extent和aspect
                if self.image_spacing is not None:
                    self.ax.imshow(label_colored, interpolation='antialiased', origin='upper',
                                  aspect='equal', extent=extent, rasterized=False)
                else:
                    self.ax.imshow(label_colored, interpolation='antialiased', origin='upper',
                                  aspect='equal', rasterized=False)
        
        # 绘制十字线：蓝色虚线 + 绿色方块交叉点
        # 注意：crosshair_x和crosshair_y是像素坐标，需要转换为物理坐标
        if self.show_crosshair and slice_img is not None:
            h, w = slice_img.shape
            
            # 将像素坐标转换为物理坐标
            if self.image_spacing is not None:
                if self.view_type == 'axial':
                    # 轴位：x对应Y方向，y对应X方向
                    crosshair_x_phys = self.crosshair_x * self.image_spacing[1]  # Y方向
                    crosshair_y_phys = self.crosshair_y * self.image_spacing[0]  # X方向
                    crosshair_size = max(h * self.image_spacing[0], w * self.image_spacing[1]) / 80
                elif self.view_type == 'coronal':
                    # 冠状位：方向标识显示左=A(前)，右=P(后)，所以x对应Y方向（前后A-P）
                    # 图片已上下翻转，y坐标直接使用（翻转后，顶部对应z=0）
                    crosshair_x_phys = self.crosshair_x * self.image_spacing[1]  # Y方向（前后A-P）
                    crosshair_y_phys = self.crosshair_y * self.image_spacing[2]  # Z方向（上下S-I，翻转后）
                    crosshair_size = max(h * self.image_spacing[2], w * self.image_spacing[1]) / 80
                elif self.view_type == 'sagittal':
                    # 矢状位：x对应X方向（左右R-L），y对应Z方向（上下S-I）
                    # 图片已上下翻转，y坐标直接使用（翻转后，顶部对应z=0）
                    crosshair_x_phys = self.crosshair_x * self.image_spacing[1]  # X方向（左右R-L）
                    crosshair_y_phys = self.crosshair_y * self.image_spacing[2]  # Z方向（上下S-I，翻转后）
                    crosshair_size = max(h * self.image_spacing[2], w * self.image_spacing[1]) / 80
                else:
                    crosshair_x_phys = self.crosshair_x
                    crosshair_y_phys = self.crosshair_y
                    crosshair_size = max(h, w) / 80
            else:
                crosshair_x_phys = self.crosshair_x
                crosshair_y_phys = self.crosshair_y
                crosshair_size = max(h, w) / 80
            
            # 水平线：蓝色虚线
            self.ax.axhline(y=crosshair_y_phys, color='#0080FF', linewidth=1.5, 
                          alpha=0.9, linestyle='--', dashes=(5, 5))
            # 垂直线：蓝色虚线
            self.ax.axvline(x=crosshair_x_phys, color='#0080FF', linewidth=1.5, 
                          alpha=0.9, linestyle='--', dashes=(5, 5))
            
            # 交叉点：绿色方块
            self.ax.add_patch(Rectangle(
                (crosshair_x_phys - crosshair_size/2, crosshair_y_phys - crosshair_size/2),
                crosshair_size, crosshair_size,
                facecolor='#00FF00', edgecolor='#00FF00', linewidth=1,
                alpha=0.9, zorder=10
            ))
        
        # 绘制方向标识（在图像边缘，使用物理坐标）
        if self.direction_labels:
            h, w = slice_img.shape
            # 计算物理坐标范围
            if self.image_spacing is not None:
                if self.view_type == 'axial':
                    phys_w = w * self.image_spacing[1]  # X方向
                    phys_h = h * self.image_spacing[0]  # Y方向
                elif self.view_type == 'coronal':
                    phys_w = w * self.image_spacing[0]  # Y方向（A-P）
                    phys_h = h * self.image_spacing[2]  # Z方向（I-S）
                elif self.view_type == 'sagittal':
                    phys_w = w * self.image_spacing[1]  # X方向（R-L）
                    phys_h = h * self.image_spacing[2]  # Z方向（I-S）
                else:
                    phys_w = w
                    phys_h = h
            else:
                phys_w = w
                phys_h = h
            
            font_size = max(phys_h, phys_w) / 30
            # 注意：在matplotlib中，当extent=[xmin, xmax, ymin, ymax]且origin='upper'时
            # ymin对应显示顶部，ymax对应显示底部
            # 由于我们反转了extent的y值，y轴坐标也反转了
            # 所以：顶部对应较大的y值（phys_h*0.95），底部对应较小的y值（phys_h*0.05）
            # 上（显示顶部）
            if 'top' in self.direction_labels:
                # 由于y轴已反转，顶部对应较大的y值
                self.ax.text(phys_w/2, phys_h*0.95, self.direction_labels['top'], 
                           color='yellow', fontsize=font_size, 
                           ha='center', va='top', weight='bold')
            # 下（显示底部）
            if 'bottom' in self.direction_labels:
                # 由于y轴已反转，底部对应较小的y值
                self.ax.text(phys_w/2, phys_h*0.05, self.direction_labels['bottom'], 
                           color='yellow', fontsize=font_size, 
                           ha='center', va='bottom', weight='bold')
            # 左
            if 'left' in self.direction_labels:
                self.ax.text(phys_w*0.05, phys_h/2, self.direction_labels['left'], 
                           color='yellow', fontsize=font_size, 
                           ha='left', va='center', weight='bold')
            # 右
            if 'right' in self.direction_labels:
                self.ax.text(phys_w*0.95, phys_h/2, self.direction_labels['right'], 
                           color='yellow', fontsize=font_size, 
                           ha='right', va='center', weight='bold')
        
        # 绘制缩放条（Scale Bar）：绿色，右下角，显示物理尺寸
        if self.image_spacing is not None:
            h, w = slice_img.shape
            # 计算物理尺寸范围
            if self.view_type == 'axial':
                phys_w = w * self.image_spacing[1]  # X方向
                phys_h = h * self.image_spacing[0]  # Y方向
            elif self.view_type == 'coronal':
                phys_w = w * self.image_spacing[0]  # Y方向（A-P）
                phys_h = h * self.image_spacing[2]  # Z方向（I-S）
            elif self.view_type == 'sagittal':
                phys_w = w * self.image_spacing[1]  # X方向（R-L）
                phys_h = h * self.image_spacing[2]  # Z方向（I-S）
            else:
                phys_w = w
                phys_h = h
            
            # 选择合适的缩放条长度（约为图像宽度的10-20%）
            scale_length_phys = max(phys_w, phys_h) * 0.15
            # 取整到合适的值（1, 2, 5, 10, 20, 50, 100等）
            if scale_length_phys >= 100:
                scale_length_phys = round(scale_length_phys / 10) * 10
            elif scale_length_phys >= 50:
                scale_length_phys = round(scale_length_phys / 5) * 5
            elif scale_length_phys >= 20:
                scale_length_phys = round(scale_length_phys / 2) * 2
            elif scale_length_phys >= 10:
                scale_length_phys = round(scale_length_phys)
            else:
                scale_length_phys = round(scale_length_phys * 10) / 10
            
            # 缩放条位置：右下角
            scale_x_start = phys_w * 0.85
            scale_x_end = scale_x_start + scale_length_phys
            scale_y = phys_h * 0.08  # 底部向上8%
            
            # 绘制缩放条（绿色）
            self.ax.plot([scale_x_start, scale_x_end], [scale_y, scale_y], 
                        color='green', linewidth=3, zorder=10)
            # 绘制垂直端线
            line_height = phys_h * 0.02
            self.ax.plot([scale_x_start, scale_x_start], 
                        [scale_y - line_height/2, scale_y + line_height/2],
                        color='green', linewidth=3, zorder=10)
            self.ax.plot([scale_x_end, scale_x_end], 
                        [scale_y - line_height/2, scale_y + line_height/2],
                        color='green', linewidth=3, zorder=10)
            
            # 添加文字标签（显示物理尺寸）
            if scale_length_phys >= 10:
                scale_text = f"{int(scale_length_phys)} mm"
            else:
                scale_text = f"{scale_length_phys:.1f} mm"
            self.ax.text((scale_x_start + scale_x_end) / 2, scale_y + line_height * 2,
                        scale_text, color='green', fontsize=10,
                        ha='center', va='bottom', weight='bold', zorder=10)
        
        # 更新切片索引显示
        max_slice = self._get_max_slice()
        self.slice_label.setText(f"{self.current_slice + 1} of {max_slice}")
        
        self.canvas.draw()
    
    def _on_zoom_fit(self):
        """
        缩放至拟合
        重置视图，让图像完整显示在窗口中，保持正确的物理比例
        """
        if self.image_data is None:
            return
        
        slice_img = self._get_slice_data()
        if slice_img is not None:
            h, w = slice_img.shape
            
            # 重置缩放（使用物理坐标）
            self.zoom_factor = 1.0
            
            # 计算物理坐标范围（与_update_display一致）
            xlim, ylim, _, _ = self._get_physical_range(h, w)
            self.xlim_range = xlim
            self.ylim_range = ylim
            
            # 强制刷新显示
            self._update_display()
    
    def _on_slice_changed(self, value: int):
        """切片改变回调"""
        self.current_slice = value
        self._update_display()
        self.slice_changed.emit(value)
    
    def _on_scroll(self, event):
        """鼠标滚轮事件 - 切换切片"""
        if event.inaxes == self.ax:
            if event.button == 'up':
                self.set_slice(self.current_slice - 1)
            elif event.button == 'down':
                self.set_slice(self.current_slice + 1)
    
    def _on_mouse_press(self, event):
        """鼠标按下事件"""
        if event.inaxes == self.ax:
            self.mouse_button = event.button  # 记录按下的按钮
            if event.button == 1:  # 左键
                if event.xdata is not None and event.ydata is not None:
                    if self.tool_mode == 'brush':
                        # 画笔工具：绘制标签
                        self._paint_label(event.xdata, event.ydata)
                    elif self.tool_mode == 'eraser':
                        # 橡皮擦工具：清除标签
                        self._erase_label(event.xdata, event.ydata)
                    elif self.tool_mode == 'pointer':
                        # 指针工具：设置十字线位置
                        self.set_crosshair(event.xdata, event.ydata)
                        self.mouse_clicked.emit(event.xdata, event.ydata)
                    elif self.tool_mode == 'pan':
                        # 平移工具：记录起始位置
                        self.pan_start_pos = (event.xdata, event.ydata)
                self.mouse_pressed = True
                self.last_mouse_pos = (event.xdata, event.ydata)
            elif event.button == 3:  # 右键：开始调整窗宽窗位
                self.mouse_pressed = True
                self.last_mouse_pos = (event.xdata, event.ydata)
    
    def _on_mouse_release(self, event):
        """鼠标释放事件"""
        self.mouse_pressed = False
        self.mouse_button = None
        self.last_mouse_pos = None
    
    def _on_mouse_move(self, event):
        """鼠标移动事件，支持实时同步拖拽"""
        if not event.inaxes == self.ax:
            return
        
        if self.mouse_pressed and self.mouse_button is not None:
            if self.mouse_button == 1:  # 左键
                if event.xdata is not None and event.ydata is not None:
                    if self.tool_mode == 'brush':
                        # 画笔工具：连续绘制
                        self._paint_label(event.xdata, event.ydata)
                    elif self.tool_mode == 'eraser':
                        # 橡皮擦工具：连续清除
                        self._erase_label(event.xdata, event.ydata)
                    elif self.tool_mode == 'pointer':
                        # 指针工具：移动十字线（实时同步）
                        self.set_crosshair(event.xdata, event.ydata)
                        self.mouse_dragged.emit(event.xdata, event.ydata)
                    elif self.tool_mode == 'pan':
                        # 平移工具：移动视图
                        if hasattr(self, 'pan_start_pos') and self.pan_start_pos:
                            dx = event.xdata - self.pan_start_pos[0]
                            dy = event.ydata - self.pan_start_pos[1]
                            if self.xlim_range and self.ylim_range:
                                self.xlim_range = (
                                    self.xlim_range[0] - dx,
                                    self.xlim_range[1] - dx
                                )
                                self.ylim_range = (
                                    self.ylim_range[0] - dy,
                                    self.ylim_range[1] - dy
                                )
                                self._update_display()
                            self.pan_start_pos = (event.xdata, event.ydata)
            elif self.mouse_button == 3:  # 右键：缩放视图或窗宽窗位调整
                if self.tool_mode == 'zoom':
                    # 缩放工具：右键拖拽缩放
                    if self.last_mouse_pos and event.xdata is not None and event.ydata is not None:
                        if self.image_data is None:
                            return
                        
                        slice_img = self._get_slice_data()
                        if slice_img is None:
                            return
                    
                    h, w = slice_img.shape
                    
                    # 计算鼠标移动距离
                    dy = event.ydata - self.last_mouse_pos[1]
                    
                    # 右键拖拽缩放：垂直移动控制缩放
                    # 垂直向上拖拽：放大，垂直向下拖拽：缩小
                    zoom_speed = 0.02  # 缩放速度
                    zoom_delta = -dy * zoom_speed
                    self.zoom_factor *= (1.0 + zoom_delta)
                    self.zoom_factor = max(0.1, min(20.0, self.zoom_factor))  # 限制缩放范围
                    
                    # 以鼠标位置为中心缩放（使用物理坐标）
                    if self.xlim_range is None or self.ylim_range is None:
                        xlim, ylim, _, _ = self._get_physical_range(h, w)
                        self.xlim_range = xlim
                        self.ylim_range = ylim
                    
                    # 计算缩放中心（鼠标位置，相对于当前视图范围）
                    zoom_center_x = event.xdata
                    zoom_center_y = event.ydata
                    
                    # 计算当前视图范围
                    current_width = self.xlim_range[1] - self.xlim_range[0]
                    current_height = self.ylim_range[1] - self.ylim_range[0]
                    
                    # 计算新的视图范围（以鼠标位置为中心）
                    new_width = current_width / self.zoom_factor
                    new_height = current_height / self.zoom_factor
                    
                    # 以鼠标位置为中心缩放
                    self.xlim_range = (
                        zoom_center_x - new_width / 2,
                        zoom_center_x + new_width / 2
                    )
                    self.ylim_range = (
                        zoom_center_y - new_height / 2,
                        zoom_center_y + new_height / 2
                    )
                    
                    # 限制在图像范围内（物理坐标）
                    _, _, max_x, max_y = self._get_physical_range(h, w)
                    
                    if self.xlim_range[0] < 0:
                        self.xlim_range = (0, new_width)
                    if self.xlim_range[1] > max_x:
                        self.xlim_range = (max_x - new_width, max_x)
                    if self.ylim_range[0] < 0:
                        self.ylim_range = (0, new_height)
                    if self.ylim_range[1] > max_y:
                        self.ylim_range = (max_y - new_height, max_y)
                    
                    self._update_display()
                    self.last_mouse_pos = (event.xdata, event.ydata)
                else:
                    # 非缩放工具：右键调整窗宽窗位（原有逻辑）
                    if self.last_mouse_pos and event.xdata is not None and event.ydata is not None:
                        # 原有的窗宽窗位调整逻辑
                        pass
    
    def _physical_to_voxel_coords(self, x: float, y: float):
        """将物理坐标转换为体素坐标 (orig_x, orig_y, orig_z)"""
        slice_img = self._get_slice_data()
        if slice_img is None:
            return None
        
        h, w = slice_img.shape
        
        if self.image_spacing is not None:
            if self.view_type == 'axial':
                pixel_x = int(x / self.image_spacing[1])
                pixel_y = int(y / self.image_spacing[0])
                orig_x = pixel_y  # 转置后y对应X方向
                orig_y = pixel_x  # 转置后x对应Y方向
                orig_z = self.current_slice
            elif self.view_type == 'coronal':
                pixel_x = int(x / self.image_spacing[1])
                pixel_y = int(y / self.image_spacing[2])
                orig_x = self.current_slice
                orig_y = pixel_x
                orig_z = int(h - 1 - pixel_y)
            elif self.view_type == 'sagittal':
                pixel_x = int(x / self.image_spacing[1])
                pixel_y = int(y / self.image_spacing[2])
                orig_x = self.current_slice
                orig_y = pixel_x
                orig_z = int(h - 1 - pixel_y)
            else:
                return None
        else:
            pixel_x = int(x)
            pixel_y = int(y)
            if self.view_type == 'axial':
                orig_x = pixel_y
                orig_y = pixel_x
                orig_z = self.current_slice
            elif self.view_type == 'coronal':
                orig_x = self.current_slice
                orig_y = pixel_x
                orig_z = int(h - 1 - pixel_y)
            elif self.view_type == 'sagittal':
                orig_x = self.current_slice
                orig_y = pixel_x
                orig_z = int(h - 1 - pixel_y)
            else:
                return None
        
        return (orig_x, orig_y, orig_z)
    
    def _apply_brush(self, orig_x: int, orig_y: int, orig_z: int, label_value: int):
        """在指定体素坐标应用画笔（绘制或清除）"""
        H, W, D = self.label_data.shape
        if not (0 <= orig_x < W and 0 <= orig_y < H and 0 <= orig_z < D):
            return False
        
        brush_radius = self.brush_size
        for dy in range(-brush_radius, brush_radius + 1):
            for dx in range(-brush_radius, brush_radius + 1):
                if dx*dx + dy*dy <= brush_radius*brush_radius:
                    nx, ny = orig_x + dx, orig_y + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        self.label_data[ny, nx, orig_z] = label_value
        return True
    
    def _paint_label(self, x: float, y: float):
        """画笔工具：在指定位置绘制标签"""
        if self.label_data is None or self.image_data is None:
            return
        
        coords = self._physical_to_voxel_coords(x, y)
        if coords is None:
            return
        
        orig_x, orig_y, orig_z = coords
        if self._apply_brush(orig_x, orig_y, orig_z, self.active_label_id):
            self.label_changed.emit()
            self._update_display()
    
    def _erase_label(self, x: float, y: float):
        """橡皮擦工具：在指定位置清除标签"""
        if self.label_data is None or self.image_data is None:
            return
        
        coords = self._physical_to_voxel_coords(x, y)
        if coords is None:
            return
        
        orig_x, orig_y, orig_z = coords
        if self._apply_brush(orig_x, orig_y, orig_z, 0):
            self.label_changed.emit()
            self._update_display()
    
    # 信号：标签改变时发出
    label_changed = pyqtSignal()


class ThreeDViewer(QWidget):
    """3D预览视图"""
    
    view_updated = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_data = None
        self.label_data = None
        
        self.crosshair_3d = [0, 0, 0]
        self.setMinimumSize(300, 300)
        self.setStyleSheet("background-color: black;")
        
        size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        size_policy.setHeightForWidth(True)
        self.setSizePolicy(size_policy)
        try:
            import pyvista as pv
            from pyvistaqt import QtInteractor
            self.use_pyvista = True
            self.pv = pv
        except ImportError as e:
            self.use_pyvista = False
            print(f"警告：PyVista/PyVistaQt不可用，将使用matplotlib作为备选: {e}")
        
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(1)
        
        title_label = QLabel("3D View")
        title_label.setFont(QFont("Arial", 8, QFont.Bold))
        title_label.setStyleSheet("color: white;")
        title_label.setMaximumHeight(18)
        layout.addWidget(title_label)
        
        if self.use_pyvista:
            import pyvista as pv
            from pyvistaqt import QtInteractor
            
            pv.set_plot_theme('dark')
            self.plotter = QtInteractor(self, off_screen=False)
            self.plotter.background_color = 'black'
            layout.addWidget(self.plotter.interactor, stretch=1)
            self.plotter.camera_position = 'iso'
            self.plotter.camera.parallel_projection = True
            
            try:
                renderer = self.plotter.renderer
                renderer.SetUseDepthPeeling(False)
                renderer.SetUseFXAA(False)
                renderer.SetAutomaticLightCreation(True)
                try:
                    renderer.SetLineSmoothing(False)
                except:
                    pass
            except Exception as e:
                print(f"警告：设置渲染器兼容模式失败: {e}")
            
        else:
            # 备选方案：使用matplotlib
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            
            bg_color = 'black'
            self.figure = Figure(figsize=(4, 4), facecolor=bg_color)
            self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111, projection='3d')
            self.ax.set_facecolor(bg_color)
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False
            self.ax.set_proj_type('ortho')
            layout.addWidget(self.canvas, stretch=1)
        
        # 设置明显的边框（3D视图使用深色边框）
        self.setStyleSheet("""
            QWidget {
                border: 2px solid #666666;
                background-color: black;
            }
        """)
        
        # 3D交互模式（默认启用旋转，这样用户可以拖动视图）
        self.interaction_mode = 'rotate'  # 'pointer', 'rotate', 'pan', 'zoom', 'clip'
        self.clip_planes = []  # 裁剪平面列表
        
        self.setLayout(layout)
    
    def set_image_data(self, image_data: np.ndarray, spacing=None):
        """设置图像数据：不自动更新，只在refresh_view时更新"""
        self.image_data = image_data
        self.image_spacing = spacing  # 保存spacing信息用于正确的3D比例
    
    def set_label_data(self, label_data: np.ndarray):
        """设置标签数据：不自动更新，只在refresh_view时更新"""
        self.label_data = label_data
    
    def set_crosshair_3d(self, x: int, y: int, z: int):
        """设置3D十字准心位置"""
        self.crosshair_3d = [x, y, z]
        # 如果视图已经初始化，更新显示
        if hasattr(self, 'plotter') and self.plotter is not None:
            self._update_crosshair_display()
    
    def _update_3d_view(self):
        """
        更新3D视图
        使用VTK/PyVista进行3D渲染
        """
        if self.use_pyvista:
            self._update_3d_view_vtk()
        else:
            self._update_3d_view_matplotlib()
    
    def _update_3d_view_vtk(self):
        """使用VTK/PyVista更新3D视图"""
        import pyvista as pv
        
        # 清除之前的actor
        self.plotter.clear()
        self.plotter.background_color = 'black'  # 黑色背景
        
        # 启用阴影和光照效果
        # 使用简化的方法避免OpenGL shader错误，同时增强阴影深度
        try:
            # 方法1：使用PyVista的高级API（最兼容）
            self.plotter.enable_shadows()
            
            # 尝试调整阴影参数以增强阴影深度
            try:
                renderer = self.plotter.renderer
                # 设置阴影相关参数（如果支持）
                if hasattr(renderer, 'SetUseShadows'):
                    renderer.SetUseShadows(True)
                # 尝试设置阴影偏移和分辨率（增强阴影对比度）
                if hasattr(renderer, 'SetShadowMapResolution'):
                    renderer.SetShadowMapResolution(2048)  # 提高阴影分辨率，使阴影更清晰
                if hasattr(renderer, 'SetShadowBias'):
                    renderer.SetShadowBias(0.001)  # 减小阴影偏移，使阴影更贴近物体
            except:
                pass  # 如果设置失败，继续使用默认阴影
        except:
            try:
                # 方法2：使用VTK底层API
                renderer = self.plotter.renderer
                if hasattr(renderer, 'SetUseShadows'):
                    renderer.SetUseShadows(True)
            except Exception as e:
                pass  # 静默失败，避免输出过多警告
        
        # 添加光源：头灯跟随视角（像拿着手电筒），辅以极弱填充光
        try:
            # 清除默认光源
            self.plotter.renderer.RemoveAllLights()
            
            # 头灯：始终和摄像机同方向，模拟手电筒照射当前视角
            head_light = pv.Light(light_type='headlight')
            head_light.intensity = 1.1  # 视角方向的主光
            self.plotter.add_light(head_light)
            
            # 极弱填充光：仅防止完全漆黑的背面，不影响阴影方向感
            fill_light = pv.Light(position=(0, 0, -1), focal_point=(0, 0, 0),
                                  color='white', intensity=0.08)
            fill_light.positional = False
            self.plotter.add_light(fill_light)
            
        except Exception as e:
            print(f"警告：添加光源失败: {e}，使用默认光源")
        
        # 不添加地面平面（只有3D模型，没有地面）
        
        if self.image_data is None:
            self.view_updated.emit()
            return
        
        # 获取图像尺寸
        h, w, d = self.image_data.shape
        
        # 显示标签的3D表面（如果有标签数据）
        if self.label_data is not None and np.any(self.label_data > 0):
            try:
                # 使用PyVista直接处理3D数据
                # PyVista可以高效处理大型数据，自动优化渲染
                label_binary = (self.label_data > 0).astype(np.uint8)
                
                # 创建PyVista ImageData（直接从numpy数组）
                # PyVista使用(x, y, z)顺序，需要转置numpy的(z, y, x)顺序
                # 转置：从(z, y, x)到(x, y, z)
                label_transposed = np.transpose(label_binary, (2, 1, 0))
                
                # 创建ImageData（UniformGrid的新名称）
                grid = pv.ImageData()
                grid.dimensions = label_transposed.shape
                
                # 使用正确的spacing信息（如果可用），否则使用1.0
                # 注意：label_transposed是(x, y, z)顺序，spacing也需要对应
                # 原始spacing是(spacing_x, spacing_y, spacing_z)，对应numpy的(z, y, x)
                # 转置后是(x, y, z)，所以spacing应该是(spacing_z, spacing_y, spacing_x)
                if self.image_spacing is not None:
                    # image_spacing是(spacing_x, spacing_y, spacing_z)，对应(z, y, x)
                    # 转置后是(x, y, z)，所以需要反转：spacing = (spacing_z, spacing_y, spacing_x)
                    spacing_xyz = (self.image_spacing[2], self.image_spacing[1], self.image_spacing[0])
                    grid.spacing = spacing_xyz
                else:
                    grid.spacing = (1.0, 1.0, 1.0)
                    print("警告：未提供spacing信息，使用默认值(1.0, 1.0, 1.0)")
                
                grid.origin = (0.0, 0.0, 0.0)
                grid['values'] = label_transposed.flatten(order='F')  # Fortran顺序
                
                # 使用contour提取等值面
                # PyVista的contour使用VTK的marching cubes，性能优异
                mesh = grid.contour([0.5])  # 提取0.5等值面
                
                if mesh.n_points > 0:
                    # 红色表面，完全不透明，启用阴影和光照
                    # 使用简化的渲染设置，但保留阴影效果
                    try:
                        # 使用基本设置添加mesh，启用光照（阴影需要光照）
                        # 关键：不使用show_edges，不使用line_width，避免EdgesGLES30错误
                        actor = self.plotter.add_mesh(
                            mesh, 
                            color='red', 
                            opacity=1.0,  # 完全不透明
                            show_edges=False,  # 禁用边缘显示（避免EdgesGLES30错误）
                            lighting=True  # 启用光照（阴影需要光照）
                        )
                        
                        # 设置材质属性以启用阴影和双面光照
                        try:
                            if hasattr(actor, 'GetProperty'):
                                prop = actor.GetProperty()
                                # 启用双面光照（背面也能看清）
                                prop.SetBackfaceCulling(False)  # 禁用背面剔除
                                
                                # 设置光照属性（深阴影，清晰结构）
                                try:
                                    # 降低环境光和漫反射，增强阴影对比度
                                    # 让阴影更深，结构更清晰
                                    prop.SetDiffuse(0.6)  # 降低漫反射，增强阴影对比度
                                    prop.SetAmbient(0.2)  # 大幅降低环境光，让阴影更深
                                    prop.SetSpecular(0.3)  # 增加镜面反射，增强结构细节
                                    prop.SetSpecularPower(20)  # 提高镜面反射锐度，增强细节
                                except:
                                    pass  # 如果设置失败，继续使用默认光照
                                
                                # 尝试启用阴影（如果系统支持）
                                try:
                                    # 注意：阴影可能需要额外的渲染器设置
                                    if hasattr(prop, 'SetReceiveShadows'):
                                        prop.SetReceiveShadows(True)
                                    if hasattr(prop, 'SetCastShadows'):
                                        prop.SetCastShadows(True)
                                except:
                                    pass  # 阴影设置失败不影响基本渲染
                        except Exception:
                            # 材质设置失败不影响基本渲染
                            pass
                        
                    except Exception as e:
                        # 如果启用光照失败，尝试无光照模式
                        print(f"警告：启用光照失败，尝试无光照模式: {e}")
                        try:
                            # 降级到无光照模式
                            self.plotter.add_mesh(mesh, color='red', opacity=0.8, show_edges=False, lighting=False)
                        except Exception as e2:
                            print(f"错误：渲染完全失败: {e2}")
                            # 如果还是失败，跳过这个mesh
                            pass
                else:
                    print("警告：未提取到表面")
                    
            except Exception as e:
                print(f"VTK渲染失败：{e}")
                import traceback
                traceback.print_exc()
        
        # 绘制坐标轴（蓝色）和原点（绿色）
        origin = [0, 0, 0]
        axis_length = min(h, w, d) * 0.3
        
        # X轴（蓝色）- 使用基本设置，避免shader错误（不使用line_width参数）
        try:
            x_axis = pv.Line(origin, [axis_length, 0, 0])
            self.plotter.add_mesh(x_axis, color='blue', lighting=False)  # 移除line_width避免shader错误
        except Exception as e:
            print(f"警告：添加X轴失败: {e}")
        
        # Y轴（蓝色）- 使用基本设置，避免shader错误（不使用line_width参数）
        try:
            y_axis = pv.Line(origin, [0, axis_length, 0])
            self.plotter.add_mesh(y_axis, color='blue', lighting=False)  # 移除line_width避免shader错误
        except Exception as e:
            print(f"警告：添加Y轴失败: {e}")
        
        # Z轴（蓝色）- 使用基本设置，避免shader错误（不使用line_width参数）
        try:
            z_axis = pv.Line(origin, [0, 0, axis_length])
            self.plotter.add_mesh(z_axis, color='blue', lighting=False)  # 移除line_width避免shader错误
        except Exception as e:
            print(f"警告：添加Z轴失败: {e}")
        
        # 原点（绿色方块）- 使用基本设置，避免shader错误
        origin_marker = pv.Sphere(radius=axis_length*0.05, center=origin)
        self.plotter.add_mesh(origin_marker, color='#00FF00', lighting=False)  # 禁用光照和smooth_shading避免shader错误
        
        # 设置相机（等轴测视图，正交投影）
        self.plotter.camera_position = 'iso'
        self.plotter.camera.parallel_projection = True
        
        # 设置坐标轴标签（白色）- 使用try-except避免shader错误（不使用line_width参数）
        try:
            # 不使用line_width参数，避免EdgesGLES30 shader错误
            self.plotter.add_axes(labels_off=False)
        except Exception as e:
            print(f"警告：添加坐标轴失败（可能由于OpenGL兼容性问题）: {e}")
            # 如果add_axes失败，尝试使用更简单的设置
            try:
                self.plotter.add_axes(labels_off=True)  # 只显示轴线，不显示标签
            except:
                pass  # 如果还是失败，跳过坐标轴显示
        
        # 绘制3D十字准心（三条线）
        self._update_crosshair_display()
        
        # 设置默认交互模式（默认启用旋转，这样用户可以拖动视图）
        if hasattr(self, 'interaction_mode'):
            self.set_interaction_mode(self.interaction_mode)
        else:
            # 如果没有设置，默认启用旋转模式
            self.set_interaction_mode('rotate')
        
        self.view_updated.emit()
    
    def _update_crosshair_display(self):
        """更新3D十字准心显示（三条线）"""
        if not hasattr(self, 'plotter') or self.plotter is None:
            return
        
        if self.image_data is None:
            return
        
        import pyvista as pv
        
        # 移除旧的准心线和标记（如果存在）
        try:
            for name in ['crosshair_x', 'crosshair_y', 'crosshair_z', 'crosshair_center']:
                try:
                    actor = self.plotter.renderer.GetActorByName(name)
                    if actor is not None:
                        self.plotter.renderer.RemoveActor(actor)
                except:
                    pass
        except:
            pass
        
        # 获取图像尺寸
        h, w, d = self.image_data.shape
        
        # crosshair_3d存储的是物理坐标 [space_x, space_y, space_z]（numpy坐标系统）
        # 需要转换为PyVista坐标系统
        # PyVista坐标系统：x对应numpy的z（上下），y对应numpy的y（前后），z对应numpy的x（左右）
        # 所以转换：PyVista [x, y, z] = numpy [z, y, x] = [crosshair_3d[2], crosshair_3d[1], crosshair_3d[0]]
        if self.image_spacing is not None:
            # crosshair_3d已经是物理坐标，直接使用
            # 但需要转换到PyVista坐标系统
            # PyVista x = numpy z = crosshair_3d[2]
            # PyVista y = numpy y = crosshair_3d[1]
            # PyVista z = numpy x = crosshair_3d[0]
            pos = [self.crosshair_3d[2], self.crosshair_3d[1], self.crosshair_3d[0]]
        else:
            # 没有spacing时，crosshair_3d是像素坐标，也需要转换
            pos = [float(self.crosshair_3d[2]), float(self.crosshair_3d[1]), float(self.crosshair_3d[0])]
        
        # 计算准心线的长度（约为图像尺寸的20%，使其更明显）
        if self.image_spacing is not None:
            # 计算图像的最大物理尺寸（使用PyVista坐标系统）
            # PyVista坐标：x对应numpy z，y对应numpy y，z对应numpy x
            max_phys_size = max(
                d * self.image_spacing[2],  # PyVista x方向（numpy z）
                h * self.image_spacing[1],  # PyVista y方向（numpy y）
                w * self.image_spacing[0]   # PyVista z方向（numpy x）
            )
            line_length = max_phys_size * 0.20  # 20%的长度
        else:
            line_length = max(h, w, d) * 0.20
        
        # 绘制三条准心线（蓝色虚线效果，使用较粗的线）
        try:
            # X方向准心线（沿X轴，固定Y和Z）
            x_line_start = [pos[0] - line_length/2, pos[1], pos[2]]
            x_line_end = [pos[0] + line_length/2, pos[1], pos[2]]
            x_line = pv.Line(x_line_start, x_line_end)
            actor_x = self.plotter.add_mesh(x_line, color='#0080FF', line_width=3, lighting=False, name='crosshair_x')
            # 设置actor名称以便后续移除
            if hasattr(actor_x, 'SetName'):
                actor_x.SetName('crosshair_x')
            
            # Y方向准心线（沿Y轴，固定X和Z）
            y_line_start = [pos[0], pos[1] - line_length/2, pos[2]]
            y_line_end = [pos[0], pos[1] + line_length/2, pos[2]]
            y_line = pv.Line(y_line_start, y_line_end)
            actor_y = self.plotter.add_mesh(y_line, color='#0080FF', line_width=3, lighting=False, name='crosshair_y')
            if hasattr(actor_y, 'SetName'):
                actor_y.SetName('crosshair_y')
            
            # Z方向准心线（沿Z轴，固定X和Y）
            z_line_start = [pos[0], pos[1], pos[2] - line_length/2]
            z_line_end = [pos[0], pos[1], pos[2] + line_length/2]
            z_line = pv.Line(z_line_start, z_line_end)
            actor_z = self.plotter.add_mesh(z_line, color='#0080FF', line_width=3, lighting=False, name='crosshair_z')
            if hasattr(actor_z, 'SetName'):
                actor_z.SetName('crosshair_z')
            
            # 在准心位置添加一个绿色球体标记（与2D视图的绿色方块一致）
            # 球体大小约为图像最大尺寸的2%
            if self.image_spacing is not None:
                marker_size = max_phys_size * 0.02
            else:
                marker_size = max(h, w, d) * 0.02
            
            center_marker = pv.Sphere(radius=marker_size, center=pos)
            actor_center = self.plotter.add_mesh(
                center_marker, 
                color='#00FF00',  # 绿色，与2D视图一致
                lighting=False, 
                name='crosshair_center'
            )
            if hasattr(actor_center, 'SetName'):
                actor_center.SetName('crosshair_center')
                
        except Exception as e:
            # 如果line_width参数不支持，尝试不使用该参数
            try:
                # X方向准心线
                x_line_start = [pos[0] - line_length/2, pos[1], pos[2]]
                x_line_end = [pos[0] + line_length/2, pos[1], pos[2]]
                x_line = pv.Line(x_line_start, x_line_end)
                actor_x = self.plotter.add_mesh(x_line, color='#0080FF', lighting=False, name='crosshair_x')
                if hasattr(actor_x, 'SetName'):
                    actor_x.SetName('crosshair_x')
                
                # Y方向准心线
                y_line_start = [pos[0], pos[1] - line_length/2, pos[2]]
                y_line_end = [pos[0], pos[1] + line_length/2, pos[2]]
                y_line = pv.Line(y_line_start, y_line_end)
                actor_y = self.plotter.add_mesh(y_line, color='#0080FF', lighting=False, name='crosshair_y')
                if hasattr(actor_y, 'SetName'):
                    actor_y.SetName('crosshair_y')
                
                # Z方向准心线
                z_line_start = [pos[0], pos[1], pos[2] - line_length/2]
                z_line_end = [pos[0], pos[1], pos[2] + line_length/2]
                z_line = pv.Line(z_line_start, z_line_end)
                actor_z = self.plotter.add_mesh(z_line, color='#0080FF', lighting=False, name='crosshair_z')
                if hasattr(actor_z, 'SetName'):
                    actor_z.SetName('crosshair_z')
                
                # 中心标记
                if self.image_spacing is not None:
                    marker_size = max_phys_size * 0.02
                else:
                    marker_size = max(h, w, d) * 0.02
                center_marker = pv.Sphere(radius=marker_size, center=pos)
                actor_center = self.plotter.add_mesh(
                    center_marker, 
                    color='#00FF00', 
                    lighting=False, 
                    name='crosshair_center'
                )
                if hasattr(actor_center, 'SetName'):
                    actor_center.SetName('crosshair_center')
            except Exception as e2:
                print(f"警告：绘制3D十字准心失败: {e2}")
    
    def set_interaction_mode(self, mode: str):
        """设置3D交互模式"""
        self.interaction_mode = mode
        
        if not hasattr(self, 'plotter') or self.plotter is None:
            return
        
        try:
            # PyVista的交互模式设置
            if mode == 'pointer':
                # 指针模式：禁用所有交互
                self.plotter.disable()
            elif mode == 'rotate':
                # 旋转模式：启用旋转（默认左键）
                self.plotter.enable_trackball_style()
            elif mode == 'pan':
                # 平移模式：启用平移（中键或Shift+左键）
                self.plotter.enable_trackball_style()
            elif mode == 'zoom':
                # 缩放模式：启用缩放（滚轮或右键）
                self.plotter.enable_trackball_style()
            elif mode == 'clip':
                # 裁剪模式：启用裁剪平面交互
                self.plotter.enable_trackball_style()
            else:
                # 默认启用trackball风格（包含旋转、平移、缩放）
                self.plotter.enable_trackball_style()
        except Exception as e:
            print(f"设置3D交互模式失败: {e}")
    
    def add_clip_plane(self, normal=(1, 0, 0), origin=(0, 0, 0)):
        """添加裁剪平面"""
        if not hasattr(self, 'plotter') or self.plotter is None:
            return
        
        try:
            import pyvista as pv
            
            # 创建裁剪平面
            plane = pv.Plane(center=origin, direction=normal, i_size=1000, j_size=1000)
            
            # 添加到plotter - 使用最基本的设置避免shader错误
            try:
                actor = self.plotter.add_mesh(
                    plane, 
                    color='yellow', 
                    opacity=0.3, 
                    name='clip_plane',
                    lighting=False  # 禁用光照避免shader错误
                )
            except Exception as e1:
                print(f"警告：添加裁剪平面mesh失败: {e1}")
                # 如果失败，尝试更简单的设置
                try:
                    actor = self.plotter.add_mesh(plane, color='yellow', opacity=0.3)
                except Exception as e2:
                    print(f"错误：添加裁剪平面完全失败: {e2}")
                    return
            
            # 保存到列表
            if not hasattr(self, 'clip_planes'):
                self.clip_planes = []
            self.clip_planes.append({
                'plane': plane,
                'actor': actor,
                'normal': normal,
                'origin': origin
            })
            
            # 更新显示
            try:
                self.plotter.render()
            except Exception as e:
                print(f"警告：渲染裁剪平面失败: {e}")
        except Exception as e:
            print(f"添加裁剪平面失败: {e}")
    
    def remove_clip_planes(self):
        """移除所有裁剪平面"""
        if not hasattr(self, 'plotter') or self.plotter is None:
            return
        
        try:
            if not hasattr(self, 'clip_planes') or not self.clip_planes:
                return
            
            # 移除所有裁剪平面
            for clip_plane in self.clip_planes:
                # 尝试移除actor（如果存在）
                if 'actor' in clip_plane:
                    try:
                        self.plotter.remove_actor(clip_plane['actor'])
                    except:
                        pass
                # 也尝试移除plane（向后兼容）
                elif 'plane' in clip_plane:
                    try:
                        self.plotter.remove_actor(clip_plane['plane'])
                    except:
                        pass
            
            self.clip_planes.clear()
            
            # 更新显示
            try:
                self.plotter.render()
            except Exception as e:
                print(f"警告：移除裁剪平面后渲染失败: {e}")
        except Exception as e:
            print(f"移除裁剪平面失败: {e}")
    
    def _update_3d_view_matplotlib(self):
        """使用matplotlib更新3D视图（备选方案）"""
        self.ax.clear()
        self.ax.set_facecolor('black')
        
        if self.image_data is None:
            self.canvas.draw()
            self.view_updated.emit()
            return
        
        h, w, d = self.image_data.shape
        
        if self.label_data is not None and np.any(self.label_data > 0):
            try:
                from skimage import measure
                label_binary = (self.label_data > 0).astype(float)
                
                verts, faces, normals, values = measure.marching_cubes(
                    label_binary, level=0.5, spacing=(1.0, 1.0, 1.0), allow_degenerate=False
                )
                
                if len(verts) > 0 and len(faces) > 0:
                    max_faces = 5000
                    if len(faces) > max_faces:
                        step = len(faces) // max_faces
                        faces = faces[::step]
                    
                    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                    mesh = Poly3DCollection(verts[faces], alpha=0.8, facecolor='red', 
                                          edgecolor='none', linewidths=0)
                    self.ax.add_collection3d(mesh)
                    
                    margin = 10
                    self.ax.set_xlim(verts[:, 0].min() - margin, verts[:, 0].max() + margin)
                    self.ax.set_ylim(verts[:, 1].min() - margin, verts[:, 1].max() + margin)
                    self.ax.set_zlim(verts[:, 2].min() - margin, verts[:, 2].max() + margin)
            except Exception as e:
                print(f"matplotlib渲染失败：{e}")
        
        # 设置坐标轴
        self.ax.set_xlim(0, h)
        self.ax.set_ylim(0, w)
        self.ax.set_zlim(0, d)
        
        self.ax.set_xlabel('X', color='white')
        self.ax.set_ylabel('Y', color='white')
        self.ax.set_zlabel('Z', color='white')
        self.ax.tick_params(colors='white')
        
        origin = [0, 0, 0]
        axis_length = min(h, w, d) * 0.3
        
        self.ax.plot([origin[0], axis_length], [origin[1], origin[1]], [origin[2], origin[2]], 
                    'b-', linewidth=2, alpha=0.8)
        self.ax.plot([origin[0], origin[0]], [origin[1], axis_length], [origin[2], origin[2]], 
                    'b-', linewidth=2, alpha=0.8)
        self.ax.plot([origin[0], origin[0]], [origin[1], origin[1]], [origin[2], axis_length], 
                    'b-', linewidth=2, alpha=0.8)
        
        marker_size = axis_length * 0.05
        self.ax.scatter([origin[0]], [origin[1]], [origin[2]], 
                       c='#00FF00', s=marker_size*100, marker='s', alpha=0.9)
        
        self.ax.view_init(elev=20, azim=45)
        self.ax.set_proj_type('ortho')
        
        self.canvas.draw()
        self.view_updated.emit()
    
    def refresh_view(self):
        """刷新3D视图 - 公共接口（手动更新）"""
        self._update_3d_view()
    
    def reset_camera(self):
        """重置相机视角"""
        if self.use_pyvista:
            if hasattr(self, 'plotter'):
                self.plotter.camera_position = 'iso'
                self.plotter.camera.parallel_projection = True
        else:
            if self.image_data is not None and hasattr(self, 'ax'):
                self.ax.view_init(elev=20, azim=45)
                self.ax.set_proj_type('ortho')
                self.canvas.draw()


class FourViewWidget(QWidget):
    """2×2布局视图组件"""
    
    # 信号：十字线位置改变 (view_type, x, y)
    crosshair_changed = pyqtSignal(str, float, float)
    # 信号：切片改变 (view_type, slice_idx)
    slice_changed = pyqtSignal(str, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_data = None
        self.label_data = None
        self.image_spacing = None  # 保存spacing信息
        
        # 当前十字线位置（3D空间坐标）
        self.crosshair_3d = [0, 0, 0]  # [x, y, z]
        
        self._init_ui()
        self._connect_signals()
    
    def _init_ui(self):
        """初始化2×2布局 - 使用QGridLayout，紧密排布"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # 无外边距
        layout.setSpacing(0)  # 无间距
        
        # 创建2×2网格布局
        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(0, 0, 0, 0)  # 无外边距
        grid_layout.setSpacing(1)  # 最小间距（1像素，用于分隔线效果）
        
        # 设置行和列的拉伸权重（均为1，实现自适应）
        grid_layout.setRowStretch(0, 1)
        grid_layout.setRowStretch(1, 1)
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 1)
        
        # 创建视图
        self.axial_view = SliceViewer('axial')  # 行0列0：轴位
        self.sagittal_view = SliceViewer('sagittal')  # 行0列1：矢状位
        self.view_3d = ThreeDViewer()  # 行1列0：3D视图（左下角）
        self.coronal_view = SliceViewer('coronal')  # 行1列1：冠状位
        
        # 设置3D视图的宽高比策略（保持接近1:1）
        # 使用QSizePolicy确保3D视图保持宽高比
        size_policy_3d = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        size_policy_3d.setHeightForWidth(True)  # 允许根据宽度调整高度
        self.view_3d.setSizePolicy(size_policy_3d)
        
        # 添加到网格布局
        grid_layout.addWidget(self.axial_view, 0, 0)  # 行0列0（左上角）
        grid_layout.addWidget(self.coronal_view, 0, 1)  # 行0列1（右上角：冠状位）
        grid_layout.addWidget(self.view_3d, 1, 0)  # 行1列0（左下角：3D视图）
        grid_layout.addWidget(self.sagittal_view, 1, 1)  # 行1列1（右下角：矢状位）
        
        # 设置最小尺寸约束
        grid_layout.setColumnMinimumWidth(0, 300)
        grid_layout.setColumnMinimumWidth(1, 300)
        grid_layout.setRowMinimumHeight(0, 300)
        grid_layout.setRowMinimumHeight(1, 300)
        
        layout.addLayout(grid_layout)
        self.setLayout(layout)
    
    def _connect_signals(self):
        """连接信号，实现十字线联动（点击和拖拽完全同步）"""
        # 轴位视图：点击和拖拽时更新冠状位和矢状位的十字线
        self.axial_view.mouse_clicked.connect(
            lambda x, y: self._on_axial_clicked(x, y)
        )
        self.axial_view.mouse_dragged.connect(
            lambda x, y: self._on_axial_clicked(x, y)  # 拖拽时也调用相同函数
        )
        
        # 冠状位视图：点击和拖拽时更新轴位和矢状位的十字线
        self.coronal_view.mouse_clicked.connect(
            lambda x, y: self._on_coronal_clicked(x, y)
        )
        self.coronal_view.mouse_dragged.connect(
            lambda x, y: self._on_coronal_clicked(x, y)  # 拖拽时也调用相同函数
        )
        
        # 矢状位视图：点击和拖拽时更新轴位和冠状位的十字线
        self.sagittal_view.mouse_clicked.connect(
            lambda x, y: self._on_sagittal_clicked(x, y)
        )
        self.sagittal_view.mouse_dragged.connect(
            lambda x, y: self._on_sagittal_clicked(x, y)  # 拖拽时也调用相同函数
        )
        
        # 切片改变信号
        self.axial_view.slice_changed.connect(lambda idx: self.slice_changed.emit('axial', idx))
        self.coronal_view.slice_changed.connect(lambda idx: self.slice_changed.emit('coronal', idx))
        self.sagittal_view.slice_changed.connect(lambda idx: self.slice_changed.emit('sagittal', idx))
    
    def _on_axial_clicked(self, x: float, y: float):
        if self.image_data is None:
            return
        
        slice_img = self.axial_view._get_slice_data()
        if slice_img is None:
            return
        
        if self.image_spacing is not None:
            axial_pixel_x = x / self.image_spacing[0]
            axial_pixel_y = y / self.image_spacing[1]
        else:
            axial_pixel_x = x
            axial_pixel_y = y
        
        pixel_idx_x = int(axial_pixel_x)
        pixel_idx_y = int(axial_pixel_y)
        pixel_idx_z = self.axial_view.current_slice
        
        if self.image_spacing is not None:
            space_x = pixel_idx_x * self.image_spacing[0]
            space_y = pixel_idx_y * self.image_spacing[1]
            space_z = pixel_idx_z * self.image_spacing[2]
        else:
            space_x = float(pixel_idx_x)
            space_y = float(pixel_idx_y)
            space_z = float(pixel_idx_z)
        
        self.crosshair_3d = [space_x, space_y, space_z]
        self.view_3d.set_crosshair_3d(space_x, space_y, space_z)
        
        coronal_slice_idx = int(space_y / self.image_spacing[1]) if self.image_spacing else int(space_y)
        if self.coronal_view.current_slice != coronal_slice_idx:
            self.coronal_view.set_slice(coronal_slice_idx)
        coronal_cross_x = space_y
        coronal_cross_y = space_z
        self.coronal_view.set_crosshair(coronal_cross_x, coronal_cross_y)
        
        sagittal_slice_idx = int(space_x / self.image_spacing[0]) if self.image_spacing else int(space_x)
        if self.sagittal_view.current_slice != sagittal_slice_idx:
            self.sagittal_view.set_slice(sagittal_slice_idx)
        sagittal_cross_x = space_x
        sagittal_cross_y = space_z
        self.sagittal_view.set_crosshair(sagittal_cross_x, sagittal_cross_y)
        
        self.crosshair_changed.emit('axial', x, y)
    
    def _on_coronal_clicked(self, x: float, y: float):
        if self.image_data is None:
            return
        
        slice_img = self.coronal_view._get_slice_data()
        if slice_img is None:
            return
        
        if self.image_spacing is not None:
            coronal_pixel_x = x / self.image_spacing[0]
            coronal_pixel_y = y / self.image_spacing[2]
        else:
            coronal_pixel_x = x
            coronal_pixel_y = y
        
        pixel_idx_x = int(coronal_pixel_x)
        pixel_idx_y = self.coronal_view.current_slice
        pixel_idx_z = int(coronal_pixel_y)
        
        if self.image_spacing is not None:
            space_x = pixel_idx_x * self.image_spacing[0]
            space_y = pixel_idx_y * self.image_spacing[1]
            space_z = pixel_idx_z * self.image_spacing[2]
        else:
            space_x = float(pixel_idx_x)
            space_y = float(pixel_idx_y)
            space_z = float(pixel_idx_z)
        
        self.crosshair_3d = [space_x, space_y, space_z]
        self.view_3d.set_crosshair_3d(space_x, space_y, space_z)
        
        axial_slice_idx = int(space_z / self.image_spacing[2]) if self.image_spacing else int(space_z)
        if self.axial_view.current_slice != axial_slice_idx:
            self.axial_view.set_slice(axial_slice_idx)
        axial_cross_x = space_y
        axial_cross_y = space_x
        self.axial_view.set_crosshair(axial_cross_x, axial_cross_y)
        
        sagittal_slice_idx = int(space_x / self.image_spacing[0]) if self.image_spacing else int(space_x)
        if self.sagittal_view.current_slice != sagittal_slice_idx:
            self.sagittal_view.set_slice(sagittal_slice_idx)
        sagittal_cross_x = space_x
        sagittal_cross_y = space_z
        self.sagittal_view.set_crosshair(sagittal_cross_x, sagittal_cross_y)
        
        self.crosshair_changed.emit('coronal', x, y)
    
    def _on_sagittal_clicked(self, x: float, y: float):
        if self.image_data is None:
            return
        
        slice_img = self.sagittal_view._get_slice_data()
        if slice_img is None:
            return
        
        if self.image_spacing is not None:
            sagittal_pixel_x = x / self.image_spacing[1]
            sagittal_pixel_y = y / self.image_spacing[2]
        else:
            sagittal_pixel_x = x
            sagittal_pixel_y = y
        
        pixel_idx_x = self.sagittal_view.current_slice
        pixel_idx_y = int(sagittal_pixel_x)
        pixel_idx_z = int(sagittal_pixel_y)
        
        if self.image_spacing is not None:
            space_x = pixel_idx_x * self.image_spacing[0]
            space_y = pixel_idx_y * self.image_spacing[1]
            space_z = pixel_idx_z * self.image_spacing[2]
        else:
            space_x = float(pixel_idx_x)
            space_y = float(pixel_idx_y)
            space_z = float(pixel_idx_z)
        
        self.crosshair_3d = [space_x, space_y, space_z]
        self.view_3d.set_crosshair_3d(space_x, space_y, space_z)
        
        axial_slice_idx = int(space_z / self.image_spacing[2]) if self.image_spacing else int(space_z)
        if self.axial_view.current_slice != axial_slice_idx:
            self.axial_view.set_slice(axial_slice_idx)
        axial_cross_x = space_y
        axial_cross_y = space_x
        self.axial_view.set_crosshair(axial_cross_x, axial_cross_y)
        
        coronal_slice_idx = int(space_y / self.image_spacing[1]) if self.image_spacing else int(space_y)
        if self.coronal_view.current_slice != coronal_slice_idx:
            self.coronal_view.set_slice(coronal_slice_idx)
        coronal_cross_x = space_y
        coronal_cross_y = space_z
        self.coronal_view.set_crosshair(coronal_cross_x, coronal_cross_y)
        
        self.crosshair_changed.emit('sagittal', x, y)
    
    def set_image_data(self, image_data: np.ndarray, spacing=None):
        """
        设置图像数据
        
        Args:
            image_data: 图像数据 (H, W, D)
            spacing: 像素间距，可选
        """
        self.image_data = image_data
        self.image_spacing = spacing  # 保存spacing信息
        self.axial_view.set_image_data(image_data, spacing)
        self.coronal_view.set_image_data(image_data, spacing)
        self.sagittal_view.set_image_data(image_data, spacing)
        self.view_3d.set_image_data(image_data, spacing)
        
            # 初始化十字线到中心，并同步所有视图
        if image_data is not None:
            h, w, d = image_data.shape  # (Y, X, Z)
            # crosshair_3d 顺序：[X, Y, Z]
            self.crosshair_3d = [
                w // 2,
                h // 2,
                d // 2
            ]
            
            orig_x = self.crosshair_3d[0]
            orig_y = self.crosshair_3d[1]
            orig_z = self.crosshair_3d[2]
            
            self.axial_view.set_slice(orig_z)
            self.axial_view.set_crosshair(orig_x, orig_y)
            
            self.coronal_view.set_slice(orig_y)
            # Coronal的x轴对应Y方向（前后A-P），y轴对应Z方向（上下S-I）
            if self.image_spacing is not None:
                self.coronal_view.set_crosshair(orig_y * self.image_spacing[1], orig_z * self.image_spacing[2])
            else:
                self.coronal_view.set_crosshair(orig_y, orig_z)
            
            self.sagittal_view.set_slice(orig_x)
            # Sagittal的x轴对应X方向（左右R-L），y轴对应Z方向（上下S-I）
            if self.image_spacing is not None:
                self.sagittal_view.set_crosshair(orig_x * self.image_spacing[1], orig_z * self.image_spacing[2])
            else:
                self.sagittal_view.set_crosshair(orig_x, orig_z)
            
            # 初始化3D视图中的十字准心
            self.view_3d.set_crosshair_3d(orig_x, orig_y, orig_z)
    
    def set_label_data(self, label_data: np.ndarray):
        """设置标签数据"""
        self.label_data = label_data
        # 2D视图立即更新（轻量级操作）
        self.axial_view.set_label_data(label_data)
        self.coronal_view.set_label_data(label_data)
        self.sagittal_view.set_label_data(label_data)
            # 3D视图不自动更新（需要手动点击Update按钮）
        # 只设置数据，不触发更新，避免卡顿
        self.view_3d.set_label_data(label_data)
    
    def set_window(self, window_width: float, window_level: float):
        """设置窗宽窗位（所有视图）"""
        self.axial_view.set_window(window_width, window_level)
        self.coronal_view.set_window(window_width, window_level)
        self.sagittal_view.set_window(window_width, window_level)
    
    def refresh_3d_view(self):
        """
        刷新3D视图 - 公共接口
        刷新逻辑：重新加载数据、重置相机、执行体渲染更新
        """
        if self.view_3d is not None:
            # 重新设置数据（触发更新）
            if self.image_data is not None:
                self.view_3d.set_image_data(self.image_data, self.image_spacing)
            if self.label_data is not None:
                self.view_3d.set_label_data(self.label_data)
            
            # 重置相机视角
            self.view_3d.reset_camera()
            
            # 执行刷新
            self.view_3d.refresh_view()
