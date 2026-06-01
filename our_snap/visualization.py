#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
闁告瑩宕犻弽顭屼線宕?
闁告梻鍠曢崗姗€鏁?閼?閻㈩垰鍟惇顒傛喆閸℃绂堥柨娑樼墣闁拌鲸鎷?闁活厹鍨绘慨鍛婃媴?闁告劗濮锋慨鍛婃媴?3D濡澘瀚柨娑橆檧缁辨繈宕℃担鍝ユ憻缂佹崘鍎绘禒鍫ュ礉缁辨繈寮悷鐗堝€婚柡宥呮穿閻︽垿鏁?D閻炴稓绮婚埄鍐煇闁告瑩宕犻弽鐢电闁惧繑纰嶇€氭瑩宕氶崶褍顥忔鐐村▕濞?
"""

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QSizePolicy, QSlider, QDoubleSpinBox, QComboBox
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
from typing import Dict, Optional, Tuple


class SliceViewer(QWidget):
    """闁告娲戦柌婊堝礆閸モ晛顣婚悷娆忔濞?"""
    
    slice_changed = pyqtSignal(int)
    crosshair_changed = pyqtSignal(float, float)
    mouse_clicked = pyqtSignal(float, float)
    mouse_dragged = pyqtSignal(float, float)
    window_changed = pyqtSignal(float, float)
    label_edit_started = pyqtSignal()
    label_changed = pyqtSignal()
    label_edit_finished = pyqtSignal()
    
    def __init__(self, view_type: str, parent=None):
        """
        闁告帗绻傞柛鏍ㄧ墪閸ㄥ繘鎮ч崶顏堝炊?

        Args:
            view_type: 閻熸瑥妫楀ù妯肩尵鐠囪尙鈧?('axial', 'coronal', 'sagittal', '3d')
        """
        super().__init__(parent)
        self.setObjectName("SliceViewer")
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
        self.seed_points = []
        self.show_seed_points = True
        
        # 闁哄倻鎳撻幃婊堝冀閸ヮ亞妲?
        self.direction_labels = self._get_direction_labels()
        
        # 闁告帒娲ゆ竟濠囩嵁閹惰姤妗ㄩ柣鈺冾焾閸?
        self.cutting_planes = []  # 閻庢稒锚閸嬪秹宕氶崶褍顥忔鐐村▕濞肩増绌遍埄鍐х礀
        self.selected_plane = None  # 鐟滅増鎸告晶鐘绘焻婢跺鍘柣銊ュ閸ㄥ繘宕滈幓鎺楁尙闂?
        
        self._init_ui()
    
    def _get_direction_labels(self) -> dict:
        """闁兼儳鍢茶ぐ鍥棘閻熺増鍊婚柡宥呮穿閻?"""
        if self.view_type == 'axial':
            return {'top': 'A', 'bottom': 'P', 'left': 'R', 'right': 'L'}
        elif self.view_type == 'coronal':
            return {'top': 'S', 'bottom': 'I', 'left': 'A', 'right': 'P'}
        elif self.view_type == 'sagittal':
            return {'top': 'S', 'bottom': 'I', 'left': 'R', 'right': 'L'}
        else:
            return {}
    
    def _init_ui(self):
        """闁告帗绻傞柛鏍ㄦ恭I"""
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(1)
        
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(2, 2, 2, 2)
        title_layout.setSpacing(4)
        
        view_title = QLabel(self.view_type.upper())
        view_title.setObjectName("SliceViewTitle")
        view_title.setFont(QFont("Arial", 8, QFont.Bold))
        title_layout.addWidget(view_title)
        
        title_layout.addStretch()
        
        if self.direction_labels:
            dir_label = QLabel(
                f"{self.direction_labels.get('left', '')} | {self.direction_labels.get('right', '')}"
            )
            dir_label.setObjectName("SliceViewMeta")
            dir_label.setFont(QFont("Arial", 7))
            title_layout.addWidget(dir_label)
        
        btn_zoom_fit = QPushButton("Fit")
        btn_zoom_fit.setObjectName("ViewerMiniButton")
        btn_zoom_fit.setMaximumWidth(35)
        btn_zoom_fit.setMaximumHeight(18)
        btn_zoom_fit.setFont(QFont("Arial", 7))
        btn_zoom_fit.clicked.connect(self._on_zoom_fit)
        title_layout.addWidget(btn_zoom_fit)
        
        layout.addLayout(title_layout)
        
        bg_color = '#000000'
        self.figure = Figure(figsize=(4, 4), dpi=120, facecolor=bg_color)
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #000000;")
        self.ax = self.figure.add_subplot(111, facecolor=bg_color)
        self.ax.axis('off')
        
        self.zoom_factor = 1.0
        self.zoom_center_x = 0.0
        self.zoom_center_y = 0.0
        self.xlim_range = None
        self.ylim_range = None
        
        layout.addWidget(self.canvas, stretch=1)
        
        # 闁告帒娲ㄦ晶鏍閵忕姷绌块柡鍕⒔閵囨岸鏁嶉崼?24 of 47"闁? 缂佹瘱鍐ㄦ闁哄嫬澧介妵?
        self.slice_label = QLabel("0 of 0")
        self.slice_label.setObjectName("SliceCounter")
        self.slice_label.setAlignment(Qt.AlignCenter)
        self.slice_label.setFont(QFont("Arial", 8))
        self.slice_label.setMaximumHeight(18)  # 闂傚嫭鍔曢崺妤侇殗濡搫顔?
        layout.addWidget(self.slice_label)

        # 鍒囧眰婊戝姩鏉★紙姣忎釜 2D 瑙嗗浘鐙珛锛?
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_slider.setEnabled(False)
        self.slice_slider.setMaximumHeight(18)
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        layout.addWidget(self.slice_slider)
        
        self.setLayout(layout)
        
        # 閻犱礁澧介悿鍡涘及鎼淬垺鈻旈柣銊ュ缁旂喎顩?
        self.setStyleSheet("""
            QWidget#SliceViewer {
                border: 1px solid #26344a;
                border-radius: 6px;
                background-color: #000000;
            }
            QLabel#SliceViewTitle {
                color: #e6edf7;
                background-color: transparent;
                border: none;
                padding-left: 4px;
            }
            QLabel#SliceViewMeta, QLabel#SliceCounter {
                color: #9fb7d8;
                background-color: transparent;
                border: none;
            }
            QPushButton#ViewerMiniButton {
                color: #e6edf7;
                background-color: #111827;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 0;
            }
            QSlider {
                background: transparent;
                border: none;
            }
        """)
        
        # 閻犱礁澧介悿鍡涘嫉閻忓繐绻愰弰鍌溾偓?
        self.setMinimumSize(300, 300)
        
        # 閺夆晝鍋炵敮瀛樜楅悩宕囧灱濞存粌顑勫▎?
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        
        self.mouse_pressed = False
        self.mouse_button = None  # 閻犱焦婢樼紞宥夊箰婢跺鐟撻柣銊ュ鐎垫粓鏌︾槐?=鐎归潻绠撻弫顓㈡晬?=闁告瑦濞婇弫顓㈡晬?
        self.last_mouse_pos = None
        self.last_mouse_canvas_pos = None
        self.pending_pointer_sync_pos = None
        self._label_edit_active = False
    
    def set_image_data(self, image_data: np.ndarray, spacing=None):
        """
        閻犱礁澧介悿鍡涘炊閹冨壖闁轰胶澧楀畵?
        
        Args:
            image_data: 闁搞儲鍎抽崕姘跺极閻楀牆绁?(H, W, D)
            spacing: 闁稿秴绻掔粈宀勬⒒绾惧鐛?(spacing_x, spacing_y, spacing_z)闁挎稑鑻ぐ鏌ユ焻?
        """
        self.image_data = image_data
        self.image_spacing = spacing
        if image_data is not None:
            # 闁告帗绻傞柛鏍ㄧ墪瀹曞嫮鈧稒顨堥崵搴㈡媴瀹ュ洨鏋傞柛鎺撴緲濞存﹢宕撹箛搴ゅ幀闊?
            if self.view_type == 'axial':
                # 閺夌偟绻濈紞鍛喆閸℃绂堥弶鐐垫瀭闁告艾鍑界槐婵嬪锤閹邦厾鍨煎☉鏃傚枛濞撳墎鎲版担鐣屾闁?
                self.crosshair_x = image_data.shape[0] / 2.0  # 閺夌偟鏋傞柛姘€楅悗鐢垫嚀缁ㄦ煡宕㈤悞?
                self.crosshair_y = image_data.shape[1] / 2.0  # 閺夌偟鏋傞柛姘棙閻庣數鎳撶花鏌ュ储閻?
            elif self.view_type == 'coronal':
                self.crosshair_x = image_data.shape[1] / 2.0
                self.crosshair_y = image_data.shape[2] / 2.0
            elif self.view_type == 'sagittal':
                self.crosshair_x = image_data.shape[0] / 2.0
                self.crosshair_y = image_data.shape[2] / 2.0
            
            # 闂佹彃绉堕悿鍡欑磽閳哄倹鏉归柛婊冪焷闁搞儲宕樼€垫牠宕舵潏鍓х濞达綀娉曢弫銈夋偋閳哄啯鍊為柛褎鍔栭悥锝夋晬?
            self.zoom_factor = 1.0
            # 缂佹柨顑呭畵鍡涘即鐎涙ɑ鐓€闁哄嫬澧介妵?
            self._update_display()
            slice_img = self._get_slice_data()
            if slice_img is not None:
                h, w = slice_img.shape
                # 閻犱緤绱曢悾濠氭偋閳哄啯鍊為柛褎鍔栭悥锝夋嚑閸愩劍绾?
                xlim, ylim, _, _ = self._get_physical_range(h, w)
                self.xlim_range = xlim
                self.ylim_range = ylim
        
        self._update_slice_range()
        self._update_display()
    
    def set_label_data(self, label_data: np.ndarray):
        """閻犱礁澧介悿鍡涘冀閸モ晠寮悧鍫濈ウ"""
        self.label_data = label_data
        self._update_display()

    def set_seed_points(self, seed_points):
        """閻犱礁澧介悿鍡欑矓瀹ュ懐鎽嶉柣鎰嚀閸亞鎮扮槐娆愭媴閹惧墎顦遍柛褎鍔栭悥锝夋晬鐏炲墽澹愮€殿喖鍝?x,y,z), ...]闁?"""
        if seed_points is None:
            self.seed_points = []
        else:
            cleaned = []
            for p in seed_points:
                try:
                    cleaned.append((int(p[0]), int(p[1]), int(p[2])))
                except Exception:
                    continue
            self.seed_points = cleaned
        self._update_display()
    
    def set_label_color(self, label_value: int, color: tuple):
        """
        閻犱礁澧介悿鍡涘冀閸モ晜锛愬鍡楊棌
        
        Args:
            label_value: 闁哄秴娲ㄩ柛濠勩€嬬槐?, 2, 3...闁?
            color: 濡増绮忔竟?(R, G, B, A) 闁?(R, G, B)闁挎稑鑻懝鏉跨槺闁?-1
        """
        if len(color) == 3:
            color = (*color, self.label_opacity)
        else:
            # 濞ｅ洦绻勯弳鈧琑GB闁稿﹦銆嬬槐婵囨媴鐠恒劍鏆忕憸鐗堟尭婢х娀鏌呰箛鏃€顫栭幖?
            color = (color[0], color[1], color[2], self.label_opacity)
        self.label_colors[label_value] = color
        self._update_display()
    
    def set_crosshair(self, x: float, y: float, redraw: bool = True, emit_signal: bool = True):
        """閻犱礁澧介悿鍡涘础娴ｅ摜鎽熺紒鎯с仒缂嶅懐绱旂槐娆撴偋閳哄啯鍊為柛褎鍔栭悥锝夋晬?"""
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
        
        if redraw:
            self._update_display()
        if emit_signal:
            self.crosshair_changed.emit(self.crosshair_x, self.crosshair_y)
    
    def set_slice(self, slice_idx: int, emit_signal: bool = True, redraw: bool = True):
        """閻犱礁澧介悿鍡氥亹閹惧啿顤呴柛鎺戞川婢?"""
        if self.image_data is None:
            return
        
        max_slice = self._get_max_slice()
        if max_slice <= 0:
            return
        new_slice = max(0, min(int(slice_idx), max_slice - 1))
        changed = (new_slice != self.current_slice)
        self.current_slice = new_slice
        self._update_slice_range()
        if redraw:
            self._update_display()
        if emit_signal and changed:
            self.slice_changed.emit(self.current_slice)
    
    def set_tool_mode(self, mode: str):
        """閻犱礁澧介悿鍡楊啅閵夈儱寰旀俊顖椻偓宕囩"""
        self.tool_mode = mode
        # 闁哄洤鐡ㄩ弻濠兾楅悩宕囧灱闁哄秴鍢茬槐?
        if mode == 'brush':
            self.canvas.setCursor(Qt.CrossCursor)
        elif mode == 'eraser':
            self.canvas.setCursor(Qt.CrossCursor)
        elif mode == 'zoom':
            self.canvas.setCursor(Qt.OpenHandCursor)
        elif mode == 'pan':
            self.canvas.setCursor(Qt.OpenHandCursor)
        else:  # pointer
            self.canvas.setCursor(Qt.ArrowCursor)
    
    def set_brush_size(self, size: int):
        """Set brush radius in pixels."""
        self.brush_size = max(1, min(50, size))
    
    def set_active_label_id(self, label_id: int):
        """Set active label id for drawing."""
        self.active_label_id = label_id
    
    def set_label_opacity(self, opacity: float):
        """Set label overlay opacity."""
        self.label_opacity = max(0.0, min(1.0, opacity))
        # 闁哄洤鐡ㄩ弻濠囧箥闁哄牆顦崙锛勬媼閸撗呮瀭闁汇劌瀚悥锝囩驳妤ｅ啴鎳濋懠顒佺暠闂侇偄绻戝Σ鎴炴償?
        for label_value in self.label_colors:
            color = self.label_colors[label_value]
            if len(color) == 4:
                # 濞ｅ洦绻勯弳鈧琑GB闁稿﹦銆嬬槐婵嬪即鐎涙ɑ鐓€闂侇偄绻戝Σ鎴炴償?
                self.label_colors[label_value] = (color[0], color[1], color[2], self.label_opacity)
            else:
                # 婵烇綀顕ф慨鐐烘焻韫囨梹顫栭幖?
                self.label_colors[label_value] = (*color, self.label_opacity)
        self._update_display()
    
    def set_window(self, window_width: float, window_level: float):
        """閻犱礁澧介悿鍡欑玻濡も偓缂佹劖銇炵紞?"""
        self.window_width = window_width
        self.window_level = window_level
        self._update_display()
    
    def _get_max_slice(self) -> int:
        """闁兼儳鍢茶ぐ鍥嫉濠㈠爢鍐ㄧ€奸柣妤€娲﹂弳?"""
        if self.image_data is None:
            return 0
        
        if self.view_type == 'axial':
            return self.image_data.shape[2]  # Z 閺夌偞娼欓崹蹇涙偋閸ャ劍娈?
        elif self.view_type == 'coronal':
            return self.image_data.shape[1]  # X 閺夌偞娼欓崹蹇涙偋閸ャ劍娈堕柨娑樻綇oronal闁搞儱鎼悾缍抽柨?
        elif self.view_type == 'sagittal':
            return self.image_data.shape[0]  # Y 閺夌偞娼欓崹蹇涙偋閸ャ劍娈堕柨娑樻緝agittal闁搞儱鎼悾缍撮柨?
        return 0
    
    def _update_slice_range(self):
        """闁哄洤鐡ㄩ弻濠囧礆閸モ晛顣婚柤鐓庡暙濞?"""
        max_slice = self._get_max_slice()
        if max_slice > 0:
            self.current_slice = min(self.current_slice, max_slice - 1)
            if hasattr(self, "slice_slider") and self.slice_slider is not None:
                self.slice_slider.blockSignals(True)
                self.slice_slider.setEnabled(True)
                self.slice_slider.setMinimum(0)
                self.slice_slider.setMaximum(max_slice - 1)
                self.slice_slider.setValue(self.current_slice)
                self.slice_slider.blockSignals(False)
        else:
            self.current_slice = 0
            if hasattr(self, "slice_slider") and self.slice_slider is not None:
                self.slice_slider.blockSignals(True)
                self.slice_slider.setEnabled(False)
                self.slice_slider.setMinimum(0)
                self.slice_slider.setMaximum(0)
                self.slice_slider.setValue(0)
                self.slice_slider.blockSignals(False)
    
    def _get_physical_range(self, h: int, w: int):
        """閻犱緤绱曢悾濠氭偋閳哄啯鍊為柛褎鍔栭悥锝夋嚑閸愩劍绾?(xlim_range, ylim_range, max_x, max_y)"""
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

    def _ensure_view_range(self):
        slice_img = self._get_slice_data()
        if slice_img is None:
            return None
        h, w = slice_img.shape
        xlim, ylim, max_x, max_y = self._get_physical_range(h, w)
        if self.xlim_range is None or self.ylim_range is None:
            self.xlim_range = xlim
            self.ylim_range = ylim
        return h, w, max_x, max_y

    def _set_view_range(self, left: float, bottom: float, width: float, height: float):
        state = self._ensure_view_range()
        if state is None:
            return False
        h, w, max_x, max_y = state
        if max_x <= 0 or max_y <= 0:
            return False

        pixel_w = max_x / max(1, w)
        pixel_h = max_y / max(1, h)
        min_width = max(pixel_w * 6.0, max_x * 0.004)
        min_height = max(pixel_h * 6.0, max_y * 0.004)

        width = min(max(float(width), min_width), max_x)
        height = min(max(float(height), min_height), max_y)

        if width >= max_x:
            left = 0.0
        else:
            left = min(max(float(left), 0.0), max_x - width)

        if height >= max_y:
            bottom = 0.0
        else:
            bottom = min(max(float(bottom), 0.0), max_y - height)

        self.xlim_range = (left, left + width)
        self.ylim_range = (bottom, bottom + height)
        self.zoom_factor = max(1.0, min(max_x / width, max_y / height))
        return True

    def _zoom_at(self, x: float, y: float, scale: float):
        state = self._ensure_view_range()
        if state is None:
            return False

        left, right = self.xlim_range
        bottom, top = self.ylim_range
        width = right - left
        height = top - bottom
        if width <= 0 or height <= 0:
            _h, _w, max_x, max_y = state
            left, bottom, width, height = 0.0, 0.0, max_x, max_y

        rel_x = 0.5 if x is None else min(max((float(x) - left) / width, 0.0), 1.0)
        rel_y = 0.5 if y is None else min(max((float(y) - bottom) / height, 0.0), 1.0)
        scale = min(max(float(scale), 0.2), 5.0)

        new_width = width * scale
        new_height = height * scale
        anchor_x = left + rel_x * width if x is None else float(x)
        anchor_y = bottom + rel_y * height if y is None else float(y)
        new_left = anchor_x - rel_x * new_width
        new_bottom = anchor_y - rel_y * new_height

        if self._set_view_range(new_left, new_bottom, new_width, new_height):
            self._update_display()
            return True
        return False

    def _zoom_from_pointer_drag(self, event):
        if event.xdata is None or event.ydata is None:
            return
        if self.last_mouse_canvas_pos is None:
            self.last_mouse_canvas_pos = (event.x, event.y)
            return
        dy_pixels = float(event.y - self.last_mouse_canvas_pos[1])
        if abs(dy_pixels) < 1.0:
            return
        scale = float(np.exp(-dy_pixels * 0.012))
        scale = min(max(scale, 0.72), 1.38)
        self._zoom_at(event.xdata, event.ydata, scale)
        self.last_mouse_canvas_pos = (event.x, event.y)

    def _pan_from_pointer_drag(self, event):
        if self.last_mouse_canvas_pos is None:
            self.last_mouse_canvas_pos = (event.x, event.y)
            return
        state = self._ensure_view_range()
        if state is None:
            return

        inv = self.ax.transData.inverted()
        last_x, last_y = inv.transform(self.last_mouse_canvas_pos)
        current_x, current_y = inv.transform((event.x, event.y))
        dx = float(current_x - last_x)
        dy = float(current_y - last_y)
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return

        left, right = self.xlim_range
        bottom, top = self.ylim_range
        if self._set_view_range(left - dx, bottom - dy, right - left, top - bottom):
            self._update_display()
            self.last_mouse_canvas_pos = (event.x, event.y)
    
    def _get_slice_data(self) -> np.ndarray:
        """
        闁兼儳鍢茶ぐ鍥亹閹惧啿顤呴柛鎺戞川婢ф牠鎯冮崟顐ｇ闁稿秴绻戦弳鐔煎箲?
        婵炲鍔嶉崜浼存晬濮濈椃mpy闁轰焦澹嗙划宥囨閵忕姷绌垮☉?(H, W, D)闁挎稑鑻幖瀛樶煥鐏忔壆鈧冻绠戞總鏍磽韫囨洘鐣?(Y, X, Z)

        闁哄秴娲ら崳顖炲礆閸モ晛顣婚柟缁樺姇瑜板洭鏁?
        - Axial闁挎稑鐗愰柊杈ㄦ媴瀹ュ繒绀嗛柨娑欑煯缁姵绋夋繝鍌滅獢濞戞挸顑囧﹢鍛存晬鐏炵偓鈻旂紒鈧弧?Y妤犵偞濞婂浼存晬鐎涘吋娼弶鎴濈€奸柣?
          -> image_data[:, :, z] 鐎电増顨呴崺?(Y, X) 闁?D闁轰焦澹嗙划?
        - Coronal闁挎稑鐗嗛崯婵嬫偐閺堢數绉撮柨娑橆檧缁辩増绂掓惔鈥愁枀闂傚牄鍨诲﹢鍛存晬鐏炵偓鈻旂紒鈧弧?Z妤犵偞濞婂浼存晬鐎涚粯娼弶鎴濈€奸柣?
          -> image_data[y, :, :] 鐎电増顨呴崺?(X, Z) 闁?D闁轰焦澹嗙划宥夋晬瀹€鍕粯閻熸洑娴囧ù鍡欑磾鐠?(Z, X)
        - Sagittal闁挎稑鐗忛悡婵嬫偐閺堢數绉撮柨娑橆檧缁辩増绂掓惔婵囨珷闂傚牄鍨诲﹢鍛存晬鐏炵偓鈻旂紒鈧弧?Z妤犵偞濞婂浼存晬鐎涜娼弶鎴濈€奸柣?
          -> image_data[:, x, :] 鐎电増顨呴崺?(Y, Z) 闁?D闁轰焦澹嗙划宥夋晬瀹€鍕粯閻熸洑娴囧ù鍡欑磾鐠?(Z, Y)
        """
        if self.image_data is None:
            return None
        
        if self.view_type == 'axial':
            # 閺夌偟绻濈紞鍛存晬濮樿鲸绾柟鎭掑劜瑜颁線宕ｉ張鍙夋姜閺夋垵鐎奸柣妤€娴勭槐婵嬪及閸撗佷粵X-Y妤犵偞濞婂?
            # 閺夌偟绻濈紞鍛喆閸℃绂堟俊顖涚祷闁衡偓閹惧懐绀勬慨妯绘綑闁解晠寮悷鐗堝€婚柨?
            # image_data[:, :, z] 鐎电増顨呴崺?(Y, X)闁挎稑鐭傚〒鍓佹啺娴ｈ姤绁紓鍐 (X, Y) 濞寸姰鍎查搯闁活偊寮伴崜褋浠?
            return np.fliplr(self.image_data[:, :, self.current_slice].T)
        elif self.view_type == 'coronal':
            # 闁告劗濮锋慨鍛婃媴瀹ュ繒绐楅柟缁樺姇瑜板槴閺夌偞娼欓崹蹇涙偋閸ラ绀夐弶鐐垫瀭濞?(Z, X)闁挎稑鐬奸崝褔宕ユ惔婵堢憪濞戞挸顑囬悙鏇熸姜?
            # 闁哄倻鎳撻幃婊堝冀閸ヮ亞妲曢柨娑欑煯缁?I, 濞?S, 鐎?A, 闁?P闁挎稑婢戦弶鐐存綑閹煎瓨鎽抽柡鍌滄嚀閹粓鏁嶇€靛娼弶鎴炴償閹垮綊寮悷鐗堝€婚柨?
            slice_data = self.image_data[self.current_slice, :, :].T
            return np.flipud(slice_data)  # 濞戞挸锕ｇ粭鍛礄閺勫繑绁?
        elif self.view_type == 'sagittal':
            # 闁活厹鍨绘慨鍛婃媴瀹ュ繒绐楅柟缁樺姇瑜板槳閺夌偞娼欓崹蹇涙偋閸ラ绀夐柡鍕⒔閵囨瓛-Z妤犵偞濞婂浼存晬瀹€鈧崝褔宕ユ惔婵堢憪濞戞挸顑囬悙鏇熸姜?
            # image_data[:, x, :] 鐎电増顨呴崺?(Y, Z)闁挎稑鐭佸ù鍡欑磾鐠?(Z, Y) 濞寸姰鍎辩亸顕€鏌婂鍡樷枖缂佲偓閻戞ɑ鐓欓柛?
            slice_data = self.image_data[:, self.current_slice, :].T
            return np.fliplr(np.flipud(slice_data))  # 濞戞挸锕ｇ粭鍛礄閺勫繑绁?
        return None
    
    def _get_slice_label(self) -> np.ndarray:
        """闁兼儳鍢茶ぐ鍥亹閹惧啿顤呴柛鎺戞川婢ф牠鎯冮崟顒傚灱缂佹稓鍋撻弳鐔煎箲缁辨瑦绋夊涔琫t_slice_data濞ｅ洦绻冪€垫梹绋夐柤鐤彧缁?"""
        if self.label_data is None:
            return None
        
        if self.view_type == 'axial':
            # 閺夌偟绻濈紞鍛存晬濮橆偆鐟㈤柛銉﹀劤閸庢碍绌卞┑鍥х槷濞戞捇鎳涙潏鍓х闂傚洨鎲版担鑺ョギ缂?
            return np.fliplr(self.label_data[:, :, self.current_slice].T)
        elif self.view_type == 'coronal':
            # 闁告劗濮锋慨鍛婃媴瀹ュ繒绐楀☉鎾抽濞存﹢宕撹箛搴ｇ闁归晲妞掔粩鎾嚊鏉堝墽绀夐梻鍥╂啺娴ｉ鐟愬☉鎾愁儑閻愭洘娼?
            slice_label = self.label_data[self.current_slice, :, :].T
            return np.flipud(slice_label)
        elif self.view_type == 'sagittal':
            # 闁活厹鍨绘慨鍛婃媴瀹ュ繒绐楀☉鎾抽濞存﹢宕撹箛搴ｇ闁归晲妞掔粩鎾嚊鏉堝墽绀夐梻鍥╂啺娴ｉ鐟愬☉鎾愁儑閻愭洘娼?
            slice_label = self.label_data[:, self.current_slice, :].T
            return np.fliplr(np.flipud(slice_label))
        return None

    def _get_slice_seed_pixels(self):
        """闁兼儳鍢茶ぐ鍥亹閹惧啿顤呴柛鎺戞川婢ф牗绋夋繝鍕暠缂佸绉撮悺娆撴倷閻熸澘鍓肩紒杈╁Т濞兼寮介崶褍鐏欓悶娑掓櫧(col, row), ...]闁?"""
        if self.image_data is None or not self.seed_points:
            return []

        sx, sy, sz = self.image_data.shape  # axis0, axis1, axis2
        points = []

        if self.view_type == 'axial':
            z = self.current_slice
            w = sx
            for x, y, z0 in self.seed_points:
                if 0 <= x < sx and 0 <= y < sy and 0 <= z0 < sz and z0 == z:
                    # axial闁哄嫬澧介妵?image_data[:, :, z].T => (row=y, col=x)
                    points.append((w - 1 - x, y))
        elif self.view_type == 'coronal':
            x_slice = self.current_slice
            h = sz
            for x, y, z0 in self.seed_points:
                if 0 <= x < sx and 0 <= y < sy and 0 <= z0 < sz and x == x_slice:
                    # coronal闁哄嫬澧介妵?image_data[x, :, :].T 闁告劕纰俵ipud => col=y, row=(h-1-z)
                    points.append((y, h - 1 - z0))
        elif self.view_type == 'sagittal':
            y_slice = self.current_slice
            h = sz
            w = sx
            for x, y, z0 in self.seed_points:
                if 0 <= x < sx and 0 <= y < sy and 0 <= z0 < sz and y == y_slice:
                    # sagittal闁哄嫬澧介妵?image_data[:, y, :].T 闁告劕纰俵ipud => col=x, row=(h-1-z)
                    points.append((w - 1 - x, h - 1 - z0))

        return points

    def _get_slice_spacing_xy(self):
        """閺夆晜鏌ㄥú鏍亹閹惧啿顤?D闁告帒娲ㄦ晶鏍及閸撗佷粵闁秆勫姈閻栵絿鍖栫拋宕囩憮闁?x_spacing, y_spacing)闁?"""
        if self.image_spacing is None:
            return 1.0, 1.0
        if self.view_type == 'axial':
            return float(self.image_spacing[1]), float(self.image_spacing[0])
        if self.view_type == 'coronal':
            return float(self.image_spacing[1]), float(self.image_spacing[2])
        if self.view_type == 'sagittal':
            return float(self.image_spacing[1]), float(self.image_spacing[2])
        return 1.0, 1.0
    
    def _update_display(self):
        """闁哄洤鐡ㄩ弻濠囧及閸撗佷粵"""
        self.ax.clear()
        self.ax.axis('off')
        # 缂佺偓顏ら柤纭呭蔼閸庢寮?
        self.figure.patch.set_facecolor('#000000')
        self.ax.set_facecolor('#000000')
        
        if self.view_type == '3d':
            # 3D閻熸瑥妫楀ù姗€鏁嶅顓熲枖缂佲偓妤﹁法绮婚埄鍐煇闁告粌鑻崹蹇涘礈閹绘帡鎸梻?
            self._update_3d_display()
            return
        
        slice_img = self._get_slice_data()
        if slice_img is None:
            self.canvas.draw()
            return
        
        # 闁兼儳鍢茶ぐ鍥礆閸モ晛顣婚柣銊ュ閻ゅ嫰姊介崨顓熸閻?
        h, w = slice_img.shape
        
        # 閹煎瓨姊婚弫銈囩玻濡も偓缂佹劖銇炵紞?
        img_min = self.window_level - self.window_width / 2
        img_max = self.window_level + self.window_width / 2
        
        # 鐟滅増甯婄粩鎾礌閺嵮冪厒0-255
        img_normalized = np.clip((slice_img - img_min) / (img_max - img_min) * 255, 0, 255).astype(np.uint8)
        
        # 閻犱緤绱曢悾绫pect ratio濞寸姰鍎扮换姘跺箰娴ｅ搫鈷栭柣鐐叉閻︻喗绗?
        # 闁稿繑濞婇弫顓㈡偠閸℃鎺楁晬濮濈敘tplotlib闁汇劌鍨穝pect闁告瑥鍊归弳鐔煎及?(y閺夌偞娼欏畷鐔告媴瀹ュ姣愰幖? / (x閺夌偞娼欏畷鐔告媴瀹ュ姣愰幖?
        # 濠碘€冲€归悘澶愬炊閹冨壖闁哄牆鈥滈悶娑樼槏闁告帗顨愮槐婕奸弶鐐茬摠濠€涔☉鎾崇濞达絽绋勭槐婕婚弶鐐茬摠濠€浜€濞戞挸绀嬪ù?
        # 閻熸洑娴囬柣妞绘櫇閹﹤袙閺傚墽浼愭慨婵撶悼閳ユ﹢鏁嶅畝鍕粯閻熸洑绶ょ槐鐧畇pect = (y閺夌偛顕晶鍧楁偠閸℃绀嬪ù锝呯Ч閺嗚鲸鎯? / (x閺夌偛顕晶鍧楁偠閸℃绀嬪ù锝呯Ч閺嗚鲸鎯?
        # 闁告顒茬槐鐧畇pect = (y閺夌偟顥減acing) / (x閺夌偟顥減acing)
        if self.image_spacing is not None:
            if self.view_type == 'axial':
                # Axial闁挎稒顒?闁告鍓氶柨娑樼様=闁告鍓?
                aspect_ratio = self.image_spacing[1] / self.image_spacing[0]  # Y_spacing / X_spacing
            elif self.view_type == 'coronal':
                # Coronal闁挎稒顒?闁告鍓涢柨娑樼様=闁告鍓?
                aspect_ratio = self.image_spacing[2] / self.image_spacing[1]  # Z_spacing / Y_spacing
            elif self.view_type == 'sagittal':
                # Sagittal闁挎稒顒?闁告鍓氶柨娑樼様=闁告鍓?
                aspect_ratio = self.image_spacing[2] / self.image_spacing[0]  # Z_spacing / X_spacing
            else:
                aspect_ratio = 1.0
        else:
            # 濠碘€冲€归悘澶娾柦閳╁啯绠抯pacing濞ｅ洠鍓濇导鍛存晬鐏炵厧鈻忛柣顫妼閸庢氨妲愰悩宕囨Х濞?
            aspect_ratio = h / w
        
        # 濞达綀娉曢弫顦媥tent闁哄鍎遍悾鐐▕婢跺苯鈷栭柣鐐叉濞兼寮介崶顏勭槺闁?
        # 婵炲鍔嶉崜浼存晬濮橆剚韬琺atplotlib濞戞挾绀塭xtent=[xmin, xmax, ymin, ymax]
        # 鐟滅増鎮恟igin='upper'闁哄啳顔愮槐婕糾in閻庣數鎳撶花鏌ュ及閸撗佷粵濡炪倕鐖奸崕鎾晬鐎靛〉ax閻庣數鎳撶花鏌ュ及閸撗佷粵閹煎瓨娲熼崕?
        # 闁瑰灚绂掗妷銉╁几濠婂啯绂堥柛宥呯箲閺嗙喖骞戝▓鎱弶鐐茬摠濡插憡绂掑閬嶅礆閻у棝鏁嶉崼婵囧€诲☉鎾愁儏闁告梻濯寸槐姘舵晬瀹€鍕簼濞戞柨婢抦in閹煎瓨妫侀悗鐢垫嚀缁ㄧ澋闁挎稑鐦攎ax閹煎瓨妫侀悗鐢垫嚀缁ㄧ潪
        # 濞达絽妫旂拹鐔哥閸℃洜绠介柟闀愮劍閺岀喖宕ラ幋鐐靛灱閻犲洤妫欑痪顓犵闁瑰瓨鍨冲鎴︽閻熸洑妞掑锕傚箲min闁告粌鐦攎ax
        if self.image_spacing is not None:
            if self.view_type == 'axial':
                # 閺夌偟绻濈紞鍛存晬濮?Y妤犵偞濞婂浼存晬鐎规敍tent閻庤鐭粻鐒侀柛婊冾劥闁汇劌瀚晶鍧楁偠閸℃艾鐦遍柛?
                # 濞?P闁挎稑濂旂粭?A闁挎稑鏈晶宥嗙min閹煎瓨妫侀悗鐢垫嚀缁ㄧ澋闁挎稑鐗撻妴濠囨焾缁辨岸鏁嶇€靛〉ax閹煎瓨妫侀悗鐢垫嚀缁ㄧ潪闁挎稑鐗嗙花鎶芥焾缁?
                # 濞达絽鏀穜igin='upper'闁哄啳顔愮槐婕糾in閻庣數鎳撶花鍙夈亜閸洖鍔ラ柨娑樻湰婢у秵绂抶tent闁汇劌澧㈤柛濠傘偢濞撳墎鎲版担绋垮唨閺?
                extent_x = (0, w * self.image_spacing[1])  # Y闁哄倻鎳撻幃婊堟偋閳哄啯鍊為柤鐓庡暙濞?
                extent_y = (h * self.image_spacing[0], 0)    # X闁哄倻鎳撻幃婊堟偋閳哄啯鍊為柤鐓庡暙濞插潡鏁嶉崼婵嗗唨閺夌偟绐楀銈呯埣閸?0闁挎稑鑻花鎶芥焾?h*spacing闁?
                extent = [extent_x[0], extent_x[1], extent_y[0], extent_y[1]]
            elif self.view_type == 'coronal':
                # 闁告劗濮锋慨鍛婃媴瀹ュ繒绐楅柡鍌滄嚀閹粓寮介崶顏嗘濞?I, 濞?S, 鐎?A, 闁?P闁挎稑婢戦弶鐐存綑閹煎瓨鎽抽柡鍌滄嚀閹粓鏁嶇€靛娼弶鎴炴償閹垮綊寮悷鐗堝€婚柨?
                extent_x = (w * self.image_spacing[1], 0)  # Y闁哄倻鎳撻幃婊堟晬閸繂顤呴柛姘洴-P闁?
                extent_y = (h * self.image_spacing[2], 0)  # Z闁哄倻鎳撻幃婊堟晬閸粎鐟愬☉鎾斥敇-I闁挎稑鑻鑺ユ姜缁?
                extent = [extent_x[0], extent_x[1], extent_y[0], extent_y[1]]
            elif self.view_type == 'sagittal':
                # 闁活厹鍨绘慨鍛婃媴瀹ュ繒绐梈-Y妤犵偞濞婂浼存晬鐎规敍tent閻庤鐭粻鐒冮柛婊冾劥闁汇劌瀚晶鍧楁偠閸℃艾鐦遍柛?
                # 濞?S闁挎稑濂旂粭?I闁挎稑鏈晶宥嗙min閹煎瓨妫侀悗鐢垫嚀缁ㄧ澑闁挎稑鐗撻妴濠囨焾缁辨岸鏁嶇€靛〉ax閹煎瓨妫侀悗鐢垫嚀缁ㄧ澁闁挎稑鐗嗙花鎶芥焾缁?
                extent_x = (0, w * self.image_spacing[1])  # Y闁哄倻鎳撻幃婊堟偋閳哄啯鍊為柤鐓庡暙濞?
                extent_y = (h * self.image_spacing[2], 0)  # Z闁哄倻鎳撻幃婊堟偋閳哄啯鍊為柤鐓庡暙濞插潡鏁嶉崼婵嗗唨閺夌偟绐楀銈呯埣閸?0闁挎稑鑻花鎶芥焾?h*spacing闁?
                extent = [extent_x[0], extent_x[1], extent_y[0], extent_y[1]]
            else:
                extent = [0, w, h, 0]  # 闁告瑥绉峰ù鍞濋弶?
        else:
            extent = [0, w, h, 0]  # 闁告瑥绉峰ù鍞濋弶?
        
        # 闁哄嫬澧介妵姘跺炊閹冨壖 - 濞达綀娉曢弫顦媥tent閻庤鐭粻鐔兼偋閳哄啯鍊為柛褎鍔栭悥锝夋晬鐎瑰窏pect='equal'濞ｅ洦绻冪€垫棃鎮ч埡鍐╁€炴慨锝嗘煣缁?
        self.ax.imshow(img_normalized, cmap='gray', interpolation='antialiased', 
                      origin='upper', aspect='equal', extent=extent,
                      rasterized=False, filternorm=True)
        
        # 閻犱礁澧介悿鍡涘锤閹邦厾鍨奸弶鐐殿棎鐎垫牠宕舵潏鍓х闁衡偓鐎垫梻绱撻埡鍌涙澒闁挎稑濂旀繛鍥偨閵娧冣挅闁荤偛妫楀妤呭冀閸ラ绀?
        if self.xlim_range is None or self.ylim_range is None:
            # 闁告帗绻傞柛鏍ㄧ壄缁变即寮伴崜褋浠涢悗鐟版湰閺嗭綁宕堕幆褍鍓奸柨娑樼墢婢у潡鎮堕崱妤佺稄闁哄秴娴勭槐?
            self.xlim_range = (extent[0], extent[1])
            # 婵炲鍔嶉崜浼存晬濮濈劔tent闁汇劌澧㈤柛濠勫帶閸戯紕绱掕箛鎾冲唨閺夌偟绀夐柟鍨lim_range濞戞梻鍠栧〒鍓佹啺娴ｇ鍐€閺?
            self.ylim_range = (extent[3], extent[2])  # 闁告瑥绉峰ù鍞濋弶鐐殿棎鐎垫牠宕?
        
        self.ax.set_xlim(self.xlim_range[0], self.xlim_range[1])
        # y閺夌偛鐡ㄩ弻鐔煎触閹搭垳绐?
        # - axial闁挎稒纰嶇€垫粓鎮介妸锕€鐓曢悷鏇氱劍閻即宕戝顐ょ憪濞戞挸顑囬悙鏇熸姜缁辨瑦銇勯崜浣圭▔妤﹁法绐涢悘蹇撶闯闁稿﹦銆嬬槐?
        # - 闁稿繑婀圭划顒傛喆閸℃绂堥柨娑欑煯缁绘岸骞愭担绋挎枾闁哄牆顦伴弻鐔煎触?
        if self.view_type == 'axial':
            self.ax.set_ylim(self.ylim_range[0], self.ylim_range[1])
        else:
            self.ax.set_ylim(self.ylim_range[1], self.ylim_range[0])
        
        # 闁告瑧濮存慨鐐哄冀閸?
        if self.show_label and self.label_data is not None:
            slice_label = self._get_slice_label()
            if slice_label is not None and np.any(slice_label > 0):
                label_colored = np.zeros((*slice_label.shape, 4), dtype=np.float32)
                
                for label_value in np.unique(slice_label):
                    if label_value > 0:
                        if label_value in self.label_colors:
                            color = self.label_colors[label_value]
                        else:
                            # 濮掓稒顭堢紒淇卞灱婢瑰﹪鏁嶇仦鐓庘枏闁活潿鍔岀紞瀣礈瀹ュ绻戝Σ鎴炴償?
                            color = (1.0, 0.0, 0.0, self.label_opacity)  # 濞达綀娉曢弫銈堛亹閹惧啿顤呴梺顐㈢箲濡叉垶鎯?
                        
                        mask = (slice_label == label_value)
                        label_colored[mask, 0] = color[0]
                        label_colored[mask, 1] = color[1]
                        label_colored[mask, 2] = color[2]
                        # 闁诡剛绮Σ鍛婃媴鐠恒劍鏆忕憸鐗堟尭婢х娀鎯冮崟顒傚灱缂佹盯绠栬箛鏃€顫栭幖杈剧畳缂?
                        label_colored[mask, 3] = self.label_opacity
                
                # 闁哄秴娲ㄩ柛娆戝Т婵偞绋婇悢璺衡枏闁活潿鍔庡ù澶愬触瀹€鈧▓鎱箈tent闁告粌鐣畇pect
                if self.image_spacing is not None:
                    self.ax.imshow(label_colored, interpolation='antialiased', origin='upper',
                                  aspect='equal', extent=extent, rasterized=False)
                else:
                    self.ax.imshow(label_colored, interpolation='antialiased', origin='upper',
                                  aspect='equal', rasterized=False)

        # 缂備焦锚閸╂绮斿鍛憤闁绘劕缍婇悵顔界缁辨瑩妫冮幒鏃囬浌闁肩灏欓獮?+ 濮掓稑瀚竟濠冪▔缁洪箖鏁?
        if self.show_seed_points:
            seed_px = self._get_slice_seed_pixels()
            if seed_px:
                spacing_x, spacing_y = self._get_slice_spacing_xy()
                xs = np.array([p[0] * spacing_x for p in seed_px], dtype=np.float32)
                ys = np.array([p[1] * spacing_y for p in seed_px], dtype=np.float32)
                self.ax.scatter(
                    xs, ys,
                    s=64,
                    facecolors='none',
                    edgecolors='#00ffd9',
                    linewidths=1.8,
                    zorder=12
                )
                self.ax.scatter(
                    xs, ys,
                    s=22,
                    c='#ffe600',
                    edgecolors='black',
                    linewidths=0.6,
                    zorder=13
                )
        
        # 缂備焦锚閸╂宕℃担鍝ユ憻缂佹儳灏呯槐浼存嫅濠靛﹤顥忛柧蹇旀皑閸?+ 缂備浇鍎绘竟濠囧棘閻熺増鍋ュù婧垮€曞鍫曟倷?
        # 婵炲鍔嶉崜浼存晬濮濇即osshair_x闁告粌鐣眗osshair_y闁哄嫬鍓肩紒杈╁Т濞兼寮介崶椋庣闂傚洨鎲版担鑺ョギ闁圭绀嬮柣妞绘櫇閹﹪宕搁幇顓犲灱
        if self.show_crosshair and slice_img is not None:
            h, w = slice_img.shape
            
            # 閻忓繐妫楅崕姘辨閻樺弶缍忛柡宥呮穿濞村棝骞戠拹鐔兼偋閳哄啯鍊為柛褎鍔栭悥?
            if self.image_spacing is not None:
                if self.view_type == 'axial':
                    # 閺夌偟绻濈紞鍛存晬濮濊櫣鈧數鎳撶花鐬冮柡鍌滄嚀閹粓鏁嶇€靛鈧數鎳撶花鐬傞柡鍌滄嚀閹?
                    crosshair_x_phys = self.crosshair_x * self.image_spacing[1]  # Y闁哄倻鎳撻幃?
                    crosshair_y_phys = self.crosshair_y * self.image_spacing[0]  # X闁哄倻鎳撻幃?
                    crosshair_size = max(h * self.image_spacing[0], w * self.image_spacing[1]) / 80
                elif self.view_type == 'coronal':
                    # 闁告劗濮锋慨鍛婃媴瀹ュ繒绐楅柡鍌滄嚀閹粓寮介崶顏嗘闁哄嫬澧介妵姘啅?A(闁?闁挎稑鑻ぐ?P(闁?闁挎稑鏈晶宥嗙閻庣數鎳撶花鐬冮柡鍌滄嚀閹粓鏁嶉崼婵嗩枀闁告艾鐩?P闁?
                    # 闁搞儱澧芥晶鏍ь啅闊厾鐟愬☉鎾愁儑閻愭洘娼槐婕奸柛褎鍔栭悥锝夋儎鐎涙ê澶嶅ù锝堟硶閺併倝鏁嶉崼銏㈠€抽弶鐐村€甸柨娑樼焸閵嗗﹪鏌堥妸銉︽償閺€?0闁?
                    crosshair_x_phys = self.crosshair_x * self.image_spacing[1]  # Y闁哄倻鎳撻幃婊堟晬閸繂顤呴柛姘洴-P闁?
                    crosshair_y_phys = self.crosshair_y * self.image_spacing[2]  # Z闁哄倻鎳撻幃婊堟晬閸粎鐟愬☉鎾斥敇-I闁挎稑鐬奸悙鏇熸姜閹鏁?
                    crosshair_size = max(h * self.image_spacing[2], w * self.image_spacing[1]) / 80
                elif self.view_type == 'sagittal':
                    # 闁活厹鍨绘慨鍛婃媴瀹ュ繒绐梮閻庣數鎳撶花鐬傞柡鍌滄嚀閹粓鏁嶉崼婵呯闁告瑧鐖?L闁挎稑顧€缁辨技閻庣數鎳撶花鐬嗛柡鍌滄嚀閹粓鏁嶉崼婊呯憪濞戞挸鈹?I闁?
                    # 闁搞儱澧芥晶鏍ь啅闊厾鐟愬☉鎾愁儑閻愭洘娼槐婕奸柛褎鍔栭悥锝夋儎鐎涙ê澶嶅ù锝堟硶閺併倝鏁嶉崼銏㈠€抽弶鐐村€甸柨娑樼焸閵嗗﹪鏌堥妸銉︽償閺€?0闁?
                    crosshair_x_phys = self.crosshair_x * self.image_spacing[1]  # X闁哄倻鎳撻幃婊堟晬閸繀绠柛娆戠垼-L闁?
                    crosshair_y_phys = self.crosshair_y * self.image_spacing[2]  # Z闁哄倻鎳撻幃婊堟晬閸粎鐟愬☉鎾斥敇-I闁挎稑鐬奸悙鏇熸姜閹鏁?
                    crosshair_size = max(h * self.image_spacing[2], w * self.image_spacing[1]) / 80
                else:
                    crosshair_x_phys = self.crosshair_x
                    crosshair_y_phys = self.crosshair_y
                    crosshair_size = max(h, w) / 80
            else:
                crosshair_x_phys = self.crosshair_x
                crosshair_y_phys = self.crosshair_y
                crosshair_size = max(h, w) / 80
            
            # 婵ɑ娼欓柦鈺冪棯閸栵紕绐楅柦鍐╃箚婢瑰﹪鎼瑰鍗炴疇
            self.ax.axhline(y=crosshair_y_phys, color='#0080FF', linewidth=1.5, 
                          alpha=0.9, linestyle='--', dashes=(5, 5))
            # 闁搞劌鍊诲ú璺ㄧ棯閸栵紕绐楅柦鍐╃箚婢瑰﹪鎼瑰鍗炴疇
            self.ax.axvline(x=crosshair_x_phys, color='#0080FF', linewidth=1.5, 
                          alpha=0.9, linestyle='--', dashes=(5, 5))
            
            # 濞存嚎鍊曞鍫曟倷閻у摜绐楃紓浣藉劵婢瑰﹪寮悷鐗堝仴
            self.ax.add_patch(Rectangle(
                (crosshair_x_phys - crosshair_size/2, crosshair_y_phys - crosshair_size/2),
                crosshair_size, crosshair_size,
                facecolor='#00FF00', edgecolor='#00FF00', linewidth=1,
                alpha=0.9, zorder=10
            ))
        
        # 缂備焦锚閸╂寮悷鐗堝€婚柡宥呮穿閻︽垿鏁嶉崼婊冣枏闁活潿鍔忛柊閬嶅锤閹邦厾鍨奸柨娑樼焸娴尖晠宕楀鍛秬y閺夌偞娼欏鑺ユ姜?extent鐟滄澘宕幖鐑芥晬?
        if self.direction_labels:
            h, w = slice_img.shape
            if self.image_spacing is not None:
                if self.view_type == 'axial':
                    phys_w = w * self.image_spacing[1]
                    phys_h = h * self.image_spacing[0]
                elif self.view_type == 'coronal':
                    phys_w = w * self.image_spacing[0]
                    phys_h = h * self.image_spacing[2]
                elif self.view_type == 'sagittal':
                    phys_w = w * self.image_spacing[1]
                    phys_h = h * self.image_spacing[2]
                else:
                    phys_w = w
                    phys_h = h
            else:
                phys_w = w
                phys_h = h

            font_size = float(np.clip(max(phys_h, phys_w) / 30.0, 8.0, 28.0))

            # 濞?
            if 'top' in self.direction_labels:
                self.ax.text(
                    0.5, 0.98, self.direction_labels['top'],
                    transform=self.ax.transAxes,
                    color='yellow', fontsize=font_size,
                    ha='center', va='top', weight='bold', zorder=20
                )
            # 濞?
            if 'bottom' in self.direction_labels:
                self.ax.text(
                    0.5, 0.02, self.direction_labels['bottom'],
                    transform=self.ax.transAxes,
                    color='yellow', fontsize=font_size,
                    ha='center', va='bottom', weight='bold', zorder=20
                )
            # 鐎?
            if 'left' in self.direction_labels:
                self.ax.text(
                    0.02, 0.5, self.direction_labels['left'],
                    transform=self.ax.transAxes,
                    color='yellow', fontsize=font_size,
                    ha='left', va='center', weight='bold', zorder=20
                )
            # 闁?
            if 'right' in self.direction_labels:
                self.ax.text(
                    0.98, 0.5, self.direction_labels['right'],
                    transform=self.ax.transAxes,
                    color='yellow', fontsize=font_size,
                    ha='right', va='center', weight='bold', zorder=20
                )
        
        # 缂備焦锚閸╂绱撻埡鍌涙澒闁哄妲勭槐姗瞔ale Bar闁挎稑顧€缁辨壆绱掗懗顖氼棌闁挎稑鑻ぐ鍛婄▔鐎ｎ參鏁嶇仦鐐枖缂佲偓閾忕懓鈷栭柣鐐叉閺勫倻鈧?
        if self.image_spacing is not None:
            h, w = slice_img.shape
            # 閻犱緤绱曢悾濠氭偋閳哄啯鍊為悘蹇撴惈闁肩厧鍟ú?
            if self.view_type == 'axial':
                phys_w = w * self.image_spacing[1]  # X闁哄倻鎳撻幃?
                phys_h = h * self.image_spacing[0]  # Y闁哄倻鎳撻幃?
            elif self.view_type == 'coronal':
                phys_w = w * self.image_spacing[0]  # Y闁哄倻鎳撻幃婊堟晬閸?P闁?
                phys_h = h * self.image_spacing[2]  # Z闁哄倻鎳撻幃婊堟晬閸?S闁?
            elif self.view_type == 'sagittal':
                phys_w = w * self.image_spacing[1]  # X闁哄倻鎳撻幃婊堟晬閸?L闁?
                phys_h = h * self.image_spacing[2]  # Z闁哄倻鎳撻幃婊堟晬閸?S闁?
            else:
                phys_w = w
                phys_h = h
            
            # 闂侇偄顦扮€氥劑宕ラ崼銉ュ€诲▓鎴犵磽閳哄倹鏉归柡澶涚節閺嗚鲸鎯旈敂鑲╃缂佹拝缂氱拹鐔煎炊閹冨壖閻庣妫勭€规娊鎯?0-20%闁?
            scale_length_phys = max(phys_w, phys_h) * 0.15
            # 闁告瑦鐗楅弳锝夊礆閺夋寧鍊ら梺顐㈠€诲▓鎴﹀磹绾绀?, 2, 5, 10, 20, 50, 100缂佹稑顧€缁?
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
            
            # 缂傚倵鏅滈弬渚€寮堕垾鑼Т缂傚啰绐楅柛娆忓帠缁楀懐鎲?
            scale_x_start = phys_w * 0.85
            scale_x_end = scale_x_start + scale_length_phys
            scale_y = phys_h * 0.08  # 閹煎瓨娲熼崕鎾触閹存粎鐟?%
            
            # 缂備焦锚閸╂绱撻埡鍌涙澒闁哄妲勭槐娆戠磼閼愁垰顥忛柨?
            self.ax.plot([scale_x_start, scale_x_end], [scale_y, scale_y], 
                        color='green', linewidth=3, zorder=10)
            # 缂備焦锚閸╂宕归崒婊勭函缂佹柨娈?
            line_height = phys_h * 0.02
            self.ax.plot([scale_x_start, scale_x_start], 
                        [scale_y - line_height/2, scale_y + line_height/2],
                        color='green', linewidth=3, zorder=10)
            self.ax.plot([scale_x_end, scale_x_end], 
                        [scale_y - line_height/2, scale_y + line_height/2],
                        color='green', linewidth=3, zorder=10)
            
            # 婵烇綀顕ф慨鐐哄棘閸パ呮憻闁哄秴娲ㄩ柨娑樼墛濡绮堥搹鐟扳挅闁荤偛妫楅弰鍌溾偓鐢告交缁?
            if scale_length_phys >= 10:
                scale_text = f"{int(scale_length_phys)} mm"
            else:
                scale_text = f"{scale_length_phys:.1f} mm"
            self.ax.text((scale_x_start + scale_x_end) / 2, scale_y + line_height * 2,
                        scale_text, color='green', fontsize=10,
                        ha='center', va='bottom', weight='bold', zorder=10)
        
        # 闁哄洤鐡ㄩ弻濠囧礆閸モ晛顣荤紒渚垮灩缁扁晠寮伴崜褋浠?
        max_slice = self._get_max_slice()
        self.slice_label.setText(f"{self.current_slice + 1} of {max_slice}")
        
        self.canvas.draw()
    
    def _on_zoom_fit(self):
        """
        缂傚倵鏅滈弬渚€鎳涢搹顐㈢彲闁?
        闂佹彃绉堕悿鍡欐喆閸℃绂堥柨娑樼焷闁搞儲鍎抽崕姘扁偓鐟版湰閺嗭綁寮伴崜褋浠涢柛锔哄妿閻涖儵宕ｉ敐鍕幀闁挎稑濂旂换姘跺箰娴ｅ湱娑靛▓鎴︽偋閳哄啯鍊炴慨锝嗘煣缁?
        """
        if self.image_data is None:
            return
        
        if self.view_type == '3d':
            # 3D閻熸瑥妫楀ù姗€鎯冮崟顒€鐝柛?
            if self.image_data is not None:
                # 閻犱緤绱曢悾濠氬炊閹冨壖闁肩厧鍟ú?
                if self.image_spacing is not None:
                    x_range = (0, self.image_data.shape[0] * self.image_spacing[0])
                    y_range = (0, self.image_data.shape[1] * self.image_spacing[1])
                    z_range = (0, self.image_data.shape[2] * self.image_spacing[2])
                else:
                    x_range = (0, self.image_data.shape[0])
                    y_range = (0, self.image_data.shape[1])
                    z_range = (0, self.image_data.shape[2])
                
                # 閻犱礁澧介悿鍡涘锤閹邦厾鍨奸弶鐐殿棎鐎垫牠宕?
                self.ax.set_xlim(x_range)
                self.ax.set_ylim(y_range)
                self.ax.set_zlim(z_range)
                
                # 閻犲鍟弳锝囨喆閸℃绂?
                self.ax.view_init(elev=30, azim=45)
                self.canvas.draw()
            return
        
        slice_img = self._get_slice_data()
        if slice_img is not None:
            h, w = slice_img.shape
            
            # 闂佹彃绉堕悿鍡欑磽閳哄倹鏉归柨娑樼墔婵炲洭鎮介妸褍鈷栭柣鐐叉濞兼寮介崶椋庣
            self.zoom_factor = 1.0
            
            # 閻犱緤绱曢悾濠氭偋閳哄啯鍊為柛褎鍔栭悥锝夋嚑閸愩劍绾柨娑樼墔缁楀畩update_display濞戞捇鎳涙潏鍓х
            xlim, ylim, _, _ = self._get_physical_range(h, w)
            self.xlim_range = xlim
            self.ylim_range = ylim
            
            # 鐎殿喖鎼崺妤呭礆闁垮鐓€闁哄嫬澧介妵?
            self._update_display()
    
    def _on_slice_changed(self, value: int):
        """闁告帒娲ㄦ晶鏍绩閻熸澘缍侀柛銉у仩閻?"""
        self.set_slice(int(value), emit_signal=True)
    
    def _on_scroll(self, event):
        """Handle mouse wheel events for slice navigation."""
        if event.inaxes == self.ax:
            if self.tool_mode == 'zoom' and event.xdata is not None and event.ydata is not None:
                scale = 0.86 if event.button == 'up' else 1.0 / 0.86
                self._zoom_at(event.xdata, event.ydata, scale)
                return
            if event.button == 'up':
                self.set_slice(self.current_slice - 1)
            elif event.button == 'down':
                self.set_slice(self.current_slice + 1)
    
    def _update_3d_display(self):
        """闁哄洤鐡ㄩ弻?D闁哄嫬澧介妵?"""
        # 婵炴挸鎳樺▍搴ゃ亹閹惧啿顤呴弶?
        self.ax.clear()
        
        # 閻犱礁澧介悿?D閻熸瑥妫楀ù?
        if not hasattr(self.ax, 'get_proj'):
            # 濠碘€冲€归悘澶嬬▔瀹ュ棙笑3D閺夌偠鎻槐婵嬪礆濞戞绱﹀☉鎾寸▔閺屽﹪鎯?
            self.ax = self.figure.add_subplot(111, projection='3d')
        
        self.figure.patch.set_facecolor('#000000')
        self.ax.set_facecolor('#000000')
        
        # 婵☆偊寮婚妷锔叫﹂柛姘鹃檮濠€渚€宕堕幆褍鍓奸柡浣哄瀹?
        if self.image_data is None:
            self.canvas.draw()
            return
        
        # 婵☆偊寮婚妷锔叫﹂柛姘鹃檮濠€渚€寮介崶鈺呭极閻楀牆绁﹂柨娑樼墣缂佺姭鈧啿鐎婚柛鎾瑰皺缁劑寮稿鎰
        if self.label_data is None:
            self.canvas.draw()
            return
        
        # 闁圭粯鍔曡ぐ鍥╂偘缂佺姭鍓濋弳鐔煎箲?
        portal_vein = (self.label_data == 1)
        hepatic_vein = (self.label_data == 2)
        
        # 閻犱緤绱曢悾濠氭偋閳哄啯鍊為柛褎鍔栭悥?
        if self.image_spacing is not None:
            spacing = self.image_spacing
        else:
            spacing = (1.0, 1.0, 1.0)
        
        # 缂備焦锚閸╂鎮扮紒鐘墲閻?
        if np.any(portal_vein):
            # 闂傚倶鍔戝銈夋嚇婢舵稓绐楅柦鍐╃箚婢?
            self._plot_vessel_tree(portal_vein, spacing, color='#0000FF', alpha=0.6, label='PV')
        
        if np.any(hepatic_vein):
            # 闁煎弶绻堝銈夋嚇婢舵稓绐楃紒淇卞灱婢?
            self._plot_vessel_tree(hepatic_vein, spacing, color='#FF0000', alpha=0.6, label='HV')
        
        # 婵烇綀顕ф慨鐐哄炊閸欍儳浼?
        self.ax.legend()
        
        # 閻犱礁澧介悿鍡涘锤閹邦厾鍨奸弶鐐茬摠閻栵絿绮?
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        
        # 閻犱礁澧介悿鍡涘冀閸?
        self.ax.set_title('3D閻炴稓绮婚埄鍐煇')
        
        # 閻犲鍟弳锝囨喆閸℃绂?
        self.ax.view_init(elev=30, azim=45)
        
        # 缂備焦锚閸╂鎼瑰顓炵彲闁告帒娲ゆ竟濠囩嵁閹惰姤妗?
        self._plot_cutting_plane()
        
        # 濞村吋锚鐎垫彃銆掗崣澶屽帬闁诡儸鍡楀幋
        self.ax.set_rasterized(True)  # 濞达綀娉曢弫銈夊礂婢跺鍨洪柛鏍ㄧ墬鐟曞棝寮?
        self.figure.set_dpi(80)  # 闂傚嫬绉崇紞鍜睵I濞寸姰鍎茶ぐ浣诡殗濡皫鍡楀幋
        
        self.canvas.draw()
    
    def _plot_vessel_tree(self, vessel_mask, spacing, color, alpha, label):
        """缂備焦锚閸╂鎮扮紒鐘墲閻?"""
        # 闁圭粯鍔曡ぐ鍥╂偘缂佺姭鈧櫕缍忛柡?
        coords = np.where(vessel_mask)
        x = coords[0] * spacing[0]
        y = coords[1] * spacing[1]
        z = coords[2] * spacing[2]
        
        # 閻庨潧婀遍崑锝嗙閹达絿绠婚悶娑樼焸濡剧兘鏌岄崶銊у闁挎稑鑻崳铏逛焊閹寸姷甯涢柛鎺戝濞堟垿鎮欑憴鍕闂?
        total_points = len(x)
        if total_points > 20000:
            # 闂傚嫬绉归崳浼村冀瀹勬澘鐓?0000濞戞挸浠柨娑樺缁绘岸鎮惧▎鎰函濠㈣埖姘ㄧ划蹇涙嚍?
            step = total_points // 20000
            x = x[::step]
            y = y[::step]
            z = z[::step]
        elif total_points > 10000:
            # 闂傚嫬绉归崳浼村冀瀹勬澘鐓?0000濞戞挸浠?
            step = total_points // 10000
            x = x[::step]
            y = y[::step]
            z = z[::step]
        
        # 缂備焦锚閸╂鎮欓柅娑氶殬
        if len(x) > 0:
            # 濞达綀娉曢弫銈夊即閹绢喚褰柡浣哥墢濞堟垿寮悷鎵缂備焦锚閸╂鏁嶇仦浠嬪礉閻橀潧浠鍫嗗啰姣堝ù鐘劜瑜颁焦顨囧Ο鍝勮閻熸瑤鐒?
            self.ax.scatter(x, y, z, c=color, alpha=alpha, s=2, label=label, marker='.', linewidths=0)
    
    def _plot_cutting_plane(self):
        """Render configured cutting planes in the 3D view."""
        if self.image_data is not None:
            # 缂備焦锚閸╂骞嶉柡鍫濐槸閸ㄥ繘宕滈幓鎺楁尙闂?
            for plane in self.cutting_planes:
                # 闁圭粯鍔曡ぐ鍥嵁閹惰姤妗ㄩ柛娆忓€归弳?
                center = plane['center']
                normal = plane['normal']
                size = plane['size']
                color = plane['color']
                alpha = plane['alpha']
                
                # 闁汇垻鍠愰崹姘剁嵁閹惰姤妗ㄩ柣銊ュ濞叉挻绋夐妴濠囨倷?
                # 閻犱緤绱曢悾濠氱嵁閹惰姤妗ㄥ☉鎾筹功濞堟垶绋夐妶鍕殝婵繐绲煎锕傚触閹达箑娅?
                if abs(normal[0]) < 0.1:
                    # 婵炲娲滈崵搴ㄥ箳閵夈劎绠畒-z妤犵偞濞婂浼存晬鐏炵厧鈻忛柣鈶╂澒閺夌偟绻濈紞鏃€绋夐搹瑙勭▔濞戞挻鐓欓柛姘灥閹粓鏌?
                    v1 = np.array([1, 0, 0])
                else:
                    # 闁告熬绠戦崹顖涙媴鐠恒劍鏆弝閺夌偟绻濈紞鏃€绋夐搹瑙勭▔濞戞挻鐓欓柛姘灥閹粓鏌?
                    v1 = np.array([0, 1, 0])
                
                # 閻犱緤绱曢悾鑽ょ箔缁ㄢ晜绋夐弻鐔煎触閹存繃鍊婚梺鎻掗獜缁辨瑩宕归崒婊勭函濞存粌瀛╃涵鍓佺棯閸喗瀚茬紒妤冾伇濞戞挻鐓欓柛姘灥閹粓鏌岃箛銉х
                v2 = np.cross(normal, v1)
                v1 = np.cross(v2, normal)
                
                # 鐟滅増甯婄粩鎾礌閺嶃劍鐓欓柛姘灥閹粓鏌?
                v1 = v1 / np.linalg.norm(v1)
                v2 = v2 / np.linalg.norm(v2)
                
                # 闁汇垻鍠愰崹姘跺炊濞戞鍤嬪銈呭閸?
                half_size = size / 2
                p1 = center - v1 * half_size - v2 * half_size
                p2 = center + v1 * half_size - v2 * half_size
                p3 = center + v1 * half_size + v2 * half_size
                p4 = center - v1 * half_size + v2 * half_size
                
                # 闁告帗绋戠紓鎾村緞濮樺磭鐝剁憸?
                vertices = [p1, p2, p3, p4]
                poly = Poly3DCollection([vertices])
                poly.set_color(color)
                poly.set_alpha(alpha)
                poly.set_edgecolor('black')
                poly.set_linewidth(1)
                
                # 婵烇綀顕ф慨鐐哄礆閹峰矂鍙?
                self.ax.add_collection3d(poly)
        
        # 濠碘€冲€归悘澶娾柦閳╁啯绠掗柛鎺戞搐婢瑰﹪鐛幎鑺ユ〃闁挎稑鏈崸濠囧礉閻樿京顏卞☉鎾跺笡閻犱降鍊曢柦鈺呮?
        if not self.cutting_planes and self.image_data is not None:
            # 閻犱緤绱曢悾濠氬炊閹冨壖濞戞挾濡?
            center_x = self.image_data.shape[0] * (self.image_spacing[0] if self.image_spacing else 1) / 2
            center_y = self.image_data.shape[1] * (self.image_spacing[1] if self.image_spacing else 1) / 2
            center_z = self.image_data.shape[2] * (self.image_spacing[2] if self.image_spacing else 1) / 2
            
            # 闁告帗绋戠紓鎾搭渶濡潡鐛幎鑺ユ〃
            default_plane = {
                'center': np.array([center_x, center_y, center_z]),
                'normal': np.array([0, 0, 1]),  # 闁搞劌鍊诲ú鎸庣瀹革絾娼?
                'size': 100,
                'color': 'cyan',
                'alpha': 0.3
            }
            self.cutting_planes.append(default_plane)
            
            # 闂佹彃绉甸弻濠勭磼濡搫鐓?
            self._update_3d_display()
    
    def add_cutting_plane(self, center=None, normal=None, size=100, color='cyan', alpha=0.3):
        """
        婵烇綀顕ф慨鐐寸▔濞戞挻鐓€闁汇劌瀚崹蹇涘礈閹绘帡鎸梻?
        
        Args:
            center: 妤犵偞濞婂鐗堢▔缁洪箖宕搁幇顓犲灱闁挎稑鑻柡瀣矆鐠愮兇one闁告帗鐟ゆ繛鍥偨閵娿儲绂堥柛宥呯箣閼垫垼绠?
            normal: 妤犵偞濞婂鏉库枖閺囩姴娈犻柛姘灴閸ｆ椽鏁嶇仦浠嬪几濠娾偓鐠愮兇one闁告帗鐟ゆ繛鍥偨閵娾晝甯涢悹浣靛€栫涵鍓佺棯?
            size: 妤犵偞濞婂鐗堝緞瑜嶉惃?
            color: 妤犵偞濞婂鐗堬紣濠婂棗顥?
            alpha: 妤犵偞濞婂浼存焻韫囨梹顫栭幖?
        """
        if self.image_data is not None:
            # 閻犱緤绱曢悾濠氬炊閹冨壖濞戞挾濡?
            if center is None:
                center_x = self.image_data.shape[0] * (self.image_spacing[0] if self.image_spacing else 1) / 2
                center_y = self.image_data.shape[1] * (self.image_spacing[1] if self.image_spacing else 1) / 2
                center_z = self.image_data.shape[2] * (self.image_spacing[2] if self.image_spacing else 1) / 2
                center = np.array([center_x, center_y, center_z])
            
            # 濞达綀娉曢弫銈嗩渶濡灝鈻旈弴鐘叉疇
            if normal is None:
                normal = np.array([0, 0, 1])  # 闁搞劌鍊诲ú鎸庣瀹革絾娼?
            
            # 鐟滅増甯婄粩鎾礌閺嶃劎銆婄紒?
            normal = normal / np.linalg.norm(normal)
            
            # 闁告帗绋戠紓鎾诲棘閺夊潡鎸梻?
            new_plane = {
                'center': center,
                'normal': normal,
                'size': size,
                'color': color,
                'alpha': alpha
            }
            
            # 婵烇綀顕ф慨鐐哄礆閺夊灝鐎奸柛鎾村絻闁解晠妫冮姀鐘茬仚閻?
            self.cutting_planes.append(new_plane)
            
            # 闁哄洤鐡ㄩ弻濠囧及閸撗佷粵
            self._update_3d_display()
            
            return True
        return False
    
    def _on_mouse_press(self, event):
        """濮捬呭У閻栵綁骞愭径澶岀憮濞存粌顑勫▎?"""
        if event.inaxes == self.ax:
            self.mouse_button = event.button  # 閻犱焦婢樼紞宥夊箰婢跺鐟撻柣銊ュ鐎垫粓鏌?
            self.last_mouse_canvas_pos = (event.x, event.y)
            if event.button == 1:  # 鐎归潻绠撻弫?
                if event.xdata is not None and event.ydata is not None:
                    if self.tool_mode == 'brush':
                        # 闁汇垼宕甸悷顏勵啅閵夈儱寰旈柨娑欐皑缁垶宕氶懜鐢靛灱缂?
                        self._paint_label(event.xdata, event.ydata)
                    elif self.tool_mode == 'eraser':
                        # 婵绱曞В濠囧箼閿曗偓娴兼劙宕楅崙銈囩獥婵炴挸鎳樺▍搴ㄥ冀閸?
                        self._erase_label(event.xdata, event.ydata)
                    elif self.tool_mode == 'pointer':
                        # 闁圭娲幏鈥愁啅閵夈儱寰旈柨娑欎亢缂傚啫纾╅悗娑欘殘閸ゅ孩鎷呭鍥╂瀭
                        self.set_crosshair(event.xdata, event.ydata, emit_signal=False)
                        self.pending_pointer_sync_pos = (event.xdata, event.ydata)
                    elif self.tool_mode == 'pan':
                        # 妤犵偛纾簺鐎规悶鍎遍崣鍧楁晬濮樺疇銇愰弴鈩冨闯濠殿喖顑勭紞鍛磾?
                        self.pan_start_pos = (event.xdata, event.ydata)
                self.mouse_pressed = True
                self.last_mouse_pos = (event.xdata, event.ydata)
            elif event.button == 3:  # 闁告瑦濞婇弫顓㈡晬濮橆剛纾诲┑顔碱儓閻ㄧ喖寮€靛摜宕堕悗鍦櫕閻涖儲鎷?
                self.mouse_pressed = True
                self.last_mouse_pos = (event.xdata, event.ydata)
    
    def _on_mouse_release(self, event):
        """濮捬呭У閻栵綁鏌屾繝鍐╂澒濞存粌顑勫▎?"""
        pending_pointer_sync_pos = self.pending_pointer_sync_pos
        if self.mouse_button == 1 and self.tool_mode == 'pointer' and pending_pointer_sync_pos is not None:
            self.mouse_clicked.emit(pending_pointer_sync_pos[0], pending_pointer_sync_pos[1])
        self._finish_label_edit()
        self.mouse_pressed = False
        self.mouse_button = None
        self.last_mouse_pos = None
        self.last_mouse_canvas_pos = None
        self.pending_pointer_sync_pos = None
    
    def _on_mouse_move(self, event):
        """濮捬呭У閻栵絿绮旂拠鎻捫楀ù婊冾儎濞嗐垽鏁嶇仦鐐殰闁归晲绀侀悿鍕籍鐠虹儤鍊辨慨婵勫劜鐎氬骞?"""
        if not event.inaxes == self.ax:
            return
        
        if self.mouse_pressed and self.mouse_button is not None:
            if self.mouse_button == 1:  # 鐎归潻绠撻弫?
                if event.xdata is not None and event.ydata is not None:
                    if self.tool_mode == 'brush':
                        # 闁汇垼宕甸悷顏勵啅閵夈儱寰旈柨娑欎亢缁绘稓绱掔划顖炲礆?
                        self._paint_label(event.xdata, event.ydata)
                    elif self.tool_mode == 'eraser':
                        # 婵绱曞В濠囧箼閿曗偓娴兼劙宕楅崙銈囩獥閺夆晝鍋熼悽璇层€掗崨瀛樼彑
                        self._erase_label(event.xdata, event.ydata)
                    elif self.tool_mode == 'pointer':
                        # 闁圭娲幏鈥愁啅閵夈儱寰旈柨娑欐皑浜涢柛鏂诲妼瀹曞嫮鈧稒顨堥崵搴ㄦ晬閸繄鏉介柡鍐硾閹挸顫㈤妷顖滅
                        self.set_crosshair(event.xdata, event.ydata, emit_signal=False)
                        self.pending_pointer_sync_pos = (event.xdata, event.ydata)
                    elif self.tool_mode == 'zoom':
                        self._pan_from_pointer_drag(event)
                    elif self.tool_mode == 'pan':
                        self._pan_from_pointer_drag(event)
            elif self.mouse_button == 3:  # 闁告瑦濞婇弫顓㈡晬濮樿京绱氶柡鈧幑鎰板炊閻愵剙鐏楃紒鎰殔缂佹劖銇炵紞鍛嫬閸愨晜娈?
                if self.tool_mode == 'zoom':
                    self._zoom_from_pointer_drag(event)
                    return
                    # 缂傚倵鏅滈弬浣割啅閵夈儱寰旈柨娑欒壘瑜版悂鏌ㄧ€氬骞忛悾宀€绱氶柡鈧?
                    if self.last_mouse_pos and event.xdata is not None and event.ydata is not None:
                        if self.image_data is None:
                            return
                        
                        slice_img = self._get_slice_data()
                        if slice_img is None:
                            return
                    
                    h, w = slice_img.shape
                    
                    # 閻犱緤绱曢悾缁樜楅悩宕囧灱缂佸顕ф慨鈺冩崉濠?
                    dy = event.ydata - self.last_mouse_pos[1]
                    
                    # 闁告瑦濞婇弫顓㈠箯閺嶃劌顏肩紓鍌楁櫆閺備線鏁嶅顒傗偓顓㈡儎鐎垫劅鈺呭礉閵婏箑浠橀柛鎺戝缂傚寮?
                    # 闁搞劌鍊诲ú鍧楀触閹存粎鐟愰柟閿嬬墬鐎氬潡鏁嶅顓熸澒濠㈠爢宥囩闁搞劌鍊诲ú鍧楀触閹存粎鐟撻柟閿嬬墬鐎氬潡鏁嶅杈╃礆閻?
                    zoom_speed = 0.02  # 缂傚倵鏅滈弬渚€鏌呴悢宄邦唺
                    zoom_delta = -dy * zoom_speed
                    self.zoom_factor *= (1.0 + zoom_delta)
                    self.zoom_factor = max(0.1, min(20.0, self.zoom_factor))  # 闂傚嫭鍔曢崺妤冪磽閳哄倹鏉归柤鐓庡暙濞?
                    
                    # 濞寸姰鍎电槐鍫曞冀閸ワ妇绉寸紓鍐濞戞挾濡囩紓鍌楁櫆閺備線鏁嶉崼婊冣枏闁活潿鍔庢晶鍧楁偠閸℃缍忛柡宥呮祫缁?
                    if self.xlim_range is None or self.ylim_range is None:
                        xlim, ylim, _, _ = self._get_physical_range(h, w)
                        self.xlim_range = xlim
                        self.ylim_range = ylim
                    
                    # 閻犱緤绱曢悾鑽ょ磽閳哄倹鏉瑰☉鎾跺闁挎稑鐗撶槐鍫曞冀閸ワ妇绉寸紓鍐闁烩晝顭堝ù婊冮缂嶅宕滃鍫ュ炊閹规劕鐦遍柛銉ㄦ彧缁?
                    zoom_center_x = event.xdata
                    zoom_center_y = event.ydata
                    
                    # 閻犱緤绱曢悾鏄忋亹閹惧啿顤呴悷娆忔濞存﹢鎳犻崘銊︾函
                    current_width = self.xlim_range[1] - self.xlim_range[0]
                    current_height = self.ylim_range[1] - self.ylim_range[0]
                    
                    # 閻犱緤绱曢悾濠氬棘閹殿喗鐣遍悷娆忔濞存﹢鎳犻崘銊︾函闁挎稑鐗呮禍鎺撐楅悩宕囧灱濞达絽绉堕悿鍡樼▔鏉炴媽鍘煫鍥у枦缁?
                    new_width = current_width / self.zoom_factor
                    new_height = current_height / self.zoom_factor
                    
                    # 濞寸姰鍎电槐鍫曞冀閸ワ妇绉寸紓鍐濞戞挾濡囩紓鍌楁櫆閺?
                    self.xlim_range = (
                        zoom_center_x - new_width / 2,
                        zoom_center_x + new_width / 2
                    )
                    self.ylim_range = (
                        zoom_center_y - new_height / 2,
                        zoom_center_y + new_height / 2
                    )
                    
                    # 闂傚嫭鍔曢崺妤呭捶閵娿儲绂堥柛宥呯箺鐎垫牠宕堕弶鎴濇暥闁挎稑鐗忔晶鍧楁偠閸℃缍忛柡宥呮祫缁?
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
                    # 闂傚牏鍋熺紓澶愬绩閹傜矗闁稿繐鍤栫槐浼村矗閹惰姤鏆涢悹瀣暞閺嗭絿绮ｅΔ鈧紒鎰仦缂嶅懘鏁嶉崼婵嗘枾闁哄牆顦甸弰蹇曞竼闁?
                    if self.last_mouse_pos and event.xdata is not None and event.ydata is not None:
                        # 闁告鍠愬﹢渚€鎯冮崟顓犲炊閻庡湱鏅悰銉︽媴瀹ュ牏娈堕柡浣规尦閺勫繒甯?
                        pass
    
    def _physical_to_voxel_coords(self, x: float, y: float):
        """閻忓繐妫涙晶鍧楁偠閸℃缍忛柡宥呮穿濞村棝骞戠拹鐔告媴閹惧墎顦遍柛褎鍔栭悥?(orig_x, orig_y, orig_z)"""
        slice_img = self._get_slice_data()
        if slice_img is None:
            return None
        
        h, w = slice_img.shape
        
        if self.image_spacing is not None:
            if self.view_type == 'axial':
                pixel_x = int(x / self.image_spacing[1])
                pixel_y = int(y / self.image_spacing[0])
                orig_x = pixel_y  # 閺夌偟鏋傞柛姘閻庣數鎳撶花鐬傞柡鍌滄嚀閹?
                orig_y = int(w - 1 - pixel_x)  # axial 鐎甸潻顥忚ぐ鍥╃▔閻樺牅鏁嶉崼婵囧箖
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
                orig_y = int(w - 1 - pixel_x)  # sagittal 鐎甸潻顥忚ぐ鍥╃▔閻樺牅鏁嶉崼婵囧箖
                orig_z = int(h - 1 - pixel_y)
            else:
                return None
        else:
            pixel_x = int(x)
            pixel_y = int(y)
            if self.view_type == 'axial':
                orig_x = pixel_y
                orig_y = int(w - 1 - pixel_x)
                orig_z = self.current_slice
            elif self.view_type == 'coronal':
                orig_x = self.current_slice
                orig_y = pixel_x
                orig_z = int(h - 1 - pixel_y)
            elif self.view_type == 'sagittal':
                orig_x = self.current_slice
                orig_y = int(w - 1 - pixel_x)
                orig_z = int(h - 1 - pixel_y)
            else:
                return None
        
        return (orig_x, orig_y, orig_z)
    
    def _begin_label_edit(self):
        if not self._label_edit_active:
            self._label_edit_active = True
            self.label_edit_started.emit()

    def _finish_label_edit(self):
        if self._label_edit_active:
            self.label_edit_finished.emit()
        self._label_edit_active = False

    def _iter_brush_voxels(self, orig_x: int, orig_y: int, orig_z: int):
        """Yield label-array coordinates affected by the current circular brush."""
        H, W, D = self.label_data.shape
        if not (0 <= orig_x < W and 0 <= orig_y < H and 0 <= orig_z < D):
            return

        brush_radius = self.brush_size
        for dy in range(-brush_radius, brush_radius + 1):
            for dx in range(-brush_radius, brush_radius + 1):
                if dx * dx + dy * dy <= brush_radius * brush_radius:
                    nx, ny = orig_x + dx, orig_y + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        yield ny, nx, orig_z

    def _apply_brush(self, orig_x: int, orig_y: int, orig_z: int, label_value: int):
        """Apply circular brush to a single slice at the given voxel."""
        voxels = list(self._iter_brush_voxels(orig_x, orig_y, orig_z))
        if not voxels:
            return False

        if not any(self.label_data[ny, nx, nz] != label_value for ny, nx, nz in voxels):
            return False

        self._begin_label_edit()
        for ny, nx, nz in voxels:
            self.label_data[ny, nx, nz] = label_value
        return True
    
    def _paint_label(self, x: float, y: float):
        """闁汇垼宕甸悷顏勵啅閵夈儱寰旈柨娑欒壘濠€顏堝箰閸パ呮毎濞达絽绉堕悿鍡欑磼濡搫鐓戦柡宥呮川"""
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
        """婵绱曞В濠囧箼閿曗偓娴兼劙宕楅崙銈囩獥闁革负鍔嶇€垫氨鈧鐭紞鍛磾缁斿姊介妶鍡欏灱缂?"""
        if self.label_data is None or self.image_data is None:
            return
        
        coords = self._physical_to_voxel_coords(x, y)
        if coords is None:
            return
        
        orig_x, orig_y, orig_z = coords
        if self._apply_brush(orig_x, orig_y, orig_z, 0):
            self.label_changed.emit()
            self._update_display()
    
    # 濞ｅ洠鈧啿濞囬柨娑欑閻栵絿绮甸悙顒佹毉闁告瑦蓱濡炲倿宕ｉ幋婵嗘瘔
    def _get_max_slice(self) -> int:
        """Return slice count using the project voxel convention (x, y, z)."""
        if self.image_data is None:
            return 0
        if self.view_type == 'axial':
            return int(self.image_data.shape[2])
        if self.view_type == 'coronal':
            return int(self.image_data.shape[0])
        if self.view_type == 'sagittal':
            return int(self.image_data.shape[1])
        return 0

    def _get_slice_seed_pixels(self):
        """Map stored seed voxels (x, y, z) into the displayed slice pixels."""
        if self.image_data is None or not self.seed_points:
            return []

        h, w, d = self.image_data.shape
        points = []

        if self.view_type == 'axial':
            z = self.current_slice
            for x, y, z0 in self.seed_points:
                if 0 <= x < w and 0 <= y < h and 0 <= z0 < d and z0 == z:
                    points.append((h - 1 - y, x))
        elif self.view_type == 'coronal':
            y_slice = self.current_slice
            for x, y, z0 in self.seed_points:
                if 0 <= x < w and 0 <= y < h and 0 <= z0 < d and y == y_slice:
                    points.append((x, d - 1 - z0))
        elif self.view_type == 'sagittal':
            x_slice = self.current_slice
            for x, y, z0 in self.seed_points:
                if 0 <= x < w and 0 <= y < h and 0 <= z0 < d and x == x_slice:
                    points.append((h - 1 - y, d - 1 - z0))

        return points

    def _physical_to_voxel_coords(self, x: float, y: float):
        """Convert displayed physical coordinates to a voxel (x, y, z)."""
        slice_img = self._get_slice_data()
        if slice_img is None or self.image_data is None:
            return None

        h_disp, w_disp = slice_img.shape
        if self.image_spacing is not None:
            if self.view_type == 'axial':
                pixel_x = int(round(x / self.image_spacing[1]))
                pixel_y = int(round(y / self.image_spacing[0]))
                orig_x = pixel_y
                orig_y = int(w_disp - 1 - pixel_x)
                orig_z = self.current_slice
            elif self.view_type == 'coronal':
                pixel_x = int(round(x / self.image_spacing[1]))
                pixel_y = int(round(y / self.image_spacing[2]))
                orig_x = pixel_x
                orig_y = self.current_slice
                orig_z = int(h_disp - 1 - pixel_y)
            elif self.view_type == 'sagittal':
                pixel_x = int(round(x / self.image_spacing[1]))
                pixel_y = int(round(y / self.image_spacing[2]))
                orig_x = self.current_slice
                orig_y = int(w_disp - 1 - pixel_x)
                orig_z = int(h_disp - 1 - pixel_y)
            else:
                return None
        else:
            pixel_x = int(round(x))
            pixel_y = int(round(y))
            if self.view_type == 'axial':
                orig_x = pixel_y
                orig_y = int(w_disp - 1 - pixel_x)
                orig_z = self.current_slice
            elif self.view_type == 'coronal':
                orig_x = pixel_x
                orig_y = self.current_slice
                orig_z = int(h_disp - 1 - pixel_y)
            elif self.view_type == 'sagittal':
                orig_x = self.current_slice
                orig_y = int(w_disp - 1 - pixel_x)
                orig_z = int(h_disp - 1 - pixel_y)
            else:
                return None

        h, w, d = self.image_data.shape
        if not (0 <= orig_x < w and 0 <= orig_y < h and 0 <= orig_z < d):
            return None
        return (int(orig_x), int(orig_y), int(orig_z))

class ThreeDViewer(QWidget):
    """3D濡澘瀚悷娆忔濞?"""
    
    view_updated = pyqtSignal()
    
    def __init__(self, parent=None, force_matplotlib: bool = False):
        super().__init__(parent)
        self.setObjectName("ThreeDViewer")
        self.image_data = None
        self.label_data = None
        self.image_spacing = None
        self.structure_data = None
        self.force_matplotlib = bool(force_matplotlib)

        self.crosshair_3d = [0, 0, 0]
        self.setMinimumSize(300, 300)
        self.setStyleSheet("background-color: #050a12;")

        size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        size_policy.setHeightForWidth(True)
        self.setSizePolicy(size_policy)
        self.use_pyvista = False
        if not self.force_matplotlib:
            try:
                import pyvista as pv
                from pyvistaqt import QtInteractor
                self.use_pyvista = True
                self.pv = pv
            except ImportError as e:
                self.use_pyvista = False
                print(f"閻犫偓閿曗偓閹诧繝鏁嶅鍘僔ista/PyVistaQt濞戞挸绉磋ぐ鏌ユ偨缁辨繄浜搁崱鏇炩枏闁烩懇鏆昦tplotlib濞达絾绮堢拹鐔稿緞閸? {e}")

        # 婵炴挸寮堕悡瀣磽閹惧磭鎽犻柨娑欏哺娴尖晠宕楀鍡欐Ж婵炲棌鈧啿鐓曢柡鍌欏嵆閸忔﹢鏌屽鍡樼厐闁稿顒rching cubes
        self._mesh_cache_signature = None
        self._mesh_cache: Dict[int, object] = {}
        self._mesh_max_cells = 120000

        # 鐟滅増鎸告晶鐘诲礆閸パ冾棌妤犵偞濞婂浼存偐閼?
        self.clip_planes = []
        self.active_clip_plane: Optional[Dict[str, object]] = None
        self.last_cut_stats: Dict[str, object] = {}

        self._init_ui()
    
    def _init_ui(self):
        """闁告帗绻傞柛鏍ㄦ恭I"""
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(1)
        
        title_label = QLabel("3D View")
        title_label.setObjectName("ThreeDTitle")
        title_label.setFont(QFont("Arial", 8, QFont.Bold))
        title_label.setStyleSheet("color: #edf4ff; border: none; background: transparent; padding-left: 4px;")
        title_label.setMaximumHeight(18)
        layout.addWidget(title_label)
        
        if self.use_pyvista:
            import pyvista as pv
            from pyvistaqt import QtInteractor
            
            pv.set_plot_theme('dark')
            self.plotter = QtInteractor(self, off_screen=False)
            self.plotter.background_color = '#050a12'
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
                print(f"閻犫偓閿曗偓閹诧繝鏁嶅宕囩磾鐟曞棝寮婚幘铏彜闁稿繒鍘ф俊顖椻偓宕囩濠㈡儼绮剧憴? {e}")
            
        else:
            # 濠㈣泛娲径瀣厵婵℃鐗炵槐鐗堟媴鐠恒劍鏆弇atplotlib
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            
            bg_color = '#050a12'
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
        
        # 閻犱礁澧介悿鍡涘及鎼淬垺鈻旈柣銊ュ缁旂喎顩奸崱顓犵3D閻熸瑥妫楀ù妯绘媴鐠恒劍鏆忔繛锝堢簿婢瑰﹥娼忕憴鍕晬?
        self.setStyleSheet("""
            QWidget#ThreeDViewer {
                border: 1px solid #26344a;
                border-radius: 6px;
                background-color: #050a12;
            }
            QLabel#ThreeDTitle {
                color: #edf4ff;
                border: none;
                background: transparent;
                padding-left: 4px;
            }
        """)
        
        # 3D濞存嚎鍊撶花鏉课熼垾宕囩闁挎稑鐗撶划顖滄媼閵堝懏鍎欓柣顫妽濡棙娼槐婵囨交濞嗘劗澹夐柣顫妽閸╂盯宕ｆ禍鎺楀箯閺嵮冃楅悷娆忔濞存﹢鏁?
        self.interaction_mode = 'rotate'  # 'pointer', 'rotate', 'pan', 'zoom', 'clip'

        self.setLayout(layout)

    def _normalize_spacing(self) -> Tuple[float, float, float]:
        """閺夆晜鏌ㄥú?x, y, z)闁哄倻鎳撻幃娓焢acing闁挎稑鐭傜划顖滄媼?1,1,1)闁?"""
        if self.image_spacing is None or len(self.image_spacing) < 3:
            return 1.0, 1.0, 1.0
        try:
            sx = float(self.image_spacing[0])
            sy = float(self.image_spacing[1])
            sz = float(self.image_spacing[2])
            return sx if sx > 0 else 1.0, sy if sy > 0 else 1.0, sz if sz > 0 else 1.0
        except Exception:
            return 1.0, 1.0, 1.0

    def _spacing_xyz_for_pyvista(self) -> Tuple[float, float, float]:
        """
        閺夆晜鏌ㄥú鏈爕Vista闁秆勫姈閻栵絿鍖栭懡銈嗙暠spacing:
        PyVista(x, y, z) <=> numpy(axis2, axis1, axis0)
        """
        sx, sy, sz = self._normalize_spacing()
        return sz, sy, sx

    def _label_signature(self) -> Tuple:
        """閺夌偛顭烽崳铏圭棯瑜忛柛姘▌缁辨繈鎮介妸銈囪壘闁告帇鍊栭弻鍣嘺bel闁哄嫭鍎婇柛娆惷€垫煡濡?"""
        if self.label_data is None:
            return ("none",)
        label = self.label_data
        return (
            label.shape,
            int(np.count_nonzero(label)),
            int(np.sum(label, dtype=np.int64)),
            tuple(np.unique(label)[:8].tolist()),
            tuple(round(v, 4) for v in self._normalize_spacing()),
        )

    def _invalidate_mesh_cache(self):
        self._mesh_cache_signature = None
        self._mesh_cache = {}

    def get_volume_center(self) -> Tuple[float, float, float]:
        """闁兼儳鍢茶ぐ鍥亹閹惧啿顤呭ù锝嗘尰閺嗙喖骞戦懙鎴ｇ疀閸愯法绀凱yVista闁秆勫姈閻栵絿鍖栨导娆戠闁?"""
        if self.image_data is None:
            return 0.0, 0.0, 0.0
        h, w, d = self.image_data.shape
        sx, sy, sz = self._normalize_spacing()
        # PyVista: x<-d, y<-w, z<-h
        return (d * sz * 0.5, w * sy * 0.5, h * sx * 0.5)

    def _volume_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """鐟滅増鎸告晶鐘虫媴閹惧瓨娈堕柟鐟扮樁闁搞儱顕ú鍛存晬閸︻櫩Vista闁秆勫姈閻栵絿鍖栨导娆戠闁?"""
        if self.image_data is None:
            return (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        h, w, d = self.image_data.shape
        sx, sy, sz = self._normalize_spacing()
        return (0.0, d * sz, 0.0, w * sy, 0.0, h * sx)

    def _extract_surface_mesh(self, label_id: int):
        """闁圭粯鍔曡ぐ鍥箰閸パ呮毎闁哄秴娲ㄩ柣銊ュ閻℃垿宕愭總鍛婃〃闁挎稑鐗嗛悽鐜哋I閻熶椒绀佹竟鈧?闁告瑥顦靛鐑芥缁辨岸濡?"""
        if self.label_data is None:
            return None
        mask = (self.label_data == label_id)
        return self._extract_surface_mesh_from_mask(mask)

    def _extract_surface_mesh_from_mask(self, mask: np.ndarray):
        """濞寸姴绨肩花鈺呭磹缁都sk闁圭粯鍔曡ぐ鍥╃驳婢跺﹤銈稿浼村Υ?"""
        if not np.any(mask):
            return None

        coords = np.argwhere(mask)
        min_h, min_w, min_d = coords.min(axis=0)
        max_h, max_w, max_d = coords.max(axis=0)

        pad = 1
        h0 = max(int(min_h) - pad, 0)
        w0 = max(int(min_w) - pad, 0)
        d0 = max(int(min_d) - pad, 0)
        h1 = min(int(max_h) + pad + 1, self.label_data.shape[0])
        w1 = min(int(max_w) + pad + 1, self.label_data.shape[1])
        d1 = min(int(max_d) + pad + 1, self.label_data.shape[2])

        roi = mask[h0:h1, w0:w1, d0:d1].astype(np.uint8)
        if roi.size == 0 or np.count_nonzero(roi) < 8:
            return None

        import pyvista as pv
        # numpy(H,W,D) -> pv(X,Y,Z): (D,W,H)
        roi_xyz = np.transpose(roi, (2, 1, 0))
        grid = pv.ImageData()
        grid.dimensions = roi_xyz.shape
        grid.spacing = self._spacing_xyz_for_pyvista()

        sx, sy, sz = self._normalize_spacing()
        # 閻庣數鎳撶花鐫祔Vista闁秆勫姈閻? x<-d, y<-w, z<-h
        grid.origin = (d0 * sz, w0 * sy, h0 * sx)
        grid["values"] = roi_xyz.flatten(order="F")

        mesh = grid.contour([0.5])
        if mesh is None or mesh.n_points == 0:
            return None

        # 濠㈠爢鍕€渚€宕圭€ｎ喗顎栭梻鍫㈢闁圭粯鍔曞畷灞姐€掗崣澶屽帬闁诡儸鍡楀幋
        n_cells = int(getattr(mesh, "n_cells", 0))
        if n_cells > self._mesh_max_cells:
            reduction = 1.0 - (self._mesh_max_cells / float(max(n_cells, 1)))
            reduction = max(0.0, min(0.95, reduction))
            try:
                mesh = mesh.decimate_pro(reduction, preserve_topology=True)
            except Exception:
                pass

        return mesh

    def _mesh_style(self, label_id: int) -> Dict[str, object]:
        # 闁圭顦靛〒璺盒ч崒锔剧獥PV缂佷勘鍨兼竟濠囧Υ娑撶嚢闁藉啯绻嗘竟?
        if label_id == 1:
            return {"name": "PV", "color": "#ff4d4f", "opacity": 0.92}
        if label_id == 2:
            return {"name": "HV", "color": "#4096ff", "opacity": 0.88}
        return {"name": f"Label-{label_id}", "color": "#ff9f1a", "opacity": 0.85}

    def _get_mesh_bundle(self) -> Dict[int, object]:
        """闁圭顦伴悥锝囩驳閸撗勬櫢闁?濠㈣泛绉堕弫顦揺sh缂傚倹鎸搁悺銊╁Υ?"""
        signature = self._label_signature()
        if signature == self._mesh_cache_signature and self._mesh_cache:
            return self._mesh_cache

        meshes: Dict[int, object] = {}
        if self.use_pyvista and self.label_data is not None and np.any(self.label_data > 0):
            for label_id in (1, 2):
                mesh = self._extract_surface_mesh(label_id)
                if mesh is not None:
                    meshes[label_id] = mesh
            # 闁?/2婵炲备鍓濆﹢渚€寮悧鍫濈ウ闁挎稑鑻崢瑙勬償閺囩偟娼旂紒鈧ú顏呭闂傚棜鍩栫猾鐩縮k
            if not meshes:
                tmp = np.where(self.label_data > 0, 1, 0).astype(np.uint8)
                backup = self.label_data
                try:
                    self.label_data = tmp
                    mesh = self._extract_surface_mesh(1)
                    if mesh is not None:
                        meshes[99] = mesh
                finally:
                    self.label_data = backup

        self._mesh_cache_signature = signature
        self._mesh_cache = meshes
        return meshes

    def set_image_data(self, image_data: np.ndarray, spacing=None):
        """閻犱礁澧介悿鍡涘炊閹冨壖闁轰胶澧楀畵渚€鏁嶅顐ょ憹闁煎嘲袟闁哄洤鐡ㄩ弻濠囨晬鐏炶棄娑ч柛锔规殰efresh_view闁哄啳鍩栧ú鍧楀棘?"""
        self.image_data = image_data
        self.image_spacing = spacing  # 濞ｅ洦绻傞悺鈺痯acing濞ｅ洠鍓濇导鍛存偨閵娿倗鑹炬慨婵撶悼閳ユ﹢鎯?D婵絾鏌х欢?
        # 鐟滄澘宕崕姘跺矗濡搫顕ч梺顐ｈ壘閻栧爼骞囪箛鎾村殑闁活偆浜搁崫鍕唺闁瑰瓨鐗滈埞鏍⒒閺夋垵缍侀柛鏍ㄧ壄缁辨繂銆掗崨顖滃閻庢稒蓱濞茶法鈧懓顦崣?
        self._invalidate_mesh_cache()

    def set_label_data(self, label_data: np.ndarray):
        """閻犱礁澧介悿鍡涘冀閸モ晠寮悧鍫濈ウ闁挎稒鐭粭澶愭嚊婵晠寮寸€涙ɑ鐓€闁挎稑鑻ぐ褔宕烽埡鐢resh_view闁哄啳鍩栧ú鍧楀棘?"""
        self.label_data = label_data
        self._invalidate_mesh_cache()

    def set_structure_data(self, structure_data: Optional[np.ndarray]):
        """
        閻犱礁澧介悿鍡欐偘缂佺媴绱曠划銊╁几閸曨偄鐎婚悷娆欑到濞存﹢鏁?
        0闁煎啿鏈▍? 1濞戞捁顕ч崗? 2缂備礁妫楅崹搴ㄥ绩? 3闁告帒妫楀鍫曟倷楠?
        """
        self.structure_data = structure_data

    def compute_cut_statistics(self, normal, origin) -> Dict[str, object]:
        """
        閻犱緤绱曢悾濠氱嵁閹惰姤妗ㄩ柛鎺戞搐婢瑰﹦绱掗悢濠氭晬閸繄鍞ㄥù婊冪凹缂嶅妲愰悪鍛闁?        - 闁告挴鏅欑紞鎴炴媴閹剧紟?闁告帒娲▍搴㈡媴閹剧紟婵嬫晬閸粎鍠橀柛蹇撶墔婵炲洭鎮介埡鈺漛el=3濞达絾绮堢拹鐔兼嚃濠靛﹤澹堥柨娑樼焷鐎氥垺绋夊鍛憼闁革负鍔岄崹顖炲炊閻愭椽宕氭导瀛樺闂傚棜鍩栭悥锝囩驳閹惧懐绀?
        - 闁哄嫭鍎婇柛鎺戞处閺屽槤V/HV闁挎稑鐗嗛崹蹇涙鐞氳鲸绗熻閸忔﹢寮垫径搴ｆ偘缂佺姭鈧尙绉肩紒杈╁缁?
        """
        result: Dict[str, object] = {
            "available": False,
            "source": "none",
            "remaining_ml": 0.0,
            "resected_ml": 0.0,
            "total_ml": 0.0,
            "remaining_ratio": 0.0,
            "resected_ratio": 0.0,
            "severed_pv": False,
            "severed_hv": False,
            "pv_cut_voxels": 0,
            "hv_cut_voxels": 0,
            # New unified simulator fields
            "resection_rate": 0.0,
            "risk_assessment": "LOW",
            "cut_vessels": {
                "hv_count": 0,
                "pv_count": 0,
                "total_count": 0,
                "critical_vessels": [],
            },
            "cut_plane_visualization": {},
        }
        if self.label_data is None:
            return result

        n = np.asarray(normal, dtype=np.float64)
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-8:
            return result
        n = n / n_norm

        sx, sy, sz = self._normalize_spacing()
        voxel_ml = (sx * sy * sz) / 1000.0

        # 濞村吋锚閸樻稒鎷呯捄銊︽殢闁哄秴娲?闁挎稑鐗愮€氥垽鎮介妸锕€鐓曢柡鍫濐槼閸婁粙鎳樿箛鏃傚灱缂佹稒鎷濈槐姘舵晬鐏炶姤鍎婇柛鎺撶懃濞叉牠鏌呴梻鍫㈠仱濞村倿寮介崶?
        if np.any(self.label_data == 3):
            target_mask = (self.label_data == 3)
            result["source"] = "label_3_liver"
        elif np.any(self.label_data > 0):
            target_mask = (self.label_data > 0)
            result["source"] = "nonzero_label_proxy"
        else:
            target_mask = None

        if target_mask is not None and np.any(target_mask):
            # Build vessel label map for simulator: 1=PV, 2=HV, 0=others/background
            vessel_map = np.zeros_like(self.label_data, dtype=np.uint8)
            vessel_map[self.label_data == 1] = 1
            vessel_map[self.label_data == 2] = 2

            # NOTE:
            # - cut plane origin/normal are in the current 3D viewer (PyVista) coordinates:
            #   x<-axis2, y<-axis1, z<-axis0
            # - surgery_simulator assumes x<-axis2, y<-axis1, z<-axis0 with spacing=(sx,sy,sz)
            #   therefore we pass spacing as (sz, sy, sx) to stay consistent with current viewer coords.
            try:
                try:
                    from surgery_simulator import run_surgery_simulation
                except ModuleNotFoundError:
                    from our_snap.surgery_simulator import run_surgery_simulation

                sim = run_surgery_simulation(
                    liver_mask=target_mask.astype(np.uint8),
                    vessel_mask=vessel_map,
                    cut_plane_params={
                        "point": tuple(float(v) for v in origin),
                        "normal": tuple(float(v) for v in n.tolist()),
                        "point_is_mm": True,
                    },
                    spacing=(sz, sy, sx),
                )

                remaining_ml = float(sim.get("remaining_volume_ml", 0.0))
                resected_ml = float(sim.get("resected_volume_ml", 0.0))
                total_ml = remaining_ml + resected_ml
                remaining_ratio = (remaining_ml / total_ml) if total_ml > 1e-8 else 0.0
                resected_ratio = (resected_ml / total_ml) if total_ml > 1e-8 else 0.0

                result["available"] = True
                result["remaining_ml"] = remaining_ml
                result["resected_ml"] = resected_ml
                result["total_ml"] = total_ml
                result["remaining_ratio"] = remaining_ratio
                result["resected_ratio"] = resected_ratio
                result["resection_rate"] = float(sim.get("resection_rate", resected_ratio * 100.0))
                result["risk_assessment"] = str(sim.get("risk_assessment", "LOW"))
                result["cut_vessels"] = sim.get(
                    "cut_vessels",
                    {"hv_count": 0, "pv_count": 0, "total_count": 0, "critical_vessels": []},
                )
                result["cut_plane_visualization"] = sim.get("cut_plane_visualization", {})

                hv_count = int(result["cut_vessels"].get("hv_count", 0))
                pv_count = int(result["cut_vessels"].get("pv_count", 0))
                result["severed_hv"] = hv_count > 0
                result["severed_pv"] = pv_count > 0

            except Exception as e:
                print(f"閻犲鍟伴弫顦檜rgery_simulator濠㈡儼绮剧憴锕傛晬鐏炶姤绀€闂侇偊寮缁櫣鎷嬮敓鐘虫缁? {e}")

                o = np.asarray(origin, dtype=np.float64)
                plane_offset = float(np.dot(n, o))
                coords = np.argwhere(target_mask)
                # PyVista闁秆勫姈閻? x<-axis2, y<-axis1, z<-axis0
                px = coords[:, 2].astype(np.float64) * sz
                py = coords[:, 1].astype(np.float64) * sy
                pz = coords[:, 0].astype(np.float64) * sx
                signed = n[0] * px + n[1] * py + n[2] * pz - plane_offset
                keep = signed <= 0.0
                cut = ~keep
                keep_count = int(np.count_nonzero(keep))
                cut_count = int(np.count_nonzero(cut))
                total = keep_count + cut_count

                result["available"] = True
                result["remaining_ml"] = keep_count * voxel_ml
                result["resected_ml"] = cut_count * voxel_ml
                result["total_ml"] = total * voxel_ml
                if total > 0:
                    result["remaining_ratio"] = keep_count / total
                    result["resected_ratio"] = cut_count / total
                    result["resection_rate"] = result["resected_ratio"] * 100.0

                def _calc_vessel_cut(label_id: int):
                    vessel = (self.label_data == label_id)
                    if not np.any(vessel):
                        return False, 0
                    c = np.argwhere(vessel)
                    vx = c[:, 2].astype(np.float64) * sz
                    vy = c[:, 1].astype(np.float64) * sy
                    vz = c[:, 0].astype(np.float64) * sx
                    sd = n[0] * vx + n[1] * vy + n[2] * vz - plane_offset
                    on_a = int(np.count_nonzero(sd <= 0.0))
                    on_b = int(np.count_nonzero(sd > 0.0))
                    return (on_a > 0 and on_b > 0), min(on_a, on_b)

                pv_cut, pv_vox = _calc_vessel_cut(1)
                hv_cut, hv_vox = _calc_vessel_cut(2)
                result["severed_pv"] = pv_cut
                result["severed_hv"] = hv_cut
                result["pv_cut_voxels"] = pv_vox
                result["hv_cut_voxels"] = hv_vox
                result["cut_vessels"] = {
                    "hv_count": int(1 if hv_cut else 0),
                    "pv_count": int(1 if pv_cut else 0),
                    "total_count": int((1 if hv_cut else 0) + (1 if pv_cut else 0)),
                    "critical_vessels": [],
                }

            # Compatibility: keep legacy cut voxel estimate even with new simulator output
            o = np.asarray(origin, dtype=np.float64)
            plane_offset = float(np.dot(n, o))

            def _legacy_cut_vox(label_id: int) -> int:
                vessel = (self.label_data == label_id)
                if not np.any(vessel):
                    return 0
                c = np.argwhere(vessel)
                vx = c[:, 2].astype(np.float64) * sz
                vy = c[:, 1].astype(np.float64) * sy
                vz = c[:, 0].astype(np.float64) * sx
                sd = n[0] * vx + n[1] * vy + n[2] * vz - plane_offset
                on_a = int(np.count_nonzero(sd <= 0.0))
                on_b = int(np.count_nonzero(sd > 0.0))
                return int(min(on_a, on_b))

            result["pv_cut_voxels"] = int(_legacy_cut_vox(1))
            result["hv_cut_voxels"] = int(_legacy_cut_vox(2))

        self.last_cut_stats = result
        return result

    def set_clip_plane(self, normal=(1.0, 0.0, 0.0), origin=None, apply_to_mesh=True):
        """閻犱礁澧介悿?闁哄洤鐡ㄩ弻濠囧础閺囨岸鍤嬮柛鎺戞搐婢瑰﹪鐛幎鑺ユ〃闁挎稑鑻懟鐔告交閺傛寧绀€闁告帒娲ゆ竟濠勭磼閻斿濡?"""
        n = np.asarray(normal, dtype=np.float64)
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-8:
            n = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            n_norm = 1.0
        n = (n / n_norm).tolist()

        if origin is None:
            origin = list(self.get_volume_center())
        else:
            origin = [float(origin[0]), float(origin[1]), float(origin[2])]

        self.active_clip_plane = {
            "normal": tuple(n),
            "origin": tuple(origin),
            "apply_to_mesh": bool(apply_to_mesh),
        }
        stats = self.compute_cut_statistics(n, origin)
        self.refresh_view()
        return stats

    def set_crosshair_3d(self, x: int, y: int, z: int):
        """閻犱礁澧介悿?D闁告ぞ绀侀悺褔宕欓崱妤冨濞达絽绉堕悿?"""
        self.crosshair_3d = [x, y, z]
        # 濠碘€冲€归悘澶屾喆閸℃绂堢€规瓕灏欑划锟犲礆濠靛棝宕犻弽鐢电闁哄洤鐡ㄩ弻濠囧及閸撗佷粵
        if hasattr(self, 'plotter') and self.plotter is not None:
            self._update_crosshair_display()
    
    def _update_3d_view(self):
        """
        闁哄洤鐡ㄩ弻?D閻熸瑥妫楀ù?
        濞达綀娉曢弫顥糡K/PyVista閺夆晜绋?D婵炴挸寮堕悡?
        """
        if self.use_pyvista:
            self._update_3d_view_vtk()
        else:
            self._update_3d_view_matplotlib()
    
    def _update_3d_view_vtk(self):
        """濞达綀娉曢弫顥糡K/PyVista闁哄洤鐡ㄩ弻?D閻熸瑥妫楀ù姗€鏁嶉崼婵嗙€婚柡宥呮川婵炴挸寮堕悡?+ 闁告帒娲ゆ竟濠囩嵁閹惰姤妗ㄦ俊顖楀墲鐎?+ 缂傚倹鎸搁悺銊╂晬婢?"""
        import pyvista as pv

        self.plotter.clear()
        self.plotter.background_color = "black"

        if self.image_data is None:
            self.view_updated.emit()
            return

        # 闁稿繐顦遍崣搴ㄦ晬濮橆剙鎮戦悗褰掆偓娑氬枠闁?
        try:
            self.plotter.renderer.RemoveAllLights()
            head = pv.Light(light_type="headlight")
            head.intensity = 1.0
            self.plotter.add_light(head)
            fill = pv.Light(position=(0, 0, -1), focal_point=(0, 0, 0), color="white", intensity=0.1)
            fill.positional = False
            self.plotter.add_light(fill)
        except Exception:
            pass

        meshes = self._get_mesh_bundle()
        for label_id, mesh in meshes.items():
            style = self._mesh_style(label_id)
            render_mesh = mesh

            # 闁兼眹鍎遍幆搴ㄦ偨閵娿儱鐎奸柛鎾冲级鑶╅柟椋庡櫐缁辨繃鎷呯捄銊︽殢妤犵偞濞婂鎵偓鍦仦濡炲倻鎲楁担绋款梾婵☆垪鈧磭鈧?
            if self.active_clip_plane and self.active_clip_plane.get("apply_to_mesh"):
                try:
                    render_mesh = mesh.clip(
                        normal=self.active_clip_plane["normal"],
                        origin=self.active_clip_plane["origin"],
                        invert=False,
                    )
                except Exception:
                    render_mesh = mesh

            try:
                self.plotter.add_mesh(
                    render_mesh,
                    color=style["color"],
                    opacity=style["opacity"],
                    show_edges=False,
                    lighting=True,
                    name=f"mesh_{style['name']}",
                )
            except Exception:
                # 闁稿繒鍘ч柟顑啫骞戦幖?
                try:
                    self.plotter.add_mesh(
                        render_mesh,
                        color=style["color"],
                        opacity=min(float(style["opacity"]), 0.85),
                        lighting=False,
                        name=f"mesh_{style['name']}",
                    )
                except Exception as e:
                    print(f"Warning: render {style['name']} failed: {e}")

        # 缂備焦鎸婚悗顖氼啅缁辨捇宕犻弽銊︹枖缂佲偓閻氬绀勫☉鎾诡嚙閸?闁告帒妫楀鍫曟倷閻у摜绀?
        if (
            self.structure_data is not None
            and self.label_data is not None
            and self.structure_data.shape == self.label_data.shape
        ):
            try:
                # 濞戞捁顕ч崗閬嶅矗閻樻彃顫ｉ柨娑樼墛濞叉寧绂嶆担瑙勭函濡ゅ倹眉缁楀鏌呰箛鏃€顫栭柨?
                trunk_styles = {
                    1: {"color": "#ff0000", "opacity": 0.98, "name": "pv_trunk"},
                    2: {"color": "#005eff", "opacity": 0.98, "name": "hv_trunk"},
                }
                for label_id in (1, 2):
                    trunk_mask = (self.label_data == label_id) & (self.structure_data == 1)
                    if np.any(trunk_mask):
                        trunk_mesh = self._extract_surface_mesh_from_mask(trunk_mask)
                        if trunk_mesh is not None:
                            ts = trunk_styles[label_id]
                            try:
                                self.plotter.add_mesh(
                                    trunk_mesh,
                                    color=ts["color"],
                                    opacity=ts["opacity"],
                                    show_edges=False,
                                    lighting=True,
                                    name=ts["name"],
                                )
                            except Exception:
                                pass

                # 闁告帒妫楀鍫曟倷瑜版帞褰ù婊呯濮掓稑瀚竟濠囨晬?
                junc_mask = (self.structure_data == 3) & (self.label_data > 0)
                if np.any(junc_mask):
                    junc_mesh = self._extract_surface_mesh_from_mask(junc_mask)
                    if junc_mesh is not None:
                        try:
                            self.plotter.add_mesh(
                                junc_mesh,
                                color="#ffd400",
                                opacity=1.0,
                                show_edges=False,
                                lighting=True,
                                name="junction_highlight",
                            )
                        except Exception:
                            pass
            except Exception as e:
                print(f"閻犫偓閿曗偓閹诧繝鏁嶅杈╂尝闁哄瀚Ο濠傤嚕閸屾艾顕ч柡鍕⒔閵囨碍寰勬潏顐バ? {e}")

        # 闁告帒娲ゆ竟濠囩嵁閹惰姤妗ㄩ柛娆撳礌?
        self.clip_planes = []
        if self.active_clip_plane:
            try:
                bounds = self._volume_bounds()
                extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
                plane = pv.Plane(
                    center=self.active_clip_plane["origin"],
                    direction=self.active_clip_plane["normal"],
                    i_size=extent * 1.2,
                    j_size=extent * 1.2,
                )
                actor = self.plotter.add_mesh(
                    plane,
                    color="#00d3b8",
                    opacity=0.28,
                    lighting=False,
                    name="surgical_plane",
                )
                self.clip_planes.append(
                    {
                        "plane": plane,
                        "actor": actor,
                        "normal": self.active_clip_plane["normal"],
                        "origin": self.active_clip_plane["origin"],
                    }
                )
            except Exception as e:
                print(f"閻犫偓閿曗偓閹诧繝鏁嶅顒€鐎奸柛鎾村絻闁解晠妫冮姀陇顩柡灞炬尭閵囨垹鎷? {e}")

        # 闁秆勫姈閻栵絿鍖栫拋宕囩憿闁烩晛鎲″┃鈧?
        try:
            self.plotter.add_axes(labels_off=False)
        except Exception:
            try:
                self.plotter.add_axes(labels_off=True)
            except Exception:
                pass

        self.plotter.camera_position = "iso"
        self.plotter.camera.parallel_projection = True

        self._update_crosshair_display()
        self.set_interaction_mode(self.interaction_mode if hasattr(self, "interaction_mode") else "rotate")
        self.view_updated.emit()
    
    def _update_crosshair_display(self):
        """闁哄洤鐡ㄩ弻?D闁告ぞ绀侀悺褔宕欓崱妤冨闁哄嫬澧介妵姘舵晬閸粎鐟忛柡澶涚磿閸ゅ酣鏁?"""
        if not hasattr(self, 'plotter') or self.plotter is None:
            return
        
        if self.image_data is None:
            return
        
        import pyvista as pv
        
        # 缂佸顭峰▍搴ㄥ籍瑜忓▓鎴﹀礄閸℃濡囩紒鎯х仢閹蜂即寮介崶顏堟晬閸繈寮稿鍐憼闁革妇绀?
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
        
        # 闁兼儳鍢茶ぐ鍥炊閹冨壖閻忓繐鎼?
        h, w, d = self.image_data.shape
        
        # crosshair_3d閻庢稒锚閸嬪秹鎯冮崟顒佇﹂柣妞绘櫇閹﹪宕搁幇顓犲灱 [space_x, space_y, space_z]闁挎稑娼眜mpy闁秆勫姈閻栵絿鍖栭懡銈囧煚闁?
        # 闂傚洨鎲版担鑺ョギ闁圭绀婸yVista闁秆勫姈閻栵絿鍖栭懡銈囧煚
        # PyVista闁秆勫姈閻栵絿鍖栭懡銈囧煚闁挎稒顒抽悗鐢垫嚀缁ㄧ灐umpy闁汇劌澧ラ柨娑樼墔缁楀倹绋夌€ｅ墎绀嗛柨娑樼様閻庣數鎳撶花鐬mpy闁汇劌澧㈤柨娑樼墕婢х娀宕ユ惔顖滅闁挎稑鐦栭悗鐢垫嚀缁ㄧ灐umpy闁汇劌澧￠柨娑樼墕娑斿繘宕ｇ粵鍦
        # 闁瑰灚绂掗妷銊︾ギ闁瑰湱绐桺yVista [x, y, z] = numpy [z, y, x] = [crosshair_3d[2], crosshair_3d[1], crosshair_3d[0]]
        if self.image_spacing is not None:
            # crosshair_3d鐎规瓕灏欑划锟犲及婢у潡鎮堕崱妤佺稄闁哄秴娴勭槐婵嬫儎鐎涙ê澶嶅ù锝堟硶閺?
            # 濞达絽妫濆〒鍓佹啺娴ｈ姤绁柟骞垮灩閸╁yVista闁秆勫姈閻栵絿鍖栭懡銈囧煚
            # PyVista x = numpy z = crosshair_3d[2]
            # PyVista y = numpy y = crosshair_3d[1]
            # PyVista z = numpy x = crosshair_3d[0]
            pos = [self.crosshair_3d[2], self.crosshair_3d[1], self.crosshair_3d[0]]
        else:
            # 婵炲备鍓濆﹢涔籶acing闁哄啳顔愮槐婕渞osshair_3d闁哄嫬鍓肩紒杈╁Т濞兼寮介崶椋庣濞戞梻鍠栧〒鍓佹啺娴ｈ姤绁柟?
            pos = [float(self.crosshair_3d[2]), float(self.crosshair_3d[1]), float(self.crosshair_3d[0])]
        
        # 閻犱緤绱曢悾濠氬礄閸℃濡囩紒鎹愭硶濞堟垿姊归崹顔碱唺闁挎稑鐗忕€硅櫕绋夐崫鍕闁稿秴绻愰弰鍌溾偓闈涙憸濞?0%闁挎稑濂旀繛鍥礂閼稿灚绾柡鍕濡鏁?
        if self.image_spacing is not None:
            # 閻犱緤绱曢悾濠氬炊閹冨壖闁汇劌瀚〒鑸靛緞瑜忔晶鍧楁偠閸℃妲€閻庣敻娼荤槐娆愭媴鐠恒劍鏆廝yVista闁秆勫姈閻栵絿鍖栭懡銈囧煚闁?
            # PyVista闁秆勫姈閻栵綁鏁嶅铏光偓鐢垫嚀缁ㄧ灐umpy z闁挎稑鐦旈悗鐢垫嚀缁ㄧ灐umpy y闁挎稑鐦栭悗鐢垫嚀缁ㄧ灐umpy x
            max_phys_size = max(
                d * self.image_spacing[2],  # PyVista x闁哄倻鎳撻幃婊堟晬閸ь柣mpy z闁?
                h * self.image_spacing[1],  # PyVista y闁哄倻鎳撻幃婊堟晬閸ь柣mpy y闁?
                w * self.image_spacing[0]   # PyVista z闁哄倻鎳撻幃婊堟晬閸ь柣mpy x闁?
            )
            line_length = max_phys_size * 0.20  # 20%闁汇劌瀚伴弳杈ㄦ償?
        else:
            line_length = max(h, w, d) * 0.20
        
        # 缂備焦锚閸╂绋夋径瀣拫闁告垵妫楃缓鍓х棯閸栵紕绀勯柦鍐╃箚婢瑰﹪鎼瑰鍗炴疇闁轰礁鐗婇悘澶愭晬鐏炵厧鈻忛柣顫姀缁舵繄鍒掑Δ鍐╃暠缂佹儳灏呯槐?
        try:
            # X闁哄倻鎳撻幃婊堝礄閸℃濡囩紒鎯у皡缁辨瑥鈻界缓鎾存姜鏉堝墽绀夐柛銉ユ惈閻ｇ洞闁告粌顒甸柨?
            x_line_start = [pos[0] - line_length/2, pos[1], pos[2]]
            x_line_end = [pos[0] + line_length/2, pos[1], pos[2]]
            x_line = pv.Line(x_line_start, x_line_end)
            actor_x = self.plotter.add_mesh(x_line, color='#0080FF', line_width=3, lighting=False, name='crosshair_x')
            # 閻犱礁澧介悿鍝竎tor闁告艾绉惰ⅷ濞寸姰鍎扮粚鍫曞触鎼达絿鏁剧紒澶婎煼濞?
            if hasattr(actor_x, 'SetName'):
                actor_x.SetName('crosshair_x')
            
            # Y闁哄倻鎳撻幃婊堝礄閸℃濡囩紒鎯у皡缁辨瑥鈻界缓鏃€娼潏鍓х闁搞儱鎼悾缍抽柛婊冾劦闁?
            y_line_start = [pos[0], pos[1] - line_length/2, pos[2]]
            y_line_end = [pos[0], pos[1] + line_length/2, pos[2]]
            y_line = pv.Line(y_line_start, y_line_end)
            actor_y = self.plotter.add_mesh(y_line, color='#0080FF', line_width=3, lighting=False, name='crosshair_y')
            if hasattr(actor_y, 'SetName'):
                actor_y.SetName('crosshair_y')
            
            # Z闁哄倻鎳撻幃婊堝礄閸℃濡囩紒鎯у皡缁辨瑥鈻界缓鏇熸姜鏉堝墽绀夐柛銉ユ惈閻ｇ冻闁告粌顒撮柨?
            z_line_start = [pos[0], pos[1], pos[2] - line_length/2]
            z_line_end = [pos[0], pos[1], pos[2] + line_length/2]
            z_line = pv.Line(z_line_start, z_line_end)
            actor_z = self.plotter.add_mesh(z_line, color='#0080FF', line_width=3, lighting=False, name='crosshair_z')
            if hasattr(actor_z, 'SetName'):
                actor_z.SetName('crosshair_z')
            
            # 闁革负鍔岄崳顖濈疀閸愌呯Т缂傚啫娼戦柛鏃傚С缁斿瓨绋夌挒銏ゆ嚌閼碱剚鍊栧ù锝嗘尰閻栵絿鎷嬬敮顔剧濞?D閻熸瑥妫楀ù姗€鎯冮崟顓￠浌闁肩懓寮堕弻鐔煎锤濡炲墽顏遍柤鐤彧缁?
            # 闁荤偛鍟╃紞瀣緞瑜嶉惃顒傜棯閿旇儻绀嬮柛銉﹀劤閸庢岸寮靛鍫嗗啯妲€閻庨潧鎽滃▓?%
            if self.image_spacing is not None:
                marker_size = max_phys_size * 0.02
            else:
                marker_size = max(h, w, d) * 0.02
            
            center_marker = pv.Sphere(radius=marker_size, center=pos)
            actor_center = self.plotter.add_mesh(
                center_marker, 
                color='#00FF00',  # 缂備浇鍎绘竟濠囨晬鐏炶偐鐟?D閻熸瑥妫楀ù妯荤▔闁?
                lighting=False, 
                name='crosshair_center'
            )
            if hasattr(actor_center, 'SetName'):
                actor_center.SetName('crosshair_center')
                
        except Exception as e:
            # 濠碘€冲€归悘濉磇ne_width闁告瑥鍊归弳鐔哥▔瀹ュ棙鏆滈柟闀愮筏缁辨繄浜稿┑濠勬Ц濞戞挸绉虫繛鍥偨閵娿劑宕ｉ崒娑欐
            try:
                # X闁哄倻鎳撻幃婊堝礄閸℃濡囩紒?
                x_line_start = [pos[0] - line_length/2, pos[1], pos[2]]
                x_line_end = [pos[0] + line_length/2, pos[1], pos[2]]
                x_line = pv.Line(x_line_start, x_line_end)
                actor_x = self.plotter.add_mesh(x_line, color='#0080FF', lighting=False, name='crosshair_x')
                if hasattr(actor_x, 'SetName'):
                    actor_x.SetName('crosshair_x')
                
                # Y闁哄倻鎳撻幃婊堝礄閸℃濡囩紒?
                y_line_start = [pos[0], pos[1] - line_length/2, pos[2]]
                y_line_end = [pos[0], pos[1] + line_length/2, pos[2]]
                y_line = pv.Line(y_line_start, y_line_end)
                actor_y = self.plotter.add_mesh(y_line, color='#0080FF', lighting=False, name='crosshair_y')
                if hasattr(actor_y, 'SetName'):
                    actor_y.SetName('crosshair_y')
                
                # Z闁哄倻鎳撻幃婊堝礄閸℃濡囩紒?
                z_line_start = [pos[0], pos[1], pos[2] - line_length/2]
                z_line_end = [pos[0], pos[1], pos[2] + line_length/2]
                z_line = pv.Line(z_line_start, z_line_end)
                actor_z = self.plotter.add_mesh(z_line, color='#0080FF', lighting=False, name='crosshair_z')
                if hasattr(actor_z, 'SetName'):
                    actor_z.SetName('crosshair_z')
                
                # 濞戞挾濡囬柡宥呮穿
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
                print(f"閻犫偓閿曗偓閹诧繝鏁嶅杈╁笡闁?D闁告ぞ绀侀悺褔宕欓崱妤冨濠㈡儼绮剧憴? {e2}")
    
    def set_interaction_mode(self, mode: str):
        """閻犱礁澧介悿?D濞存嚎鍊撶花鏉课熼垾宕囩"""
        self.interaction_mode = mode
        
        if not hasattr(self, 'plotter') or self.plotter is None:
            return
        
        try:
            # PyVista闁汇劌瀚锔界閹烘拑浣割嚕韫囨氨绱?
            if mode == 'pointer':
                # 闁圭娲幏鈥澄熼垾宕囩闁挎稒姘ㄩ柣顫妽婢у秹寮垫径澶嬪攭濞?
                self.plotter.disable()
            elif mode == 'rotate':
                # 闁哄啫顑堝ù鍡椢熼垾宕囩闁挎稒鑹鹃幆搴ㄦ偨閵婏附顥嬮弶鐐电濮掓稒顭堢€归潻绠撻弫顓㈡晬?
                self.plotter.enable_trackball_style()
            elif mode == 'pan':
                # 妤犵偛纾簺婵☆垪鈧磭纭€闁挎稒鑹鹃幆搴ㄦ偨閵娿儵鎸紒澶夌串缁辨瑦绋夐弫顓㈠箣閺堫柈ift+鐎归潻绠撻弫顓㈡晬?
                self.plotter.enable_trackball_style()
            elif mode == 'zoom':
                # 缂傚倵鏅滈弬浣肝熼垾宕囩闁挎稒鑹鹃幆搴ㄦ偨閵娧呯礆闁衡偓閹惧懐绀勬繝濠冧亢閻ゅ棝骞嬮弽褍绀侀梺璺ㄧ
                self.plotter.enable_trackball_style()
            elif mode == 'clip':
                # 閻熶椒绀佹竟鈧俊顖椻偓宕囩闁挎稒鑹鹃幆搴ㄦ偨閵娿劑宕滈柦鈺呮濮橈附绂?
                self.plotter.enable_trackball_style()
            else:
                # 濮掓稒顭堥柛姘殢trackball濡炲瀛╅悧鎼佹晬閸繂鐦堕柛姘棆閺夌偘绀侀柦鈺冪矓濮瑰洣鑳剁紓澶愬绩閹惧懐绀?
                self.plotter.enable_trackball_style()
        except Exception as e:
            print(f"閻犱礁澧介悿?D濞存嚎鍊撶花鏉课熼垾宕囩濠㈡儼绮剧憴? {e}")
    
    def add_clip_plane(self, normal=(1, 0, 0), origin=(0, 0, 0)):
        """婵烇綀顕ф慨?闁哄洤鐡ㄩ弻濠囧礆閸パ冾棌妤犵偞濞婂浼存晬瀹€鍕笡閻犱降鍊曠花鏌ユ偨閵娿儱鐓傛俊顖椻偓宕団偓椋庢啑娴ｇ顥呴柕?"""
        return self.set_clip_plane(normal=normal, origin=origin, apply_to_mesh=True)
    
    def remove_clip_planes(self):
        """缂佸顭峰▍搴ㄥ箥闁哄牆顦抽柛鎾绘尙闂?"""
        self.active_clip_plane = None
        self.clip_planes = []
        self.last_cut_stats = {}
        if self.use_pyvista:
            self.refresh_view()
    
    def _update_3d_view_matplotlib(self):
        """濞达綀娉曢弫顦揳tplotlib闁哄洤鐡ㄩ弻?D閻熸瑥妫楀ù姗€鏁嶉崼婵嬫焻婢跺鐓欐俊妤€鐗炵槐?"""
        self.ax.clear()
        self.ax.set_facecolor('black')
        
        if self.image_data is None:
            self.canvas.draw()
            self.view_updated.emit()
            return
        
        h, w, d = self.image_data.shape
        sx, sy, sz = self._normalize_spacing()
        # 濞戞挸渚梱Vista濞ｅ洦绻冪€垫梹绋夐柤宄邦嚟濞堟垿鎮ч埡鍐╁€為柛褎鍔栭悥锝夋嚑閸愩劍绾柨?        # x <- axis2(z), y <- axis1(y), z <- axis0(x)
        x_max = float(d) * sz
        y_max = float(w) * sy
        z_max = float(h) * sx
        
        rendered = False
        all_min = None
        all_max = None
        if self.label_data is not None and np.any(self.label_data > 0):
            try:
                from skimage import measure
                label_styles = {
                    1: ("#ff4d4f", 0.90),  # PV
                    2: ("#4096ff", 0.88),  # HV
                }

                for label_id, (color, opacity) in label_styles.items():
                    label_mask = (self.label_data == label_id)
                    voxels = int(np.count_nonzero(label_mask))
                    if voxels < 8:
                        continue

                    # 濞达綀娉曢弫顦揳rching_cubes闁汇劌澧晅ep_size闁稿鍩冨鍡涘礆濞嗘挻顎栭梺鎻掓处閻楅亶鍨惧┑鎾剁濞ｅ洦绻冪€垫梹娼婚悙鍓佹焾鐟滆埇鍨烘笟濠勭闂侇剙鐏濋崢銈夋⒕韫囨梹绨氱痪鏉挎捣婢?
                    if voxels > 2_500_000:
                        step_size = 4
                    elif voxels > 800_000:
                        step_size = 3
                    elif voxels > 180_000:
                        step_size = 2
                    else:
                        step_size = 1

                    # 闂佹彃娲ㄩ弫銈嗙▔瀹ョ湥Vista濞戞捇鎳涚€靛憡鐣遍弶鐐存尦閵嗗孩鎯旇箛搴ｇ憿spacing闁挎稑鐭傛导鈺呭礂瀹ヤ焦绮嶆晶搴ㄧ嵁閸愭彃顕ч柍?                    mask_xyz = np.transpose(label_mask.astype(np.uint8), (2, 1, 0))
                    verts, faces, _normals, _values = measure.marching_cubes(
                        mask_xyz,
                        level=0.5,
                        spacing=(sz, sy, sx),
                        allow_degenerate=False,
                        step_size=step_size,
                    )
                    if len(verts) == 0 or len(faces) == 0:
                        continue

                    mesh = Poly3DCollection(
                        verts[faces],
                        alpha=opacity,
                        facecolor=color,
                        edgecolor='none',
                        linewidths=0,
                    )
                    self.ax.add_collection3d(mesh)
                    rendered = True

                    vmin = verts.min(axis=0)
                    vmax = verts.max(axis=0)
                    if all_min is None:
                        all_min = vmin
                        all_max = vmax
                    else:
                        all_min = np.minimum(all_min, vmin)
                        all_max = np.maximum(all_max, vmax)
            except Exception as e:
                print(f"matplotlib render failed: {e}")

        # 閻犱礁澧介悿鍡涘锤閹邦厾鍨奸弶鐐殿棎鐎垫牠宕舵潏鍓х闁哄牆顦拌啯闁搞劌顑嗗鍌炴嚂濮樺崬濡介柛鎺斿鑶╅柛銊ヮ儓缁旂喖鎮惧畝瀣耿闁告熬绠戦崹顖炲及閸撗佷粵闁稿繈鍔忛梺鎻掑嚱缁?
        if rendered and all_min is not None and all_max is not None:
            margin = 8.0
            self.ax.set_xlim(max(0.0, float(all_min[0]) - margin), min(x_max, float(all_max[0]) + margin))
            self.ax.set_ylim(max(0.0, float(all_min[1]) - margin), min(y_max, float(all_max[1]) + margin))
            self.ax.set_zlim(max(0.0, float(all_min[2]) - margin), min(z_max, float(all_max[2]) + margin))
        else:
            self.ax.set_xlim(0.0, x_max)
            self.ax.set_ylim(0.0, y_max)
            self.ax.set_zlim(0.0, z_max)

        # 闁稿繑濞婂Λ瀵哥磾閹寸偟澹愬☉鎾抽濞兼寮介崶顏堝彙闁哄秴娲﹂弫鐐烘晬瀹€鍐ㄥ灡閺夆晜鍘紋Vista濡炲瀛╅悧?
        self.ax.grid(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        self.ax.set_zlabel("")
        try:
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False
            self.ax.xaxis.pane.set_edgecolor((0, 0, 0, 0))
            self.ax.yaxis.pane.set_edgecolor((0, 0, 0, 0))
            self.ax.zaxis.pane.set_edgecolor((0, 0, 0, 0))
        except Exception:
            pass
        try:
            self.ax.set_box_aspect((max(1.0, x_max), max(1.0, y_max), max(1.0, z_max)))
        except Exception:
            pass

        # 濞寸姰鍎茶啯闁搞劌顑勯懙鎴ｇ疀閸愵亞甯涢柛鎺曟硾瀹曞嫮鈧稒顨呴崳顖炲及閻曞倻绀勯柦鍐╃箘閸?缂備浇娉曢崑锝夋晬?
        if rendered and all_min is not None and all_max is not None:
            center = 0.5 * (all_min + all_max)
            span = float(np.max(all_max - all_min))
        else:
            center = np.array([x_max * 0.5, y_max * 0.5, z_max * 0.5], dtype=np.float64)
            span = float(min(x_max, y_max, z_max))
        line_length = max(12.0, 0.35 * max(1.0, span))
        half = line_length * 0.5
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        self.ax.plot([cx - half, cx + half], [cy, cy], [cz, cz], color="#0080FF", linewidth=2.0, alpha=0.95)
        self.ax.plot([cx, cx], [cy - half, cy + half], [cz, cz], color="#0080FF", linewidth=2.0, alpha=0.95)
        self.ax.plot([cx, cx], [cy, cy], [cz - half, cz + half], color="#0080FF", linewidth=2.0, alpha=0.95)
        self.ax.scatter([cx], [cy], [cz], c="#00FF00", s=240, marker="o", alpha=0.95, depthshade=False)

        # 闁搞儱鎼悾楣冨箳閵夈劎绠甈yVista闁汇劌瀚悺鎴炴姜绾惧鎯?        self.ax.view_init(elev=24, azim=225)
        self.ax.set_proj_type('ortho')
        
        self.canvas.draw()
        self.view_updated.emit()
    
    def refresh_view(self):
        """Refresh the 3D view."""
        self._update_3d_view()
    
    def reset_camera(self):
        """闂佹彃绉堕悿鍡涙儎閸涘﹥绨氶悷娆忔"""
        if self.use_pyvista:
            if hasattr(self, 'plotter'):
                self.plotter.camera_position = 'iso'
                self.plotter.camera.parallel_projection = True
        else:
            if self.image_data is not None and hasattr(self, 'ax'):
                self.ax.view_init(elev=20, azim=45)
                self.ax.set_proj_type('ortho')
                self.canvas.draw()

    def closeEvent(self, event):
        """闂佹彃锕ラ弬涔俆K/QtInteractor閻犙冨缁噣鏁嶅畝鍕級闁稿繐绉堕悰銉╁矗閿濆懎褰犻梻鍌涘€靛☉鎾筹梗缁楀懘寮崶銊ф殭闁伙絾鐟?"""
        if getattr(self, "use_pyvista", False):
            plotter = getattr(self, "plotter", None)
            if plotter is not None:
                try:
                    rw = getattr(plotter, "ren_win", None)
                    if rw is None:
                        rw = getattr(plotter, "render_window", None)
                    if rw is not None and hasattr(rw, "Finalize"):
                        rw.Finalize()
                except Exception:
                    pass
                try:
                    plotter.close()
                except Exception:
                    pass
                self.plotter = None
        super().closeEvent(event)


class FourViewWidget(QWidget):
    """2閼?閻㈩垰鍟惇顒傛喆閸℃绂堢紓浣稿濞?"""
    
    # 濞ｅ洠鈧啿濞囬柨娑欒壘瀹曞嫮鈧稒顨堥崵搴㈡媴瀹ュ洨鏋傞柡鈧悷鏉跨秮 (view_type, x, y)
    crosshair_changed = pyqtSignal(str, float, float)
    # 濞ｅ洠鈧啿濞囬柨娑欒壘閸ㄥ繘鎮ч崶銊︽毉闁?(view_type, slice_idx)
    slice_changed = pyqtSignal(str, int)
    
    def __init__(self, parent=None, force_matplotlib_3d: bool = False):
        super().__init__(parent)
        self.image_data = None
        self.label_data = None
        self.image_spacing = None  # 濞ｅ洦绻傞悺鈺痯acing濞ｅ洠鍓濇导?
        self.seed_points = []
        self.force_matplotlib_3d = bool(force_matplotlib_3d)
        self._syncing_views = False
        self._pending_3d_crosshair = None
        self._3d_crosshair_update_pending = False
        
        # 鐟滅増鎸告晶鐘诲础娴ｅ摜鎽熺紒鎯с仒缂嶅懐绱旂槐?D缂佸本妞藉Λ鍧楀锤閹邦厾鍨奸柨?
        self.crosshair_3d = [0, 0, 0]  # [x, y, z]
        
        self._init_ui()
        self._connect_signals()
    
    def _init_ui(self):
        """Initialize the 2x2 four-view layout."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(1)

        grid_layout.setRowStretch(0, 1)
        grid_layout.setRowStretch(1, 1)
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 1)

        # Build four panes explicitly to avoid partial initialization.
        self.axial_view = SliceViewer('axial')
        self.coronal_view = SliceViewer('coronal')
        self.sagittal_view = SliceViewer('sagittal')
        self.view_3d = ThreeDViewer(force_matplotlib=self.force_matplotlib_3d)

        size_policy_3d = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        size_policy_3d.setHeightForWidth(True)
        self.view_3d.setSizePolicy(size_policy_3d)

        grid_layout.addWidget(self.axial_view, 0, 0)
        grid_layout.addWidget(self.coronal_view, 0, 1)
        grid_layout.addWidget(self.view_3d, 1, 0)
        grid_layout.addWidget(self.sagittal_view, 1, 1)

        grid_layout.setColumnMinimumWidth(0, 300)
        grid_layout.setColumnMinimumWidth(1, 300)
        grid_layout.setRowMinimumHeight(0, 300)
        grid_layout.setRowMinimumHeight(1, 300)

        layout.addLayout(grid_layout)
        self.setLayout(layout)

    
    def _connect_signals(self):
        """閺夆晝鍋炵敮瀛樼┍閳ュ啿濞囬柨娑樿嫰閻ゅ嫰鎮抽弶鍨）閻庢稒顨堥崵搴ㄦ嚂閺傚灝袟闁挎稑鐗忛崑锝夊礄鐠囧弶瀚查柟閿嬬墬鐎氳法鈧懓鑻崣蹇涘触鐏為箖鏁?"""
        # 閺夌偟绻濈紞鍛喆閸℃绂堥柨娑欐皑閸嬶綁宕欑拠鍙夊闁归攱鐗楃€氬潡寮懜鍨函闁哄倹婢橀崯婵嬫偐閺堢數绉撮柛婊冪灱閻撴繈鎮╅張鐢电Т闁汇劌瀚畷鍕偓娑欘殘閸?
        self.axial_view.mouse_clicked.connect(
            lambda x, y: self._on_axial_clicked(x, y)
        )
        self.axial_view.mouse_dragged.connect(
            lambda x, y: self._on_axial_clicked(x, y)  # 闁归攱鐗楃€氬潡寮張鐢电槏閻犲鍟伴弫銈夋儎缁嬫寧鍊遍柛鎴ｅГ閺?
        )
        
        # 闁告劗濮锋慨鍛婃媴瀹ュ牓宕堕幘鍛獥闁绘劗鎳撻崵顕€宕仦鎯х彌闁圭柉濮ゅ鍌炲即鐎涙ɑ鐓€閺夌偟绻濈紞鍛村椽瀹€鈧悡婵嬫偐閺堢數绉撮柣銊ュ瀹曞嫮鈧稒顨堥崵?
        self.coronal_view.mouse_clicked.connect(
            lambda x, y: self._on_coronal_clicked(x, y)
        )
        self.coronal_view.mouse_dragged.connect(
            lambda x, y: self._on_coronal_clicked(x, y)  # 闁归攱鐗楃€氬潡寮張鐢电槏閻犲鍟伴弫銈夋儎缁嬫寧鍊遍柛鎴ｅГ閺?
        )
        
        # 闁活厹鍨绘慨鍛婃媴瀹ュ牓宕堕幘鍛獥闁绘劗鎳撻崵顕€宕仦鎯х彌闁圭柉濮ゅ鍌炲即鐎涙ɑ鐓€閺夌偟绻濈紞鍛村椽鐏炶棄鏆嗛柣妯绘构缂嶅懘鎯冮崟顐㈢）閻庢稒顨堥崵?
        self.sagittal_view.mouse_clicked.connect(
            lambda x, y: self._on_sagittal_clicked(x, y)
        )
        self.sagittal_view.mouse_dragged.connect(
            lambda x, y: self._on_sagittal_clicked(x, y)  # 闁归攱鐗楃€氬潡寮張鐢电槏閻犲鍟伴弫銈夋儎缁嬫寧鍊遍柛鎴ｅГ閺?
        )
        
        # 鍒囧眰鑱斿姩锛堟粴杞?婊戝姩鏉★級
        self.axial_view.slice_changed.connect(self._on_axial_slice_changed)
        self.coronal_view.slice_changed.connect(self._on_coronal_slice_changed)
        self.sagittal_view.slice_changed.connect(self._on_sagittal_slice_changed)

    def _get_spacing_values(self):
        if self.image_spacing is None or len(self.image_spacing) < 3:
            return 1.0, 1.0, 1.0
        try:
            s0 = float(self.image_spacing[0]) if float(self.image_spacing[0]) > 0 else 1.0
            s1 = float(self.image_spacing[1]) if float(self.image_spacing[1]) > 0 else 1.0
            s2 = float(self.image_spacing[2]) if float(self.image_spacing[2]) > 0 else 1.0
            return s0, s1, s2
        except Exception:
            return 1.0, 1.0, 1.0

    def _clip_voxel(self, voxel_x: int, voxel_y: int, voxel_z: int):
        if self.image_data is None:
            return 0, 0, 0
        h, w, d = self.image_data.shape  # [Y, X, Z]
        vx = int(np.clip(int(voxel_x), 0, max(0, w - 1)))
        vy = int(np.clip(int(voxel_y), 0, max(0, h - 1)))
        vz = int(np.clip(int(voxel_z), 0, max(0, d - 1)))
        return vx, vy, vz

    def _current_voxel(self):
        if self.image_data is None:
            return 0, 0, 0
        h, w, d = self.image_data.shape
        if self.crosshair_3d is None or len(self.crosshair_3d) != 3:
            return w // 2, h // 2, d // 2
        return self._clip_voxel(self.crosshair_3d[0], self.crosshair_3d[1], self.crosshair_3d[2])

    def _update_views_from_voxel(self, voxel_x: int, voxel_y: int, voxel_z: int, force: bool = False):
        """Sync all 2D views and 3D crosshair from a voxel coordinate."""
        if self.image_data is None:
            return False
        vx, vy, vz = self._clip_voxel(voxel_x, voxel_y, voxel_z)
        if not force and self.crosshair_3d == [vx, vy, vz]:
            return False
        self.crosshair_3d = [vx, vy, vz]

        h, w, d = self.image_data.shape
        self._syncing_views = True
        try:
            # Axial: slice=Z, col=Y, row=X
            self.axial_view.set_slice(vz, emit_signal=False, redraw=False)
            self.axial_view.crosshair_x = float((h - 1) - vy)
            self.axial_view.crosshair_y = float(vx)
            self.axial_view._update_display()

            # Coronal: slice=Y, col=X, row=Z(reversed due flipud)
            self.coronal_view.set_slice(vy, emit_signal=False, redraw=False)
            self.coronal_view.crosshair_x = float(vx)
            self.coronal_view.crosshair_y = float((d - 1) - vz)
            self.coronal_view._update_display()

            # Sagittal: slice=X, col=Y, row=Z(reversed due flipud)
            self.sagittal_view.set_slice(vx, emit_signal=False, redraw=False)
            self.sagittal_view.crosshair_x = float((h - 1) - vy)
            self.sagittal_view.crosshair_y = float((d - 1) - vz)
            self.sagittal_view._update_display()
        finally:
            self._syncing_views = False

        self._schedule_3d_crosshair_update(vx, vy, vz)
        return True

    def _schedule_3d_crosshair_update(self, voxel_x: int, voxel_y: int, voxel_z: int):
        """Coalesce 3D crosshair refreshes so pointer drags stay responsive."""
        self._pending_3d_crosshair = (int(voxel_x), int(voxel_y), int(voxel_z))
        if self._3d_crosshair_update_pending:
            return
        self._3d_crosshair_update_pending = True
        QTimer.singleShot(0, self._flush_3d_crosshair_update)

    def _flush_3d_crosshair_update(self):
        self._3d_crosshair_update_pending = False
        if self._pending_3d_crosshair is None:
            return
        vx, vy, vz = self._pending_3d_crosshair
        self._pending_3d_crosshair = None
        self.view_3d.set_crosshair_3d(vx, vy, vz)

    def _on_axial_clicked(self, x: float, y: float):
        if self.image_data is None:
            return
        s0, s1, _s2 = self._get_spacing_values()
        col = int(round(float(x) / s1))  # Y
        row = int(round(float(y) / s0))  # X
        h, _w, _d = self.image_data.shape
        voxel_x = row
        voxel_y = (h - 1) - col
        voxel_z = self.axial_view.current_slice
        if self._update_views_from_voxel(voxel_x, voxel_y, voxel_z):
            self.crosshair_changed.emit('axial', x, y)

    def _on_coronal_clicked(self, x: float, y: float):
        if self.image_data is None:
            return
        _s0, s1, s2 = self._get_spacing_values()
        col = int(round(float(x) / s1))  # X
        row = int(round(float(y) / s2))  # Z(reversed)
        _h, _w, d = self.image_data.shape
        voxel_x = col
        voxel_y = self.coronal_view.current_slice
        voxel_z = (d - 1) - row
        if self._update_views_from_voxel(voxel_x, voxel_y, voxel_z):
            self.crosshair_changed.emit('coronal', x, y)

    def _on_sagittal_clicked(self, x: float, y: float):
        if self.image_data is None:
            return
        _s0, s1, s2 = self._get_spacing_values()
        col = int(round(float(x) / s1))  # Y
        row = int(round(float(y) / s2))  # Z(reversed)
        h, _w, d = self.image_data.shape
        voxel_x = self.sagittal_view.current_slice
        voxel_y = (h - 1) - col
        voxel_z = (d - 1) - row
        if self._update_views_from_voxel(voxel_x, voxel_y, voxel_z):
            self.crosshair_changed.emit('sagittal', x, y)

    def _on_axial_slice_changed(self, idx: int):
        if self.image_data is None or self._syncing_views:
            return
        vx, vy, _vz = self._current_voxel()
        self._update_views_from_voxel(vx, vy, int(idx))
        self.slice_changed.emit('axial', int(idx))

    def _on_coronal_slice_changed(self, idx: int):
        if self.image_data is None or self._syncing_views:
            return
        vx, _vy, vz = self._current_voxel()
        self._update_views_from_voxel(vx, int(idx), vz)
        self.slice_changed.emit('coronal', int(idx))

    def _on_sagittal_slice_changed(self, idx: int):
        if self.image_data is None or self._syncing_views:
            return
        _vx, vy, vz = self._current_voxel()
        self._update_views_from_voxel(int(idx), vy, vz)
        self.slice_changed.emit('sagittal', int(idx))
    
    def set_image_data(self, image_data: np.ndarray, spacing=None):
        """
        閻犱礁澧介悿鍡涘炊閹冨壖闁轰胶澧楀畵?
        
        Args:
            image_data: 闁搞儲鍎抽崕姘跺极閻楀牆绁?(H, W, D)
            spacing: 闁稿秴绻掔粈宀勬⒒绾惧鐛╅柨娑樿嫰瑜版煡鏌?
        """
        self.image_data = image_data
        self.image_spacing = spacing  # 濞ｅ洦绻傞悺鈺痯acing濞ｅ洠鍓濇导?
        self.axial_view.set_image_data(image_data, spacing)
        self.coronal_view.set_image_data(image_data, spacing)
        self.sagittal_view.set_image_data(image_data, spacing)
        self.view_3d.set_image_data(image_data, spacing)
        self.set_seed_points(self.seed_points)
        
        # 鍒濆鍖栧埌浣撲腑蹇冿紝骞堕┍鍔ㄥ洓瑙嗗浘鍚屾
        if image_data is not None:
            h, w, d = image_data.shape  # [Y, X, Z]
            center_x = w // 2
            center_y = h // 2
            center_z = d // 2
            self._update_views_from_voxel(center_x, center_y, center_z, force=True)
    
    def set_label_data(self, label_data: np.ndarray):
        """閻犱礁澧介悿鍡涘冀閸モ晠寮悧鍫濈ウ"""
        self.label_data = label_data
        # 2D閻熸瑥妫楀ù妯肩博鐎ｎ亜绁柡鍥х摠閺屽﹪鏁嶉崼锝勬唉闂佹彃绻掓鍥箼瀹ュ嫮绋婇柨?
        self.axial_view.set_label_data(label_data)
        self.coronal_view.set_label_data(label_data)
        self.sagittal_view.set_label_data(label_data)
            # 3D閻熸瑥妫楀ù妯荤▔瀹ュ牆娈伴柛鏂诲妽濞插潡寮敮顔剧闂傚洨鎲版担鐟邦杹闁告柣鍔庨崑锝夊礄缁獤date闁圭顦甸幐鎶芥晬?
        # 闁告瑧绱旈弳鐔煎箲缁辨繃绋夊鍩挎洟宕ｉ幋鐐寸函闁哄倸搴滅槐婵嬫焼閸喖甯抽柛妤嬬節閵?
        self.view_3d.set_label_data(label_data)

    def set_seed_points(self, seed_points):
        """闁告碍鍨崇粭浣圭▔?D閻熸瑥妫楀ù姗€宕ョ仦鍓х矓瀹ュ懐鎽嶉柣鎰秺閻濐喗绂?"""
        self.seed_points = list(seed_points) if seed_points is not None else []
        self.axial_view.set_seed_points(self.seed_points)
        self.coronal_view.set_seed_points(self.seed_points)
        self.sagittal_view.set_seed_points(self.seed_points)
    
    def set_window(self, window_width: float, window_level: float):
        """閻犱礁澧介悿鍡欑玻濡も偓缂佹劖銇炵紞鍛存晬閸喎顣查柡鍫濐槼闁搞儲鎷濈槐?"""
        self.axial_view.set_window(window_width, window_level)
        self.coronal_view.set_window(window_width, window_level)
        self.sagittal_view.set_window(window_width, window_level)
    
    def refresh_3d_view(self):
        """
        闁告帡鏀遍弻?D閻熸瑥妫楀ù?- 闁稿繐褰欓柟鎭掑劚瑜?
        闁告帡鏀遍弻濠囨焻閺勫繒甯嗛柨娑欏哺閸ｆ悂寮弶鍨潱閺夌偠濮ら弳鐔煎箲娓氣偓閸ｅ摜绱斿ù澶愬嫉閹帊鐒︽晶鐣屾偘鐏炶偐绉兼繛鎾冲级閻撳寮寸€涙ɑ鐓€
        """
        if self.view_3d is not None:
            # 闂佹彃绉甸弻濠勬媼閸撗呮瀭闁轰胶澧楀畵渚€鏁嶉崼锛版洟宕ｉ幋鐐寸函闁哄倸搴滅槐?
            if self.image_data is not None:
                self.view_3d.set_image_data(self.image_data, self.image_spacing)
            if self.label_data is not None:
                self.view_3d.set_label_data(self.label_data)
            
            # 闂佹彃绉堕悿鍡涙儎閸涘﹥绨氶悷娆忔
            self.view_3d.reset_camera()
            
            # 闁圭瑳鍡涘礆闁垮鐓€
            self.view_3d.refresh_view()
