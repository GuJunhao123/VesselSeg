#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主界面布局模块 (集成 SQL Server 病例归档功能)
功能：实现界面布局，左侧工具栏，中间视图，右侧参数，以及数据库归档/回溯
"""
from case_data_service import CaseDataService
import numpy as np
import os
import shutil
import tempfile
import pandas as pd
from datetime import datetime
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QMessageBox, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
    QGroupBox, QColorDialog, QComboBox, QProgressDialog,
    QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem, QLineEdit,
    QSizePolicy, QDialog, QFormLayout, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QInputDialog, QTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QFont
from typing import Optional

# === 引入你的自定义模块 ===
from image_io import ImageIO
from visualization import FourViewWidget
from segmentation import ActiveContourSegmentation
from ai_interface import AISegmentationInterface
from database_index import find_patient_dirs_522, build_patient_index_522, SeriesEntry
from excel_reader import ExcelDataReader

# === [新增] 引入数据库管理模块 ===
# 请确保 database.py 和 config.py 在同一目录下
try:
    from database import DatabaseManager
    DB_AVAILABLE = True
except ImportError:
    print("Warning: database.py not found. Archiving features will be disabled.")
    DB_AVAILABLE = False


class SegmentationWorker(QThread):
    """分割算法工作线程 - 避免界面冻结"""
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    
    def __init__(self, algorithm, image_data, seed_points, params):
        super().__init__()
        self.algorithm = algorithm
        self.image_data = image_data
        self.seed_points = seed_points
        self.params = params
    
    def run(self):
        try:
            self.algorithm.set_image(self.image_data)
            self.algorithm.set_seed_points(self.seed_points)
            
            if self.params['method'] == 'region_competition':
                mask = self.algorithm.region_competition(
                    iterations=self.params['iterations'],
                    smoothness=self.params['smoothness'],
                    expansion_force=self.params['expansion_force']
                )
            else:  # edge_based
                mask = self.algorithm.edge_based(
                    iterations=self.params['iterations'],
                    smoothness=self.params['smoothness'],
                    edge_weight=self.params['edge_weight']
                )
            
            self.finished.emit(mask)
        except Exception as e:
            self.error.emit(str(e))


class PatientIndexWorker(QThread):
    """单病人索引扫描线程（懒加载）"""
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, patient_dir: str):
        super().__init__()
        self.patient_dir = patient_dir

    def run(self):
        try:
            entries = build_patient_index_522(
                self.patient_dir,
                stop_checker=self.isInterruptionRequested,
            )
            self.finished.emit(entries)
        except Exception as e:
            self.error.emit(str(e))


class CaseHistoryDialog(QDialog):
    """[新增] 病例历史记录搜索弹窗"""
    case_selected = pyqtSignal(str, str) # signal(original_path, mask_path)

    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db = db_manager
        self.setWindowTitle("病例历史档案 (SQL Server)")
        self.setMinimumSize(600, 400)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        
        # 搜索区
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("搜索 (姓名/编号):"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("输入关键词回车搜索...")
        self.search_input.returnPressed.connect(self._on_search)
        search_layout.addWidget(self.search_input)
        
        btn_search = QPushButton("搜索")
        btn_search.clicked.connect(self._on_search)
        search_layout.addWidget(btn_search)
        layout.addLayout(search_layout)

        # 列表区
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.list_widget)

        # 提示
        layout.addWidget(QLabel("提示: 双击列表项以加载历史病例"))
        
        self.setLayout(layout)
        # 初始加载所有记录
        self._on_search()

    def _on_search(self):
        keyword = self.search_input.text().strip()
        self.list_widget.clear()
        
        try:
            # 调用 database.py 中的搜索方法
            # 如果 database.py 没有 get_patient_history 方法，请确保添加它
            if hasattr(self.db, 'get_patient_history'):
                records = self.db.get_patient_history(keyword)
            else:
                # 兼容旧版 database.py 的 get_all_records
                records = self.db.get_all_records() # 假设这是返回所有记录
                if keyword:
                    records = [r for r in records if keyword in str(r.get('patient_name', '')) or keyword in str(r.get('patient_code', ''))]

            if not records:
                self.list_widget.addItem("未找到相关记录")
                return

            for row in records:
                # 兼容字典或Row对象
                p_name = row.get('patient_name') or "未知"
                p_code = row.get('patient_code') or "N/A"
                c_date = row.get('created_at')
                note = row.get('diagnosis_note') or ""
                
                if isinstance(c_date, str):
                    date_str = c_date
                elif c_date:
                    date_str = c_date.strftime("%Y-%m-%d %H:%M")
                else:
                    date_str = "未知时间"

                display_text = f"[{date_str}] {p_name} ({p_code}) - {note}"
                
                item = QListWidgetItem(display_text)
                # 存储路径数据
                item.setData(Qt.UserRole, row.get('original_path'))
                item.setData(Qt.UserRole + 1, row.get('mask_path'))
                self.list_widget.addItem(item)
                
        except Exception as e:
            self.list_widget.addItem(f"查询出错: {e}")

    def _on_item_double_clicked(self, item):
        orig_path = item.data(Qt.UserRole)
        mask_path = item.data(Qt.UserRole + 1)
        
        if orig_path and os.path.exists(orig_path):
            self.case_selected.emit(orig_path, mask_path)
            self.accept()
        else:
            QMessageBox.warning(self, "错误", f"原始文件不存在:\n{orig_path}")


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("医学影像分割软件")
        self.setGeometry(100, 100, 1600, 1000)
        
        # 初始化模块
        self.image_io = ImageIO()
        self.segmentation = ActiveContourSegmentation()
        self.ai_interface = AISegmentationInterface()
        
        # === [新增] 初始化数据库管理器 ===
        self.db = None
        if DB_AVAILABLE:
            try:
                self.db = DatabaseManager()
                print("DatabaseManager initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize DatabaseManager: {e}")
        
        # 初始化数据服务
        self.data_service = CaseDataService()

        # 数据存储
        self.current_image_data = None
        self.current_label_data = None
        self.current_metadata = None
        self.current_image_path = None  # 当前图像文件路径（用于显示Layer名称）
        self.current_entry = None  # 当前序列的SeriesEntry（如果是从数据库加载的）
        self.seed_points = []
        self.label_list = []  # 标签列表
        self.active_label_id = 1  # 活动标签ID

        # 当前病人/检查/序列信息（从DICOM或数据库获取）
        self.current_patient_info = {}

        # 保存上次选择的数据库路径
        self.last_db_root = None

        # 撤销/重做历史
        self.undo_history = []  # 撤销历史栈
        self.redo_history = []  # 重做历史栈
        self.max_history_size = 50  # 最大历史记录数
        
        # UI初始化
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI布局"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        central_widget.setLayout(main_layout)
        
        # ========== 左侧：工具栏区（4个模块） ==========
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, stretch=0)
        
        # ========== 中间：2×2视图区 ==========
        center_panel = self._create_center_panel()
        main_layout.addWidget(center_panel, stretch=4)
        
        # ========== 右侧：参数设置区 ==========
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, stretch=1)
    
    def _create_left_panel(self) -> QWidget:
        """创建左侧工具栏（4个模块）"""
        panel = QWidget()
        # 左侧面板加宽一些，给数据库树和按钮留出可见空间
        panel.setMinimumWidth(400)
        panel.setMaximumWidth(500)
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(5)
        panel.setLayout(layout)
        
        # ===== 1. Main Toolbar =====
        main_toolbar_group = QGroupBox("Main Toolbar")
        main_toolbar_layout = QVBoxLayout()
        main_toolbar_layout.setSpacing(3)
        
        # 工具按钮组
        tools_layout = QHBoxLayout()
        tools_layout.setSpacing(2)
        
        # Pointer（指针工具）
        self.btn_pointer = QPushButton("↖")
        self.btn_pointer.setToolTip("Pointer Tool")
        self.btn_pointer.setCheckable(True)
        self.btn_pointer.setChecked(True)
        self.btn_pointer.setFixedSize(30, 30)
        self.btn_pointer.clicked.connect(lambda: self._on_tool_selected('pointer'))
        tools_layout.addWidget(self.btn_pointer)
        
        # Zoom（缩放工具）
        self.btn_zoom = QPushButton("🔍")
        self.btn_zoom.setToolTip("Zoom Tool")
        self.btn_zoom.setCheckable(True)
        self.btn_zoom.setFixedSize(30, 30)
        self.btn_zoom.clicked.connect(lambda: self._on_tool_selected('zoom'))
        tools_layout.addWidget(self.btn_zoom)
        
        # Pan（平移工具）
        self.btn_pan = QPushButton("✋")
        self.btn_pan.setToolTip("Pan Tool")
        self.btn_pan.setCheckable(True)
        self.btn_pan.setFixedSize(30, 30)
        self.btn_pan.clicked.connect(lambda: self._on_tool_selected('pan'))
        tools_layout.addWidget(self.btn_pan)
        
        # Brush（画笔工具）
        self.btn_brush = QPushButton("🖌")
        self.btn_brush.setToolTip("Brush Tool")
        self.btn_brush.setCheckable(True)
        self.btn_brush.setFixedSize(30, 30)
        self.btn_brush.clicked.connect(lambda: self._on_tool_selected('brush'))
        tools_layout.addWidget(self.btn_brush)
        
        # Eraser（橡皮擦工具）
        self.btn_eraser = QPushButton("🧹")
        self.btn_eraser.setToolTip("Eraser Tool")
        self.btn_eraser.setCheckable(True)
        self.btn_eraser.setFixedSize(30, 30)
        self.btn_eraser.clicked.connect(lambda: self._on_tool_selected('eraser'))
        tools_layout.addWidget(self.btn_eraser)
        
        # Save Screenshot（保存截图）
        self.btn_screenshot = QPushButton("📷")
        self.btn_screenshot.setToolTip("Save Screenshot")
        self.btn_screenshot.setFixedSize(30, 30)
        self.btn_screenshot.clicked.connect(self._on_save_screenshot)
        tools_layout.addWidget(self.btn_screenshot)
        
        main_toolbar_layout.addLayout(tools_layout)
        
        # 画笔大小设置
        brush_size_layout = QHBoxLayout()
        brush_size_layout.addWidget(QLabel("Brush Size:"))
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setMinimum(1)
        self.brush_size_spin.setMaximum(50)
        self.brush_size_spin.setValue(3)
        self.brush_size_spin.valueChanged.connect(self._on_brush_size_changed)
        brush_size_layout.addWidget(self.brush_size_spin)
        main_toolbar_layout.addLayout(brush_size_layout)
        
        # 撤销/重做按钮
        undo_redo_layout = QHBoxLayout()
        undo_redo_layout.setSpacing(2)
        self.btn_undo = QPushButton("↶")
        self.btn_undo.setToolTip("Undo (Ctrl+Z)")
        self.btn_undo.setFixedSize(30, 30)
        self.btn_undo.clicked.connect(self._on_undo)
        undo_redo_layout.addWidget(self.btn_undo)
        
        self.btn_redo = QPushButton("↷")
        self.btn_redo.setToolTip("Redo (Ctrl+Y)")
        self.btn_redo.setFixedSize(30, 30)
        self.btn_redo.clicked.connect(self._on_redo)
        undo_redo_layout.addWidget(self.btn_redo)
        main_toolbar_layout.addLayout(undo_redo_layout)
        
        # 文件操作按钮
        btn_load_image = QPushButton("Load Image")
        btn_load_image.clicked.connect(self._on_load_image)
        main_toolbar_layout.addWidget(btn_load_image)

        # 新增：加载DICOM文件夹按钮（支持无后缀DICOM序列）
        btn_load_dicom_folder = QPushButton("Load DICOM Folder")
        btn_load_dicom_folder.clicked.connect(self._on_load_dicom_folder)
        main_toolbar_layout.addWidget(btn_load_dicom_folder)
        
        btn_load_label = QPushButton("Load Label")
        btn_load_label.clicked.connect(self._on_load_label)
        main_toolbar_layout.addWidget(btn_load_label)
        
        btn_save_label = QPushButton("Save Label")
        btn_save_label.clicked.connect(self._on_save_label)
        main_toolbar_layout.addWidget(btn_save_label)
        
        main_toolbar_group.setLayout(main_toolbar_layout)
        layout.addWidget(main_toolbar_group)
        
        # 当前工具模式
        self.current_tool_mode = 'pointer'
        # 种子点模式标志
        self.seed_mode_enabled = False
        
        # ===== 2. Cursor Inspector =====
        cursor_group = QGroupBox("Cursor Inspector")
        cursor_layout = QVBoxLayout()
        cursor_layout.setSpacing(3)
        
        # Cursor position
        self.cursor_pos_label = QLabel("Cursor position (x,y,z): --")
        cursor_layout.addWidget(self.cursor_pos_label)
        
        # Intensity under cursor
        self.cursor_intensity_label = QLabel("Intensity under cursor:")
        self.cursor_layer_label = QLabel("  Layer: --")
        self.cursor_intensity_value_label = QLabel("  Intensity: --")
        cursor_layout.addWidget(self.cursor_intensity_label)
        cursor_layout.addWidget(self.cursor_layer_label)
        cursor_layout.addWidget(self.cursor_intensity_value_label)
        
        # Label under cursor
        self.cursor_label_label = QLabel("Label under cursor:")
        self.cursor_label_value_label = QLabel("  Label: --")
        self.btn_clear_label = QPushButton("Clear Label")
        self.btn_clear_label.setMaximumHeight(25)
        self.btn_clear_label.clicked.connect(self._on_clear_label)
        cursor_layout.addWidget(self.cursor_label_label)
        cursor_layout.addWidget(self.cursor_label_value_label)
        cursor_layout.addWidget(self.btn_clear_label)
        
        cursor_group.setLayout(cursor_layout)
        layout.addWidget(cursor_group)

        # ===== 2.5 Patient Info =====
        patient_group = QGroupBox("Patient Info")
        patient_layout = QVBoxLayout()
        patient_layout.setSpacing(2)

        self.pi_id_label = QLabel("ID: --")
        self.pi_name_label = QLabel("Name: --")
        self.pi_sex_label = QLabel("Sex: --")
        self.pi_birth_label = QLabel("Birth: --")
        self.pi_study_date_label = QLabel("Study Date: --")
        self.pi_modality_label = QLabel("Modality: --")
        self.pi_series_label = QLabel("Series: --")

        for w in [
            self.pi_id_label,
            self.pi_name_label,
            self.pi_sex_label,
            self.pi_birth_label,
            self.pi_study_date_label,
            self.pi_modality_label,
            self.pi_series_label,
        ]:
            w.setMinimumWidth(200)
            patient_layout.addWidget(w)

        patient_group.setLayout(patient_layout)
        layout.addWidget(patient_group)
        
        # ===== 3. Segmentation Labels =====
        labels_group = QGroupBox("Segmentation Labels")
        labels_layout = QVBoxLayout()
        labels_layout.setSpacing(3)
        
        # Active label（活动标签）
        active_label_layout = QHBoxLayout()
        active_label_layout.setSpacing(5)
        self.active_label_color_widget = QWidget()
        self.active_label_color_widget.setFixedSize(20, 20)
        self.active_label_color_widget.setStyleSheet("background-color: red; border: 1px solid black;")
        active_label_layout.addWidget(self.active_label_color_widget)
        self.active_label_name_label = QLabel("Active label: Label 1")
        active_label_layout.addWidget(self.active_label_name_label)
        active_label_layout.addStretch()
        labels_layout.addLayout(active_label_layout)
        
        # Paint over（绘制覆盖标签）
        self.paint_over_label = QLabel("Paint over: Label 1")
        labels_layout.addWidget(self.paint_over_label)
        
        # Overall label opacity（标签透明度滑块）
        opacity_layout = QHBoxLayout()
        opacity_layout.setSpacing(5)
        opacity_layout.addWidget(QLabel("Overall label opacity:"))
        self.label_opacity_slider = QSlider(Qt.Horizontal)
        self.label_opacity_slider.setMinimum(0)
        self.label_opacity_slider.setMaximum(100)
        self.label_opacity_slider.setValue(50)
        self.label_opacity_slider.valueChanged.connect(self._on_label_opacity_changed)
        opacity_layout.addWidget(self.label_opacity_slider)
        self.label_opacity_value_label = QLabel("50")
        self.label_opacity_value_label.setMinimumWidth(30)
        opacity_layout.addWidget(self.label_opacity_value_label)
        labels_layout.addLayout(opacity_layout)
        
        # 标签列表
        self.label_list_widget = QListWidget()
        self.label_list_widget.setMaximumHeight(100)
        self.label_list_widget.itemSelectionChanged.connect(self._on_label_selection_changed)
        labels_layout.addWidget(self.label_list_widget)
        
        # 添加标签按钮
        btn_add_label = QPushButton("Add Label")
        btn_add_label.clicked.connect(self._on_add_label)
        labels_layout.addWidget(btn_add_label)
        
        # 删除标签按钮
        btn_delete_label = QPushButton("Delete Label")
        btn_delete_label.clicked.connect(self._on_delete_label)
        labels_layout.addWidget(btn_delete_label)
        
        # 标签颜色按钮
        btn_label_color = QPushButton("Label Color")
        btn_label_color.clicked.connect(self._on_label_color)
        labels_layout.addWidget(btn_label_color)
        
        labels_group.setLayout(labels_layout)
        layout.addWidget(labels_group)
        
        # 初始化活动标签
        self.active_label_id = 1
        
        # ===== 4. 3D操作按钮 =====
        toolbar_3d_group = QGroupBox("3D操作")
        toolbar_3d_layout = QVBoxLayout()
        toolbar_3d_layout.setSpacing(3)
        
        # Update（更新3D视图）
        btn_update_3d = QPushButton("🔄 Update")
        btn_update_3d.setToolTip("Update 3D View")
        btn_update_3d.setFixedSize(120, 30)
        btn_update_3d.clicked.connect(self._on_update_3d_view_clicked)
        toolbar_3d_layout.addWidget(btn_update_3d)
        
        # 种子点相关按钮
        seed_layout = QHBoxLayout()
        self.btn_add_seed = QPushButton("Add Seed")
        self.btn_add_seed.setFixedSize(120, 30)
        self.btn_add_seed.setCheckable(True)
        self.btn_add_seed.setToolTip("Click then add seeds by clicking in views")
        self.btn_add_seed.clicked.connect(self._on_toggle_seed_mode)
        seed_layout.addWidget(self.btn_add_seed)
        
        self.btn_clear_seeds = QPushButton("Clear Seeds")
        self.btn_clear_seeds.setFixedSize(120, 30)
        self.btn_clear_seeds.setToolTip("Clear all seed points")
        self.btn_clear_seeds.clicked.connect(self._on_clear_seeds)
        seed_layout.addWidget(self.btn_clear_seeds)
        toolbar_3d_layout.addLayout(seed_layout)
        
        # 种子点数量显示
        self.seed_count_label = QLabel("Seeds: 0")
        self.seed_count_label.setToolTip("Current seed count")
        toolbar_3d_layout.addWidget(self.seed_count_label)
        
        # 分割相关按钮
        btn_segment_3d = QPushButton("Segment 3D")
        btn_segment_3d.clicked.connect(self._on_segment_3d)
        toolbar_3d_layout.addWidget(btn_segment_3d)
        
        # AI自动分割按钮
        btn_ai_seg = QPushButton("AI自动分割")
        btn_ai_seg.setFixedSize(120, 30)
        btn_ai_seg.clicked.connect(self._on_ai_segmentation)
        toolbar_3d_layout.addWidget(btn_ai_seg)
        
        toolbar_3d_group.setLayout(toolbar_3d_layout)
        layout.addWidget(toolbar_3d_group)

        # ===== 5. 数据库对接（522范式） - 现在以弹窗形式打开 =====
        btn_open_db_dialog = QPushButton("打开数据库 (522)")
        btn_open_db_dialog.setToolTip("打开数据库选择弹窗以浏览和加载序列")
        btn_open_db_dialog.clicked.connect(self._on_open_db_dialog)
        layout.addWidget(btn_open_db_dialog)
        
        # ===== 6. [新增] 病例归档/PACS系统 =====
        archive_group = QGroupBox("病例归档 (PACS)")
        archive_layout = QVBoxLayout()
        
        self.btn_save_case_db = QPushButton("保存病例到库")
        self.btn_save_case_db.setToolTip("将当前影像及分割结果归档到数据库")
        self.btn_save_case_db.clicked.connect(self._on_save_case_to_db)
        archive_layout.addWidget(self.btn_save_case_db)
        
        self.btn_search_case_db = QPushButton("搜索历史病例")
        self.btn_search_case_db.setToolTip("搜索数据库中的历史分割记录并回溯")
        self.btn_search_case_db.clicked.connect(self._on_search_case_history)
        archive_layout.addWidget(self.btn_search_case_db)
        
        archive_group.setLayout(archive_layout)
        layout.addWidget(archive_group)
        
        layout.addStretch()
        
        return panel
    
    def _create_center_panel(self) -> QWidget:
        """创建中间2×2视图区"""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        panel.setLayout(layout)
        
        # 创建2×2视图组件
        self.four_view = FourViewWidget()
        self.four_view.crosshair_changed.connect(self._on_crosshair_changed)
        self.four_view.slice_changed.connect(self._on_slice_changed)
        layout.addWidget(self.four_view)
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """创建右侧参数设置区（只保留分割算法参数）"""
        panel = QWidget()
        panel.setMaximumWidth(220)
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(5)
        panel.setLayout(layout)
        
        # 分割算法参数组
        algo_group = QGroupBox("分割算法参数")
        algo_layout = QVBoxLayout()
        algo_layout.setSpacing(5)
        
        # 算法选择
        algo_layout.addWidget(QLabel("算法类型:"))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["Region Competition", "Edge-Based"])
        algo_layout.addWidget(self.algo_combo)
        
        # 迭代次数
        algo_layout.addWidget(QLabel("迭代次数:"))
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setMinimum(10)
        self.iterations_spin.setMaximum(1000)
        self.iterations_spin.setValue(100)
        algo_layout.addWidget(self.iterations_spin)
        
        # 平滑度
        algo_layout.addWidget(QLabel("平滑度:"))
        self.smoothness_spin = QDoubleSpinBox()
        self.smoothness_spin.setMinimum(0.1)
        self.smoothness_spin.setMaximum(10.0)
        self.smoothness_spin.setValue(1.0)
        self.smoothness_spin.setSingleStep(0.1)
        algo_layout.addWidget(self.smoothness_spin)
        
        # 扩张力（仅Region Competition）
        algo_layout.addWidget(QLabel("扩张力:"))
        self.expansion_spin = QDoubleSpinBox()
        self.expansion_spin.setMinimum(-5.0)
        self.expansion_spin.setMaximum(5.0)
        self.expansion_spin.setValue(1.0)
        self.expansion_spin.setSingleStep(0.1)
        algo_layout.addWidget(self.expansion_spin)
        
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)
        
        layout.addStretch()
        
        return panel
    
    # ========== 文件操作槽函数 ==========
    
    def _on_load_image(self):
        """加载影像"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "加载影像", "",
            "NIfTI文件 (*.nii *.nii.gz);;DICOM文件 (*.dcm);;NRRD文件 (*.nrrd);;所有文件 (*.*)"
        )
        
        if filepath:
            try:
                data, metadata = self.image_io.load_image(filepath)
                if data is not None:
                    self.current_image_data = data
                    self.current_metadata = metadata
                    self.current_image_path = filepath  # 保存文件路径
                    # 尝试从metadata构建entry（用于后续自动定位）
                    self.current_entry = self._build_entry_from_metadata(metadata, filepath)
                    
                    # 传递spacing信息（如果可用）
                    spacing = metadata.get('spacing') if metadata else None
                    self.four_view.set_image_data(data, spacing)
                    
                    # 初始化标签数据
                    self.current_label_data = np.zeros(data.shape, dtype=np.uint8)
                    self.four_view.set_label_data(self.current_label_data)
                    # 设置活动标签ID
                    self.four_view.axial_view.set_active_label_id(self.active_label_id)
                    self.four_view.coronal_view.set_active_label_id(self.active_label_id)
                    self.four_view.sagittal_view.set_active_label_id(self.active_label_id)
                    # 清空历史记录
                    self.undo_history.clear()
                    self.redo_history.clear()
                    # 清空种子点
                    self.seed_points = []
                    self._update_seed_count()
                    # 关闭种子点模式
                    self.seed_mode_enabled = False
                    self.btn_add_seed.setChecked(False)
                    self.btn_add_seed.setText("Add Seed")
                    self.btn_add_seed.setStyleSheet("")

                    # 更新病人信息显示（若有DICOM信息则展示）
                    # 如果构建了entry，使用entry信息（更完整）；否则只使用metadata
                    if self.current_entry:
                        self._update_patient_info_from_entry(self.current_entry, metadata)
                    else:
                        self._update_patient_info_from_metadata(metadata)
                    
                    QMessageBox.information(self, "成功", f"影像加载成功\n尺寸: {data.shape}")
                else:
                    QMessageBox.warning(self, "错误", "影像加载失败")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载影像时发生错误:\n{e}")

    # ========== [新增] 病例归档相关函数 ==========

    def _on_save_case_to_db(self):
        """保存病例到数据库（归档）"""
        if not self.db:
            QMessageBox.warning(self, "错误", "数据库连接未初始化，无法保存。")
            return

        if self.current_image_data is None:
            QMessageBox.warning(self, "提示", "请先加载并处理一张影像")
            return

        # 1. 确认病人ID和姓名
        p_id = self.current_patient_info.get("PatientID", "")
        p_name = self.current_patient_info.get("PatientName", "")
        
        # 如果自动提取为空，让用户输入
        if not p_id or p_id == "--" or p_id == "Unknown":
            text, ok = QInputDialog.getText(self, "输入病人编号", "无法自动识别病人编号，请输入:")
            if ok and text:
                p_id = text
            else:
                return # 用户取消
        
        if not p_name or p_name == "--" or p_name == "Unknown":
            # 尝试去数据库查名字
            if len(p_id) > 2:
                # 这是一个简单的尝试，如果数据库支持根据ID查名字
                try:
                    conn = self.db.get_connection()
                    if conn:
                        cur = conn.cursor()
                        cur.execute("SELECT name FROM patient_info WHERE patient_code = ?", (p_id,))
                        res = cur.fetchone()
                        if res:
                            p_name = res[0]
                        conn.close()
                except:
                    pass
            
            if not p_name or p_name == "Unknown":
                text, ok = QInputDialog.getText(self, "输入病人姓名", f"请输入病人 {p_id} 的姓名:")
                if ok and text:
                    p_name = text
                else:
                    p_name = "Unknown"

        # 2. 获取备注
        note, ok = QInputDialog.getText(self, "诊断备注", "请输入诊断备注 (可选):")
        if not ok:
            note = ""

        # 3. 保存Mask到临时文件
        # 必须先把内存里的 mask 存成文件，DatabaseManager 才能归档它
        try:
            temp_dir = tempfile.gettempdir()
            temp_mask_path = os.path.join(temp_dir, f"temp_mask_{datetime.now().strftime('%H%M%S')}.nii.gz")
            
            # 使用 ImageIO 保存 mask
            # 注意：mask 数据必须是整数类型
            mask_data = self.current_label_data.astype(np.uint8) if self.current_label_data is not None else np.zeros_like(self.current_image_data, dtype=np.uint8)
            
            self.image_io.save_label_map(mask_data, temp_mask_path, self.current_metadata)
            
            # 4. 调用数据库保存
            # 注意：current_image_path 必须存在。如果是从 DICOM 文件夹加载的，它是一个目录路径。
            # DatabaseManager 的 save_record 通常期望一个文件路径用于复制。
            # 如果是目录，shutil.copy2 会报错。我们需要特殊处理。
            
            img_path_to_save = self.current_image_path
            is_folder = os.path.isdir(img_path_to_save)
            
            if is_folder:
                # 如果是文件夹（DICOM序列），我们可能需要压缩它或者只记录路径
                # 为了简单起见，如果 DatabaseManager 只是复制文件，我们这里暂不支持文件夹归档的物理复制
                # 或者，我们可以把整个文件夹打包。
                # 这里的简单策略：提示用户，DICOM文件夹模式仅支持记录路径，不支持物理备份（除非修改 database.py）
                # 为了兼容性，我们只传路径字符串，让 database.py 决定怎么处理 (如果它只存路径不复制，那就没问题)
                pass 

            print(f"Archiving: {p_id}, {p_name}")
            
            success = self.db.save_record(
                p_code=p_id,
                p_name=p_name,
                current_img_path=img_path_to_save,
                mask_save_path=temp_mask_path,
                note=note
            )
            
            if success:
                QMessageBox.information(self, "成功", f"病例 {p_name} 已成功归档！")
            else:
                QMessageBox.warning(self, "失败", "归档失败，请检查日志或数据库连接。")
                
            # 清理临时文件
            if os.path.exists(temp_mask_path):
                os.remove(temp_mask_path)
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存过程中发生错误:\n{e}")

    def _on_search_case_history(self):
        """搜索历史病例"""
        if not self.db:
            QMessageBox.warning(self, "错误", "数据库连接未初始化。")
            return
            
        dialog = CaseHistoryDialog(self.db, self)
        dialog.case_selected.connect(self._on_history_case_loaded)
        dialog.exec_()

    def _on_history_case_loaded(self, original_path, mask_path):
        """从历史记录加载病例的回调"""
        print(f"Loading history case: {original_path}, {mask_path}")
        
        try:
            # 1. 加载原图
            # 判断原图是文件还是文件夹 (DICOM)
            if os.path.isdir(original_path):
                # 是 DICOM 文件夹
                data, metadata = self.image_io.load_image(original_path)
            else:
                # 是单个文件 (nii.gz)
                data, metadata = self.image_io.load_image(original_path)
            
            if data is None:
                QMessageBox.warning(self, "错误", "无法加载原始影像文件")
                return
                
            self.current_image_data = data
            self.current_metadata = metadata
            self.current_image_path = original_path
            
            # 传递 spacing
            spacing = metadata.get('spacing') if metadata else None
            self.four_view.set_image_data(data, spacing)
            
            # 2. 加载 Mask
            if os.path.exists(mask_path):
                mask_data = self.image_io.load_label_map(mask_path)
                if mask_data is not None:
                    # 确保尺寸匹配
                    if mask_data.shape == data.shape:
                        self.current_label_data = mask_data
                    else:
                        print(f"Warning: Mask shape {mask_data.shape} mismatch image {data.shape}")
                        self.current_label_data = np.zeros(data.shape, dtype=np.uint8)
                else:
                    self.current_label_data = np.zeros(data.shape, dtype=np.uint8)
            else:
                self.current_label_data = np.zeros(data.shape, dtype=np.uint8)
            
            self.four_view.set_label_data(self.current_label_data)
            
            # 更新信息显示
            self._update_patient_info_from_metadata(metadata)
            
            # 从路径中获取 p_code，或者从搜索结果中获取
            p_code = self.get_current_patient_code() 
            
            # 触发数据库联动展示
            self.refresh_patient_tables(p_code)
            
            QMessageBox.information(self, "成功", "历史病例加载完成")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载历史病例失败:\n{e}")

    def _on_connect_database(self):
        """
        连接 522 数据库根目录（通常选择 522/data）

        改为“懒加载”策略：这里只列出病人目录，不做全库扫描。
        """
        folder_path = QFileDialog.getExistingDirectory(
            self, "选择数据库根目录（建议选择 522/data）", ""
        )
        if not folder_path:
            return

        self.db_root = folder_path
        self.db_series_list.clear()
        self.db_entries = []

        patient_dirs = find_patient_dirs_522(folder_path)
        for pdir in patient_dirs:
            pid = os.path.basename(pdir)
            p_item = QTreeWidgetItem()
            p_item.setText(0, pid)
            p_item.setToolTip(0, f"Patient dir: {pdir}\n展开以加载该病人的序列列表")
            # UserRole 中存 patient_dir，并标记“未加载”
            p_item.setData(0, Qt.UserRole, {"type": "patient", "patient_dir": pdir, "loaded": False})
            # 放一个占位子节点，让 UI 显示可展开箭头
            p_item.addChild(QTreeWidgetItem(["(expand to load...)"]))
            self.db_series_list.addTopLevelItem(p_item)

        self.db_status_label.setText(f"DB: connected ({len(patient_dirs)} patients)\n{folder_path}")
        self.db_series_list.expandToDepth(0)

    def _on_db_item_expanded(self, item: QTreeWidgetItem):
        """懒加载：展开病人节点时扫描该病人的序列（SeriesUID粒度）"""
        payload = item.data(0, Qt.UserRole)
        if not isinstance(payload, dict):
            return
        if payload.get("type") != "patient":
            return
        if payload.get("loaded"):
            return

        patient_dir = payload.get("patient_dir")
        if not patient_dir:
            return

        # 清掉占位
        item.takeChildren()

        # 若上一个病人扫描还在跑，先请求中断
        if self.patient_worker is not None and self.patient_worker.isRunning():
            self.patient_worker.requestInterruption()

        progress = QProgressDialog(f"正在扫描该病人序列...\n{os.path.basename(patient_dir)}", "取消", 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        self.patient_worker = PatientIndexWorker(patient_dir)

        def _done(entries):
            progress.close()
            # 直接将序列作为患者节点的子节点（不显示Study层级）
            for e in entries:
                leaf = QTreeWidgetItem()
                # 第0列：Patient ID（使用序列ID或描述）
                series_desc = (e.series_description or "").strip()
                uid_short = (e.series_uid or "")[-8:] if e.series_uid else ""
                series_text = f"{e.series_id or ''} {series_desc} ({uid_short})".strip()
                if not series_text:
                    series_text = os.path.basename(e.series_dir) if e.series_dir else "Series"
                leaf.setText(0, series_text)
                # 第1列：Patient Name
                leaf.setText(1, e.patient_name or "")
                # 第2列：Study ID（保留信息但不作为层级）
                leaf.setText(2, e.study_id or "")
                # 第3列：Study Date
                leaf.setText(3, e.study_date or "")
                # 第4列：Modality
                leaf.setText(4, e.modality or "")
                # 第5列：Series Description（详细信息）
                leaf.setText(5, series_text)
                leaf.setData(0, Qt.UserRole, e)
                leaf.setToolTip(0, e.tooltip_text)
                item.addChild(leaf)

            payload["loaded"] = True
            item.setData(0, Qt.UserRole, payload)
            item.setExpanded(True)

        def _err(err):
            progress.close()
            QMessageBox.critical(self, "错误", f"扫描该病人序列失败:\n{err}")

        self.patient_worker.finished.connect(_done)
        self.patient_worker.error.connect(_err)
        progress.canceled.connect(self.patient_worker.requestInterruption)
        self.patient_worker.start()

    def _on_view_patient_info(self):
        """弹窗查看当前图像的病人信息"""
        info = self.current_patient_info or {}
        if not info:
            QMessageBox.information(self, "信息", "当前没有已加载图像的病人信息。\n请先加载一个DICOM序列/图像。")
            return

        dialog = PatientInfoDialog(info, self)
        dialog.exec_()

    def _on_load_dicom_folder(self):
        """
        加载DICOM文件夹
        
        说明：
        - 用于处理类似 522/data/.../ST*/SE*/IM* 这种「无后缀」DICOM 文件结构
        - 现在支持选择上层目录（病人/Study），会自动向下递归找到第一个可加载的 SE 目录
        """
        folder_path = QFileDialog.getExistingDirectory(
            self, "选择DICOM文件夹", ""
        )

        if not folder_path:
            return

        # 尝试直接加载选中目录，如果失败则向下递归查找可加载的 SE 目录
        target_dir = None
        data = None
        metadata = None

        def try_load(dirpath: str):
            try:
                d, m = self.image_io.load_image(dirpath)
                return d, m
            except Exception:
                return None, None

        data, metadata = try_load(folder_path)
        if data is None:
            for dirpath, _dirnames, _filenames in os.walk(folder_path):
                data, metadata = try_load(dirpath)
                if data is not None:
                    target_dir = dirpath
                    break

        if data is None:
            QMessageBox.warning(self, "错误", "未找到可加载的DICOM序列\n请选中具体的 SE 文件夹或包含 DICOM 的目录。")
            return

        # 如果找到了递归目录但未设置 target_dir，说明就是选中的那层
        if target_dir is None:
            target_dir = folder_path

        try:
            self.current_image_data = data
            self.current_metadata = metadata
            self.current_image_path = target_dir
            # 尝试从metadata构建entry（用于后续自动定位）
            self.current_entry = self._build_entry_from_metadata(metadata, target_dir)

            # 传递spacing信息（如果可用）
            spacing = metadata.get('spacing') if metadata else None
            self.four_view.set_image_data(data, spacing)
            # 初始化标签数据
            self.current_label_data = np.zeros(data.shape, dtype=np.uint8)
            self.four_view.set_label_data(self.current_label_data)
            # 设置活动标签ID
            self.four_view.axial_view.set_active_label_id(self.active_label_id)
            self.four_view.coronal_view.set_active_label_id(self.active_label_id)
            self.four_view.sagittal_view.set_active_label_id(self.active_label_id)
            # 清空历史记录和种子点，与普通加载逻辑保持一致
            self.undo_history.clear()
            self.redo_history.clear()
            self.seed_points = []
            self._update_seed_count()
            self.seed_mode_enabled = False
            self.btn_add_seed.setChecked(False)
            self.btn_add_seed.setText("Add Seed")
            self.btn_add_seed.setStyleSheet("")
            # 更新病人信息显示
            # 如果构建了entry，使用entry信息（更完整）；否则只使用metadata
            if self.current_entry:
                self._update_patient_info_from_entry(self.current_entry, metadata)
            else:
                self._update_patient_info_from_metadata(metadata)

            QMessageBox.information(self, "成功", f"DICOM文件夹加载成功\n实际加载目录: {target_dir}\n尺寸: {data.shape}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载DICOM文件夹时发生错误:\n{e}")

    def _on_load_selected_db_series(self, *_args):
        """加载数据库列表中选中的序列"""
        item = self.db_series_list.currentItem()
        if not item:
            QMessageBox.information(self, "提示", "请先在列表中选择一个序列")
            return

        # QTreeWidgetItem: 数据存放在第0列的 UserRole 中
        entry = item.data(0, Qt.UserRole)
        # 只有叶子节点才会绑定 entry；点到病人/检查节点时给出提示
        if not entry or not hasattr(entry, "file_list"):
            QMessageBox.information(self, "提示", "请展开并选择具体序列（叶子节点）再加载")
            return

        try:
            # 精确到单个 DICOM Series（使用具体的文件列表）
            data, metadata = self.image_io.load_dicom_series(entry.file_list)
            if data is not None:
                self.current_image_data = data
                self.current_metadata = metadata
                # 使用"序列所在目录"作为路径显示信息
                self.current_image_path = entry.series_dir
                self.current_entry = entry  # 保存当前序列的entry信息

                spacing = metadata.get('spacing') if metadata else None
                self.four_view.set_image_data(data, spacing)

                self.current_label_data = np.zeros(data.shape, dtype=np.uint8)
                self.four_view.set_label_data(self.current_label_data)

                self.four_view.axial_view.set_active_label_id(self.active_label_id)
                self.four_view.coronal_view.set_active_label_id(self.active_label_id)
                self.four_view.sagittal_view.set_active_label_id(self.active_label_id)

                self.undo_history.clear()
                self.redo_history.clear()
                self.seed_points = []
                self._update_seed_count()
                self.seed_mode_enabled = False
                self.btn_add_seed.setChecked(False)
                self.btn_add_seed.setText("Add Seed")
                self.btn_add_seed.setStyleSheet("")

                # 使用数据库中的entry信息（优先）更新病人显示
                self._update_patient_info_from_entry(entry, metadata)

                QMessageBox.information(self, "成功", f"数据库序列加载成功\n{entry.display_name}\n尺寸: {data.shape}")
            else:
                QMessageBox.warning(self, "错误", "数据库序列加载失败")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载数据库序列时发生错误:\n{e}")

    def _update_patient_info_from_metadata(self, metadata: Optional[dict]):
        """根据ImageIO返回的metadata中的dicom_info更新病人信息"""
        info = {}
        if metadata and isinstance(metadata, dict):
            info = metadata.get("dicom_info") or {}
        
        # 如果dicom_info为空或信息不完整，尝试从当前图像路径中提取基本信息
        if self.current_image_path:
            path_parts = os.path.normpath(self.current_image_path).split(os.sep)
            # 尝试从路径中提取患者ID（如果还没有）
            if not info.get("PatientID"):
                for part in reversed(path_parts):
                    # 更宽松的匹配：1-4位数字都可能是患者ID
                    if part.isdigit() and 1 <= len(part) <= 4:
                        info["PatientID"] = part
                        break
            
            # 尝试从路径中提取Series信息（如果还没有）
            if not info.get("SeriesDescription"):
                # 查找SE*目录
                for part in reversed(path_parts):
                    if part.startswith("SE") and len(part) > 2:
                        # 检查SE后面的部分是否是数字
                        suffix = part[2:]
                        if suffix.isdigit():
                            info["SeriesDescription"] = part
                            break
                        # 也支持SE后面没有数字的情况
                        elif not suffix:
                            info["SeriesDescription"] = part
                            break
            
            # 尝试从路径中提取Study信息（如果还没有）
            if not info.get("StudyID"):
                # 查找ST*目录
                for part in reversed(path_parts):
                    if part.startswith("ST") and len(part) > 2:
                        # 检查ST后面的部分是否是数字
                        suffix = part[2:]
                        if suffix.isdigit():
                            info["StudyID"] = part
                            break
                        # 也支持ST后面没有数字的情况
                        elif not suffix:
                            info["StudyID"] = part
                            break
        
        # 确保至少有一些信息被提取
        self._apply_patient_info(info)

    def _update_patient_info_from_entry(self, entry: SeriesEntry, metadata: Optional[dict]):
        """综合数据库entry和metadata中的dicom_info更新病人信息"""
        info = {}
        if metadata and isinstance(metadata, dict):
            info = metadata.get("dicom_info") or {}
            # 如果dicom_info是空字典，确保info是字典而不是None
            if not isinstance(info, dict):
                info = {}

        # 覆盖/补充字段，以数据库中的信息为优先
        if entry:
            if entry.patient_id:
                info["PatientID"] = entry.patient_id
            if entry.patient_name:
                info["PatientName"] = entry.patient_name
            if entry.study_date:
                info["StudyDate"] = entry.study_date
            if entry.modality:
                info["Modality"] = entry.modality
            if entry.series_description:
                info["SeriesDescription"] = entry.series_description
            if entry.series_uid:
                info["SeriesInstanceUID"] = entry.series_uid
            if entry.study_id:
                info["StudyID"] = entry.study_id
            if entry.series_id:
                info["SeriesNumber"] = entry.series_id

        # 如果信息不完整，尝试从路径中补充
        if self.current_image_path:
            path_parts = os.path.normpath(self.current_image_path).split(os.sep)
            # 尝试从路径中提取患者ID（如果还没有）
            if not info.get("PatientID"):
                for part in reversed(path_parts):
                    # 更宽松的匹配：1-4位数字都可能是患者ID
                    if part.isdigit() and 1 <= len(part) <= 4:
                        info["PatientID"] = part
                        break
            
            # 尝试从路径中提取Series信息（如果还没有）
            if not info.get("SeriesDescription"):
                # 查找SE*目录
                for part in reversed(path_parts):
                    if part.startswith("SE") and len(part) > 2:
                        # 检查SE后面的部分是否是数字
                        suffix = part[2:]
                        if suffix.isdigit():
                            info["SeriesDescription"] = part
                            break
                        # 也支持SE后面没有数字的情况
                        elif not suffix:
                            info["SeriesDescription"] = part
                            break
            
            # 尝试从路径中提取Study信息（如果还没有）
            if not info.get("StudyID"):
                # 查找ST*目录
                for part in reversed(path_parts):
                    if part.startswith("ST") and len(part) > 2:
                        # 检查ST后面的部分是否是数字
                        suffix = part[2:]
                        if suffix.isdigit():
                            info["StudyID"] = part
                            break
                        # 也支持ST后面没有数字的情况
                        elif not suffix:
                            info["StudyID"] = part
                            break
        
        # 确保至少有一些信息被提取
        self._apply_patient_info(info)

    def _apply_patient_info(self, info: dict):
        """将info中的字段同步到界面显示"""
        self.current_patient_info = info or {}

        def get(k, default="--"):
            v = self.current_patient_info.get(k)
            return v if (v is not None and v != "") else default

        self.pi_id_label.setText(f"ID: {get('PatientID')}")
        self.pi_name_label.setText(f"Name: {get('PatientName')}")
        self.pi_sex_label.setText(f"Sex: {get('PatientSex')}")
        self.pi_birth_label.setText(f"Birth: {get('PatientBirthDate')}")
        self.pi_study_date_label.setText(f"Study Date: {get('StudyDate')}")
        self.pi_modality_label.setText(f"Modality: {get('Modality')}")

        series_text = f"{get('SeriesNumber', '')} {get('SeriesDescription', '')}".strip()
        if not series_text:
            series_text = get("SeriesInstanceUID")
        self.pi_series_label.setText(f"Series: {series_text}")

    def _on_db_filter_changed(self, text: str):
        """数据库树过滤：隐藏不匹配的叶子，并自动隐藏空的父节点"""
        q = (text or "").strip().lower()

        def match_item(it: QTreeWidgetItem) -> bool:
            # 任何列包含关键字即匹配；空关键字则全部匹配
            if not q:
                return True
            for col in range(self.db_series_list.columnCount()):
                if (it.text(col) or "").lower().find(q) != -1:
                    return True
            return False

        # 只对已有节点进行可见性设置
        for i in range(self.db_series_list.topLevelItemCount()):
            p_item = self.db_series_list.topLevelItem(i)
            p_visible = False

            for j in range(p_item.childCount()):
                s_item = p_item.child(j)
                s_visible = False

                for k in range(s_item.childCount()):
                    leaf = s_item.child(k)
                    ok = match_item(leaf) or match_item(s_item) or match_item(p_item)
                    leaf.setHidden(not ok)
                    if ok:
                        s_visible = True

                # 如果 study 本身匹配，也显示所有叶子
                if match_item(s_item) and q:
                    for k in range(s_item.childCount()):
                        s_item.child(k).setHidden(False)
                    s_visible = True

                s_item.setHidden(not s_visible)
                if s_visible:
                    p_visible = True

            # 如果 patient 本身匹配，也显示全部
            if match_item(p_item) and q:
                for j in range(p_item.childCount()):
                    p_item.child(j).setHidden(False)
                    s_item = p_item.child(j)
                    for k in range(s_item.childCount()):
                        s_item.child(k).setHidden(False)
                p_visible = True

            p_item.setHidden(not p_visible)

        # 有过滤关键字时自动展开，方便看到结果
        if q:
            self.db_series_list.expandAll()
    
    def _on_load_label(self):
        """加载标签"""
        if self.current_image_data is None:
            QMessageBox.warning(self, "警告", "请先加载影像")
            return
        
        filepath, _ = QFileDialog.getOpenFileName(
            self, "加载标签", "",
            "NIfTI文件 (*.nii *.nii.gz);;所有文件 (*.*)"
        )
        
        if filepath:
            try:
                label_data = self.image_io.load_label_map(filepath)
                if label_data is not None:
                    if label_data.shape == self.current_image_data.shape:
                        self.current_label_data = label_data
                        self.four_view.set_label_data(label_data)
                        QMessageBox.information(self, "成功", "标签加载成功")
                    else:
                        QMessageBox.warning(self, "错误", 
                            f"标签尺寸不匹配\n影像: {self.current_image_data.shape}\n标签: {label_data.shape}")
                else:
                    QMessageBox.warning(self, "错误", "标签加载失败")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载标签时发生错误:\n{e}")
    
    def _on_save_label(self):
        """保存标签"""
        if self.current_label_data is None or np.all(self.current_label_data == 0):
            QMessageBox.warning(self, "警告", "没有可保存的标签数据")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "保存标签", "",
            "NIfTI文件 (*.nii *.nii.gz);;所有文件 (*.*)"
        )
        
        if filepath:
            try:
                success = self.image_io.save_label_map(
                    self.current_label_data, 
                    filepath, 
                    self.current_metadata
                )
                if success:
                    QMessageBox.information(self, "成功", "标签保存成功")
                else:
                    QMessageBox.warning(self, "错误", "标签保存失败")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存标签时发生错误:\n{e}")
    
    def _on_save_screenshot(self):
        """保存截图：保存当前视图"""
        if self.current_image_data is None:
            QMessageBox.warning(self, "警告", "没有可保存的图像")
            return
        
        # 选择保存位置和文件名
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self, "保存截图", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        
        if not filepath:
            return
        
        # 让用户选择保存哪个视图
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QRadioButton, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("选择要保存的视图")
        layout = QVBoxLayout()
        
        rb_axial = QRadioButton("Axial View (轴位视图)")
        rb_coronal = QRadioButton("Coronal View (冠状位视图)")
        rb_sagittal = QRadioButton("Sagittal View (矢状位视图)")
        rb_3d = QRadioButton("3D View (3D视图)")
        rb_all = QRadioButton("All Views (所有视图 - 2x2拼接)")
        
        rb_axial.setChecked(True)
        layout.addWidget(rb_axial)
        layout.addWidget(rb_coronal)
        layout.addWidget(rb_sagittal)
        layout.addWidget(rb_3d)
        layout.addWidget(rb_all)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() != QDialog.Accepted:
            return
        
        try:
            if rb_axial.isChecked():
                self._save_view_screenshot('axial', filepath)
            elif rb_coronal.isChecked():
                self._save_view_screenshot('coronal', filepath)
            elif rb_sagittal.isChecked():
                self._save_view_screenshot('sagittal', filepath)
            elif rb_3d.isChecked():
                self._save_view_screenshot('3d', filepath)
            elif rb_all.isChecked():
                # 保存所有视图（保存为单个图像，拼接在一起）
                self._save_all_views_screenshot(filepath)
            
            QMessageBox.information(self, "成功", "截图已保存")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存截图时出错:\n{e}")
    
    def _save_view_screenshot(self, view_type: str, filepath: str):
        """保存单个视图的截图"""
        import os
        
        # 根据视图类型选择对应的视图组件
        if view_type == 'axial':
            viewer = self.four_view.axial_view
        elif view_type == 'coronal':
            viewer = self.four_view.coronal_view
        elif view_type == 'sagittal':
            viewer = self.four_view.sagittal_view
        elif view_type == '3d':
            viewer = self.four_view.three_d_view
        else:
            raise ValueError(f"Unknown view type: {view_type}")
        
        # 获取视图的canvas
        if hasattr(viewer, 'canvas'):
            # 2D视图：使用matplotlib的canvas
            canvas = viewer.canvas
            # 保存为图像
            canvas.print_figure(filepath, dpi=150, bbox_inches='tight')
        elif hasattr(viewer, 'plotter'):
            # 3D视图：使用PyVista的plotter
            plotter = viewer.plotter
            plotter.screenshot(filepath)
        else:
            raise ValueError(f"Cannot save screenshot for view type: {view_type}")
    
    def _save_all_views_screenshot(self, filepath: str):
        """保存所有视图的截图（拼接在一起）"""
        import os
        import tempfile
        
        try:
            from PIL import Image
        except ImportError:
            QMessageBox.warning(self, "警告", "需要安装Pillow库才能保存拼接截图\n请运行: pip install Pillow")
            return
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 保存各个视图到临时文件
            temp_files = {}
            for view_type in ['axial', 'coronal', 'sagittal', '3d']:
                temp_file = os.path.join(temp_dir, f"{view_type}.png")
                self._save_view_screenshot(view_type, temp_file)
                temp_files[view_type] = temp_file
            
            # 读取所有图像
            images = {}
            for view_type, temp_file in temp_files.items():
                if os.path.exists(temp_file):
                    images[view_type] = Image.open(temp_file)
            
            if not images:
                raise ValueError("没有成功保存任何视图")
            
            # 计算拼接后的尺寸（2x2布局）
            max_width = max(img.width for img in images.values())
            max_height = max(img.height for img in images.values())
            
            # 创建新图像（2x2布局）
            result = Image.new('RGB', (max_width * 2, max_height * 2), color='white')
            
            # 拼接图像（按照布局：左上=axial, 右上=sagittal, 左下=3d, 右下=coronal）
            if 'axial' in images:
                result.paste(images['axial'], (0, 0))
            if 'sagittal' in images:
                result.paste(images['sagittal'], (max_width, 0))
            if '3d' in images:
                result.paste(images['3d'], (0, max_height))
            if 'coronal' in images:
                result.paste(images['coronal'], (max_width, max_height))
            
            # 保存结果
            result.save(filepath, quality=95)
        finally:
            # 清理临时文件
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # ========== 种子点操作函数 ==========
    
    def _on_toggle_seed_mode(self):
        """切换种子点模式"""
        self.seed_mode_enabled = self.btn_add_seed.isChecked()
        if self.seed_mode_enabled:
            self.btn_add_seed.setText("Seed Mode: ON")
            self.btn_add_seed.setStyleSheet("background-color: lightgreen;")
        else:
            self.btn_add_seed.setText("Add Seed")
            self.btn_add_seed.setStyleSheet("")
    
    def _on_clear_seeds(self):
        """清除所有种子点"""
        if len(self.seed_points) == 0:
            QMessageBox.information(self, "Info", "No seed points to clear.")
            return
        
        reply = QMessageBox.question(self, "Confirm", f"Clear all {len(self.seed_points)} seed points?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.seed_points = []
            self._update_seed_count()
            QMessageBox.information(self, "Info", "All seed points cleared.")
    
    def _update_seed_count(self):
        """更新种子点数量显示"""
        count = len(self.seed_points)
        self.seed_count_label.setText(f"Seeds: {count}")
        if count > 0:
            self.seed_count_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.seed_count_label.setStyleSheet("")
    
    # ========== 分割操作槽函数 ==========
    
    def _on_segment_3d(self):
        """Segment 3D - 主动轮廓分割"""
        if self.current_image_data is None:
            QMessageBox.warning(self, "警告", "请先加载影像")
            return
        
        if len(self.seed_points) == 0:
            QMessageBox.warning(self, "警告", "请先设置种子点（在视图中点击）")
            return
        
        # 获取算法参数
        method = 'region_competition' if self.algo_combo.currentIndex() == 0 else 'edge_based'
        params = {
            'method': method,
            'iterations': self.iterations_spin.value(),
            'smoothness': self.smoothness_spin.value(),
            'expansion_force': self.expansion_spin.value(),
            'edge_weight': 1.0
        }
        
        # 显示进度对话框
        progress = QProgressDialog("正在执行分割...", "取消", 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        # 创建工作线程
        worker = SegmentationWorker(self.segmentation, self.current_image_data, 
                                    self.seed_points, params)
        worker.finished.connect(lambda mask: self._on_segmentation_finished(mask, progress))
        worker.error.connect(lambda err: self._on_segmentation_error(err, progress))
        worker.start()
    
    def _on_segmentation_finished(self, mask: np.ndarray, progress: QProgressDialog):
        """分割完成回调"""
        progress.close()
        self.current_label_data = mask
        self.four_view.set_label_data(mask)
        QMessageBox.information(self, "成功", "分割完成")
    
    def _on_segmentation_error(self, error: str, progress: QProgressDialog):
        """分割错误回调"""
        progress.close()
        QMessageBox.critical(self, "错误", f"分割失败:\n{error}")
    
    def _on_ai_segmentation(self):
        """AI自动分割"""
        if self.current_image_data is None:
            QMessageBox.warning(self, "警告", "请先加载影像")
            return
        
        # 选择ONNX模型文件
        model_path, _ = QFileDialog.getOpenFileName(
            self, "选择AI模型", "",
            "ONNX模型 (*.onnx);;所有文件 (*.*)"
        )
        
        if not model_path:
            return
        
        try:
            # 加载模型
            if not self.ai_interface.load_model(model_path):
                QMessageBox.critical(self, "错误", "模型加载失败\n请检查模型文件是否正确")
                return
            
            # 显示进度对话框
            progress = QProgressDialog("正在执行AI分割...", "取消", 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # 执行分割
            try:
                mask = self.ai_interface.segment(self.current_image_data)
                progress.close()
                
                if mask is not None:
                    if mask.dtype != np.uint8:
                        mask = mask.astype(np.uint8)
                    
                    self.current_label_data = mask
                    self.four_view.set_label_data(mask)
                    QMessageBox.information(self, "成功", "AI分割完成")
                else:
                    QMessageBox.warning(self, "错误", "AI分割失败，未生成有效掩码")
            except Exception as e:
                progress.close()
                QMessageBox.critical(self, "错误", f"AI分割执行失败:\n{e}")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"AI分割过程发生错误:\n{e}")
    
    # ========== 标签管理槽函数 ==========
    
    def _on_add_label(self):
        """添加标签"""
        label_id = len(self.label_list) + 1
        color = (1.0, 0.0, 0.0, 1.0) if label_id == 1 else (0.0, 1.0, 0.0, 1.0)
        self.label_list.append({
            'id': label_id,
            'name': f'Label {label_id}',
            'color': color
        })
        
        item = QListWidgetItem(f"Label {label_id}")
        self.label_list_widget.addItem(item)
        
        # 设置为活动标签
        self.active_label_id = label_id
        self._update_active_label_display()
        # 更新所有视图的活动标签ID
        self.four_view.axial_view.set_active_label_id(label_id)
        self.four_view.coronal_view.set_active_label_id(label_id)
        self.four_view.sagittal_view.set_active_label_id(label_id)
    
    def _on_delete_label(self):
        """删除标签"""
        current_item = self.label_list_widget.currentItem()
        if current_item:
            row = self.label_list_widget.row(current_item)
            self.label_list_widget.takeItem(row)
            if row < len(self.label_list):
                del self.label_list[row]
    
    def _on_label_color(self):
        """标签颜色设置"""
        current_item = self.label_list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "警告", "请先选择标签")
            return
        
        row = self.label_list_widget.row(current_item)
        if row < len(self.label_list):
            current_color = self.label_list[row]['color']
            qcolor = QColor(int(current_color[0]*255), int(current_color[1]*255), int(current_color[2]*255))
        else:
            qcolor = QColor(255, 0, 0)
        
        color = QColorDialog.getColor(qcolor, self, "选择标签颜色")
        if color.isValid():
            if row < len(self.label_list):
                r, g, b = color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0
                self.label_list[row]['color'] = (r, g, b, 1.0)
                # 更新视图中的标签颜色
                label_id = self.label_list[row]['id']
                self.four_view.axial_view.set_label_color(label_id, (r, g, b, 1.0))
                self.four_view.coronal_view.set_label_color(label_id, (r, g, b, 1.0))
                self.four_view.sagittal_view.set_label_color(label_id, (r, g, b, 1.0))
                # 如果是活动标签，更新颜色方块
                if label_id == self.active_label_id:
                    self._update_active_label_display()
    
    def _on_label_selection_changed(self):
        """标签选择改变回调"""
        current_item = self.label_list_widget.currentItem()
        if current_item:
            row = self.label_list_widget.row(current_item)
            if row < len(self.label_list):
                self.active_label_id = self.label_list[row]['id']
                self._update_active_label_display()
    
    def _update_active_label_display(self):
        """更新活动标签显示"""
        # 查找活动标签
        active_label = None
        for label in self.label_list:
            if label['id'] == self.active_label_id:
                active_label = label
                break
        
        if active_label:
            self.active_label_name_label.setText(f"Active label: {active_label['name']}")
            self.paint_over_label.setText(f"Paint over: {active_label['name']}")
            # 更新颜色方块
            r, g, b = int(active_label['color'][0]*255), int(active_label['color'][1]*255), int(active_label['color'][2]*255)
            self.active_label_color_widget.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border: 1px solid black;")
            # 更新所有视图的活动标签ID
            self.four_view.axial_view.set_active_label_id(self.active_label_id)
            self.four_view.coronal_view.set_active_label_id(self.active_label_id)
            self.four_view.sagittal_view.set_active_label_id(self.active_label_id)
        else:
            self.active_label_name_label.setText("Active label: --")
            self.paint_over_label.setText("Paint over: --")
    
    def _on_label_opacity_changed(self, value):
        """标签透明度改变回调"""
        opacity = value / 100.0
        self.label_opacity_value_label.setText(str(value))
        # 更新所有视图的标签透明度
        self.four_view.axial_view.label_opacity = opacity
        self.four_view.coronal_view.label_opacity = opacity
        self.four_view.sagittal_view.label_opacity = opacity
        # 更新显示
        if self.current_label_data is not None:
            self.four_view.set_label_data(self.current_label_data)
    
    # ========== 视图相关槽函数 ==========
    
    def _on_crosshair_changed(self, view_type: str, x: float, y: float):
        """十字线改变回调 - 更新Cursor Inspector"""
        if self.current_image_data is None:
            return
        
        # 如果处于种子点模式，添加种子点
        if self.seed_mode_enabled:
            # crosshair_3d 可能是物理坐标；转换为体素索引
            spacing = self.four_view.image_spacing
            if spacing:
                vx = int(round(self.four_view.crosshair_3d[0] / spacing[0]))
                vy = int(round(self.four_view.crosshair_3d[1] / spacing[1]))
                vz = int(round(self.four_view.crosshair_3d[2] / spacing[2]))
            else:
                vx = int(round(self.four_view.crosshair_3d[0]))
                vy = int(round(self.four_view.crosshair_3d[1]))
                vz = int(round(self.four_view.crosshair_3d[2]))

            # 检查坐标是否有效（体素索引）
            if (0 <= vx < self.current_image_data.shape[1] and
                0 <= vy < self.current_image_data.shape[0] and
                0 <= vz < self.current_image_data.shape[2]):
                seed_point = (vx, vy, vz)
                # 避免重复添加相同的种子点
                if seed_point not in self.seed_points:
                    self.seed_points.append(seed_point)
                    self._update_seed_count()
                    # 在状态栏或标签中显示提示（不弹出消息框）
                    print(f"Seed added at voxel ({vx}, {vy}, {vz}); total: {len(self.seed_points)}")
        
        # 根据视图类型计算3D坐标
        if view_type == 'axial':
            z = self.four_view.axial_view.current_slice
            orig_x = int(self.four_view.crosshair_3d[0])
            orig_y = int(self.four_view.crosshair_3d[1])
            orig_z = int(self.four_view.crosshair_3d[2])
            
            # 更新Cursor position
            self.cursor_pos_label.setText(f"Cursor position (x,y,z): {orig_x} {orig_y} {orig_z}")
            
            # 获取像素值
            if (0 <= orig_x < self.current_image_data.shape[1] and
                0 <= orig_y < self.current_image_data.shape[0] and
                0 <= orig_z < self.current_image_data.shape[2]):
                value = self.current_image_data[orig_y, orig_x, orig_z]
                # 更新Layer名称（从文件路径提取）
                layer_name = self.current_image_path.split('/')[-1].split('\\')[-1] if self.current_image_path else "volume"
                self.cursor_layer_label.setText(f"  Layer: {layer_name}")
                self.cursor_intensity_value_label.setText(f"  Intensity: {int(value)}")
                
                # 获取Label值
                if self.current_label_data is not None:
                    label_value = self.current_label_data[orig_y, orig_x, orig_z]
                    self.cursor_label_value_label.setText(f"  Label: {int(label_value)}")
                else:
                    self.cursor_label_value_label.setText("  Label: 0")
        
        elif view_type == 'coronal':
            orig_x = int(self.four_view.crosshair_3d[0])
            orig_y = int(self.four_view.crosshair_3d[1])
            orig_z = int(self.four_view.crosshair_3d[2])
            
            self.cursor_pos_label.setText(f"Cursor position (x,y,z): {orig_x} {orig_y} {orig_z}")
            
            if (0 <= orig_x < self.current_image_data.shape[1] and
                0 <= orig_y < self.current_image_data.shape[0] and
                0 <= orig_z < self.current_image_data.shape[2]):
                value = self.current_image_data[orig_y, orig_x, orig_z]
                layer_name = self.current_image_path.split('/')[-1].split('\\')[-1] if self.current_image_path else "volume"
                self.cursor_layer_label.setText(f"  Layer: {layer_name}")
                self.cursor_intensity_value_label.setText(f"  Intensity: {int(value)}")
                
                if self.current_label_data is not None:
                    label_value = self.current_label_data[orig_y, orig_x, orig_z]
                    self.cursor_label_value_label.setText(f"  Label: {int(label_value)}")
                else:
                    self.cursor_label_value_label.setText("  Label: 0")
        
        elif view_type == 'sagittal':
            orig_x = int(self.four_view.crosshair_3d[0])
            orig_y = int(self.four_view.crosshair_3d[1])
            orig_z = int(self.four_view.crosshair_3d[2])
            
            self.cursor_pos_label.setText(f"Cursor position (x,y,z): {orig_x} {orig_y} {orig_z}")
            
            if (0 <= orig_x < self.current_image_data.shape[1] and
                0 <= orig_y < self.current_image_data.shape[0] and
                0 <= orig_z < self.current_image_data.shape[2]):
                value = self.current_image_data[orig_y, orig_x, orig_z]
                layer_name = self.current_image_path.split('/')[-1].split('\\')[-1] if self.current_image_path else "volume"
                self.cursor_layer_label.setText(f"  Layer: {layer_name}")
                self.cursor_intensity_value_label.setText(f"  Intensity: {int(value)}")
                
                if self.current_label_data is not None:
                    label_value = self.current_label_data[orig_y, orig_x, orig_z]
                    self.cursor_label_value_label.setText(f"  Label: {int(label_value)}")
                else:
                    self.cursor_label_value_label.setText("  Label: 0")
    
    def _on_slice_changed(self, view_type: str, slice_idx: int):
        """切片改变回调"""
        # 可以在这里实现其他联动逻辑
        pass
    
    def _on_clear_label(self):
        """清除当前十字线位置的标签"""
        if self.current_label_data is None:
            return
        
        orig_x = int(self.four_view.crosshair_3d[0])
        orig_y = int(self.four_view.crosshair_3d[1])
        orig_z = int(self.four_view.crosshair_3d[2])
        
        if (0 <= orig_x < self.current_label_data.shape[1] and
            0 <= orig_y < self.current_label_data.shape[0] and
            0 <= orig_z < self.current_label_data.shape[2]):
            # 保存到撤销历史
            self._save_to_history()
            self.current_label_data[orig_y, orig_x, orig_z] = 0
            self.four_view.set_label_data(self.current_label_data)
            # 更新Cursor Inspector
            self.cursor_label_value_label.setText("  Label: 0")
    
    def _on_tool_selected(self, tool_mode: str):
        """工具选择回调"""
        self.current_tool_mode = tool_mode
        # 更新按钮状态
        self.btn_pointer.setChecked(tool_mode == 'pointer')
        self.btn_zoom.setChecked(tool_mode == 'zoom')
        self.btn_pan.setChecked(tool_mode == 'pan')
        self.btn_brush.setChecked(tool_mode == 'brush')
        self.btn_eraser.setChecked(tool_mode == 'eraser')
        
        # 更新所有视图的工具模式
        self.four_view.axial_view.set_tool_mode(tool_mode)
        self.four_view.coronal_view.set_tool_mode(tool_mode)
        self.four_view.sagittal_view.set_tool_mode(tool_mode)
    
    def _on_brush_size_changed(self, value: int):
        """画笔大小改变回调"""
        self.four_view.axial_view.set_brush_size(value)
        self.four_view.coronal_view.set_brush_size(value)
        self.four_view.sagittal_view.set_brush_size(value)
    
    def _on_label_changed(self):
        """标签改变回调 - 保存到撤销历史"""
        if self.current_label_data is not None:
            self._save_to_history()
    
    def _save_to_history(self):
        """保存当前标签状态到撤销历史"""
        if self.current_label_data is not None:
            # 深拷贝当前标签数据
            label_copy = self.current_label_data.copy()
            self.undo_history.append(label_copy)
            # 限制历史记录数量
            if len(self.undo_history) > self.max_history_size:
                self.undo_history.pop(0)
            # 清空重做历史
            self.redo_history.clear()
    
    def _on_undo(self):
        """撤销操作"""
        if len(self.undo_history) == 0:
            return
        
        if self.current_label_data is not None:
            # 当前状态保存到重做历史
            self.redo_history.append(self.current_label_data.copy())
        
        # 恢复上一个状态
        self.current_label_data = self.undo_history.pop()
        self.four_view.set_label_data(self.current_label_data)
    
    def _on_redo(self):
        """重做操作"""
        if len(self.redo_history) == 0:
            return
        
        if self.current_label_data is not None:
            # 当前状态保存到撤销历史
            self.undo_history.append(self.current_label_data.copy())
        
        # 恢复下一个状态
        self.current_label_data = self.redo_history.pop()
        self.four_view.set_label_data(self.current_label_data)
    
    def _on_update_3d_view_clicked(self):
        """更新3D视图按钮点击事件"""
        if self.current_image_data is None:
            QMessageBox.warning(self, "警告", "请先加载影像")
            return
        
        try:
            # 调用FourViewWidget的刷新方法
            self.four_view.refresh_3d_view()
            QMessageBox.information(self, "成功", "3D视图已更新")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更新3D视图时发生错误:\n{e}")

    def _build_entry_from_metadata(self, metadata: Optional[dict], filepath: str = None) -> Optional[SeriesEntry]:
        """从metadata中提取DICOM信息，构建SeriesEntry（用于自动定位和显示）"""
        dicom_info = {}
        if metadata and isinstance(metadata, dict):
            dicom_info = metadata.get("dicom_info") or {}
        
        # 提取关键字段
        patient_id = None
        if dicom_info:
            patient_id = dicom_info.get("PatientID") or dicom_info.get("0010|0020")
        
        # 如果metadata中没有patient_id，尝试从文件路径中提取
        if not patient_id and filepath:
            path_parts = os.path.normpath(filepath).split(os.sep)
            # 查找可能的患者ID（数字，通常在路径中）
            for part in reversed(path_parts):
                if part.isdigit() and len(part) <= 3:  # 患者ID通常是1-3位数字
                    patient_id = part
                    break
        
        # 如果还是没有patient_id，但至少有一些DICOM信息，仍然创建entry（用于显示）
        # 如果没有patient_id也没有其他信息，返回None
        if not patient_id and not dicom_info:
            return None
        
        # 如果没有patient_id，使用路径中的目录名或"Unknown"
        if not patient_id:
            if filepath:
                path_parts = os.path.normpath(filepath).split(os.sep)
                for part in reversed(path_parts):
                    if part and part != "." and part != "..":
                        patient_id = part
                        break
            if not patient_id:
                patient_id = "Unknown"
        
        # 从dicom_info中提取其他字段
        study_id = None
        series_uid = None
        patient_name = None
        study_date = None
        modality = None
        series_description = None
        series_id = None
        
        if dicom_info:
            study_id = dicom_info.get("StudyID") or dicom_info.get("0020|0010")
            series_uid = dicom_info.get("SeriesInstanceUID") or dicom_info.get("0020|000E")
            patient_name = dicom_info.get("PatientName") or dicom_info.get("0010|0010")
            study_date = dicom_info.get("StudyDate") or dicom_info.get("0008|0020")
            modality = dicom_info.get("Modality") or dicom_info.get("0008|0060")
            series_description = dicom_info.get("SeriesDescription") or dicom_info.get("0008|103E")
            series_id = dicom_info.get("SeriesNumber") or dicom_info.get("0020|0011")
        
        # 构建SeriesEntry（部分字段可能为空）
        series_dir = filepath if filepath else ""
        if os.path.isfile(series_dir):
            series_dir = os.path.dirname(series_dir)
        
        return SeriesEntry(
            patient_id=str(patient_id),
            study_id=str(study_id) if study_id else None,
            series_id=str(series_id) if series_id else None,
            series_uid=str(series_uid) if series_uid else None,
            series_dir=series_dir,
            file_list=[],  # 从文件加载时，不需要文件列表
            patient_name=patient_name,
            study_date=study_date,
            modality=modality,
            series_description=series_description,
        )
    
    def _on_open_db_dialog(self):
        """打开数据库选择弹窗"""
        # 传递上次选择的数据库路径（如果存在）
        dialog = DatabaseSelectionDialog(self, db_root=self.last_db_root)
        dialog.series_selected.connect(self._on_db_series_loaded)
        # 如果已经加载了序列（无论是从数据库还是从文件），传递当前序列信息以便自动定位
        if self.current_entry:
            dialog.set_current_series(self.current_entry)
        # 如果当前有图像路径，也传递用于路径定位
        if self.current_image_path:
            dialog.set_current_image_path(self.current_image_path)
        
        # 显示对话框
        dialog.exec_()
        
        # 保存本次选择的数据库路径
        if dialog.db_root:
            self.last_db_root = dialog.db_root

    def _on_db_series_loaded(self, result: dict):
        """处理从数据库选择弹窗返回的已加载序列"""
        data = result["data"]
        metadata = result["metadata"]
        entry = result["entry"]

        self.current_image_data = data
        self.current_metadata = metadata
        self.current_image_path = entry.series_dir # 使用序列所在目录作为路径显示信息
        self.current_entry = entry  # 保存当前序列的entry信息

        spacing = metadata.get('spacing') if metadata else None
        self.four_view.set_image_data(data, spacing)

        self.current_label_data = np.zeros(data.shape, dtype=np.uint8)
        self.four_view.set_label_data(self.current_label_data)

        self.four_view.axial_view.set_active_label_id(self.active_label_id)
        self.four_view.coronal_view.set_active_label_id(self.active_label_id)
        self.four_view.sagittal_view.set_active_label_id(self.active_label_id)

        self.undo_history.clear()
        self.redo_history.clear()
        self.seed_points = []
        self._update_seed_count()
        self.seed_mode_enabled = False
        self.btn_add_seed.setChecked(False)
        self.btn_add_seed.setText("Add Seed")
        self.btn_add_seed.setStyleSheet("")

        # 使用数据库中的entry信息（优先）更新病人显示
        self._update_patient_info_from_entry(entry, metadata)

        QMessageBox.information(self, "成功", f"数据库序列加载成功\n{entry.display_name}\n尺寸: {data.shape}")

class PatientInfoDialog(QDialog):
    """显示病人详细信息的弹窗"""
    def __init__(self, info: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("病人 / 检查 / 序列信息")
        self.setMinimumWidth(400) # 设置最小宽度

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # 定义要显示的字段及其对应的中文名称
        fields = {
            "PatientID": "病人 ID",
            "PatientName": "病人姓名",
            "PatientSex": "病人性别",
            "PatientBirthDate": "出生日期",
            "StudyDate": "检查日期",
            "StudyID": "检查 ID",
            "StudyDescription": "检查描述",
            "Modality": "模态",
            "SeriesNumber": "序列号",
            "SeriesDescription": "序列描述",
            "SeriesInstanceUID": "序列实例 UID",
            "StudyInstanceUID": "检查实例 UID",
            "BodyPartExamined": "检查部位",
        }

        for key, display_name in fields.items():
            value = info.get(key, "N/A")
            label = QLabel(str(value))
            label.setWordWrap(True) # 允许文本换行
            form_layout.addRow(f"{display_name}:", label)
        
        layout.addLayout(form_layout)
        
        # 添加一个关闭按钮
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

        self.setLayout(layout)

class ExcelInfoDialog(QDialog):
    """显示Excel表格中患者信息的对话框，支持选择文件和Sheet，自动定位患者行"""
    
    def __init__(self, patient_id: str, excel_root: str, excel_data: dict = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"患者表格信息 - {patient_id}")
        self.setMinimumSize(1000, 600)
        
        self.patient_id = patient_id
        self.excel_root = excel_root
        self.current_file = None
        self.current_sheet = None
        self.df = None
        self.auto_select_sheet = None  # 用于自动选择Sheet
        
        layout = QVBoxLayout()
        
        # 标题和患者ID显示
        title_layout = QHBoxLayout()
        title_label = QLabel(f"患者ID: {patient_id}")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        layout.addLayout(title_layout)
        
        # 文件选择区域
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("选择Excel文件:"))
        self.file_combo = QComboBox()
        self.file_combo.setMinimumWidth(300)
        self.file_combo.currentTextChanged.connect(self._on_file_changed)
        file_layout.addWidget(self.file_combo)
        
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._on_browse_file)
        file_layout.addWidget(btn_browse)
        file_layout.addStretch()
        layout.addLayout(file_layout)
        
        # Sheet选择区域
        sheet_layout = QHBoxLayout()
        sheet_layout.addWidget(QLabel("选择Sheet:"))
        self.sheet_combo = QComboBox()
        self.sheet_combo.setMinimumWidth(200)
        self.sheet_combo.currentTextChanged.connect(self._on_sheet_changed)
        sheet_layout.addWidget(self.sheet_combo)
        sheet_layout.addStretch()
        layout.addLayout(sheet_layout)
        
        # 表格视图
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        layout.addWidget(self.table, stretch=1)
        
        # 状态信息
        self.status_label = QLabel("请选择Excel文件和Sheet")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)
        
        # 按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # 初始化：扫描Excel文件
        self._scan_excel_files()
        
        # 如果有excel_data，尝试自动选择对应的文件和Sheet
        if excel_data:
            self._auto_select_file_and_sheet(excel_data)
    
    def _scan_excel_files(self):
        """扫描Excel文件目录，填充文件列表"""
        if not os.path.exists(self.excel_root):
            self.status_label.setText(f"错误: 目录不存在 - {self.excel_root}")
            return
        
        excel_files = []
        for file in os.listdir(self.excel_root):
            # 跳过Excel临时文件（以~$开头的文件）
            if file.startswith('~$'):
                continue
            
            file_path = os.path.join(self.excel_root, file)
            if os.path.isfile(file_path):
                ext = os.path.splitext(file)[1].lower()
                if ext in ['.xlsx', '.xls']:
                    excel_files.append((file, file_path))
        
        if not excel_files:
            self.status_label.setText(f"未找到Excel文件 (目录: {self.excel_root})")
            return
        
        # 按文件名排序
        excel_files.sort(key=lambda x: x[0])
        
        self.file_combo.clear()
        for file, file_path in excel_files:
            self.file_combo.addItem(file, file_path)
    
    def _auto_select_file_and_sheet(self, excel_data: dict):
        """根据excel_data自动选择对应的文件和Sheet"""
        if '_source_file' in excel_data:
            source_file = excel_data['_source_file']
            # 在文件列表中查找
            for i in range(self.file_combo.count()):
                if self.file_combo.itemText(i) == source_file:
                    self.file_combo.setCurrentIndex(i)
                    break
            
            # 保存Sheet信息，等待文件加载完成后再选择
            if '_source_sheet' in excel_data:
                self.auto_select_sheet = excel_data['_source_sheet']
    
    def _on_browse_file(self):
        """浏览选择Excel文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择Excel文件", self.excel_root,
            "Excel Files (*.xlsx *.xls);;All Files (*.*)"
        )
        if file_path:
            # 添加到列表或直接选择
            file_name = os.path.basename(file_path)
            # 检查是否已在列表中
            found = False
            for i in range(self.file_combo.count()):
                if self.file_combo.itemData(i) == file_path:
                    self.file_combo.setCurrentIndex(i)
                    found = True
                    break
            
            if not found:
                # 添加到列表
                self.file_combo.addItem(file_name, file_path)
                self.file_combo.setCurrentIndex(self.file_combo.count() - 1)
    
    def _on_file_changed(self, file_name: str):
        """当文件选择改变时，加载Sheet列表"""
        if not file_name:
            return
        
        file_path = self.file_combo.currentData()
        if not file_path or not os.path.exists(file_path):
            return
        
        self.current_file = file_path
        
        try:
            excel_file = pd.ExcelFile(file_path)
            self.sheet_combo.clear()
            for sheet_name in excel_file.sheet_names:
                self.sheet_combo.addItem(sheet_name)
            
            # 如果有自动选择的Sheet，尝试选择它
            if self.auto_select_sheet:
                for i in range(self.sheet_combo.count()):
                    if self.sheet_combo.itemText(i) == self.auto_select_sheet:
                        self.sheet_combo.setCurrentIndex(i)
                        self.auto_select_sheet = None  # 清除标记，避免重复
                        return
                # 如果没找到，选择第一个
                if self.sheet_combo.count() > 0:
                    self.sheet_combo.setCurrentIndex(0)
            elif self.sheet_combo.count() > 0:
                # 没有自动选择，选择第一个
                self.sheet_combo.setCurrentIndex(0)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"读取Excel文件失败:\n{e}")
            self.status_label.setText(f"错误: {e}")
    
    def _on_sheet_changed(self, sheet_name: str):
        """当Sheet选择改变时，加载表格数据并定位患者行"""
        if not sheet_name or not self.current_file:
            return
        
        self.current_sheet = sheet_name
        
        try:
            # 读取Sheet数据
            self.df = pd.read_excel(self.current_file, sheet_name=sheet_name)
            
            # 显示表格
            self._display_table()
            
            # 定位患者行
            self._locate_patient_row()
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"读取Sheet失败:\n{e}")
            self.status_label.setText(f"错误: {e}")
    
    def _display_table(self):
        """在表格视图中显示DataFrame"""
        if self.df is None or self.df.empty:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        
        # 设置行列数
        rows, cols = self.df.shape
        self.table.setRowCount(rows)
        self.table.setColumnCount(cols)
        
        # 设置表头
        headers = [str(col) for col in self.df.columns]
        self.table.setHorizontalHeaderLabels(headers)
        
        # 填充数据
        for i in range(rows):
            for j in range(cols):
                value = self.df.iloc[i, j]
                item = QTableWidgetItem(str(value) if pd.notna(value) else "")
                self.table.setItem(i, j, item)
        
        # 调整列宽
        self.table.resizeColumnsToContents()
    
    def _locate_patient_row(self):
        """定位并高亮显示患者对应的行"""
        if self.df is None or self.df.empty:
            return
        
        # 尝试找到患者ID列（扩展更多可能的列名）
        patient_id_columns = [
            '患者ID', 'Patient ID', 'PatientID', '序号', '编号', 'ID', 'id',
            '患者编号', '病人ID', '病人编号', 'PatientID', 'patient_id',
            '患者序号', '病人序号'
        ]
        patient_id_col = None
        
        # 先精确匹配
        for col in self.df.columns:
            col_str = str(col).strip()
            if col_str in patient_id_columns:
                patient_id_col = col
                break
        
        # 如果没找到，尝试模糊匹配（包含"ID"或"id"或"编号"的列）
        if patient_id_col is None:
            for col in self.df.columns:
                col_str = str(col).strip().lower()
                if 'id' in col_str or '编号' in col_str or '序号' in col_str:
                    patient_id_col = col
                    break
        
        # 如果还是没找到，尝试第一列
        if patient_id_col is None:
            patient_id_col = self.df.columns[0]
        
        # 查找匹配的行
        found_row = None
        patient_id_normalized = str(self.patient_id).strip()
        
        for idx, row in self.df.iterrows():
            try:
                # 获取行中的ID值
                row_value = row[patient_id_col]
                if pd.isna(row_value):
                    continue
                
                row_id = str(row_value).strip()
                row_id_normalized = row_id.lstrip('0')  # 去掉前导零
                patient_id_normalized_no_zero = patient_id_normalized.lstrip('0')
                
                # 尝试多种匹配方式
                if (row_id == patient_id_normalized or  # 完全匹配
                    row_id_normalized == patient_id_normalized or  # 去掉前导零后匹配
                    patient_id_normalized_no_zero == row_id or  # 患者ID去掉前导零后匹配
                    row_id_normalized == patient_id_normalized_no_zero or  # 都去掉前导零后匹配
                    row_id.endswith(patient_id_normalized) or  # 行ID以患者ID结尾
                    patient_id_normalized.endswith(row_id)):  # 患者ID以行ID结尾
                    found_row = idx
                    break
            except Exception as e:
                # 如果某行出错，继续查找下一行
                continue
        
        if found_row is not None:
            # 滚动到该行并选中
            self.table.selectRow(found_row)
            if self.table.item(found_row, 0):
                self.table.scrollToItem(self.table.item(found_row, 0))
            # 高亮显示（设置背景色）
            for col in range(self.table.columnCount()):
                item = self.table.item(found_row, col)
                if item:
                    item.setBackground(QColor(255, 255, 0))  # 黄色背景
            
            self.status_label.setText(f"已定位到患者 {self.patient_id} (第 {found_row + 1} 行)")
        else:
            # 显示更详细的调试信息
            self.status_label.setText(
                f"未找到患者 {self.patient_id} 对应的行 "
                f"(搜索列: {patient_id_col}, 共 {len(self.df)} 行)"
            )

class DatabaseSelectionDialog(QDialog):
    """用于选择数据库序列的独立弹窗"""
    series_selected = pyqtSignal(object) # 信号：当用户选择并加载一个序列时发出

    def __init__(self, parent=None, db_root=None):
        super().__init__(parent)
        self.setWindowTitle("选择数据库序列")
        self.setMinimumSize(800, 600) # 设置弹窗的最小尺寸

        self.db_root = db_root  # 如果提供了db_root，直接使用
        self.db_entries = [] # type: list[SeriesEntry]
        self.patient_worker = None # 用于懒加载病人序列的工作线程
        self.image_io = ImageIO() # 需要ImageIO来加载选中的序列
        self.db_worker = None # 数据库索引扫描线程
        self.db_scan_progress = None # 数据库扫描进度条
        self.excel_reader = None # Excel表格读取器
        self.excel_root = None # Excel文件所在目录
        self.current_series_entry = None  # 当前已加载的序列entry（用于自动定位）
        self.current_image_path = None  # 当前图像路径（用于路径定位）

        self._init_ui()
        
        # 如果提供了db_root，自动连接数据库
        if self.db_root and os.path.isdir(self.db_root):
            self._connect_database_auto(self.db_root)

    def _init_ui(self):
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # 数据库连接部分
        conn_group = QGroupBox("数据库连接")
        conn_layout = QHBoxLayout()
        self.btn_connect_db = QPushButton("Connect DB Root")
        self.btn_connect_db.setToolTip("选择 522/data 作为数据库根目录，并扫描可加载的序列")
        self.btn_connect_db.clicked.connect(self._on_connect_database)
        conn_layout.addWidget(self.btn_connect_db)

        self.db_status_label = QLabel("DB: not connected")
        self.db_status_label.setWordWrap(True)
        conn_layout.addWidget(self.db_status_label)
        conn_layout.addStretch() # 填充剩余空间
        conn_group.setLayout(conn_layout)
        main_layout.addWidget(conn_group)

        # 数据库列表和过滤部分
        db_list_group = QGroupBox("数据库序列列表")
        db_list_layout = QVBoxLayout()

        self.db_filter_edit = QLineEdit()
        self.db_filter_edit.setPlaceholderText("Filter... (e.g. 117 / CT / 2024 / name)")
        self.db_filter_edit.textChanged.connect(self._on_db_filter_changed)
        db_list_layout.addWidget(self.db_filter_edit)

        self.db_series_list = QTreeWidget()
        self.db_series_list.setColumnCount(6)
        self.db_series_list.setHeaderLabels([
            "Patient ID", "Patient Name", "Study ID",
            "Study Date", "Modality", "Series Desc"
        ])
        self.db_series_list.setToolTip("双击或点击 Load Selected 加载选中的序列(SE*)")
        self.db_series_list.itemDoubleClicked.connect(self._on_load_selected_db_series)
        self.db_series_list.itemExpanded.connect(self._on_db_item_expanded)
        self.db_series_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.db_series_list.setIndentation(12)
        self.db_series_list.setRootIsDecorated(True)  # 确保显示展开/折叠箭头
        self.db_series_list.setItemsExpandable(True)  # 确保项目可展开
        header = self.db_series_list.header()
        header.setStretchLastSection(True)
        db_list_layout.addWidget(self.db_series_list, stretch=1) # 列表占据大部分空间

        btns_layout = QHBoxLayout()
        self.btn_db_expand_all = QPushButton("Expand All")
        self.btn_db_expand_all.clicked.connect(self.db_series_list.expandAll)
        btns_layout.addWidget(self.btn_db_expand_all)
        self.btn_db_collapse_all = QPushButton("Collapse")
        self.btn_db_collapse_all.clicked.connect(lambda: self.db_series_list.collapseToDepth(0))
        btns_layout.addWidget(self.btn_db_collapse_all)
        db_list_layout.addLayout(btns_layout)

        self.btn_load_db_series = QPushButton("Load Selected")
        self.btn_load_db_series.clicked.connect(self._on_load_selected_db_series)
        db_list_layout.addWidget(self.btn_load_db_series)

        self.btn_view_excel_info = QPushButton("查看表格信息")
        self.btn_view_excel_info.setToolTip("查看选中患者对应的Excel表格信息（可选择患者、检查或序列节点）")
        self.btn_view_excel_info.clicked.connect(self._on_view_excel_info)
        db_list_layout.addWidget(self.btn_view_excel_info)

        db_list_group.setLayout(db_list_layout)
        main_layout.addWidget(db_list_group, stretch=1) # 列表组占据剩余空间

        # 底部按钮
        bottom_buttons_layout = QHBoxLayout()
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        bottom_buttons_layout.addStretch()
        bottom_buttons_layout.addWidget(btn_cancel)
        main_layout.addLayout(bottom_buttons_layout)
    
    # ========== 数据库相关槽函数（在弹窗内部使用） ==========
    
    def set_current_series(self, entry: SeriesEntry):
        """设置当前已加载的序列，用于自动定位"""
        self.current_series_entry = entry
        # 如果数据库已经连接，立即尝试定位
        if self.db_root and self.db_series_list.topLevelItemCount() > 0:
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(200, self._locate_current_series)
    
    def set_current_image_path(self, image_path: str):
        """设置当前图像路径，用于路径定位"""
        self.current_image_path = image_path
        # 如果数据库已经连接，立即尝试定位
        if self.db_root and self.db_series_list.topLevelItemCount() > 0:
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(200, lambda: self._locate_by_path(image_path))
    
    def _locate_by_path(self, image_path: str):
        """通过文件路径定位病人和序列"""
        if not image_path or not self.db_root:
            return
        
        # 标准化路径
        image_path = os.path.normpath(os.path.abspath(image_path))
        db_root = os.path.normpath(os.path.abspath(self.db_root))
        
        # 检查路径是否在数据库根目录下
        if not image_path.startswith(db_root):
            return
        
        # 从路径中提取病人ID（例如：.../data/83-88/83/... 或 .../data/83/...）
        rel_path = os.path.relpath(image_path, db_root)
        path_parts = rel_path.split(os.sep)
        
        # 查找病人目录（通常是数字，如 83, 84 等）
        patient_id = None
        for i, part in enumerate(path_parts):
            if part.isdigit():
                patient_id = part
                break
        
        if not patient_id:
            return
        
        # 查找对应的病人节点
        patient_item = None
        for i in range(self.db_series_list.topLevelItemCount()):
            item = self.db_series_list.topLevelItem(i)
            data = item.data(0, Qt.UserRole)
            if data and isinstance(data, dict) and data.get("type") == "patient":
                patient_dir = data.get("patient_dir", "")
                dir_basename = os.path.basename(patient_dir).strip()
                if dir_basename == patient_id:
                    patient_item = item
                    break
        
        if not patient_item:
            return
        
        # 展开病人节点（触发懒加载）
        if not patient_item.isExpanded():
            patient_item.setExpanded(True)
            # 等待懒加载完成后，通过路径匹配序列
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(1000, lambda: self._locate_series_by_path(patient_item, image_path))
        else:
            # 已经展开，直接定位
            self._locate_series_by_path(patient_item, image_path)
    
    def _locate_series_by_path(self, patient_item, target_path: str):
        """在已展开的病人节点中，通过路径匹配序列"""
        target_path = os.path.normpath(os.path.abspath(target_path))
        
        # 遍历所有序列节点
        for i in range(patient_item.childCount()):
            series_item = patient_item.child(i)
            entry = series_item.data(0, Qt.UserRole)
            
            if not entry or not isinstance(entry, SeriesEntry):
                continue
            
            # 通过目录路径匹配
            if entry.series_dir:
                entry_dir = os.path.normpath(os.path.abspath(entry.series_dir))
                # 检查目标路径是否在序列目录中，或序列目录在目标路径中
                if (entry_dir == target_path or 
                    target_path.startswith(entry_dir) or 
                    entry_dir.startswith(target_path)):
                    # 找到匹配的序列
                    self.db_series_list.setCurrentItem(series_item)
                    self.db_series_list.scrollToItem(series_item, QAbstractItemView.PositionAtCenter)
                    patient_item.setExpanded(True)
                    return
    
    def _locate_current_series(self):
        """定位并选中当前已加载的序列节点"""
        if not self.current_series_entry:
            return
        
        entry = self.current_series_entry
        patient_id = str(entry.patient_id).strip() if entry.patient_id else None
        series_dir = entry.series_dir if entry.series_dir else None
        
        # 如果patient_id为空，尝试从series_dir路径中提取
        if not patient_id and series_dir:
            # 尝试从路径中提取患者ID（例如：.../83-88/83/... 或 .../data/83/...）
            path_parts = os.path.normpath(series_dir).split(os.sep)
            # 查找可能的患者ID（数字）
            for part in reversed(path_parts):
                if part.isdigit():
                    patient_id = part
                    break
        
        if not patient_id:
            return
        
        # 找到患者节点（支持多种匹配方式）
        patient_item = None
        for i in range(self.db_series_list.topLevelItemCount()):
            item = self.db_series_list.topLevelItem(i)
            data = item.data(0, Qt.UserRole)
            if data and isinstance(data, dict) and data.get("type") == "patient":
                patient_dir = data.get("patient_dir", "")
                dir_basename = os.path.basename(patient_dir).strip()
                # 尝试多种匹配方式
                if (dir_basename == patient_id or 
                    dir_basename.endswith(patient_id) or 
                    patient_id.endswith(dir_basename) or
                    dir_basename in patient_id or
                    patient_id in dir_basename):
                    patient_item = item
                    break
        
        if not patient_item:
            return
        
        # 如果患者节点未展开，先展开它（这会触发懒加载）
        if not patient_item.isExpanded():
            patient_item.setExpanded(True)
            # 等待懒加载完成，使用QTimer延迟定位
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(1000, lambda: self._locate_series_after_expand(patient_item, entry))
        else:
            # 已经展开，直接定位
            self._locate_series_in_patient(patient_item, entry)
    
    def _locate_series_after_expand(self, patient_item, target_entry):
        """在患者节点展开后定位序列"""
        if patient_item.isExpanded():
            self._locate_series_in_patient(patient_item, target_entry)
        else:
            # 如果还没展开，再等一会儿
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(500, lambda: self._locate_series_after_expand(patient_item, target_entry))
    
    def _locate_series_in_patient(self, patient_item, target_entry):
        """在已展开的患者节点中查找并选中序列（序列直接是患者节点的子节点）"""
        target_series_uid = str(target_entry.series_uid).strip() if target_entry.series_uid else None
        target_series_dir = target_entry.series_dir if target_entry.series_dir else None
        
        # 标准化目标目录路径用于比较
        if target_series_dir:
            target_series_dir = os.path.normpath(os.path.abspath(target_series_dir))
        
        # 遍历所有子节点（直接是序列节点）
        for i in range(patient_item.childCount()):
            series_item = patient_item.child(i)
            entry = series_item.data(0, Qt.UserRole)
            
            if not entry or not isinstance(entry, SeriesEntry):
                continue
            
            # 匹配方式1：通过SeriesUID匹配（最准确）
            if target_series_uid:
                entry_series_uid = str(entry.series_uid).strip() if entry.series_uid else None
                if entry_series_uid and entry_series_uid == target_series_uid:
                    # 找到匹配的序列，选中并滚动到它
                    self.db_series_list.setCurrentItem(series_item)
                    self.db_series_list.scrollToItem(series_item, QAbstractItemView.PositionAtCenter)
                    patient_item.setExpanded(True)
                    return
            
            # 匹配方式2：通过目录路径匹配（备用方案）
            if target_series_dir and entry.series_dir:
                entry_series_dir = os.path.normpath(os.path.abspath(entry.series_dir))
                # 检查路径是否相同或一个包含另一个
                if entry_series_dir == target_series_dir or \
                   entry_series_dir in target_series_dir or \
                   target_series_dir in entry_series_dir:
                    # 找到匹配的序列
                    self.db_series_list.setCurrentItem(series_item)
                    self.db_series_list.scrollToItem(series_item, QAbstractItemView.PositionAtCenter)
                    patient_item.setExpanded(True)
                    return

    def _connect_database_auto(self, folder_path: str):
        """
        自动连接数据库（不弹出对话框），用于恢复上次的数据库连接
        """
        if not folder_path or not os.path.isdir(folder_path):
            return
        
        self.db_root = folder_path
        self.db_series_list.clear()
        self.db_entries = []
        
        # 设置Excel根目录（但不立即加载，延迟到需要时再加载）
        self.excel_root = folder_path
        if os.path.basename(folder_path) == "data":
            self.excel_root = os.path.dirname(folder_path)
        
        # 不立即初始化Excel读取器，延迟加载
        self.excel_reader = None

        patient_dirs = find_patient_dirs_522(folder_path)
        
        if not patient_dirs:
            self.db_status_label.setText(f"DB: 未找到病人目录\n{folder_path}")
            return

        for pdir in patient_dirs:
            pid = os.path.basename(pdir)
            p_item = QTreeWidgetItem()
            p_item.setText(0, pid)
            p_item.setToolTip(0, f"Patient dir: {pdir}\n展开以加载该病人的序列列表")
            p_item.setData(0, Qt.UserRole, {"type": "patient", "patient_dir": pdir, "loaded": False})
            placeholder = QTreeWidgetItem(["(expand to load...)"])
            placeholder.setData(0, Qt.UserRole, None)
            p_item.addChild(placeholder)
            self.db_series_list.addTopLevelItem(p_item)

        status_text = f"DB: connected ({len(patient_dirs)} patients)\n{folder_path}"
        self.db_status_label.setText(status_text)
        
        # 延迟执行定位，确保UI更新完成
        from PyQt5.QtCore import QTimer
        # 尝试通过路径定位病人（优先）
        if self.current_image_path:
            QTimer.singleShot(200, lambda: self._locate_by_path(self.current_image_path))
        # 如果已加载了序列，尝试自动定位
        elif self.current_series_entry:
            QTimer.singleShot(200, self._locate_current_series)

    def _on_connect_database(self):
        """
        连接 522 数据库根目录（通常选择 522/data）

        采用"懒加载"策略：这里只列出病人目录，不做全库扫描。
        """
        folder_path = QFileDialog.getExistingDirectory(
            self, "选择数据库根目录（建议选择 522/data）", self.db_root or ""
        )
        if not folder_path:
            return

        self.db_root = folder_path
        self.db_series_list.clear()
        self.db_entries = []

        # 设置Excel根目录（但不立即加载，延迟到需要时再加载）
        # Excel文件通常在522目录下，而不是data目录下
        # 如果folder_path是.../522/data，则Excel在.../522目录下
        self.excel_root = folder_path
        if os.path.basename(folder_path) == "data":
            self.excel_root = os.path.dirname(folder_path)  # 上一级目录（522目录）
        
        # 不立即初始化Excel读取器，延迟加载
        self.excel_reader = None

        patient_dirs = find_patient_dirs_522(folder_path)
        
        if not patient_dirs:
            QMessageBox.warning(self, "警告", 
                f"在选定的目录中未找到病人目录。\n\n"
                f"请确保选择的是 522/data 目录（应包含 83-88、117-135 等范围文件夹）。\n\n"
                f"当前选择的路径：\n{folder_path}")
            self.db_status_label.setText(f"DB: 未找到病人目录\n{folder_path}")
            return
        
        for pdir in patient_dirs:
            pid = os.path.basename(pdir)
            p_item = QTreeWidgetItem()
            p_item.setText(0, pid)
            p_item.setToolTip(0, f"Patient dir: {pdir}\n展开以加载该病人的序列列表")
            # UserRole 中存 patient_dir，并标记“未加载”
            p_item.setData(0, Qt.UserRole, {"type": "patient", "patient_dir": pdir, "loaded": False})
            # 放一个占位子节点，让 UI 显示可展开箭头
            placeholder = QTreeWidgetItem(["(expand to load...)"])
            placeholder.setData(0, Qt.UserRole, None)  # 占位节点没有数据
            p_item.addChild(placeholder)
            self.db_series_list.addTopLevelItem(p_item)

        status_text = f"DB: connected ({len(patient_dirs)} patients)\n{folder_path}"
        self.db_status_label.setText(status_text)
        # 不要自动展开，让用户手动点击展开箭头
        # self.db_series_list.expandToDepth(0)
        
        # 延迟执行定位，确保UI更新完成
        from PyQt5.QtCore import QTimer
        # 尝试通过路径定位病人（优先）
        if self.current_image_path:
            QTimer.singleShot(200, lambda: self._locate_by_path(self.current_image_path))
        # 如果已加载了序列，尝试自动定位
        elif self.current_series_entry:
            QTimer.singleShot(200, self._locate_current_series)

    def _on_db_item_expanded(self, item: QTreeWidgetItem):
        """懒加载：展开病人节点时扫描该病人的序列（SeriesUID粒度）"""
        payload = item.data(0, Qt.UserRole)
        if not isinstance(payload, dict):
            return
        if payload.get("type") != "patient":
            return
        if payload.get("loaded"):
            return

        patient_dir = payload.get("patient_dir")
        if not patient_dir:
            return

        # 清掉占位
        item.takeChildren()

        # 若上一个病人扫描还在跑，先请求中断
        if self.patient_worker is not None and self.patient_worker.isRunning():
            self.patient_worker.requestInterruption()

        progress = QProgressDialog(f"正在扫描该病人序列...\n{os.path.basename(patient_dir)}", "取消", 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        self.patient_worker = PatientIndexWorker(patient_dir)

        def _done(entries):
            progress.close()
            # 直接将序列作为患者节点的子节点（不显示Study层级）
            for e in entries:
                leaf = QTreeWidgetItem()
                # 第0列：Patient ID（使用序列ID或描述）
                series_desc = (e.series_description or "").strip()
                uid_short = (e.series_uid or "")[-8:] if e.series_uid else ""
                series_text = f"{e.series_id or ''} {series_desc} ({uid_short})".strip()
                if not series_text:
                    series_text = os.path.basename(e.series_dir) if e.series_dir else "Series"
                leaf.setText(0, series_text)
                # 第1列：Patient Name
                leaf.setText(1, e.patient_name or "")
                # 第2列：Study ID（保留信息但不作为层级）
                leaf.setText(2, e.study_id or "")
                # 第3列：Study Date
                leaf.setText(3, e.study_date or "")
                # 第4列：Modality
                leaf.setText(4, e.modality or "")
                # 第5列：Series Description（详细信息）
                leaf.setText(5, series_text)
                leaf.setData(0, Qt.UserRole, e)
                leaf.setToolTip(0, e.tooltip_text)
                item.addChild(leaf)

            payload["loaded"] = True
            item.setData(0, Qt.UserRole, payload)
            item.setExpanded(True)

        def _err(err):
            progress.close()
            QMessageBox.critical(self, "错误", f"扫描该病人序列失败:\n{err}")

        self.patient_worker.finished.connect(_done)
        self.patient_worker.error.connect(_err)
        progress.canceled.connect(self.patient_worker.requestInterruption)
        self.patient_worker.start()

    def _on_load_selected_db_series(self, *_args):
        """加载数据库列表中选中的序列"""
        item = self.db_series_list.currentItem()
        if not item:
            QMessageBox.information(self, "提示", "请先在列表中选择一个序列")
            return

        # QTreeWidgetItem: 数据存放在第0列的 UserRole 中
        entry = item.data(0, Qt.UserRole)
        # 只有叶子节点才会绑定 entry；点到病人/检查节点时给出提示
        if not entry or not hasattr(entry, "file_list"):
            QMessageBox.information(self, "提示", "请展开并选择具体序列（叶子节点）再加载")
            return

        try:
            # 精确到单个 DICOM Series（使用具体的文件列表）
            data, metadata = self.image_io.load_dicom_series(entry.file_list)
            if data is not None:
                # 成功加载后发出信号，并关闭对话框
                self.series_selected.emit({"data": data, "metadata": metadata, "entry": entry})
                self.accept()
            else:
                QMessageBox.warning(self, "错误", "数据库序列加载失败")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载数据库序列时发生错误:\n{e}")

    def _on_view_excel_info(self):
        """查看选中患者对应的Excel表格信息（支持从任何层级节点提取患者ID）"""
        # 如果Excel读取器未初始化，且Excel根目录存在，则延迟初始化
        if not self.excel_reader:
            if not self.excel_root or not os.path.exists(self.excel_root):
                QMessageBox.information(self, "提示", "未设置Excel目录，请先连接数据库")
                return
            try:
                self.excel_reader = ExcelDataReader(self.excel_root)
            except Exception as e:
                QMessageBox.warning(self, "错误", f"加载Excel表格失败: {e}")
                return
        
        if not self.excel_reader.has_data():
            QMessageBox.information(self, "提示", "Excel表格为空或未找到数据")
            return
        
        # 获取当前选中的项目
        item = self.db_series_list.currentItem()
        if not item:
            QMessageBox.information(self, "提示", "请先选择一个项目（患者、检查或序列）")
            return
        
        # 尝试从选中的项目获取患者ID
        patient_id = None
        
        # 方法1: 从当前节点的数据中获取
        data = item.data(0, Qt.UserRole)
        
        if data and isinstance(data, dict):
            # 如果是患者节点
            if data.get("type") == "patient":
                patient_dir = data.get("patient_dir", "")
                patient_id = os.path.basename(patient_dir)
        elif hasattr(data, "patient_id"):
            # 如果data是SeriesEntry对象（序列节点）
            patient_id = data.patient_id
        
        # 方法2: 如果没找到，向上遍历树结构找到患者节点
        if not patient_id:
            current = item
            while current:
                node_data = current.data(0, Qt.UserRole)
                if node_data and isinstance(node_data, dict):
                    if node_data.get("type") == "patient":
                        patient_dir = node_data.get("patient_dir", "")
                        patient_id = os.path.basename(patient_dir)
                        break
                elif hasattr(node_data, "patient_id") and node_data.patient_id:
                    patient_id = node_data.patient_id
                    break
                # 向上遍历到父节点
                current = current.parent()
        
        # 方法3: 如果还是没找到，尝试从顶层节点（患者节点）的文本中提取
        if not patient_id:
            # 找到顶层节点（患者节点）
            top_item = item
            while top_item.parent():
                top_item = top_item.parent()
            # 顶层节点的第一列通常是患者ID
            text = top_item.text(0)
            if text and text.strip() and not text.startswith("("):  # 排除占位符
                patient_id = text.strip()
        
        if not patient_id:
            QMessageBox.information(self, "提示", "无法确定患者ID，请选择患者、检查或序列节点")
            return
        
        # 从Excel中获取患者信息（用于自动选择）
        excel_data = None
        if self.excel_reader:
            excel_data = self.excel_reader.get_patient_info(patient_id)
        
        # 检查excel_root是否存在
        if not self.excel_root or not os.path.exists(self.excel_root):
            QMessageBox.warning(self, "警告", f"Excel目录不存在: {self.excel_root}")
            return
        
        # 显示表格信息对话框（即使没有excel_data也显示，让用户手动选择）
        dialog = ExcelInfoDialog(patient_id, self.excel_root, excel_data, self)
        dialog.exec_()

    def _on_db_filter_changed(self, text: str):
        """数据库树过滤：隐藏不匹配的叶子，并自动隐藏空的父节点"""
        q = (text or "").strip().lower()

        def match_item(it: QTreeWidgetItem) -> bool:
            # 任何列包含关键字即匹配；空关键字则全部匹配
            if not q:
                return True
            for col in range(self.db_series_list.columnCount()):
                if (it.text(col) or "").lower().find(q) != -1:
                    return True
            return False

        # 只对已有节点进行可见性设置
        for i in range(self.db_series_list.topLevelItemCount()):
            p_item = self.db_series_list.topLevelItem(i)
            p_visible = False

            for j in range(p_item.childCount()):
                s_item = p_item.child(j)
                s_visible = False

                for k in range(s_item.childCount()):
                    leaf = s_item.child(k)
                    ok = match_item(leaf) or match_item(s_item) or match_item(p_item)
                    leaf.setHidden(not ok)
                    if ok:
                        s_visible = True

                # 如果 study 本身匹配，也显示所有叶子
                if match_item(s_item) and q:
                    for k in range(s_item.childCount()):
                        s_item.child(k).setHidden(False)
                    s_visible = True

                s_item.setHidden(not s_visible)
                if s_visible:
                    p_visible = True

            # 如果 patient 本身匹配，也显示全部
            if match_item(p_item) and q:
                for j in range(p_item.childCount()):
                    p_item.child(j).setHidden(False)
                    s_item = p_item.child(j)
                    for k in range(s_item.childCount()):
                        s_item.child(k).setHidden(False)
                p_visible = True

            p_item.setHidden(not p_visible)

        # 有过滤关键字时自动展开，方便看到结果
        if q:
            self.db_series_list.expandAll()

    def get_current_patient_code(self):
        """获取当前患者代码"""
        # 从 current_patient_info 中提取 PatientID
        if self.current_patient_info:
            p_code = self.current_patient_info.get("PatientID", "")
            if p_code and p_code not in ["--", "Unknown"]:
                return p_code
        return ""

    def refresh_patient_tables(self, p_code):
        """ˢ���Ҳ����������չʾ��"""
        stat_dict, check_dict = self.data_service.get_patient_excel_records(p_code)

        # ����ͳ�Ʊ� (self.table_stat �� QTableWidget)
        self.table_stat.setRowCount(0)
        if stat_dict:
            for key, value in stat_dict.items():
                # �ų����ڲ� ID �� �ظ��ı����
                if key.lower() in ['id', '���']: continue 
                row = self.table_stat.rowCount()
                self.table_stat.insertRow(row)
                self.table_stat.setItem(row, 0, QTableWidgetItem(str(key)))
                self.table_stat.setItem(row, 1, QTableWidgetItem(str(value)))

        # ���¼���� (self.table_check �� QTableWidget)
        self.table_check.setRowCount(0)
        if check_dict:
            for key, value in check_dict.items():
                if key.lower() in ['id', '���˱��']: continue
                row = self.table_check.rowCount()
                self.table_check.insertRow(row)
                self.table_check.setItem(row, 0, QTableWidgetItem(str(key)))
                self.table_check.setItem(row, 1, QTableWidgetItem(str(value)))
