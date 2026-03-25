#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主界面布局模块 (集成病例归档功能)
功能：实现界面布局，左侧工具栏，中间视图，右侧参数，以及病例归档/回溯
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
        self.setWindowTitle("病例历史档案 (日志文件)")
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
            if hasattr(self.db, 'get_patient_history'):
                records = self.db.get_patient_history(keyword)
            else:
                # 兼容旧版 database.py 的 get_all_records
                records = self.db.get_all_records()
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
        self.ai_interface = None  # 延迟初始化，加快启动速度
        
        # 后台预加载AI模型
        self._preload_ai_model()
        
        # === [新增] 初始化数据库管理器 ===
        self.db = None
        if DB_AVAILABLE:
            try:
                from database import DatabaseManager
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
        
        # 左侧面板宽度设置，保持灵活性
        panel.setMinimumWidth(300)
        panel.setMaximumWidth(400)
        
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
        archive_group = QGroupBox("病例归档 (日志文件)")
        archive_layout = QVBoxLayout()
        
        self.btn_save_case_db = QPushButton("保存病例到库")
        self.btn_save_case_db.setToolTip("将当前影像及分割结果归档到日志文件")
        self.btn_save_case_db.clicked.connect(self._on_save_case_to_db)
        archive_layout.addWidget(self.btn_save_case_db)
        
        self.btn_search_case_db = QPushButton("搜索历史病例")
        self.btn_search_case_db.setToolTip("搜索日志文件中的历史分割记录并回溯")
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
    
    def _on_save_screenshot(self):
        """保存截图"""
        try:
            filepath, _ = QFileDialog.getSaveFileName(
                self, "保存截图", "",
                "PNG文件 (*.png);;JPEG文件 (*.jpg);;所有文件 (*.*)"
            )
            if filepath:
                # 实现截图保存逻辑
                # 这里可以使用Qt的截图功能
                screen = self.four_view.grab()
                screen.save(filepath)
                QMessageBox.information(self, "成功", "截图保存成功")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存截图时发生错误:\n{e}")

    def _on_brush_size_changed(self, value):
        """画笔大小改变"""
        pass

    def _on_undo(self):
        """撤销操作"""
        pass

    def _on_redo(self):
        """重做操作"""
        pass

    def _on_load_label(self):
        """加载标签"""
        pass

    def _on_save_label(self):
        """保存标签"""
        pass

    def _on_clear_label(self):
        """清除标签"""
        pass

    def _on_label_opacity_changed(self, value):
        """标签透明度改变"""
        pass

    def _on_label_selection_changed(self):
        """标签选择改变"""
        pass

    def _on_add_label(self):
        """添加标签"""
        pass

    def _on_delete_label(self):
        """删除标签"""
        pass

    def _on_label_color(self):
        """标签颜色选择"""
        pass

    def _on_update_3d_view_clicked(self):
        """更新3D视图"""
        pass

    def _on_toggle_seed_mode(self):
        """切换种子点模式"""
        pass

    def _on_clear_seeds(self):
        """清除种子点"""
        pass

    def _on_segment_3d(self):
        """3D分割"""
        pass

    def _on_ai_segmentation(self):
        """AI自动分割"""
        if self.current_image_data is None:
            QMessageBox.warning(self, "提示", "请先加载一张影像")
            return

        # 显示状态栏提示
        self.statusBar().showMessage("正在初始化AI分割...")
        
        # 确保AI接口已初始化
        if not self.ai_interface:
            self.statusBar().showMessage("正在加载AI模型，请稍候...")
            try:
                # 延迟导入，减少启动时的模块加载
                from ai_interface import AISegmentationInterface
                self.ai_interface = AISegmentationInterface()
                
                # 模型路径
                model_path = os.path.join(os.path.dirname(__file__), 'model', 'best_model.pth')
                
                # 加载模型
                if not self.ai_interface.load_model(model_path):
                    # 尝试使用 pre4.py 直接进行分割
                    self.statusBar().showMessage("模型加载失败，尝试使用 pre4.py 直接分割...")
                    try:
                        # 导入 pre4 模块
                        import sys
                        import os
                        sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
                        from pre4 import run_inference
                        
                        # 保存当前图像为临时文件
                        import tempfile
                        import nibabel as nib
                        
                        with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tmp:
                            tmp_path = tmp.name
                        
                        # 保存临时 NIfTI 文件
                        tmp_nii = nib.Nifti1Image(self.current_image_data, affine=np.eye(4))
                        nib.save(tmp_nii, tmp_path)
                        
                        # 修改 pre4.py 的输入路径
                        import pre4
                        pre4.INPUT_IMAGE = tmp_path
                        
                        # 执行分割
                        run_inference()
                        
                        # 清理临时文件
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                        
                        self.statusBar().showMessage("AI分割完成，结果已保存到 vis_results 目录")
                        return
                    except Exception as e:
                        self.statusBar().showMessage("AI分割失败")
                        QMessageBox.critical(self, "错误", f"使用 pre4.py 分割失败:\n{e}")
                        return
            except Exception as e:
                self.statusBar().showMessage("AI分割初始化失败")
                QMessageBox.critical(self, "错误", f"AI分割初始化失败:\n{e}")
                return

        # 执行分割
        self.statusBar().showMessage("正在执行AI分割，请稍候...")
        try:
            mask = self.ai_interface.segment(self.current_image_data)

            if mask is not None:
                # 更新当前标签数据
                self.current_label_data = mask
                # 更新视图
                self.four_view.set_label_data(mask)
                self.statusBar().showMessage("AI分割完成")
            else:
                self.statusBar().showMessage("AI分割失败")
                QMessageBox.critical(self, "错误", "AI分割失败")

        except Exception as e:
            self.statusBar().showMessage("AI分割过程中发生错误")
            QMessageBox.critical(self, "错误", f"AI分割过程中发生错误:\n{e}")

    def _on_open_db_dialog(self):
        """打开数据库对话框"""
        pass

    def _on_tool_selected(self, tool):
        """选择工具"""
        pass

    def _on_crosshair_changed(self, x, y, z):
        """十字准线变化"""
        pass

    def _on_slice_changed(self, view, slice_idx):
        """切片变化"""
        pass

    def _update_seed_count(self):
        """更新种子点数量"""
        pass
    
    def _preload_ai_model(self):
        """后台预加载AI模型"""
        class ModelPreloadWorker(QThread):
            finished = pyqtSignal(bool)
            
            def __init__(self, parent):
                super().__init__(parent)
                self.parent = parent
            
            def run(self):
                try:
                    # 延迟导入，减少启动时的模块加载
                    from ai_interface import AISegmentationInterface
                    
                    # 初始化AI接口
                    ai_interface = AISegmentationInterface()
                    
                    # 模型路径
                    model_path = os.path.join(os.path.dirname(__file__), 'model', 'best_model.pth')
                    
                    # 尝试加载模型
                    if os.path.exists(model_path):
                        ai_interface.load_model(model_path)
                    
                    # 保存到父对象
                    self.parent.ai_interface = ai_interface
                    self.finished.emit(True)
                except Exception as e:
                    print(f"模型预加载失败: {e}")
                    self.finished.emit(False)
        
        # 创建并启动预加载线程
        self.preload_worker = ModelPreloadWorker(self)
        self.preload_worker.finished.connect(self._on_model_preloaded)
        self.preload_worker.start()
    
    def _on_model_preloaded(self, success):
        """模型预加载完成回调"""
        if success:
            print("AI模型预加载成功")
        else:
            print("AI模型预加载失败")

    def _build_entry_from_metadata(self, metadata, path):
        """从元数据构建entry"""
        return None

    def _apply_patient_info(self, info):
        """应用病人信息"""
        pass

    def get_current_patient_code(self):
        """获取当前病人代码"""
        return ""

    def refresh_patient_tables(self, p_code):
        """刷新病人表格"""
        pass

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
        """保存病例到日志文件（归档）"""
        if not self.db:
            QMessageBox.warning(self, "错误", "日志文件管理器未初始化，无法保存。")
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
        try:
            temp_dir = tempfile.gettempdir()
            temp_mask_path = os.path.join(temp_dir, f"temp_mask_{datetime.now().strftime('%H%M%S')}.nii.gz")
            
            # 使用 ImageIO 保存 mask
            mask_data = self.current_label_data.astype(np.uint8) if self.current_label_data is not None else np.zeros_like(self.current_image_data, dtype=np.uint8)
            
            self.image_io.save_label_map(mask_data, temp_mask_path, self.current_metadata)
            
            # 4. 调用日志文件保存
            img_path_to_save = self.current_image_path
            is_folder = os.path.isdir(img_path_to_save)
            
            if is_folder:
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
                QMessageBox.warning(self, "失败", "归档失败，请检查日志或文件权限。")
                
            # 清理临时文件
            if os.path.exists(temp_mask_path):
                os.remove(temp_mask_path)
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存过程中发生错误:\n{e}")

    def _on_search_case_history(self):
        """搜索历史病例"""
        if not self.db:
            QMessageBox.warning(self, "错误", "日志文件管理器未初始化。")
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

        改为"懒加载"策略：这里只列出病人目录，不做全库扫描。
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
            # UserRole 中存 patient_dir，并标记"未加载"
            p_item.setData(0, Qt.UserRole, {"type": "patient", "patient_dir": pdir, "loaded": False})
            # 放一个占位子节点，让 UI 显示可展开箭头
            placeholder_item = QTreeWidgetItem(["(expand to load...)"])
            p_item.addChild(placeholder_item)
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
        
        # 应用病人信息
        self._apply_patient_info(info)
