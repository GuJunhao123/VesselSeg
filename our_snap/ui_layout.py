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
    QAbstractItemView, QInputDialog, QTextEdit, QSplitter, QScrollArea,
    QFrame, QApplication
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

            # 兜底：若主算法输出为空，自动尝试一轮更稳健的region competition
            if mask is None or not np.any(mask):
                fallback_iters = max(int(self.params.get('iterations', 100)), 120)
                fallback_smooth = max(float(self.params.get('smoothness', 1.0)), 1.0)
                fallback_expand = max(float(self.params.get('expansion_force', 1.0)), 1.2)
                mask = self.algorithm.region_competition(
                    iterations=fallback_iters,
                    smoothness=fallback_smooth,
                    expansion_force=fallback_expand
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
    case_selected = pyqtSignal(str, str)  # signal(original_path, mask_path)

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
                    records = [r for r in records if
                               keyword in str(r.get('patient_name', '')) or keyword in str(r.get('patient_code', ''))]

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


class DatabaseDialog(QDialog):
    """数据库对话框 - 用于浏览和选择DICOM序列"""

    def __init__(self, db_root: str, parent=None):
        super().__init__(parent)
        self.db_root = db_root
        self.selected_entry = None
        self.setWindowTitle("数据库浏览器")
        self.setMinimumSize(800, 600)
        self._init_ui()
        self._load_patient_dirs()

    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()

        # 顶部信息
        info_label = QLabel(f"数据库路径: {self.db_root}")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)

        # 左侧：病人列表
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("病人列表:"))
        self.patient_list = QListWidget()
        self.patient_list.itemClicked.connect(self._on_patient_selected)
        left_layout.addWidget(self.patient_list)
        left_panel.setLayout(left_layout)
        splitter.addWidget(left_panel)

        # 右侧：序列列表
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("序列列表:"))
        self.series_list = QListWidget()
        self.series_list.itemDoubleClicked.connect(self._on_series_double_clicked)
        right_layout.addWidget(self.series_list)
        right_panel.setLayout(right_layout)
        splitter.addWidget(right_panel)

        # 设置分割器比例
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

        # 底部按钮
        button_layout = QHBoxLayout()
        btn_load = QPushButton("加载选中序列")
        btn_load.clicked.connect(self._on_load_clicked)
        button_layout.addWidget(btn_load)

        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(btn_cancel)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def _load_patient_dirs(self):
        """加载病人目录列表"""
        self.patient_list.clear()
        try:
            patient_dirs = find_patient_dirs_522(self.db_root)
            for patient_dir in patient_dirs:
                patient_name = os.path.basename(patient_dir)
                item = QListWidgetItem(patient_name)
                item.setData(Qt.UserRole, patient_dir)
                self.patient_list.addItem(item)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载病人列表失败:\n{e}")

    def _on_patient_selected(self, item):
        """病人选择改变"""
        patient_dir = item.data(Qt.UserRole)
        self._load_series_list(patient_dir)

    def _load_series_list(self, patient_dir: str):
        """加载序列列表"""
        self.series_list.clear()
        try:
            # 创建进度对话框
            progress = QProgressDialog("正在扫描序列...", "取消", 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            # 创建工作线程扫描序列
            worker = PatientIndexWorker(patient_dir)
            worker.finished.connect(self._on_series_scan_finished)
            worker.error.connect(self._on_series_scan_error)

            # 保存进度对话框引用
            self._scan_progress = progress

            worker.start()

        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载序列列表失败:\n{e}")

    def _on_series_scan_finished(self, entries):
        """序列扫描完成"""
        if hasattr(self, '_scan_progress'):
            self._scan_progress.close()

        for entry in entries:
            item = QListWidgetItem(entry.display_name)
            item.setData(Qt.UserRole, entry)
            item.setToolTip(entry.tooltip_text)
            self.series_list.addItem(item)

    def _on_series_scan_error(self, error_msg):
        """序列扫描错误"""
        if hasattr(self, '_scan_progress'):
            self._scan_progress.close()
        QMessageBox.warning(self, "错误", f"序列扫描失败:\n{error_msg}")

    def _on_series_double_clicked(self, item):
        """双击序列项"""
        self.selected_entry = item.data(Qt.UserRole)
        self.accept()

    def _on_load_clicked(self):
        """加载按钮点击"""
        current_item = self.series_list.currentItem()
        if current_item:
            self.selected_entry = current_item.data(Qt.UserRole)
            self.accept()
        else:
            QMessageBox.warning(self, "提示", "请先选择一个序列")

    def get_selected_entry(self):
        """获取选中的序列条目"""
        return self.selected_entry


class LabelViewerWindow(QWidget):
    """独立标签查看窗口（用于医生标签/AI标签并行对比）。"""

    def __init__(self, title: str, role_text: str, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowTitle(title)
        self.setMinimumSize(900, 700)

        self.role_text = role_text
        self.image_io = ImageIO()
        self.base_image_data = None
        self.base_spacing = None
        self.current_label_data = None

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel(self.role_text))
        self.source_label = QLabel("标签来源: 未设置")
        self.source_label.setWordWrap(True)
        top_bar.addWidget(self.source_label, stretch=1)

        btn_load = QPushButton("加载标签文件")
        btn_load.clicked.connect(self._on_load_label_file)
        top_bar.addWidget(btn_load)

        btn_clear = QPushButton("清空标签")
        btn_clear.clicked.connect(self._on_clear_label)
        top_bar.addWidget(btn_clear)

        btn_update_3d = QPushButton("更新3D")
        btn_update_3d.clicked.connect(self._on_update_3d)
        top_bar.addWidget(btn_update_3d)
        layout.addLayout(top_bar)

        self.four_view = FourViewWidget()
        layout.addWidget(self.four_view, stretch=1)

    def _refresh_3d_safely(self):
        """刷新3D视图（失败时静默，避免打断主流程）。"""
        try:
            self.four_view.refresh_3d_view()
        except Exception:
            pass

    def set_base_image(self, image_data, spacing=None):
        """设置基础影像（两个窗口保持一致）。"""
        if image_data is None:
            return

        self.base_image_data = np.asarray(image_data)
        self.base_spacing = spacing
        self.four_view.set_image_data(self.base_image_data, spacing)

        if self.current_label_data is None or self.current_label_data.shape != self.base_image_data.shape:
            self.current_label_data = np.zeros(self.base_image_data.shape, dtype=np.uint8)
            self.four_view.set_label_data(self.current_label_data)

        self._refresh_3d_safely()

    def set_label_data(self, label_data, source_text: str = "") -> bool:
        """设置标签数据。"""
        if self.base_image_data is None:
            QMessageBox.warning(self, "提示", "请先在主窗口加载影像。")
            return False
        if label_data is None:
            return False

        arr = np.asarray(label_data).astype(np.uint8)
        if arr.shape != self.base_image_data.shape:
            QMessageBox.warning(
                self,
                "错误",
                f"标签尺寸与影像不匹配\n影像尺寸: {self.base_image_data.shape}\n标签尺寸: {arr.shape}",
            )
            return False

        self.current_label_data = np.copy(arr)
        self.four_view.set_label_data(self.current_label_data)
        if source_text:
            self.source_label.setText(f"标签来源: {source_text}")

        self._refresh_3d_safely()
        return True

    def _on_load_label_file(self):
        """从文件加载标签。"""
        if self.base_image_data is None:
            QMessageBox.warning(self, "提示", "请先在主窗口加载影像。")
            return

        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "加载标签",
            "",
            "NIfTI文件 (*.nii *.nii.gz);;NRRD文件 (*.nrrd);;所有文件 (*.*)",
        )
        if not filepath:
            return

        try:
            label_data = self.image_io.load_label_map(filepath)
            if label_data is None:
                QMessageBox.warning(self, "错误", "标签加载失败")
                return
            if self.set_label_data(label_data, source_text=os.path.basename(filepath)):
                QMessageBox.information(self, "成功", "标签加载成功")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载标签失败:\n{e}")

    def _on_clear_label(self):
        """清空当前窗口标签。"""
        if self.base_image_data is None:
            return

        self.current_label_data = np.zeros(self.base_image_data.shape, dtype=np.uint8)
        self.four_view.set_label_data(self.current_label_data)
        self.source_label.setText("标签来源: 已清空")
        self._refresh_3d_safely()

    def _on_update_3d(self):
        """手动刷新3D。"""
        try:
            self.four_view.refresh_3d_view()
        except Exception as e:
            QMessageBox.warning(self, "提示", f"刷新3D失败:\n{e}")


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("医学影像分割软件")
        # 使用更保守的默认尺寸，避免在小屏幕上被系统强制为“近似全屏”
        self.setGeometry(120, 80, 1280, 820)

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
        self.ai_model_variant = 'mobileuvit_l'
        self.ai_decomposition_method = 'percentile_50'
        self.last_vessel_analysis = {}
        self.last_cut_stats = {}
        self._last_seed = None
        self.doctor_label_window = None
        self.ai_label_window = None
        self.last_doctor_label_data = None
        self.last_ai_label_data = None

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

        # ========== 左侧：工具栏区（使用滚动区域） ==========
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, stretch=1)

        # ========== 中间：2×2视图区 ==========
        center_panel = self._create_center_panel()
        main_layout.addWidget(center_panel, stretch=4)

        # ========== 右侧：参数设置区（使用滚动区域） ==========
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, stretch=1)

    def _create_left_panel(self) -> QWidget:
        """创建左侧工具栏（带滚动区域，解决内容过多问题）"""
        # 创建一个容器widget
        container = QWidget()
        container.setMinimumWidth(280)
        container.setMaximumWidth(350)

        # 主布局
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 滚动区域的内容widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(5, 5, 5, 5)
        content_layout.setSpacing(8)

        # ===== 1. Main Toolbar =====
        main_toolbar_group = QGroupBox("Main Toolbar")
        main_toolbar_layout = QVBoxLayout()
        main_toolbar_layout.setSpacing(5)

        # 工具按钮组 - 使用网格布局避免重叠
        from PyQt5.QtWidgets import QGridLayout
        tools_grid = QGridLayout()
        tools_grid.setSpacing(3)

        # Pointer（指针工具）
        self.btn_pointer = QPushButton("↖")
        self.btn_pointer.setToolTip("Pointer Tool")
        self.btn_pointer.setCheckable(True)
        self.btn_pointer.setChecked(True)
        self.btn_pointer.setFixedSize(35, 35)
        self.btn_pointer.clicked.connect(lambda: self._on_tool_selected('pointer'))
        tools_grid.addWidget(self.btn_pointer, 0, 0)

        # Zoom（缩放工具）
        self.btn_zoom = QPushButton("🔍")
        self.btn_zoom.setToolTip("Zoom Tool")
        self.btn_zoom.setCheckable(True)
        self.btn_zoom.setFixedSize(35, 35)
        self.btn_zoom.clicked.connect(lambda: self._on_tool_selected('zoom'))
        tools_grid.addWidget(self.btn_zoom, 0, 1)

        # Pan（平移工具）
        self.btn_pan = QPushButton("✋")
        self.btn_pan.setToolTip("Pan Tool")
        self.btn_pan.setCheckable(True)
        self.btn_pan.setFixedSize(35, 35)
        self.btn_pan.clicked.connect(lambda: self._on_tool_selected('pan'))
        tools_grid.addWidget(self.btn_pan, 0, 2)

        # Brush（画笔工具）
        self.btn_brush = QPushButton("🖌")
        self.btn_brush.setToolTip("Brush Tool")
        self.btn_brush.setCheckable(True)
        self.btn_brush.setFixedSize(35, 35)
        self.btn_brush.clicked.connect(lambda: self._on_tool_selected('brush'))
        tools_grid.addWidget(self.btn_brush, 1, 0)

        # Eraser（橡皮擦工具）
        self.btn_eraser = QPushButton("🧹")
        self.btn_eraser.setToolTip("Eraser Tool")
        self.btn_eraser.setCheckable(True)
        self.btn_eraser.setFixedSize(35, 35)
        self.btn_eraser.clicked.connect(lambda: self._on_tool_selected('eraser'))
        tools_grid.addWidget(self.btn_eraser, 1, 1)

        # Save Screenshot（保存截图）
        self.btn_screenshot = QPushButton("📷")
        self.btn_screenshot.setToolTip("Save Screenshot")
        self.btn_screenshot.setFixedSize(35, 35)
        self.btn_screenshot.clicked.connect(self._on_save_screenshot)
        tools_grid.addWidget(self.btn_screenshot, 1, 2)

        main_toolbar_layout.addLayout(tools_grid)

        # 撤销/重做按钮
        undo_redo_layout = QHBoxLayout()
        undo_redo_layout.setSpacing(5)
        self.btn_undo = QPushButton("↶ Undo")
        self.btn_undo.setToolTip("Undo (Ctrl+Z)")
        self.btn_undo.setFixedHeight(30)
        self.btn_undo.clicked.connect(self._on_undo)
        undo_redo_layout.addWidget(self.btn_undo)

        self.btn_redo = QPushButton("↷ Redo")
        self.btn_redo.setToolTip("Redo (Ctrl+Y)")
        self.btn_redo.setFixedHeight(30)
        self.btn_redo.clicked.connect(self._on_redo)
        undo_redo_layout.addWidget(self.btn_redo)
        main_toolbar_layout.addLayout(undo_redo_layout)

        # 画笔大小设置
        brush_size_layout = QHBoxLayout()
        brush_size_layout.setSpacing(5)
        brush_size_layout.addWidget(QLabel("画笔大小:"))
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setMinimum(1)
        self.brush_size_spin.setMaximum(50)
        self.brush_size_spin.setValue(3)
        self.brush_size_spin.valueChanged.connect(self._on_brush_size_changed)
        brush_size_layout.addWidget(self.brush_size_spin)
        brush_size_layout.addStretch()
        main_toolbar_layout.addLayout(brush_size_layout)

        # 文件操作按钮
        btn_load_image = QPushButton("📂 加载影像")
        btn_load_image.setFixedHeight(32)
        btn_load_image.clicked.connect(self._on_load_image)
        main_toolbar_layout.addWidget(btn_load_image)

        btn_load_dicom_folder = QPushButton("📁 加载DICOM文件夹")
        btn_load_dicom_folder.setFixedHeight(32)
        btn_load_dicom_folder.clicked.connect(self._on_load_dicom_folder)
        main_toolbar_layout.addWidget(btn_load_dicom_folder)

        btn_load_label = QPushButton("🏷️ 加载标签")
        btn_load_label.setFixedHeight(32)
        btn_load_label.clicked.connect(self._on_load_label)
        main_toolbar_layout.addWidget(btn_load_label)

        btn_save_label = QPushButton("💾 保存标签")
        btn_save_label.setFixedHeight(32)
        btn_save_label.clicked.connect(self._on_save_label)
        main_toolbar_layout.addWidget(btn_save_label)

        btn_compare_labels = QPushButton("🪟 双窗口标签对比")
        btn_compare_labels.setFixedHeight(32)
        btn_compare_labels.clicked.connect(self._on_open_dual_label_windows)
        main_toolbar_layout.addWidget(btn_compare_labels)

        main_toolbar_group.setLayout(main_toolbar_layout)
        content_layout.addWidget(main_toolbar_group)

        # 当前工具模式
        self.current_tool_mode = 'pointer'
        self.seed_mode_enabled = False

        # ===== 2. Cursor Inspector =====
        cursor_group = QGroupBox("光标信息")
        cursor_layout = QVBoxLayout()
        cursor_layout.setSpacing(5)

        # Cursor position
        self.cursor_pos_label = QLabel("位置 (x,y,z): --")
        self.cursor_pos_label.setWordWrap(True)
        cursor_layout.addWidget(self.cursor_pos_label)

        # Intensity under cursor
        self.cursor_intensity_label = QLabel("像素值:")
        self.cursor_layer_label = QLabel("  图层: --")
        self.cursor_intensity_value_label = QLabel("  强度: --")
        cursor_layout.addWidget(self.cursor_intensity_label)
        cursor_layout.addWidget(self.cursor_layer_label)
        cursor_layout.addWidget(self.cursor_intensity_value_label)

        # Label under cursor
        self.cursor_label_label = QLabel("标签值:")
        self.cursor_label_value_label = QLabel("  标签: --")
        self.btn_clear_label = QPushButton("清除标签")
        self.btn_clear_label.setFixedHeight(30)
        self.btn_clear_label.clicked.connect(self._on_clear_label)
        cursor_layout.addWidget(self.cursor_label_label)
        cursor_layout.addWidget(self.cursor_label_value_label)
        cursor_layout.addWidget(self.btn_clear_label)

        cursor_group.setLayout(cursor_layout)
        content_layout.addWidget(cursor_group)

        # ===== 3. Patient Info =====
        patient_group = QGroupBox("病人信息")
        patient_layout = QVBoxLayout()
        patient_layout.setSpacing(3)

        self.pi_id_label = QLabel("ID: --")
        self.pi_name_label = QLabel("姓名: --")
        self.pi_sex_label = QLabel("性别: --")
        self.pi_birth_label = QLabel("出生日期: --")
        self.pi_study_date_label = QLabel("检查日期: --")
        self.pi_modality_label = QLabel("模态: --")
        self.pi_series_label = QLabel("序列: --")

        for w in [
            self.pi_id_label,
            self.pi_name_label,
            self.pi_sex_label,
            self.pi_birth_label,
            self.pi_study_date_label,
            self.pi_modality_label,
            self.pi_series_label,
        ]:
            w.setWordWrap(True)
            patient_layout.addWidget(w)

        patient_group.setLayout(patient_layout)
        content_layout.addWidget(patient_group)

        # ===== 4. Segmentation Labels =====
        labels_group = QGroupBox("分割标签")
        labels_layout = QVBoxLayout()
        labels_layout.setSpacing(5)

        # Active label（活动标签）
        active_label_layout = QHBoxLayout()
        active_label_layout.setSpacing(5)
        self.active_label_color_widget = QWidget()
        self.active_label_color_widget.setFixedSize(20, 20)
        self.active_label_color_widget.setStyleSheet("background-color: red; border: 1px solid black;")
        active_label_layout.addWidget(self.active_label_color_widget)
        self.active_label_name_label = QLabel("活动标签: Label 1")
        active_label_layout.addWidget(self.active_label_name_label)
        active_label_layout.addStretch()
        labels_layout.addLayout(active_label_layout)

        # Paint over（绘制覆盖标签）
        self.paint_over_label = QLabel("绘制覆盖: Label 1")
        labels_layout.addWidget(self.paint_over_label)

        # Overall label opacity（标签透明度滑块）
        opacity_layout = QHBoxLayout()
        opacity_layout.setSpacing(5)
        opacity_layout.addWidget(QLabel("标签透明度:"))
        self.label_opacity_slider = QSlider(Qt.Horizontal)
        self.label_opacity_slider.setMinimum(0)
        self.label_opacity_slider.setMaximum(100)
        self.label_opacity_slider.setValue(50)
        self.label_opacity_slider.valueChanged.connect(self._on_label_opacity_changed)
        opacity_layout.addWidget(self.label_opacity_slider)
        self.label_opacity_value_label = QLabel("50%")
        self.label_opacity_value_label.setMinimumWidth(40)
        opacity_layout.addWidget(self.label_opacity_value_label)
        labels_layout.addLayout(opacity_layout)

        # 标签列表
        self.label_list_widget = QListWidget()
        self.label_list_widget.setMaximumHeight(100)
        self.label_list_widget.itemSelectionChanged.connect(self._on_label_selection_changed)
        labels_layout.addWidget(self.label_list_widget)

        # 标签操作按钮 - 水平布局
        label_buttons_layout = QHBoxLayout()
        label_buttons_layout.setSpacing(5)

        btn_add_label = QPushButton("添加标签")
        btn_add_label.setFixedHeight(30)
        btn_add_label.clicked.connect(self._on_add_label)
        label_buttons_layout.addWidget(btn_add_label)

        btn_delete_label = QPushButton("删除标签")
        btn_delete_label.setFixedHeight(30)
        btn_delete_label.clicked.connect(self._on_delete_label)
        label_buttons_layout.addWidget(btn_delete_label)

        btn_label_color = QPushButton("标签颜色")
        btn_label_color.setFixedHeight(30)
        btn_label_color.clicked.connect(self._on_label_color)
        label_buttons_layout.addWidget(btn_label_color)

        labels_layout.addLayout(label_buttons_layout)

        labels_group.setLayout(labels_layout)
        content_layout.addWidget(labels_group)

        # 初始化活动标签
        self.active_label_id = 1

        # ===== 5. 3D操作按钮 =====
        toolbar_3d_group = QGroupBox("3D操作")
        toolbar_3d_layout = QVBoxLayout()
        toolbar_3d_layout.setSpacing(5)

        # Update（更新3D视图）
        btn_update_3d = QPushButton("🔄 更新3D视图")
        btn_update_3d.setFixedHeight(32)
        btn_update_3d.clicked.connect(self._on_update_3d_view_clicked)
        toolbar_3d_layout.addWidget(btn_update_3d)

        # 种子点相关按钮 - 水平布局
        seed_layout = QHBoxLayout()
        seed_layout.setSpacing(5)
        self.btn_add_seed = QPushButton("➕ 添加种子点")
        self.btn_add_seed.setFixedHeight(32)
        self.btn_add_seed.setCheckable(True)
        self.btn_add_seed.clicked.connect(self._on_toggle_seed_mode)
        seed_layout.addWidget(self.btn_add_seed)

        self.btn_clear_seeds = QPushButton("🗑️ 清除种子点")
        self.btn_clear_seeds.setFixedHeight(32)
        self.btn_clear_seeds.clicked.connect(self._on_clear_seeds)
        seed_layout.addWidget(self.btn_clear_seeds)
        toolbar_3d_layout.addLayout(seed_layout)

        # 种子点数量显示
        self.seed_count_label = QLabel("种子点数量: 0")
        toolbar_3d_layout.addWidget(self.seed_count_label)

        # 分割相关按钮
        btn_label_stats = QPushButton("📊 标签统计")
        btn_label_stats.setFixedHeight(32)
        btn_label_stats.clicked.connect(self._on_label_statistics)
        toolbar_3d_layout.addWidget(btn_label_stats)

        btn_segment_3d = QPushButton("✂️ 3D分割")
        btn_segment_3d.setFixedHeight(32)
        btn_segment_3d.clicked.connect(self._on_segment_3d)
        toolbar_3d_layout.addWidget(btn_segment_3d)

        btn_ai_seg = QPushButton("🤖 AI自动分割")
        btn_ai_seg.setFixedHeight(32)
        btn_ai_seg.clicked.connect(self._on_ai_segmentation)
        toolbar_3d_layout.addWidget(btn_ai_seg)

        toolbar_3d_group.setLayout(toolbar_3d_layout)
        content_layout.addWidget(toolbar_3d_group)

        # ===== 6. 量化分析 =====
        quant_analysis_group = QGroupBox("量化分析")
        quant_analysis_layout = QVBoxLayout()
        quant_analysis_layout.setSpacing(5)

        btn_quant_analysis = QPushButton("📈 运行量化分析")
        btn_quant_analysis.setFixedHeight(32)
        btn_quant_analysis.clicked.connect(self._on_quantitative_analysis)
        quant_analysis_layout.addWidget(btn_quant_analysis)

        quant_analysis_group.setLayout(quant_analysis_layout)
        content_layout.addWidget(quant_analysis_group)

        # ===== 7. 3D可视化 =====
        three_d_visualization_group = QGroupBox("3D可视化")
        three_d_visualization_layout = QVBoxLayout()
        three_d_visualization_layout.setSpacing(5)

        btn_3d_visualization = QPushButton("🎨 显示3D血管树")
        btn_3d_visualization.setFixedHeight(32)
        btn_3d_visualization.clicked.connect(self._on_3d_visualization)
        three_d_visualization_layout.addWidget(btn_3d_visualization)

        btn_cutting_plane = QPushButton("✂️ 添加切割平面")
        btn_cutting_plane.setFixedHeight(32)
        btn_cutting_plane.clicked.connect(self._on_add_cutting_plane)
        three_d_visualization_layout.addWidget(btn_cutting_plane)

        three_d_visualization_group.setLayout(three_d_visualization_layout)
        content_layout.addWidget(three_d_visualization_group)

        # ===== 8. 报告生成 =====
        report_group = QGroupBox("报告生成")
        report_layout = QVBoxLayout()
        report_layout.setSpacing(5)

        btn_generate_report = QPushButton("📄 生成PDF报告")
        btn_generate_report.setFixedHeight(32)
        btn_generate_report.clicked.connect(self._on_generate_report)
        report_layout.addWidget(btn_generate_report)

        report_group.setLayout(report_layout)
        content_layout.addWidget(report_group)

        # ===== 9. 数据库对接（522范式） =====
        btn_open_db_dialog = QPushButton("🗄️ 打开数据库 (522)")
        btn_open_db_dialog.setFixedHeight(32)
        btn_open_db_dialog.clicked.connect(self._on_open_db_dialog)
        content_layout.addWidget(btn_open_db_dialog)

        # ===== 10. 病例归档/PACS系统 =====
        archive_group = QGroupBox("病例归档")
        archive_layout = QVBoxLayout()
        archive_layout.setSpacing(5)

        self.btn_save_case_db = QPushButton("💾 保存病例到库")
        self.btn_save_case_db.setFixedHeight(32)
        self.btn_save_case_db.clicked.connect(self._on_save_case_to_db)
        archive_layout.addWidget(self.btn_save_case_db)

        self.btn_search_case_db = QPushButton("🔍 搜索历史病例")
        self.btn_search_case_db.setFixedHeight(32)
        self.btn_search_case_db.clicked.connect(self._on_search_case_history)
        archive_layout.addWidget(self.btn_search_case_db)

        archive_group.setLayout(archive_layout)
        content_layout.addWidget(archive_group)

        # 添加弹性空间，让内容可以滚动到底部
        content_layout.addStretch()

        # 设置滚动区域的内容
        scroll_area.setWidget(content_widget)

        # 将滚动区域添加到主布局
        main_layout.addWidget(scroll_area)

        return container

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
        """创建右侧参数设置区（分割算法参数 + 窗宽窗位）"""
        container = QWidget()
        container.setMaximumWidth(250)
        container.setMinimumWidth(220)

        # 主布局
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 滚动区域的内容widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(5, 5, 5, 5)
        content_layout.setSpacing(8)

        # 窗宽窗位参数组
        window_group = QGroupBox("窗宽窗位")
        window_layout = QVBoxLayout()
        window_layout.setSpacing(5)

        # 窗宽
        window_layout.addWidget(QLabel("窗宽:"))
        self.window_width_spin = QDoubleSpinBox()
        self.window_width_spin.setMinimum(1.0)
        self.window_width_spin.setMaximum(4000.0)
        self.window_width_spin.setValue(400.0)
        self.window_width_spin.setSingleStep(10.0)
        self.window_width_spin.valueChanged.connect(self._on_window_width_changed)
        window_layout.addWidget(self.window_width_spin)

        # 窗位
        window_layout.addWidget(QLabel("窗位:"))
        self.window_level_spin = QDoubleSpinBox()
        self.window_level_spin.setMinimum(-2000.0)
        self.window_level_spin.setMaximum(2000.0)
        self.window_level_spin.setValue(0.0)
        self.window_level_spin.setSingleStep(10.0)
        self.window_level_spin.valueChanged.connect(self._on_window_level_changed)
        window_layout.addWidget(self.window_level_spin)

        # 自动窗宽窗位按钮
        btn_auto_window = QPushButton("自动")
        btn_auto_window.setFixedHeight(30)
        btn_auto_window.clicked.connect(self._on_auto_window)
        window_layout.addWidget(btn_auto_window)

        # 默认窗宽窗位按钮
        btn_default_window = QPushButton("默认")
        btn_default_window.setFixedHeight(30)
        btn_default_window.clicked.connect(self._on_default_window)
        window_layout.addWidget(btn_default_window)

        window_group.setLayout(window_layout)
        content_layout.addWidget(window_group)

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
        content_layout.addWidget(algo_group)

        content_layout.addStretch()

        # 设置滚动区域的内容
        scroll_area.setWidget(content_widget)

        # 将滚动区域添加到主布局
        main_layout.addWidget(scroll_area)

        return container

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
        if hasattr(self.four_view, 'axial_view'):
            self.four_view.axial_view.set_brush_size(value)
            self.four_view.coronal_view.set_brush_size(value)
            self.four_view.sagittal_view.set_brush_size(value)

    def _on_undo(self):
        """撤销操作"""
        if self.undo_history:
            # 保存当前状态到重做栈
            if self.current_label_data is not None:
                self.redo_history.append(np.copy(self.current_label_data))
                # 限制重做栈大小
                if len(self.redo_history) > self.max_history_size:
                    self.redo_history.pop(0)

            # 从撤销栈恢复
            self.current_label_data = self.undo_history.pop()
            self.four_view.set_label_data(self.current_label_data)
            self._refresh_structure_from_current_labels()

    def _on_redo(self):
        """重做操作"""
        if self.redo_history:
            # 保存当前状态到撤销栈
            if self.current_label_data is not None:
                self.undo_history.append(np.copy(self.current_label_data))
                # 限制撤销栈大小
                if len(self.undo_history) > self.max_history_size:
                    self.undo_history.pop(0)

            # 从重做栈恢复
            self.current_label_data = self.redo_history.pop()
            self.four_view.set_label_data(self.current_label_data)
            self._refresh_structure_from_current_labels()

    def _get_current_spacing(self):
        """获取当前体数据 spacing。"""
        if self.current_metadata and 'spacing' in self.current_metadata:
            return self.current_metadata['spacing']
        return None

    def _ensure_dual_label_windows(self):
        """确保医生/AI两个独立标签窗口存在。"""
        if self.doctor_label_window is None:
            self.doctor_label_window = LabelViewerWindow("医生标签窗口", "医生标签", parent=None)
            self.doctor_label_window.destroyed.connect(
                lambda _obj=None: setattr(self, 'doctor_label_window', None)
            )

        if self.ai_label_window is None:
            self.ai_label_window = LabelViewerWindow("AI标签窗口", "AI标签", parent=None)
            self.ai_label_window.destroyed.connect(
                lambda _obj=None: setattr(self, 'ai_label_window', None)
            )

    def _sync_dual_windows_base_image(self, reset_labels: bool = False):
        """把主窗口当前影像同步到两个独立窗口。"""
        if self.current_image_data is None:
            return

        spacing = self._get_current_spacing()
        windows = [self.doctor_label_window, self.ai_label_window]
        for window in windows:
            if window is None:
                continue
            window.set_base_image(self.current_image_data, spacing)
            if reset_labels:
                empty_label = np.zeros(self.current_image_data.shape, dtype=np.uint8)
                window.set_label_data(empty_label, source_text="已清空")

    def _set_doctor_compare_label(self, label_data, source_text: str = ""):
        """更新医生标签窗口内容，并缓存最近一次医生标签。"""
        if label_data is None:
            return
        self.last_doctor_label_data = np.asarray(label_data).astype(np.uint8).copy()
        if self.doctor_label_window is not None:
            self._sync_dual_windows_base_image(reset_labels=False)
            self.doctor_label_window.set_label_data(self.last_doctor_label_data, source_text=source_text)

    def _set_ai_compare_label(self, label_data, source_text: str = ""):
        """更新AI标签窗口内容，并缓存最近一次AI标签。"""
        if label_data is None:
            return
        self.last_ai_label_data = np.asarray(label_data).astype(np.uint8).copy()
        if self.ai_label_window is not None:
            self._sync_dual_windows_base_image(reset_labels=False)
            self.ai_label_window.set_label_data(self.last_ai_label_data, source_text=source_text)

    def _on_open_dual_label_windows(self):
        """打开医生标签窗口 + AI标签窗口。"""
        if self.current_image_data is None:
            QMessageBox.warning(self, "提示", "请先加载影像，再打开双窗口对比。")
            return

        self._ensure_dual_label_windows()
        self._sync_dual_windows_base_image(reset_labels=False)

        if self.last_doctor_label_data is not None:
            self.doctor_label_window.set_label_data(self.last_doctor_label_data, source_text="医生标签(最近加载)")

        if self.last_ai_label_data is not None:
            self.ai_label_window.set_label_data(self.last_ai_label_data, source_text="AI标签(最近生成)")
        elif self.current_label_data is not None and np.any(self.current_label_data):
            self.ai_label_window.set_label_data(self.current_label_data, source_text="主窗口当前标签")

        self.doctor_label_window.show()
        self.ai_label_window.show()

        base_x = max(self.x() + 20, 0)
        base_y = max(self.y() + 40, 0)
        self.doctor_label_window.move(base_x, base_y)
        self.ai_label_window.move(base_x + self.doctor_label_window.width() + 20, base_y)

        self.doctor_label_window.raise_()
        self.ai_label_window.raise_()
        self.doctor_label_window.activateWindow()

    def _on_load_label(self):
        """加载标签"""
        if self.current_image_data is None:
            QMessageBox.warning(self, "提示", "请先加载一张影像")
            return

        filepath, _ = QFileDialog.getOpenFileName(
            self, "加载标签", "",
            "NIfTI文件 (*.nii *.nii.gz);;NRRD文件 (*.nrrd);;所有文件 (*.*)"
        )

        if filepath:
            try:
                label_data = self.image_io.load_label_map(filepath)
                if label_data is not None:
                    # 检查尺寸是否匹配
                    if label_data.shape == self.current_image_data.shape:
                        # 保存当前状态到撤销栈
                        if self.current_label_data is not None:
                            self.undo_history.append(np.copy(self.current_label_data))
                            # 限制撤销栈大小
                            if len(self.undo_history) > self.max_history_size:
                                self.undo_history.pop(0)

                        # 更新标签数据
                        self.current_label_data = label_data
                        self.four_view.set_label_data(label_data)
                        self._refresh_structure_from_current_labels()
                        # 清空重做栈
                        self.redo_history.clear()
                        self._set_doctor_compare_label(label_data, source_text=os.path.basename(filepath))
                        QMessageBox.information(self, "成功", "标签加载成功")
                    else:
                        QMessageBox.warning(self, "错误",
                                            f"标签尺寸与影像不匹配\n影像尺寸: {self.current_image_data.shape}\n标签尺寸: {label_data.shape}")
                else:
                    QMessageBox.warning(self, "错误", "标签加载失败")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载标签时发生错误:\n{e}")

    def _on_save_label(self):
        """保存标签"""
        if self.current_label_data is None:
            QMessageBox.warning(self, "提示", "没有标签数据可保存")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "保存标签", "",
            "NIfTI文件 (*.nii.gz);;NRRD文件 (*.nrrd);;所有文件 (*.*)"
        )

        if filepath:
            try:
                self.image_io.save_label_map(self.current_label_data, filepath, self.current_metadata)
                QMessageBox.information(self, "成功", "标签保存成功")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存标签时发生错误:\n{e}")

    def _on_clear_label(self):
        """清除标签"""
        if self.current_label_data is not None:
            # 保存当前状态到撤销栈
            self.undo_history.append(np.copy(self.current_label_data))
            # 限制撤销栈大小
            if len(self.undo_history) > self.max_history_size:
                self.undo_history.pop(0)

            # 清空标签
            self.current_label_data = np.zeros(self.current_image_data.shape, dtype=np.uint8)
            self.four_view.set_label_data(self.current_label_data)
            self._reset_analysis_state()
            # 清空重做栈
            self.redo_history.clear()
            QMessageBox.information(self, "成功", "标签已清除")

    def _on_label_opacity_changed(self, value):
        """标签透明度改变"""
        opacity = value / 100.0
        self.label_opacity_value_label.setText(f"{value}%")

        # 更新所有视图的标签透明度
        if hasattr(self.four_view, 'axial_view'):
            try:
                self.four_view.axial_view.set_label_opacity(opacity)
                self.four_view.coronal_view.set_label_opacity(opacity)
                self.four_view.sagittal_view.set_label_opacity(opacity)
            except Exception as e:
                print(f"设置标签透明度时发生错误: {e}")

    def _on_window_width_changed(self, value):
        """窗宽改变"""
        if hasattr(self.four_view, 'axial_view'):
            try:
                window_level = self.window_level_spin.value()
                self.four_view.axial_view.set_window(value, window_level)
                self.four_view.coronal_view.set_window(value, window_level)
                self.four_view.sagittal_view.set_window(value, window_level)
            except Exception as e:
                print(f"设置窗宽时发生错误: {e}")

    def _on_window_level_changed(self, value):
        """窗位改变"""
        if hasattr(self.four_view, 'axial_view'):
            try:
                window_width = self.window_width_spin.value()
                self.four_view.axial_view.set_window(window_width, value)
                self.four_view.coronal_view.set_window(window_width, value)
                self.four_view.sagittal_view.set_window(window_width, value)
            except Exception as e:
                print(f"设置窗位时发生错误: {e}")

    def _on_auto_window(self):
        """自动计算窗宽窗位"""
        if self.current_image_data is None:
            QMessageBox.warning(self, "提示", "请先加载一张影像")
            return

        try:
            # 计算图像的统计信息
            img_data = self.current_image_data
            min_val = np.min(img_data)
            max_val = np.max(img_data)
            mean_val = np.mean(img_data)
            std_val = np.std(img_data)

            # 自动计算窗宽窗位
            window_width = std_val * 4  # 4倍标准差
            window_level = mean_val

            # 更新界面
            self.window_width_spin.setValue(window_width)
            self.window_level_spin.setValue(window_level)

            # 应用到视图
            if hasattr(self.four_view, 'axial_view'):
                self.four_view.axial_view.set_window(window_width, window_level)
                self.four_view.coronal_view.set_window(window_width, window_level)
                self.four_view.sagittal_view.set_window(window_width, window_level)

            QMessageBox.information(self, "成功",
                                    f"自动窗宽窗位已设置\n窗宽: {window_width:.1f}\n窗位: {window_level:.1f}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"自动计算窗宽窗位时发生错误:\n{e}")

    def _on_default_window(self):
        """默认窗宽窗位"""
        if self.current_image_data is None:
            QMessageBox.warning(self, "提示", "请先加载一张影像")
            return

        try:
            # 设置默认窗宽窗位
            window_width = 400.0  # 默认窗宽
            window_level = 0.0  # 默认窗位

            # 更新界面
            self.window_width_spin.setValue(window_width)
            self.window_level_spin.setValue(window_level)

            # 应用到视图
            if hasattr(self.four_view, 'axial_view'):
                self.four_view.axial_view.set_window(window_width, window_level)
                self.four_view.coronal_view.set_window(window_width, window_level)
                self.four_view.sagittal_view.set_window(window_width, window_level)

            QMessageBox.information(self, "成功",
                                    f"默认窗宽窗位已设置\n窗宽: {window_width:.1f}\n窗位: {window_level:.1f}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"设置默认窗宽窗位时发生错误:\n{e}")

    def _on_label_statistics(self):
        """标签统计"""
        if self.current_label_data is None:
            QMessageBox.warning(self, "提示", "请先加载或创建标签")
            return

        try:
            # 计算所有标签的统计信息
            unique_labels = np.unique(self.current_label_data)
            unique_labels = unique_labels[unique_labels > 0]  # 排除背景(0)

            if len(unique_labels) == 0:
                QMessageBox.information(self, "提示", "当前没有标签数据")
                return

            stats_text = "标签统计信息:\n\n"

            for label_id in unique_labels:
                label_mask = (self.current_label_data == label_id)
                voxel_count = np.sum(label_mask)

                # 计算物理体积
                if self.current_metadata and 'spacing' in self.current_metadata:
                    spacing = self.current_metadata['spacing']
                    voxel_volume = spacing[0] * spacing[1] * spacing[2]  # mm³
                    volume_mm3 = voxel_count * voxel_volume
                    volume_cm3 = volume_mm3 / 1000.0  # cm³
                    volume_text = f"{volume_cm3:.2f} cm³ ({volume_mm3:.2f} mm³)"
                else:
                    volume_text = f"{voxel_count} 体素"

                # 计算表面积（近似）
                from scipy import ndimage
                # 使用形态学梯度计算表面积
                edges = ndimage.binary_dilation(label_mask) ^ label_mask
                surface_pixels = np.sum(edges)

                if self.current_metadata and 'spacing' in self.current_metadata:
                    spacing = self.current_metadata['spacing']
                    pixel_area = spacing[0] * spacing[1]  # mm²
                    surface_area_mm2 = surface_pixels * pixel_area
                    surface_area_cm2 = surface_area_mm2 / 100.0  # cm²
                    area_text = f"{surface_area_cm2:.2f} cm² ({surface_area_mm2:.2f} mm²)"
                else:
                    area_text = f"{surface_pixels} 像素"

                stats_text += f"标签 {label_id}:\n"
                stats_text += f"  体积: {volume_text}\n"
                stats_text += f"  表面积: {area_text}\n"
                stats_text += f"  体素数: {voxel_count}\n\n"

            QMessageBox.information(self, "标签统计", stats_text)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算标签统计时发生错误:\n{e}")

    def _on_label_selection_changed(self):
        """标签选择改变"""
        selected_items = self.label_list_widget.selectedItems()
        if selected_items:
            item = selected_items[0]
            label_id = int(item.data(Qt.UserRole))
            self.active_label_id = label_id
            self.active_label_name_label.setText(f"活动标签: Label {label_id}")
            self.paint_over_label.setText(f"绘制覆盖: Label {label_id}")
            # 更新视图中的活动标签
            if hasattr(self.four_view, 'axial_view'):
                self.four_view.axial_view.set_active_label_id(label_id)
                self.four_view.coronal_view.set_active_label_id(label_id)
                self.four_view.sagittal_view.set_active_label_id(label_id)

    def _on_add_label(self):
        """添加标签"""
        # 生成新标签ID
        new_label_id = len(self.label_list) + 1
        # 添加到标签列表
        self.label_list.append(new_label_id)
        # 更新标签列表控件
        item = QListWidgetItem(f"Label {new_label_id}")
        item.setData(Qt.UserRole, new_label_id)
        self.label_list_widget.addItem(item)
        # 选择新标签
        self.label_list_widget.setCurrentItem(item)
        # 更新活动标签
        self.active_label_id = new_label_id
        self.active_label_name_label.setText(f"活动标签: Label {new_label_id}")
        self.paint_over_label.setText(f"绘制覆盖: Label {new_label_id}")
        # 更新视图中的活动标签
        if hasattr(self.four_view, 'axial_view'):
            self.four_view.axial_view.set_active_label_id(new_label_id)
            self.four_view.coronal_view.set_active_label_id(new_label_id)
            self.four_view.sagittal_view.set_active_label_id(new_label_id)

    def _on_delete_label(self):
        """删除标签"""
        selected_items = self.label_list_widget.selectedItems()
        if selected_items:
            item = selected_items[0]
            label_id = int(item.data(Qt.UserRole))
            # 从标签列表中移除
            if label_id in self.label_list:
                self.label_list.remove(label_id)
            # 从列表控件中移除
            self.label_list_widget.takeItem(self.label_list_widget.row(item))
            # 如果删除的是当前活动标签，选择第一个标签
            if self.active_label_id == label_id:
                if self.label_list:
                    new_label_id = self.label_list[0]
                    self.active_label_id = new_label_id
                    self.active_label_name_label.setText(f"活动标签: Label {new_label_id}")
                    self.paint_over_label.setText(f"绘制覆盖: Label {new_label_id}")
                    # 更新视图中的活动标签
                    if hasattr(self.four_view, 'axial_view'):
                        self.four_view.axial_view.set_active_label_id(new_label_id)
                        self.four_view.coronal_view.set_active_label_id(new_label_id)
                        self.four_view.sagittal_view.set_active_label_id(new_label_id)
                else:
                    # 如果没有标签了，设置为默认值
                    self.active_label_id = 1
                    self.active_label_name_label.setText("活动标签: Label 1")
                    self.paint_over_label.setText("绘制覆盖: Label 1")

    def _on_label_color(self):
        """标签颜色选择"""
        color = QColorDialog.getColor()
        if color.isValid():
            # 更新活动标签颜色
            color_str = color.name()
            self.active_label_color_widget.setStyleSheet(f"background-color: {color_str}; border: 1px solid black;")
            # 更新视图中的标签颜色
            if hasattr(self.four_view, 'axial_view'):
                self.four_view.axial_view.set_label_color(self.active_label_id, color_str)
                self.four_view.coronal_view.set_label_color(self.active_label_id, color_str)
                self.four_view.sagittal_view.set_label_color(self.active_label_id, color_str)

    def _on_update_3d_view_clicked(self, notify: bool = True):
        """更新主界面内嵌3D视图。"""
        if self.current_image_data is None:
            QMessageBox.warning(self, "提示", "请先加载一张影像")
            return False

        if not (hasattr(self, 'four_view') and hasattr(self.four_view, 'view_3d')):
            QMessageBox.warning(self, "提示", "未检测到3D视图组件，请检查可视化模块配置。")
            return False

        try:
            spacing = self.current_metadata['spacing'] if (
                self.current_metadata and 'spacing' in self.current_metadata
            ) else None
            self.four_view.view_3d.set_image_data(self.current_image_data, spacing)
            self.four_view.view_3d.set_label_data(self.current_label_data)
            structure_map = None
            if getattr(self, 'last_vessel_analysis', None):
                structure_map = self.last_vessel_analysis.get("structure_map")
            self.four_view.view_3d.set_structure_data(structure_map)
            self.four_view.view_3d.refresh_view()
            self.statusBar().showMessage("3D视图已更新", 3000)
            if notify:
                QMessageBox.information(self, "成功", "3D视图已更新（左下角视窗）。")
            return True
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更新3D视图时发生错误:\n{e}")
            return False

    def _on_toggle_seed_mode(self):
        """切换种子点模式"""
        self.seed_mode_enabled = self.btn_add_seed.isChecked()
        if self.seed_mode_enabled:
            # 种子点依赖点击同步十字线，强制切到指针模式避免点了无效
            self._on_tool_selected('pointer')
            self.btn_add_seed.setText("✅ 种子点模式开启")
            self.btn_add_seed.setStyleSheet("background-color: #d1fae5; color: #065f46;")
            self.statusBar().showMessage("种子点模式已开启：已切换到指针工具，请在三视图点击添加种子点", 5000)
        else:
            self.btn_add_seed.setText("➕ 添加种子点")
            self.btn_add_seed.setStyleSheet("")
            self.statusBar().showMessage("种子点模式已关闭", 2000)

    def _on_clear_seeds(self):
        """清除种子点"""
        self.seed_points = []
        self._update_seed_count()
        self.statusBar().showMessage("已清除全部种子点", 3000)
        QMessageBox.information(self, "提示", "种子点已清除")

    def _on_segment_3d(self):
        """3D分割"""
        if self.current_image_data is None:
            QMessageBox.warning(self, "提示", "请先加载影像")
            return

        # 过滤并校验种子点，避免越界导致无结果
        if self.seed_points is None:
            self.seed_points = []
        valid_seeds = []
        sx, sy, sz = self.current_image_data.shape
        for seed in self.seed_points:
            try:
                x, y, z = int(seed[0]), int(seed[1]), int(seed[2])
            except Exception:
                continue
            if 0 <= x < sx and 0 <= y < sy and 0 <= z < sz:
                valid_seeds.append((x, y, z))
        # 去重保持顺序
        valid_seeds = list(dict.fromkeys(valid_seeds))
        self.seed_points = valid_seeds
        self._update_seed_count()

        if len(self.seed_points) == 0:
            QMessageBox.warning(self, "提示", "请先添加至少一个种子点")
            return

        # 获取参数
        algo_name = self.algo_combo.currentText().strip().lower()
        method = 'region_competition' if 'region' in algo_name else 'edge_based'
        params = {
            'method': method,
            'iterations': int(self.iterations_spin.value()),
            'smoothness': float(self.smoothness_spin.value()),
            'expansion_force': float(self.expansion_spin.value()),
            'edge_weight': 1.0,
        }

        # 进度对话框
        progress = QProgressDialog("正在执行3D分割，请稍候...", None, 0, 0, self)
        progress.setWindowTitle("3D分割")
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.show()

        # 后台线程执行，避免界面卡死
        worker = SegmentationWorker(
            algorithm=self.segmentation,
            image_data=self.current_image_data,
            seed_points=list(self.seed_points),
            params=params
        )
        self._segment_worker = worker

        worker.finished.connect(lambda mask: self._on_segment_3d_finished(mask, progress))
        worker.error.connect(lambda err: self._on_segment_3d_error(err, progress))
        worker.start()

    def _on_segment_3d_finished(self, mask, progress_dialog: QProgressDialog):
        """3D分割完成回调"""
        progress_dialog.close()
        if mask is None or not np.any(mask):
            QMessageBox.warning(self, "提示", "3D分割未得到有效结果")
            self.statusBar().showMessage("3D分割完成，但无有效区域", 3000)
            return

        # 保存撤销历史
        if self.current_label_data is not None:
            self.undo_history.append(np.copy(self.current_label_data))
            if len(self.undo_history) > self.max_history_size:
                self.undo_history.pop(0)
            self.redo_history.clear()

        label_id = int(self.active_label_id) if hasattr(self, 'active_label_id') else 1
        updated = np.copy(self.current_label_data) if self.current_label_data is not None else np.zeros_like(mask, dtype=np.uint8)
        updated[mask > 0] = label_id
        self.current_label_data = updated
        self.four_view.set_label_data(updated)
        self._refresh_structure_from_current_labels()

        self.statusBar().showMessage("3D分割完成", 3000)
        QMessageBox.information(
            self,
            "3D分割完成",
            f"已完成3D分割并写入当前标签(Label {label_id})。\n分割体素数: {int(np.sum(mask > 0))}"
        )

    def _on_segment_3d_error(self, error_msg: str, progress_dialog: QProgressDialog):
        """3D分割失败回调"""
        progress_dialog.close()
        self.statusBar().showMessage("3D分割失败", 3000)
        QMessageBox.critical(self, "错误", f"3D分割失败:\n{error_msg}")

    def _on_ai_segmentation(self):
        """AI自动分割"""
        if self.current_image_data is None:
            QMessageBox.warning(self, "提示", "请先加载一张影像")
            return

        # 显示AI分割参数选择对话框
        if not self._show_ai_segmentation_dialog():
            return

        # 进度提示（避免用户误以为未执行）
        progress = QProgressDialog("正在初始化AI自动分割...", None, 0, 0, self)
        progress.setWindowTitle("AI自动分割")
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.show()
        QApplication.processEvents()
        self.statusBar().showMessage("正在初始化AI分割...")

        # 确保AI接口已初始化
        if not self.ai_interface:
            progress.setLabelText("正在加载AI模型，请稍候...")
            QApplication.processEvents()
            self.statusBar().showMessage("正在加载AI模型，请稍候...")
            try:
                # 延迟导入，减少启动时的模块加载
                from ai_interface import AISegmentationInterface
                self.ai_interface = AISegmentationInterface()
            except Exception as e:
                self.statusBar().showMessage("AI分割初始化失败")
                QMessageBox.critical(self, "错误", f"AI分割初始化失败:\n{e}")
                progress.close()
                return

        # 重新加载模型（使用新选择的参数）
        try:
            progress.setLabelText("正在加载AI模型参数...")
            QApplication.processEvents()
            self.statusBar().showMessage("正在加载AI模型，请稍候...")
            model_path = os.path.join(os.path.dirname(__file__), 'model', 'best_model.pth')

            # 应用用户选择参数
            self.ai_interface.set_model_variant(self.ai_model_variant)
            self.ai_interface.set_decomposition_method(self.ai_decomposition_method)

            # 加载模型
            if not self.ai_interface.load_model(model_path):
                self.statusBar().showMessage("AI分割失败")
                QMessageBox.critical(self, "错误", "AI模型加载失败")
                progress.close()
                return
        except Exception as e:
            self.statusBar().showMessage("AI分割初始化失败")
            QMessageBox.critical(self, "错误", f"AI模型加载失败:\n{e}")
            progress.close()
            return

        # 执行分割
        progress.setLabelText("正在进行图像预处理、推理和后处理...")
        QApplication.processEvents()
        self.statusBar().showMessage("正在执行AI分割，请稍候...")
        try:
            # 仅在成功后替换标签，避免失败时丢失已有结果
            previous_label = np.copy(self.current_label_data) if self.current_label_data is not None else None

            # 执行分割
            spacing = self.current_metadata['spacing'] if (
                self.current_metadata and 'spacing' in self.current_metadata
            ) else (1.0, 1.0, 1.0)
            mask = self.ai_interface.segment(
                self.current_image_data,
                spacing=spacing,
                metadata=self.current_metadata,
            )

            if mask is not None:
                # 更新当前标签数据
                self.current_label_data = mask
                # 更新视图
                self.four_view.set_label_data(mask)
                self.statusBar().showMessage("AI分割完成")

                # 保存血管结构识别结果（主干/细支/分叉/5级）
                self._refresh_structure_from_current_labels()
                self._set_ai_compare_label(mask, source_text="AI自动分割结果")

                # 反馈简要统计
                pv_count = int(np.sum(mask == 1))
                hv_count = int(np.sum(mask == 2))
                pv_l5 = self.last_vessel_analysis.get("pv", {}).get("level_counts", {}).get("level_5_plus", 0)
                hv_l5 = self.last_vessel_analysis.get("hv", {}).get("level_counts", {}).get("level_5_plus", 0)
                QMessageBox.information(
                    self,
                    "AI分割结果",
                    f"分割完成。\n\n"
                    f"门静脉(PV)体素: {pv_count}\n"
                    f"肝静脉(HV)体素: {hv_count}\n"
                    f"PV 5级及以上微小分支体素: {pv_l5}\n"
                    f"HV 5级及以上微小分支体素: {hv_l5}"
                )
            else:
                if previous_label is not None:
                    self.current_label_data = previous_label
                    self.four_view.set_label_data(previous_label)
                self.statusBar().showMessage("AI分割失败")
                QMessageBox.critical(self, "错误", "AI分割失败")

        except Exception as e:
            self.statusBar().showMessage("AI分割过程中发生错误")
            QMessageBox.critical(self, "错误", f"AI分割过程中发生错误:\n{e}")
        finally:
            progress.close()

    def _show_ai_segmentation_dialog(self):
        """显示AI分割参数选择对话框"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QComboBox, QPushButton, QHBoxLayout, QGroupBox, \
            QFormLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("AI分割参数设置")
        dialog.setMinimumSize(400, 250)

        layout = QVBoxLayout()

        # 模型参数组
        model_group = QGroupBox("模型参数")
        model_layout = QFormLayout()

        # 模型变体选择
        model_variant_combo = QComboBox()
        model_variant_combo.addItems(['mobileuvit_l (大型模型 - 高精度)', 'mobileuvit (小型模型 - 高速度)'])
        model_variant_combo.setCurrentIndex(0)
        model_layout.addRow("模型变体:", model_variant_combo)

        # 血管分解方法选择
        decomposition_combo = QComboBox()
        decomposition_combo.addItems([
            'percentile_50 (推荐)',
            'adaptive',
            'fixed_1.0',
            'fixed_1.5',
            'fixed_2.0',
            'fixed_2.5',
            'median',
            'percentile_25',
            'percentile_75'
        ])
        decomposition_combo.setCurrentIndex(0)
        model_layout.addRow("血管分解方法:", decomposition_combo)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # 添加说明
        info_label = QLabel(
            "说明:\n"
            "• mobileuvit_l: 大型模型，精度更高\n"
            "• mobileuvit: 小型模型，速度更快\n"
            "• percentile_50: 使用50%百分位数作为阈值\n"
            "• adaptive: 自适应阈值，适合不同图像"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # 按钮布局
        button_layout = QHBoxLayout()
        btn_ok = QPushButton("开始分割")
        btn_ok.clicked.connect(dialog.accept)
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(dialog.reject)
        button_layout.addWidget(btn_ok)
        button_layout.addWidget(btn_cancel)
        layout.addLayout(button_layout)

        dialog.setLayout(layout)

        # 显示对话框
        if dialog.exec_() == QDialog.Accepted:
            # 获取用户选择的参数
            model_variant = model_variant_combo.currentText().split()[0]
            decomposition_method = decomposition_combo.currentText().split()[0]

            # 先保存参数；若AI接口已初始化则同步更新
            self.ai_model_variant = model_variant
            self.ai_decomposition_method = decomposition_method
            if self.ai_interface is not None:
                self.ai_interface.set_model_variant(model_variant)
                self.ai_interface.set_decomposition_method(decomposition_method)

            return True
        else:
            return False

    def _load_series_entry(self, entry):
        """从数据库条目加载序列"""
        try:
            # 根据entry中的file_list加载序列
            if entry.file_list:
                data, metadata = self.image_io.load_dicom_series(entry.file_list)
            else:
                # 回退到直接加载目录
                data, metadata = self.image_io.load_image(entry.series_dir)

            if data is not None:
                self.current_image_data = data
                self.current_metadata = metadata
                self.current_entry = entry
                self.current_image_path = entry.series_dir

                # 传递spacing信息（如果可用）
                spacing = metadata.get('spacing') if metadata else None
                self.four_view.set_image_data(data, spacing)

                # 初始化标签数据
                self.current_label_data = np.zeros(data.shape, dtype=np.uint8)
                self.four_view.set_label_data(self.current_label_data)
                self.last_doctor_label_data = None
                self.last_ai_label_data = None
                self._sync_dual_windows_base_image(reset_labels=True)
                self._reset_analysis_state()

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
                self.btn_add_seed.setText("➕ 添加种子点")
                self.btn_add_seed.setStyleSheet("")

                # 更新病人信息显示
                self._update_patient_info_from_entry(entry, metadata)

                QMessageBox.information(self, "成功", f"序列加载成功\n尺寸: {data.shape}")
            else:
                QMessageBox.warning(self, "错误", "序列加载失败")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载序列时发生错误:\n{e}")

    def _update_patient_info_from_entry(self, entry, metadata):
        """从数据库条目更新病人信息"""
        # 使用entry中的信息
        self.pi_id_label.setText(f"ID: {entry.patient_id}")
        self.pi_name_label.setText(f"姓名: {entry.patient_name or '--'}")
        self.pi_sex_label.setText("性别: --")
        self.pi_birth_label.setText("出生日期: --")
        self.pi_study_date_label.setText(f"检查日期: {entry.study_date or '--'}")
        self.pi_modality_label.setText(f"模态: {entry.modality or '--'}")
        self.pi_series_label.setText(f"序列: {entry.series_id}")

        # 保存当前病人信息
        self.current_patient_info = {
            "PatientID": entry.patient_id,
            "PatientName": entry.patient_name or "Unknown",
            "StudyDate": entry.study_date,
            "Modality": entry.modality,
            "SeriesID": entry.series_id
        }

    def _update_patient_info_from_metadata(self, metadata):
        """从元数据更新病人信息"""
        dicom_info = metadata.get('dicom_info', {})

        self.pi_id_label.setText(f"ID: {dicom_info.get('PatientID', '--')}")
        self.pi_name_label.setText(f"姓名: {dicom_info.get('PatientName', '--')}")
        self.pi_sex_label.setText(f"性别: {dicom_info.get('PatientSex', '--')}")
        self.pi_birth_label.setText(f"出生日期: {dicom_info.get('PatientBirthDate', '--')}")
        self.pi_study_date_label.setText(f"检查日期: {dicom_info.get('StudyDate', '--')}")
        self.pi_modality_label.setText(f"模态: {dicom_info.get('Modality', '--')}")
        self.pi_series_label.setText(f"序列: {dicom_info.get('SeriesNumber', '--')}")

        # 保存当前病人信息
        self.current_patient_info = {
            "PatientID": dicom_info.get('PatientID', "Unknown"),
            "PatientName": dicom_info.get('PatientName', "Unknown"),
            "PatientSex": dicom_info.get('PatientSex'),
            "PatientBirthDate": dicom_info.get('PatientBirthDate'),
            "StudyDate": dicom_info.get('StudyDate'),
            "Modality": dicom_info.get('Modality'),
            "SeriesID": dicom_info.get('SeriesNumber')
        }

    def _on_open_db_dialog(self):
        """打开数据库对话框"""
        # 选择数据库根目录
        db_root = QFileDialog.getExistingDirectory(
            self, "选择数据库根目录",
            self.last_db_root if self.last_db_root else ""
        )

        if not db_root:
            return

        # 保存选择的路径
        self.last_db_root = db_root

        # 创建并显示数据库对话框
        dialog = DatabaseDialog(db_root, self)
        if dialog.exec_() == QDialog.Accepted:
            # 用户选择了一个序列
            entry = dialog.get_selected_entry()
            if entry:
                self._load_series_entry(entry)

    def _on_tool_selected(self, tool):
        """选择工具"""
        # 重置所有工具按钮
        self.btn_pointer.setChecked(False)
        self.btn_zoom.setChecked(False)
        self.btn_pan.setChecked(False)
        self.btn_brush.setChecked(False)
        self.btn_eraser.setChecked(False)

        # 设置当前工具
        self.current_tool_mode = tool

        # 激活选中的工具按钮
        if tool == 'pointer':
            self.btn_pointer.setChecked(True)
        elif tool == 'zoom':
            self.btn_zoom.setChecked(True)
        elif tool == 'pan':
            self.btn_pan.setChecked(True)
        elif tool == 'brush':
            self.btn_brush.setChecked(True)
        elif tool == 'eraser':
            self.btn_eraser.setChecked(True)

        # 更新视图中的工具模式
        if hasattr(self.four_view, 'axial_view'):
            self.four_view.axial_view.set_tool_mode(tool)
            self.four_view.coronal_view.set_tool_mode(tool)
            self.four_view.sagittal_view.set_tool_mode(tool)

    def _on_quantitative_analysis(self):
        """运行量化分析"""
        if self.current_image_data is None or self.current_label_data is None:
            QMessageBox.warning(self, "提示", "请先加载影像并进行分割")
            return

        try:
            # 延迟导入，减少启动时的模块加载
            from quantitative_analysis import QuantitativeAnalyzer

            # 显示状态栏提示
            self.statusBar().showMessage("正在进行量化分析...")

            # 初始化量化分析器
            analyzer = QuantitativeAnalyzer()
            spacing = self.current_metadata['spacing'] if (
                self.current_metadata and 'spacing' in self.current_metadata
            ) else (1.0, 1.0, 1.0)

            # 提取门静脉和肝静脉掩码
            portal_vein_mask = (self.current_label_data == 1)
            hepatic_vein_mask = (self.current_label_data == 2)
            vessel_union = portal_vein_mask | hepatic_vein_mask
            if not np.any(vessel_union):
                QMessageBox.warning(self, "提示", "当前标签中未检测到PV/HV（标签1/2），请先完成血管分割。")
                self.statusBar().showMessage("量化分析中止：缺少血管标签", 3000)
                return

            # 肝脏掩码优先使用标签3，若不存在则用血管膨胀近似
            liver_mask = (self.current_label_data == 3)
            if not np.any(liver_mask):
                try:
                    from scipy import ndimage
                    if np.any(vessel_union):
                        liver_mask = ndimage.binary_dilation(vessel_union, iterations=20)
                    else:
                        liver_mask = np.ones_like(self.current_label_data, dtype=bool)
                except Exception:
                    liver_mask = np.ones_like(self.current_label_data, dtype=bool)

            # 肿瘤掩码：默认标签4（可扩展）
            tumor_mask = (self.current_label_data == 4)

            # 计算肝脏分段
            segmentation = analyzer.couinaud_segmentation(
                liver_mask,
                portal_vein_mask,
                hepatic_vein_mask
            )

            # 计算体积
            volumes = analyzer.calculate_volumes(segmentation, voxel_size=spacing)

            # 提取血管量化指标
            vascular_metrics = analyzer.extract_vascular_metrics(
                portal_vein_mask,
                hepatic_vein_mask,
                voxel_size=spacing,
                liver_mask=liver_mask,
                vessel_analysis=getattr(self, 'last_vessel_analysis', None)
            )

            # 肿瘤-血管距离评估
            tumor_vessel_assessment = analyzer.assess_tumor_vessel_distance(
                tumor_mask,
                portal_vein_mask,
                hepatic_vein_mask,
                voxel_size=spacing
            )

            # 手术风险评估
            risk_assessment = analyzer.assess_surgical_risk(
                volumes,
                vascular_metrics,
                tumor_vessel_assessment
            )

            # 显示分析结果
            result_text = "量化分析结果:\n\n"
            result_text += "肝脏体积分析:\n"
            for key, value in volumes.items():
                if key != 'total_liver':
                    result_text += f"{key.replace('segment_', '段 ')}: {value:.2f} ml\n"
            result_text += f"总体积: {volumes.get('total_liver', 0):.2f} ml\n\n"

            result_text += "血管量化指标:\n"
            for key, value in vascular_metrics.items():
                if isinstance(value, float):
                    result_text += f"{key}: {value:.2f}\n"
                else:
                    result_text += f"{key}: {value}\n"

            result_text += "\n肿瘤-血管距离评估:\n"
            min_dist = tumor_vessel_assessment.get("min_distance", float("inf"))
            mean_dist = tumor_vessel_assessment.get("mean_distance", float("inf"))
            if np.isfinite(min_dist):
                result_text += f"最小距离: {min_dist:.2f} mm\n"
                result_text += f"平均距离: {mean_dist:.2f} mm\n"
            else:
                result_text += "最小距离: 未知\n平均距离: 未知\n"
            result_text += f"风险等级: {tumor_vessel_assessment.get('risk_level', '未知')}\n"
            result_text += f"风险评分: {tumor_vessel_assessment.get('risk_score', 0)}\n"

            result_text += "\n手术风险评估:\n"
            result_text += f"整体风险: {risk_assessment.get('overall_risk', '未知')}\n"
            result_text += f"建议: {risk_assessment.get('recommendation', '无')}\n"

            self._show_long_text_dialog("量化分析结果", result_text)

            # 保存分析结果
            self.quantitative_results = {
                'volumes': volumes,
                'vascular_metrics': vascular_metrics,
                'tumor_vessel_assessment': tumor_vessel_assessment,
                'risk_assessment': risk_assessment,
                'couinaud_segmentation': segmentation.astype(np.uint8)
            }
            if getattr(self, 'last_cut_stats', None):
                self.quantitative_results['cut_stats'] = self.last_cut_stats

            # 显示状态栏提示
            self.statusBar().showMessage("量化分析完成")

        except Exception as e:
            self.statusBar().showMessage("量化分析失败")
            QMessageBox.critical(self, "错误", f"量化分析失败:\n{e}")

    def _on_3d_visualization(self):
        """显示3D血管树"""
        if self.current_image_data is None or self.current_label_data is None:
            QMessageBox.warning(self, "提示", "请先加载影像并进行分割")
            return

        try:
            ok = self._on_update_3d_view_clicked(notify=False)
            if ok:
                self.statusBar().showMessage("3D血管树已在主界面左下角显示", 4000)
                QMessageBox.information(
                    self,
                    "3D可视化",
                    "3D血管树已更新并显示在主界面左下角视图。\n\n提示：可先点击任意2D切片定位，再执行切割平面模拟。"
                )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"3D可视化失败:\n{e}")

    def _on_add_cutting_plane(self):
        """添加切割平面"""
        if self.current_image_data is None or self.current_label_data is None:
            QMessageBox.warning(self, "提示", "请先加载影像并进行分割")
            return

        if not self._on_update_3d_view_clicked(notify=False):
            return

        viewer = self.four_view.view_3d

        # 1) 法向轴选择
        axis_options = [
            "X轴法向（切YZ平面）",
            "Y轴法向（切XZ平面）",
            "Z轴法向（切XY平面）",
        ]
        axis_choice, ok = QInputDialog.getItem(
            self,
            "切割平面方向",
            "请选择切割平面的法向方向：",
            axis_options,
            0,
            False,
        )
        if not ok:
            return

        normal_map = {
            axis_options[0]: (1.0, 0.0, 0.0),
            axis_options[1]: (0.0, 1.0, 0.0),
            axis_options[2]: (0.0, 0.0, 1.0),
        }
        normal = normal_map[axis_choice]

        # 2) 偏移量
        offset_mm, ok = QInputDialog.getDouble(
            self,
            "切割平面偏移",
            "沿法向偏移量（mm，正负都可）：",
            0.0,
            -500.0,
            500.0,
            1,
        )
        if not ok:
            return

        # 3) 是否应用模型裁切（视觉上切掉一侧）
        apply_choice = QMessageBox.question(
            self,
            "切割显示模式",
            "是否将切割平面直接应用到3D模型（仅保留一侧显示）？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        apply_to_mesh = (apply_choice == QMessageBox.Yes)

        center = viewer.get_volume_center()
        origin = [
            center[0] + normal[0] * offset_mm,
            center[1] + normal[1] * offset_mm,
            center[2] + normal[2] * offset_mm,
        ]

        try:
            stats = viewer.set_clip_plane(normal=normal, origin=origin, apply_to_mesh=apply_to_mesh)
            self.last_cut_stats = stats
            self.statusBar().showMessage("切割平面已更新", 3000)

            source_desc = "无可用体积统计"
            if stats.get("source") == "label_3_liver":
                source_desc = "体积统计依据：标签3（肝脏）"
            elif stats.get("source") == "nonzero_label_proxy":
                source_desc = "体积统计依据：非零标签近似（未检测到肝脏标签）"

            if stats.get("available"):
                summary = (
                    f"切割模拟完成\n\n"
                    f"{source_desc}\n"
                    f"剩余体积: {stats.get('remaining_ml', 0.0):.2f} ml\n"
                    f"切除体积: {stats.get('resected_ml', 0.0):.2f} ml\n"
                    f"剩余比例: {stats.get('remaining_ratio', 0.0) * 100:.1f}%\n\n"
                    f"切断识别:\n"
                    f"- 门静脉(PV): {'是' if stats.get('severed_pv') else '否'}\n"
                    f"- 肝静脉(HV): {'是' if stats.get('severed_hv') else '否'}"
                )
            else:
                summary = (
                    "切割平面已添加，但当前标签不足以进行体积统计。\n"
                    "请先完成分割或加载有效标签。"
                )

            QMessageBox.information(self, "切割平面结果", summary)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"添加切割平面失败:\n{e}")

    def _on_generate_report(self):
        """生成PDF报告"""
        if self.current_image_data is None or self.current_label_data is None:
            QMessageBox.warning(self, "提示", "请先加载影像并进行分割")
            return

        try:
            from report_generator import ReportGenerator

            # 显示状态栏提示
            self.statusBar().showMessage("正在生成PDF报告...")

            # 准备报告数据
            report_data = {
                'patient_id': self.current_patient_info.get('PatientID', 'N/A'),
                'exam_date': self.current_patient_info.get('StudyDate', ''),
                'segmentation_method': 'AI自动分割' if self.ai_interface is not None else '手动/传统分割',
                'model_variant': getattr(self, 'ai_model_variant', 'mobileuvit_l'),
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # 添加量化分析结果
            if hasattr(self, 'quantitative_results'):
                report_data.update(self.quantitative_results)
            else:
                # 若用户未手动点“运行量化分析”，自动补一次
                self._on_quantitative_analysis()
                if hasattr(self, 'quantitative_results'):
                    report_data.update(self.quantitative_results)

            # 添加AI血管结构识别摘要
            if hasattr(self, 'last_vessel_analysis') and self.last_vessel_analysis:
                report_data['vessel_analysis'] = self.last_vessel_analysis

            # 添加切割模拟统计
            if hasattr(self, 'last_cut_stats') and self.last_cut_stats:
                report_data['cut_stats'] = self.last_cut_stats

            # 添加图像
            if self.current_image_data is not None:
                # 提取中间切片作为示例
                middle_slice = self.current_image_data.shape[2] // 2
                slice_data = self.current_image_data[:, :, middle_slice]
                label_slice = self.current_label_data[:, :, middle_slice]
                report_data['images'] = {
                    '肝脏影像 (中间切片)': slice_data,
                    '血管分割标签 (中间切片)': label_slice
                }

            # 生成报告
            generator = ReportGenerator()

            # 选择保存路径
            from PyQt5.QtWidgets import QFileDialog
            filepath, _ = QFileDialog.getSaveFileName(
                self, "保存PDF报告", "", "PDF文件 (*.pdf)"
            )

            if filepath:
                success = generator.generate_report(report_data, filepath)
                if success:
                    QMessageBox.information(self, "成功", f"PDF报告生成成功:\n{filepath}")
                    self.statusBar().showMessage("PDF报告生成成功")
                else:
                    QMessageBox.critical(self, "错误", "PDF报告生成失败")
                    self.statusBar().showMessage("PDF报告生成失败")

        except Exception as e:
            self.statusBar().showMessage("PDF报告生成失败")
            QMessageBox.critical(self, "错误", f"PDF报告生成失败:\n{e}")

    def _on_crosshair_changed(self, view_type, x, y):
        """十字准线变化"""
        if self.current_image_data is None:
            return

        try:
            # 计算z坐标
            z = 0
            if view_type == 'axial':
                if hasattr(self.four_view, 'axial_view'):
                    z = self.four_view.axial_view.current_slice
            elif view_type == 'coronal':
                if hasattr(self.four_view, 'coronal_view'):
                    z = self.four_view.coronal_view.current_slice
            elif view_type == 'sagittal':
                if hasattr(self.four_view, 'sagittal_view'):
                    z = self.four_view.sagittal_view.current_slice

            # 种子点模式：点击三视图时记录当前3D十字准心对应体素
            if self.seed_mode_enabled and hasattr(self, 'four_view'):
                try:
                    vx = int(self.four_view.sagittal_view.current_slice)
                    vy = int(self.four_view.coronal_view.current_slice)
                    vz = int(self.four_view.axial_view.current_slice)
                    seed = (vx, vy, vz)
                    # 去重+轻微防抖
                    if seed != self._last_seed and seed not in self.seed_points:
                        self.seed_points.append(seed)
                        self._last_seed = seed
                        self._update_seed_count()
                        self.statusBar().showMessage(f"已添加种子点: {seed}", 1500)
                except Exception:
                    pass

            # 更新光标位置标签
            self.cursor_pos_label.setText(f"位置 (x,y,z): ({int(x)}, {int(y)}, {int(z)})")

            # 更新光标下的强度值
            if (0 <= x < self.current_image_data.shape[0] and
                    0 <= y < self.current_image_data.shape[1] and
                    0 <= z < self.current_image_data.shape[2]):
                intensity = self.current_image_data[int(x), int(y), int(z)]
                self.cursor_intensity_value_label.setText(f"  强度: {intensity:.1f}")
                self.cursor_layer_label.setText(f"  图层: 影像")

                # 更新光标下的标签值
                if self.current_label_data is not None:
                    label_value = self.current_label_data[int(x), int(y), int(z)]
                    self.cursor_label_value_label.setText(f"  标签: {label_value}")
                else:
                    self.cursor_label_value_label.setText("  标签: --")
        except Exception as e:
            print(f"更新十字准线时发生错误: {e}")

    def _on_slice_changed(self, view, slice_idx):
        """切片变化"""
        if self.current_image_data is None:
            return

        try:
            # 获取当前十字准线位置
            if hasattr(self.four_view, 'axial_view'):
                axial_slice = self.four_view.axial_view.current_slice
                coronal_slice = self.four_view.coronal_view.current_slice
                sagittal_slice = self.four_view.sagittal_view.current_slice

                # 根据变化的视图更新其他视图
                if view == 'axial':
                    # 轴位切片变化，更新冠状位和矢状位的十字准线
                    self.four_view.coronal_view.set_slice(axial_slice)
                    self.four_view.sagittal_view.set_slice(axial_slice)
                elif view == 'coronal':
                    # 冠状位切片变化，更新轴位和矢状位的十字准线
                    self.four_view.axial_view.set_slice(coronal_slice)
                    self.four_view.sagittal_view.set_slice(coronal_slice)
                elif view == 'sagittal':
                    # 矢状位切片变化，更新轴位和冠状位的十字准线
                    self.four_view.axial_view.set_slice(sagittal_slice)
                    self.four_view.coronal_view.set_slice(sagittal_slice)
        except Exception as e:
            print(f"更新切片时发生错误: {e}")

    def _update_seed_count(self):
        """更新种子点数量"""
        count = len(self.seed_points) if self.seed_points is not None else 0
        self.seed_count_label.setText(f"种子点数量: {count}")
        if hasattr(self, 'four_view'):
            try:
                self.four_view.set_seed_points(self.seed_points)
            except Exception as e:
                print(f"同步种子点高亮失败: {e}")

    def _show_long_text_dialog(self, title: str, text: str):
        """用于显示长文本结果，支持滚动查看，避免信息被截断。"""
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(900, 700)

        layout = QVBoxLayout(dialog)

        text_view = QTextEdit(dialog)
        text_view.setReadOnly(True)
        text_view.setLineWrapMode(QTextEdit.NoWrap)
        text_view.setPlainText(text)
        layout.addWidget(text_view)

        btn_close = QPushButton("关闭", dialog)
        btn_close.clicked.connect(dialog.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignRight)

        dialog.exec_()

    def _reset_analysis_state(self):
        """在新病例/清空标签时重置AI结构分析和切割状态。"""
        self.last_vessel_analysis = {}
        self.last_cut_stats = {}
        if hasattr(self, 'four_view') and hasattr(self.four_view, 'view_3d'):
            try:
                self.four_view.view_3d.set_structure_data(None)
                self.four_view.view_3d.remove_clip_planes()
            except Exception:
                pass

    def _refresh_structure_from_current_labels(self):
        """基于当前标签更新血管结构识别（无需重新跑模型）。"""
        if self.current_label_data is None:
            self._reset_analysis_state()
            return
        if not np.any((self.current_label_data == 1) | (self.current_label_data == 2)):
            self._reset_analysis_state()
            return
        try:
            if self.ai_interface is None:
                from ai_interface import AISegmentationInterface
                self.ai_interface = AISegmentationInterface()
            spacing = self.current_metadata['spacing'] if (
                self.current_metadata and 'spacing' in self.current_metadata
            ) else (1.0, 1.0, 1.0)
            self.last_vessel_analysis = self.ai_interface.analyze_vessel_structure(
                self.current_label_data, spacing=spacing
            )
            if hasattr(self, 'four_view') and hasattr(self.four_view, 'view_3d'):
                self.four_view.view_3d.set_structure_data(
                    self.last_vessel_analysis.get("structure_map")
                )
        except Exception as e:
            print(f"基于标签更新结构识别失败: {e}")

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
                    self.last_doctor_label_data = None
                    self.last_ai_label_data = None
                    self._sync_dual_windows_base_image(reset_labels=True)
                    self._reset_analysis_state()
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
                    self.btn_add_seed.setText("➕ 添加种子点")
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
                return  # 用户取消

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
            mask_data = self.current_label_data.astype(
                np.uint8) if self.current_label_data is not None else np.zeros_like(self.current_image_data,
                                                                                    dtype=np.uint8)

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
            self._refresh_structure_from_current_labels()
            if self.current_label_data is not None and np.any(self.current_label_data):
                self.last_doctor_label_data = np.asarray(self.current_label_data).astype(np.uint8).copy()
            else:
                self.last_doctor_label_data = None
            self.last_ai_label_data = None
            self._sync_dual_windows_base_image(reset_labels=True)
            if self.last_doctor_label_data is not None and self.doctor_label_window is not None:
                self.doctor_label_window.set_label_data(self.last_doctor_label_data, source_text="历史病例标签")

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
            self.last_doctor_label_data = None
            self.last_ai_label_data = None
            self._sync_dual_windows_base_image(reset_labels=True)
            self._reset_analysis_state()
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
            self.btn_add_seed.setText("➕ 添加种子点")
            self.btn_add_seed.setStyleSheet("")
            # 更新病人信息显示
            # 如果构建了entry，使用entry信息（更完整）；否则只使用metadata
            if self.current_entry:
                self._update_patient_info_from_entry(self.current_entry, metadata)
            else:
                self._update_patient_info_from_metadata(metadata)

            QMessageBox.information(self, "成功",
                                    f"DICOM文件夹加载成功\n实际加载目录: {target_dir}\n尺寸: {data.shape}")
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
                self.last_doctor_label_data = None
                self.last_ai_label_data = None
                self._sync_dual_windows_base_image(reset_labels=True)
                self._reset_analysis_state()

                self.four_view.axial_view.set_active_label_id(self.active_label_id)
                self.four_view.coronal_view.set_active_label_id(self.active_label_id)
                self.four_view.sagittal_view.set_active_label_id(self.active_label_id)

                self.undo_history.clear()
                self.redo_history.clear()
                self.seed_points = []
                self._update_seed_count()
                self.seed_mode_enabled = False
                self.btn_add_seed.setChecked(False)
                self.btn_add_seed.setText("➕ 添加种子点")
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


class PatientInfoDialog(QDialog):
    """病人信息查看对话框"""

    def __init__(self, info: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("病人信息")
        self.setMinimumSize(400, 300)

        layout = QVBoxLayout()

        # 创建表格显示信息
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["字段", "值"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # 填充数据
        items = [
            ("病人ID", info.get("PatientID", "--")),
            ("病人姓名", info.get("PatientName", "--")),
            ("性别", info.get("PatientSex", "--")),
            ("出生日期", info.get("PatientBirthDate", "--")),
            ("检查日期", info.get("StudyDate", "--")),
            ("模态", info.get("Modality", "--")),
            ("序列号", info.get("SeriesNumber", "--")),
            ("序列描述", info.get("SeriesDescription", "--")),
            ("机构", info.get("InstitutionName", "--")),
            ("医生", info.get("ReferringPhysicianName", "--")),
        ]

        table.setRowCount(len(items))
        for i, (key, value) in enumerate(items):
            table.setItem(i, 0, QTableWidgetItem(key))
            table.setItem(i, 1, QTableWidgetItem(str(value)))

        layout.addWidget(table)

        # 关闭按钮
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

        self.setLayout(layout)
