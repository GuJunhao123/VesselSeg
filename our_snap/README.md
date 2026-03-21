# 医学影像分割软件

## 系统说明

这是一个基于 PyQt5 的医学影像分割软件，支持 DICOM、NIfTI、NRRD 等格式的医学影像加载、可视化、标注和分割。

## 系统要求

- Python 3.7 或更高版本
- Windows 操作系统（推荐 Windows 10/11）

## 安装步骤

### 1. 安装 Python 依赖

在 `our_snap` 文件夹下打开命令行，执行：

```bash
pip install -r requirements.txt
```

或者如果使用虚拟环境：

```bash
# 创建虚拟环境（可选）
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行程序

```bash
python main.py
```

## 功能特性

- **多格式支持**：支持 DICOM、NIfTI、NRRD 等医学影像格式
- **三视图可视化**：轴向、冠状、矢状三个视图同步显示
- **手动标注**：支持画笔、橡皮擦等工具进行手动标注
- **主动轮廓分割**：基于主动轮廓算法的半自动分割
- **AI 自动分割**：支持 ONNX 模型的 AI 自动分割
- **数据库管理**：支持 522 数据库格式的序列管理和加载
- **Excel 信息查看**：支持查看患者对应的 Excel 表格信息

## 文件结构

```
our_snap/
├── main.py              # 主程序入口
├── ui_layout.py         # 主界面布局和逻辑
├── image_io.py          # 影像读写模块
├── visualization.py     # 可视化模块
├── segmentation.py      # 分割算法模块
├── ai_interface.py      # AI 接口模块
├── database_index.py    # 数据库索引模块
├── excel_reader.py      # Excel 读取模块
├── requirements.txt     # Python 依赖清单
└── README.md           # 本文件
```

## 使用说明

### 加载影像

1. **从文件加载**：点击 "Load Image" 按钮，选择单个影像文件
2. **从文件夹加载**：点击 "Load DICOM Folder" 按钮，选择 DICOM 序列文件夹
3. **从数据库加载**：点击 "Open DB Dialog" 按钮，连接数据库并选择序列

### 标注操作

1. 在标签输入框中输入标签名称
2. 使用画笔工具在图像上绘制标注
3. 使用橡皮擦工具擦除标注
4. 可以保存标注为 NIfTI 格式

### 分割功能

1. **主动轮廓分割**：添加种子点后，点击 "Active Contour" 按钮
2. **AI 分割**：加载 ONNX 模型后，点击 "AI Segment" 按钮

## 注意事项

1. **数据库路径**：数据库路径需要在程序中手动选择，支持 522 数据库格式
2. **Excel 文件**：Excel 文件需要放在数据库根目录下（与 data 文件夹同级）
3. **模型文件**：AI 模型需要是 ONNX 格式
4. **系统路径**：所有路径都是相对路径或用户选择，没有硬编码路径

## 常见问题

### Q: 程序无法启动？
A: 检查是否已安装所有依赖：`pip install -r requirements.txt`

### Q: 无法加载 DICOM 文件？
A: 确保 DICOM 文件完整，且目录结构正确（如 SE*/IM* 格式）

### Q: 数据库连接失败？
A: 确保选择的路径是数据库根目录（通常包含 data 文件夹）

## 技术支持

如有问题，请检查：
1. Python 版本是否符合要求
2. 所有依赖是否已正确安装
3. 文件路径是否正确
4. 控制台输出的错误信息

## 版本信息

- 版本：1.0
- 更新日期：2024

