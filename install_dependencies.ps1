param(
    [switch]$CpuOnly,
    [switch]$OnnxGpu
)

$ErrorActionPreference = "Stop"

Write-Host "VesselSeg dependency installer" -ForegroundColor Cyan
Write-Host "This script installs dependencies into the currently active Python environment." -ForegroundColor Yellow
Write-Host ""

$python = $env:PYTHON
if ([string]::IsNullOrWhiteSpace($python)) {
    $python = "python"
}

Write-Host "Checking Python..." -ForegroundColor Cyan
& $python --version

Write-Host "Upgrading pip / wheel / setuptools..." -ForegroundColor Cyan
& $python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing PyTorch..." -ForegroundColor Cyan
if ($CpuOnly) {
    Write-Host "Mode: CPU-only PyTorch" -ForegroundColor Yellow
    & $python -m pip install --upgrade "torch==2.5.1+cpu" "torchvision==0.20.1+cpu" --index-url https://download.pytorch.org/whl/cpu
}
else {
    Write-Host "Mode: CUDA 12.4 PyTorch, recommended for NVIDIA GPU" -ForegroundColor Yellow
    & $python -m pip install --upgrade "torch==2.5.1+cu124" "torchvision==0.20.1+cu124" --index-url https://download.pytorch.org/whl/cu124
}

Write-Host "Installing core project dependencies..." -ForegroundColor Cyan
$packages = @(
    "numpy>=1.26,<3",
    "scipy>=1.11",
    "scikit-image>=0.22",
    "scikit-learn>=1.3",
    "matplotlib>=3.8",
    "pillow>=10.0",
    "pandas>=2.1",
    "openpyxl>=3.1",
    "xlrd>=2.0",
    "PyQt5==5.15.11",
    "nibabel>=5.2",
    "nilearn>=0.10",
    "SimpleITK>=2.3",
    "monai>=1.4",
    "timm>=1.0",
    "pyvista==0.44.1",
    "pyvistaqt==0.11.2",
    "vtk==9.3.1",
    "reportlab>=4.0",
    "python-docx>=1.1",
    "pydicom>=2.4",
    "pynrrd>=1.0",
    "tqdm>=4.66",
    "requests>=2.31"
)

if ($OnnxGpu) {
    Write-Host "Installing ONNX Runtime GPU edition..." -ForegroundColor Yellow
    $packages += "onnxruntime-gpu>=1.18"
}
else {
    Write-Host "Installing ONNX Runtime CPU edition..." -ForegroundColor Yellow
    $packages += "onnxruntime>=1.18"
}

foreach ($pkg in $packages) {
    Write-Host "Installing $pkg" -ForegroundColor DarkCyan
    & $python -m pip install --upgrade $pkg
}

Write-Host ""
Write-Host "Verifying important imports..." -ForegroundColor Cyan
& $python -c "import numpy, scipy, skimage, sklearn, matplotlib, pandas, PyQt5, nibabel, nilearn, SimpleITK, monai, timm, pyvista, pyvistaqt, vtk, reportlab, docx, torch; print('All core imports OK'); print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

Write-Host ""
Write-Host "Done. You can now run:" -ForegroundColor Green
Write-Host "  python .\our_snap\main.py" -ForegroundColor Green
Write-Host ""
Write-Host "Tips:" -ForegroundColor Yellow
Write-Host "  GPU install: powershell -ExecutionPolicy Bypass -File .\install_dependencies.ps1"
Write-Host "  CPU install: powershell -ExecutionPolicy Bypass -File .\install_dependencies.ps1 -CpuOnly"
Write-Host "  ONNX GPU:    powershell -ExecutionPolicy Bypass -File .\install_dependencies.ps1 -OnnxGpu"
