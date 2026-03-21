import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    CropForegroundd, SpatialPadd, ScaleIntensityRanged, NormalizeIntensityd, 
    EnsureTyped
)
from network.cotu import UNet3D 
from monai.inferers import SlidingWindowInferer
from datetime import datetime

# ========================= 基础配置 =========================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/qjl/mnt/qjl/Mobile-U-ViT/result_msd_vessel_3cls_dec/20260214_101918/best_model.pth" 
INPUT_IMAGE = "/home/qjl/mnt/qjl/Task08/imagesTr/hepaticvessel_437.nii.gz"
INPUT_LABEL = "/home/qjl/mnt/qjl/Task08/labelsTr/hp437.nii.gz"
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = f"./vis_results/{current_time}"
PATCH_SIZE = (96, 96, 96)
NUM_CLASSES = 3

os.makedirs(RESULT_DIR, exist_ok=True)

# ========================= 1. 叠加可视化 (使用原始 CT 强度) =========================
def visualize_overlay_clean(img, pred, path):
    """
    img: 原始 HU 值的切片
    pred: 预测类别
    """
    z = img.shape[-1] // 2
    img_slice = img[0, :, :, z]
    pred_slice = pred[0, :, :, z]

    # HV (1) -> 红色, PV (2) -> 蓝色
    colors = [(0,0,0,0), (1,0,0,0.7), (0,0,1,0.7)] 
    cmap_custom = ListedColormap(colors)

    plt.figure(figsize=(10, 10))
    
    # 或者直接让 plt 根据百分位自动调整，这里设置 vmin, vmax 使其看起来像标准 CT
    plt.imshow(img_slice, cmap='gray') 
    
    masked_pred = np.ma.masked_where(pred_slice == 0, pred_slice)
    plt.imshow(masked_pred, cmap=cmap_custom, interpolation='nearest', vmin=0, vmax=2)
    
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()
    print(f"2D 原始强度叠加图已保存: {path}")


def visualize_gt_overlay(img, gt, path):
    """
    img: 原始 HU 值的切片
    gt: ground truth 标签
    """
    z = img.shape[-1] // 2
    img_slice = img[0, :, :, z]
    gt_slice = gt[0, :, :, z]  # GT 是原始标签，取第一个通道

    # HV (1) -> 红色, PV (2) -> 蓝色
    colors = [(0,0,0,0), (1,0,0,0.7), (0,0,1,0.7)] 
    cmap_custom = ListedColormap(colors)

    plt.figure(figsize=(10, 10))
    
    plt.imshow(img_slice, cmap='gray') 
    
    # 将GT标签转换为彩色覆盖层
    masked_gt = np.ma.masked_where(gt_slice == 0, gt_slice)
    plt.imshow(masked_gt, cmap=cmap_custom, interpolation='nearest', vmin=0, vmax=2)
    
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()
    print(f"2D GT 叠加图已保存: {path}")



# ========================= 2. 推理主程序 =========================
from skimage import morphology
from scipy import ndimage
from skimage.filters import threshold_otsu
class GenerateVesselDecompositiond:
    """
    输出：
    channel 0: global label (0=bg,1=HV,2=PV)
    channel 1: trunk
    channel 2: small
    channel 3: junction
    """
    def __init__(self, keys, iterations=2, decomposition_method="adaptive"):
        self.keys = keys
        self.iterations = iterations
        self.decomposition_method = decomposition_method  # "adaptive", "fixed_1.0", "fixed_1.5", "fixed_2.0", "fixed_2.5", "median", "percentile_25", "percentile_75"

    def _decompose(self, mask):
        z = np.zeros_like(mask, dtype=np.uint8)
        
        # 1. 检查输入掩码是否为空
        if not np.any(mask):
            return z, z, z

        skeleton = morphology.skeletonize(mask)
        
        # 2. 【关键修复】检查骨架是否为空（极细血管可能无法提取骨架）
        if not np.any(skeleton):
            # 无法提取骨架，整体视为 small（细血管）
            return z, mask.astype(np.uint8), z

        edt = ndimage.distance_transform_edt(mask)
        radii = edt[skeleton]

        # 3. 【关键修复】检查 radii 是否为空（双重保险）
        if radii.size == 0:
            return z, mask.astype(np.uint8), z

        # 4. 根据方法选择阈值
        if self.decomposition_method.startswith("fixed_"):
            # 使用固定阈值
            fixed_val = float(self.decomposition_method.split("_")[1])
            th = fixed_val
        elif self.decomposition_method == "adaptive":
            # 使用自适应阈值（Otsu方法）
            if np.unique(radii).size > 1:
                th = threshold_otsu(radii)
            else:
                th = radii.max() + 1e-5
        elif self.decomposition_method == "median":
            # 使用中位数作为阈值
            th = np.median(radii)
        elif self.decomposition_method.startswith("percentile_"):
            # 使用百分位数作为阈值
            percentile_val = int(self.decomposition_method.split("_")[1])
            th = np.percentile(radii, percentile_val)
        else:
            # 默认使用adaptive方法
            if np.unique(radii).size > 1:
                th = threshold_otsu(radii)
            else:
                th = radii.max() + 1e-5

        _, idx = ndimage.distance_transform_edt(~skeleton, return_indices=True)
        prop_r = edt[tuple(idx)]

        trunk = (prop_r >= th) & mask
        non_trunk = mask & (~trunk)

        # 5. 【关键修复】检查 trunk 是否为空
        if not np.any(trunk):
            # 没有主干，全是 small
            return z, mask.astype(np.uint8), z

        trunk_dil = ndimage.binary_dilation(trunk, iterations=self.iterations)
        junction = trunk_dil & non_trunk
        small = non_trunk & (~junction)

        return trunk.astype(np.uint8), small.astype(np.uint8), junction.astype(np.uint8)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            lab = d[k][0].numpy().astype(np.uint8)

            global_lab = lab
            vessel_mask = lab > 0

            t, s, j = self._decompose(vessel_mask)
            d[k] = torch.from_numpy(
                np.stack([global_lab, t, s, j], axis=0)
            ).float()
        return d

def run_inference():
    # A. 加载模型
    from monai.networks.nets.dynunet import DynUNet
    # from network.MobileUViT_3D import mobileuvit_l
    from network.ori import OriNet
    from network.superlightnet import NormalU_Net
    from monai.networks.nets.swin_unetr import SwinUNETR
    from monai.networks.nets.basic_unet import BasicUNet
    from monai.networks.nets.basic_unetplusplus import BasicUnetPlusPlus
    from monai.networks.nets.unetr import UNETR
    from monai.networks.nets.vnet import VNet
    from network.model1 import mobileuvit_l
    # model = mobileuvit_l(inch=1,out_channel=NUM_CLASSES).to(DEVICE)
    # model = NormalU_Net(init_channels=1,class_nums=NUM_CLASSES,depths_unidirectional='small').to(DEVICE)
    # model = UNETR(
    #         in_channels=1,
    #         out_channels=NUM_CLASSES,
    #         img_size=PATCH_SIZE,
    #     ).to(DEVICE)
    # model = VNet(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)
    # model = BasicUnetPlusPlus(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)
    # model = OriNet(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)
    # model = UNet3D(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)
    model = mobileuvit_l(inch=1,).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # B. 空间变换 (两者共用，确保像素对齐)
    # 这一部分决定了图像的形状和裁剪位置
    spatial_ops = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LPI"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        CropForegroundd(keys=["image", "label"], source_key="label", select_fn=lambda x: x > 0),
        SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE),
        GenerateVesselDecompositiond(keys=["label"],decomposition_method='percentile_50'),
    ]

    # C. 推理专用 Transform (包含归一化)
    inf_transforms = Compose(spatial_ops + [
        ScaleIntensityRanged(keys="image", a_min=0, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ])

    # D. 可视化专用 Transform (不包含归一化，保留原始 HU 值)
    vis_transforms = Compose(spatial_ops + [
        EnsureTyped(keys=["image", "label"]),
    ])

    # 执行变换
    data_inf = inf_transforms({"image": INPUT_IMAGE, "label": INPUT_LABEL})
    data_vis = vis_transforms({"image": INPUT_IMAGE, "label": INPUT_LABEL})
    
    input_tensor = data_inf["image"].unsqueeze(0).to(DEVICE)

    # E. 滑窗推理
    inferer = SlidingWindowInferer(roi_size=PATCH_SIZE, sw_batch_size=1, overlap=0.6, mode="gaussian")
    
    print("正在进行滑窗推理...")
    with torch.no_grad():
        output = inferer(input_tensor, model)
        pred_label = torch.argmax(output[0], dim=1).cpu().numpy() # [1, H, W, D]

    # F. 2D 可视化 (传入 data_vis["image"]，即原始 CT 强度)
    base_name = os.path.basename(INPUT_IMAGE).split('.')[0]
    vis_path = os.path.join(RESULT_DIR, f"{base_name}_overlay_raw_HU.png")
    visualize_overlay_clean(data_vis["image"].cpu().numpy(), pred_label, vis_path)

    
    gt_vis_path = os.path.join(RESULT_DIR, f"{base_name}_gt_overlay_raw_HU.png")
    # 从data_vis获取原始标签 (第0个通道是全局标签)
    gt_data = data_vis["label"][0:1]  # 取第一个通道作为GT
    visualize_gt_overlay(data_vis["image"].cpu().numpy(), gt_data.cpu().numpy(), gt_vis_path)
    
    
    # G. 还原 NIfTI 到原始 512x512 空间 (逻辑保持不变)
    print("正在还原 NIfTI 维度...")
    orig_nii = nib.load(INPUT_IMAGE)
    from nilearn.image import resample_to_img
    
    pred_nii_cropped = nib.Nifti1Image(
        pred_label[0].astype(np.uint8), 
        data_inf["image"].meta["affine"]
    )
    
    final_pred_nii = resample_to_img(
        pred_nii_cropped, 
        orig_nii, 
        interpolation='nearest'
    )

    save_path = os.path.join(RESULT_DIR, f"{base_name}_full_res.nii.gz")
    nib.save(final_pred_nii, save_path)
    print(f"全尺寸 NIfTI 已保存: {save_path}")

if __name__ == "__main__":
    run_inference()