import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import monai
import torch.nn as nn
import torch.nn.functional as F

from monai.data import CacheDataset, DataLoader, list_data_collate, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    CropForegroundd, SpatialPadd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, ScaleIntensityRanged,
    NormalizeIntensityd, EnsureTyped, AsDiscrete,RandCropByLabelClassesd
)
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.metrics import ConfusionMatrixMetric
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism
from tqdm import tqdm
from datetime import datetime

# =========================================================
# 基础配置
# =========================================================
set_determinism(42)

DATA_DIR = "/home/qjl/mnt/qjl/Task08"
PATCH_SIZE = (96, 96, 96)
NUM_SAMPLES = 4
BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
NUM_WORKERS = 12
MAX_EPOCHS = 600
LR = 1e-3
DEVICE = torch.device("cuda:0")

RESULT_ROOT = "./result_msd_vessel_3cls_param"
os.makedirs(RESULT_ROOT, exist_ok=True)

# =========================================================
# 数据划分（thin slice）
# =========================================================
def get_thin_slice_files(data_dir, thickness_threshold=2.0):
    json_path = os.path.join(data_dir, "dataset.json")
    all_files = monai.data.load_decathlon_datalist(
        json_path, data_list_key="training", base_dir=data_dir
    )
    thin_files = []
    for item in all_files:
        try:
            img_nii = nib.load(item["image"])
            if round(img_nii.header.get_zooms()[2], 2) <= thickness_threshold:
                thin_files.append(item)
        except FileNotFoundError:
            # print(f"Warning: Skipping file {item['image']} as it does not exist.")
            continue
    print(f"Thin-slice cases: {len(thin_files)}")
    return thin_files

thin_files = get_thin_slice_files(DATA_DIR)
def get_msd_dicts_random(thin_files, train_ratio=0.8, seed=42):
    import random
    random.seed(seed)
    shuffled = thin_files.copy()
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    return shuffled[:n_train], shuffled[n_train:]

train_files, val_files = get_msd_dicts_random(thin_files, train_ratio=0.8, seed=42)

# for i, val_file in enumerate(val_files):
#     image_path = val_file["image"]
#     image_filename = os.path.basename(image_path)
#     print(f"{i+1}: {image_filename}")

# =========================================================
# Vessel Decomposition（4 通道 label）
# =========================================================
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
    def __init__(self, keys, iterations=2):
        self.keys = keys
        self.iterations = iterations

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

        # 4. 【关键修复】正确的 Otsu 条件判断
        if np.unique(radii).size > 1:  # 原代码是 np.unique(radii.size) > 1，这是错的！
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

# =========================================================
# Transform
# =========================================================
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0,1.0,1.0),
             mode=("bilinear", "nearest")),
    CropForegroundd(keys=["image", "label"], source_key="label",
                    select_fn=lambda x: x > 0),
    SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE),
    RandCropByLabelClassesd(
        keys=["image", "label"],
        label_key="label",
        spatial_size=PATCH_SIZE,
        ratios=[0.2, 0.4, 0.4],   # BG, HV, PV
        num_samples=NUM_SAMPLES,
        num_classes=3,
    ),
    RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=[0,1,2]),
    RandRotate90d(keys=["image", "label"], prob=0.2, spatial_axes=[0,1]),
    ScaleIntensityRanged(
        keys="image", a_min=0, a_max=400,
        b_min=0.0, b_max=1.0, clip=True
    ),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    GenerateVesselDecompositiond(keys=["label"]),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0,1.0,1.0),
             mode=("bilinear", "nearest")),
    CropForegroundd(keys=["image", "label"], source_key="label",
                    select_fn=lambda x: x > 0),
    SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE),
    ScaleIntensityRanged(
        keys="image", a_min=0, a_max=400,
        b_min=0.0, b_max=1.0, clip=True
    ),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    GenerateVesselDecompositiond(keys=["label"]),
    EnsureTyped(keys=["image", "label"]),
])

train_ds = CacheDataset(train_files, train_transforms,
                        cache_rate=0, num_workers=NUM_WORKERS)
val_ds = CacheDataset(val_files, val_transforms,
                      cache_rate=0, num_workers=NUM_WORKERS)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=NUM_WORKERS,
    collate_fn=list_data_collate
)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

# =========================================================
# 模型
# =========================================================
# from network.model1 import mobileuvit_l
from network.ori_tafr import orinet_tafr
model = orinet_tafr(in_channels=1,out_channels=3).to(DEVICE)

# 计算并打印模型参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params/1e6:,}")
print(f"Trainable parameters: {trainable_params/1e6:,}")

# =========================================================
# Loss
# =========================================================
class PhysicsConsistentSingleSystemLoss(nn.Module):
    def __init__(
        self,
        lambda_trunk=0.2,
        lambda_small=0.1,
        lambda_junc=0.05,
        lambda_mutex=0.05,
        lambda_junc_decouple=0.02,
        lambda_inclusion=0.05,
    ):
        super().__init__()

        # 修改：使用正确的参数设置
        self.global_loss = DiceCELoss(
            include_background=True,
            to_onehot_y=True,        # 目标将被自动转换为one-hot格式
            softmax=True,            # 预测使用softmax
            batch=True
        )
        self.trunk_dice = DiceLoss(
            to_onehot_y=False, 
            sigmoid=True, 
            batch=True
        )
        self.small_dice = DiceLoss(
            to_onehot_y=False,
            sigmoid=True,
            batch=True
        )
        self.junc_bce = nn.BCEWithLogitsLoss()

        self.lambda_trunk = lambda_trunk
        self.lambda_small = lambda_small
        self.lambda_junc = lambda_junc
        self.lambda_mutex = lambda_mutex
        self.lambda_junc_decouple = lambda_junc_decouple
        self.lambda_inclusion = lambda_inclusion

    def _vessel_system_loss(self, p_global, p_trunk, p_small, p_junc,
                           y_global, y_trunk, y_small, y_junc):
        """
        对于单个血管系统的损失计算
        p_global: [B, 3, H, W, D] - 全局预测 (背景, HV, PV)
        其他预测: [B, 1, H, W, D] - 各角色预测
        """
        # ---------------- Global (Multi-class) ----------------
        # 当to_onehot_y=True时，y_global应该保持整数标签[B, H, W, D]格式
        l_global = self.global_loss(p_global, y_global.long())

        # 获取HV和PV的概率图用于后续约束
        with torch.no_grad():
            global_probs = torch.softmax(p_global, dim=1)  # [B, 3, H, W, D]
            # HV概率图
            hv_gate = global_probs[:, 1:2]  # [B, 1, H, W, D]
            # PV概率图  
            pv_gate = global_probs[:, 2:3]  # [B, 1, H, W, D]
            # 血管总体概率图
            vessel_gate = hv_gate + pv_gate  # [B, 1, H, W, D]

        # ---------------- Trunk ----------------
        l_trunk = self.trunk_dice(p_trunk, y_trunk)

        # ---------------- Small (residual w.r.t trunk) ----------------
        p_trunk_sig = torch.sigmoid(p_trunk)
        p_small_sig = torch.sigmoid(p_small)

        # Small的目标是血管区域内但非trunk的部分
        vessel_gate_detach = vessel_gate.detach()
        small_target = vessel_gate_detach * (1.0 - p_trunk_sig)
        l_small = F.l1_loss(p_small_sig, small_target)

        # ---------------- Junction ----------------
        p_junc_sig = torch.sigmoid(p_junc)
        l_junc = self.junc_bce(p_junc, y_junc.float())

        # =================================================
        # Regularizations
        # =================================================

        # 1. trunk-small mutual exclusion within vessel region
        l_mutex = torch.mean(
            p_trunk_sig * p_small_sig * vessel_gate_detach
        )

        # 2. junction decoupling from other components
        l_junc_decouple = (
            torch.mean(p_junc_sig * p_trunk_sig.detach()) +
            torch.mean(p_junc_sig * p_small_sig.detach())
        )

        # 3. inclusion (components ⊆ vessel)
        l_inclusion = torch.mean(
            F.relu(
                p_trunk_sig + p_small_sig + p_junc_sig
                - vessel_gate_detach
            )
        )

        total = (
            l_global
            + self.lambda_trunk * l_trunk
            + self.lambda_small * l_small
            + self.lambda_junc * l_junc
            + self.lambda_mutex * l_mutex
            + self.lambda_junc_decouple * l_junc_decouple
            + self.lambda_inclusion * l_inclusion
        )

        return total, l_global

    def forward(self, outputs, labels):
        """
        outputs: (logits_global, trunk, small, junc)
            logits_global: [B, 3, H, W, D]
            trunk, small, junc: [B, 1, H, W, D]
        labels: [B, 4, H, W, D]
            channel 0: global (0=bg, 1=HV, 2=PV)
            channel 1: trunk
            channel 2: small
            channel 3: junction
        """
        logits_global, trunk, small, junc = outputs
        y_global = labels[:, 0:1]  # [B, H, W, D] - 整数标签
        y_trunk = labels[:, 1:2]  # [B, 1, H, W, D]
        y_small = labels[:, 2:3]  # [B, 1, H, W, D] 
        y_junc = labels[:, 3:4]   # [B, 1, H, W, D]

        total_loss, l_global = self._vessel_system_loss(
            logits_global, trunk, small, junc,
            y_global, y_trunk, y_small, y_junc
        )

        return total_loss, l_global


criterion = PhysicsConsistentSingleSystemLoss(
    lambda_trunk=0.2,
    lambda_small=0.1, 
    lambda_junc=0.05,
    lambda_mutex=0.05,
    lambda_junc_decouple=0.02,
    lambda_inclusion=0.05
)

def compute_loss(outputs, labels):
    loss, global_loss = criterion(outputs, labels)
    return loss, global_loss

# =========================================================
# Metrics & Post-processing
# =========================================================
# 假设 NUM_CLASSES = 3 (0=BG, 1=HV, 2=PV)
post_pred = AsDiscrete(argmax=True, to_onehot=3)
post_label = AsDiscrete(to_onehot=3)

# 使用 mean_batch 以便获取每个类别的独立指标 [HV, PV]
dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean_batch", percentile=95)
conf_matrix_metric = ConfusionMatrixMetric(
    include_background=False,
    metric_name=["sensitivity", "specificity", "precision"],
    reduction="mean_batch"
)

inferer = SlidingWindowInferer(
    roi_size=PATCH_SIZE, sw_batch_size=1,
    overlap=0.6, mode="gaussian"
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

# ========================= 实验目录 =========================
exp_time = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir = os.path.join(RESULT_ROOT, exp_time)
vis_dir = os.path.join(exp_dir, "vis")
os.makedirs(vis_dir, exist_ok=True)

results_csv_path = os.path.join(exp_dir, "validation_results.csv")
with open(results_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Avg_Dice', 'HV_Dice', 'PV_Dice', 'HV_HD95', 'PV_HD95', 
                     'Mean_Sens', 'Mean_Prec'])

best_avg = 0.0
best_epoch = 0

# 添加缺失的变量定义
best_vis_png = os.path.join(vis_dir, "best_visualization.png")
best_hv_nii = os.path.join(vis_dir, "best_hv_prediction.nii.gz")
best_pv_nii = os.path.join(vis_dir, "best_pv_prediction.nii.gz")

# ========================= 可视化函数 =========================
def save_nifti(data, affine, path):
    nii = nib.Nifti1Image(data.astype(np.uint8), affine)
    nib.save(nii, path)


def visualize_mid_slice(img, lab, pred, path):
    z = img.shape[-1] // 2
    plt.figure(figsize=(12, 4))
    plt.subplot(131);
    plt.imshow(img[0, 0, :, :, z], cmap='gray');
    plt.title("Image")
    plt.subplot(132);
    plt.imshow(lab[0, 0, :, :, z]);
    plt.title("GT")
    plt.subplot(133);
    plt.imshow(pred[0, 0, :, :, z]);
    plt.title("Pred")
    plt.savefig(path);
    plt.close()

# =========================================================
# Training
# =========================================================
best_dice = 0.0
start_time = datetime.now()
print(f"开始训练: {start_time}")

for epoch in range(MAX_EPOCHS):
    model.train()
    epoch_loss = 0
    torch.cuda.empty_cache()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        x = batch["image"].to(DEVICE)
        y = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x)
        loss, _ = compute_loss(outputs, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()

    if (epoch + 1) % 20 == 0 or epoch == 0:
        model.eval()
        dice_metric.reset()
        hd95_metric.reset()
        conf_matrix_metric.reset()
        
        first_cache = None

        with torch.no_grad():
            for i, val in enumerate(tqdm(val_loader, desc="Validating")):
                val_inputs = val["image"].to(DEVICE)
                # 标签：我们只关注 channel 0 (Global Label: 0,1,2)
                val_labels = val["label"][:, 0:1].to(DEVICE) 

                # 推理：只取模型的第一个输出 (logits_global)
                val_outputs = inferer(val_inputs, lambda z: model(z)[0])

                # 后处理并计算指标
                preds_list = [post_pred(i) for i in decollate_batch(val_outputs)]
                labels_list = [post_label(i) for i in decollate_batch(val_labels)]

                dice_metric(y_pred=preds_list, y=labels_list)
                hd95_metric(y_pred=preds_list, y=labels_list)
                conf_matrix_metric(y_pred=preds_list, y=labels_list)

                if i == 0:
                    first_cache = (val_inputs.cpu(), val["label"].cpu(), val_outputs.cpu())

        # ---------------- 聚合结果 ----------------
        # 聚合结果 [Class 1, Class 2]
        dices = dice_metric.aggregate()
        hd95s = hd95_metric.aggregate()
        cms = conf_matrix_metric.aggregate()  # [Metric, Class]

        d_hv, d_pv = dices[0].item(), dices[1].item()
        h_hv, h_pv = hd95s[0].item(), hd95s[1].item()
            # 处理 HD95 的 inf 情况
        h_hv = h_hv if not np.isinf(h_hv) else 100.0
        h_pv = h_pv if not np.isinf(h_pv) else 100.0

        s_hv, s_pv = cms[0][0].item(), cms[0][1].item()  # Sensitivity
        p_hv, p_pv = cms[2][0].item(), cms[2][1].item()  # Precision

        avg_dice = (d_hv + d_pv) / 2

        print(f"[Epoch {epoch+1}] Avg Dice: {avg_dice:.4f} | HV: {d_hv:.4f} | PV: {d_pv:.4f}")
        print(f"HD95 - HV: {h_hv:.2f}, PV: {h_pv:.2f} | Precision - HV: {p_hv:.4f}, PV: {p_pv:.4f}")

        # ---------------- 记录 CSV ----------------
        file_exists = os.path.isfile(results_csv_path)
        with open(results_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(
                        ['epoch', 'avg_dice', 'hv_dice', 'pv_dice', 'hv_hd95', 'pv_hd95', 'mean_sens', 'mean_prec'])
                # 计算平均敏感度和精确度
                mean_sens = (s_hv + s_pv) / 2
                mean_prec = (p_hv + p_pv) / 2
                writer.writerow([epoch, avg_dice, d_hv, d_pv, h_hv, h_pv, mean_sens, mean_prec])

        # ---------------- 保存最佳模型与可视化 ----------------
        if avg_dice > best_dice:
                best_dice = avg_dice
                torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))
                # 可视化第一例
                val_inputs, val_labels, val_outputs = first_cache
        
                # 从验证数据集中获取meta信息
                v_case = val_ds[0]
                img_v = val_inputs  # 使用缓存的输入
                
                # 获取预测结果
                pred_v = torch.argmax(val_outputs, dim=1, keepdim=True)
                
                # 可视化
                visualize_mid_slice(
                    img_v.cpu(), 
                    val_labels.cpu(),  # 注意这里的维度调整
                    pred_v.cpu(),
                    os.path.join(vis_dir, "best_v.png")
                )

                # 从meta信息中获取affine矩阵
                if hasattr(v_case['image'], 'meta') and 'affine' in v_case['image'].meta:
                    affine = v_case['image'].meta['affine']
                elif 'image_meta_dict' in v_case and 'affine' in v_case['image_meta_dict']:
                    affine = v_case['image_meta_dict']['affine']
                else:
                    # 如果都没有找到，则使用单位矩阵作为默认值
                    affine = np.eye(4)

                save_nifti(pred_v[0, 0].cpu().numpy(), affine, os.path.join(vis_dir, "best_pred.nii.gz"))

        dice_metric.reset()
        hd95_metric.reset()
        conf_matrix_metric.reset()

print("Training Finished.")