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
from monai.inferers import SlidingWindowInferer
from datetime import datetime
from functools import partial
from timm.models.layers import DropPath

# ========================= 模型定义 =========================
class Residual(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvUtr(torch.nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, kernel=3):
        super(ConvUtr, self).__init__()
        self.block = torch.nn.Sequential(
            *[torch.nn.Sequential(
                Residual(torch.nn.Sequential(
                    torch.nn.Conv3d(ch_in, ch_in, kernel_size=(kernel, kernel, kernel), groups=ch_in, padding=(kernel // 2, kernel // 2, kernel // 2)),
                    torch.nn.GELU(),
                    torch.nn.InstanceNorm3d(ch_in)
                )),
                Residual(torch.nn.Sequential(
                    torch.nn.Conv3d(ch_in, ch_in * 4, kernel_size=(1, 1, 1)),
                    torch.nn.GELU(),
                    torch.nn.InstanceNorm3d(ch_in * 4),
                    torch.nn.Conv3d(ch_in * 4, ch_in, kernel_size=(1, 1, 1)),
                    torch.nn.GELU(),
                    torch.nn.InstanceNorm3d(ch_in)
                )),
            ) for i in range(depth)]
        )
        self.up = torch.nn.Sequential(
            torch.nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.InstanceNorm3d(ch_out),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x


class Embeddings(torch.nn.Module):
    def __init__(self, inch=1, dims=[8, 16, 32], depths=[1, 1, 3], kernels=[3, 3, 7]):
        super(Embeddings, self).__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv3d(inch, dims[0], kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.InstanceNorm3d(dims[0]),
            torch.nn.ReLU(inplace=True)
        )
        self.layer1 = ConvUtr(dims[0], dims[0], depth=depths[0], kernel=kernels[0])
        self.layer2 = ConvUtr(dims[0], dims[1], depth=depths[1], kernel=kernels[1])
        self.layer3 = ConvUtr(dims[1], dims[2], depth=depths[2], kernel=kernels[2])
        self.down = torch.nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x0 = self.stem(x)
        x0 = self.layer1(x0)

        x1 = self.down(x0)
        x1 = self.layer2(x1)

        x2 = self.down(x1)
        x2 = self.layer3(x2)

        return x2, (x0, x1, x2)


class Mlp(torch.nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=torch.nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = torch.nn.Linear(hidden_features, out_features)
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(torch.nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=torch.nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = torch.nn.Conv3d(in_features, hidden_features, 3, padding=1, groups=in_features)
        self.act = act_layer()
        self.fc2 = torch.nn.Conv3d(hidden_features, out_features, 3, padding=1, groups=in_features)
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalSparseAttn(torch.nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

        self.sr = sr_ratio
        if self.sr > 1:
            self.sampler = torch.nn.AvgPool3d(1, sr_ratio)
            kernel_size = sr_ratio
            self.LocalProp = torch.nn.ConvTranspose3d(dim, dim, kernel_size, stride=sr_ratio, groups=dim)
            self.norm = torch.nn.LayerNorm(dim)
        else:
            self.sampler = torch.nn.Identity()
            self.upsample = torch.nn.Identity()
            self.norm = torch.nn.Identity()

    def forward(self, x, H: int, W: int, D: int):
        B, N, C = x.shape
        if self.sr > 1.:
            x = x.transpose(1, 2).reshape(B, C, H, W, D)
            x = self.sampler(x)
            x = x.flatten(2).transpose(1, 2)

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        if self.sr > 1:
            x = x.permute(0, 2, 1).reshape(B, C, int(H / self.sr), int(W / self.sr), int(D / self.sr))
            x = self.LocalProp(x)
            x = x.reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    


class LocalAgg(torch.nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=torch.nn.GELU):
        super().__init__()
        self.pos_embed = torch.nn.Conv3d(dim, dim, 9, padding=4, groups=dim)
        self.norm1 = torch.nn.InstanceNorm3d(dim)
        self.conv1 = torch.nn.Conv3d(dim, dim, 1)
        self.conv2 = torch.nn.Conv3d(dim, dim, 1)
        self.attn = torch.nn.Conv3d(dim, dim, 9, padding=4, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()
        self.norm2 = torch.nn.InstanceNorm3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.sg = torch.nn.Sigmoid()

    def forward(self, x):
        x = x + x * (self.sg(self.pos_embed(x)) - 0.5)
        x = x + x * (self.sg(self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))) - 0.5)
        x = x + x * (self.sg(self.drop_path(self.mlp(self.norm2(x)))) - 0.5)
        return x


class SelfAttn(torch.nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=torch.nn.GELU, norm_layer=torch.nn.LayerNorm, sr_ratio=1.):
        super().__init__()
        self.pos_embed = torch.nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = GlobalSparseAttn(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W, D = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, D))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W, D)
        return x


class LKLGLBlock(torch.nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=torch.nn.GELU, norm_layer=torch.nn.LayerNorm, sr_ratio=1.):
        super().__init__()

        if sr_ratio > 1:
            self.LocalAgg = LocalAgg(dim, mlp_ratio, drop, drop_path, act_layer)
        else:
            self.LocalAgg = torch.nn.Identity()

        self.SelfAttn = SelfAttn(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer,
                                 norm_layer, sr_ratio)

    def forward(self, x):
        x = self.LocalAgg(x)
        x = self.SelfAttn(x)
        return x


class up_conv(torch.nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="trilinear"),
            torch.nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.InstanceNorm3d(ch_out),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class conv_block(torch.nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.InstanceNorm3d(ch_out),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.InstanceNorm3d(ch_out),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class MobileUViT(torch.nn.Module):
    def __init__(self,
                 inch=1,
                 dims=[16, 32, 64, 128],
                 depths=[1, 1, 3, 3, 3],
                 kernels=[3, 3, 7],
                 embed_dim=256):
        super(MobileUViT, self).__init__()
        self.patch_embeddings = Embeddings(inch=inch, dims=dims[:3], depths=depths[:3], kernels=kernels)

        self.lklgl_bottleneck = torch.nn.ModuleList([
            LKLGLBlock(
                dim=dims[2], num_heads=embed_dim // 64, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=0.1, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), sr_ratio=2)
            for _ in range(depths[3])])

        self.transformer_bottleneck = torch.nn.ModuleList([
            LKLGLBlock(
                dim=dims[3], num_heads=embed_dim // 64, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=0., norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), sr_ratio=1)
            for _ in range(depths[4])])

        self.down = torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.expend_dims = conv_block(ch_in=dims[2], ch_out=dims[3])
        self.reduce_dims = conv_block(ch_in=dims[3], ch_out=dims[2])

        self.Up_conv4 = conv_block(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = conv_block(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = conv_block(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Up1 = up_conv(ch_in=dims[0], ch_out=dims[0])

        main_c = dims[0]  # 16

        # ================= Global Multi-class Head =================
        # 0=background, 1=HV, 2=PV
        self.global_head = torch.nn.Conv3d(main_c, 3, 1)

        # ================= Auxiliary Decoder =================
        self.aux_decoder = torch.nn.Sequential(
            torch.nn.Conv3d(main_c, main_c // 2, 3, padding=1, bias=False),
            torch.nn.GroupNorm(8, main_c // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(main_c // 2, main_c // 4, 3, padding=1, bias=False),
        )
        aux_c = main_c // 4

        # ================= Structural Role Heads =================
        self.trunk_head = torch.nn.Conv3d(aux_c, 1, 1)
        self.small_head = torch.nn.Conv3d(aux_c, 1, 1)
        self.junc_head  = torch.nn.Conv3d(aux_c, 1, 1)

        # ================= Topology Gate =================
        self.trunk_for_gate = torch.nn.Conv3d(aux_c, main_c, 3, padding=3, dilation=3, bias=False)
        self.gate_act = torch.nn.Sigmoid()
        
        
    def forward(self, x):
        x, skip = self.patch_embeddings(x)
        x = self.down(x)

        for blk in self.lklgl_bottleneck:
            x = blk(x)
        x = self.expend_dims(x)

        for blk in self.transformer_bottleneck:
            x = blk(x)
        x = self.reduce_dims(x)

        # Decoder（共享）
        x = self.Up_conv4(torch.cat((x, self.down(skip[2])), dim=1))
        x = self.Up3(x)
        x = self.Up_conv3(torch.cat((x, self.down(skip[1])), dim=1))
        x = self.Up2(x)
        x = self.Up_conv2(torch.cat((x, self.down(skip[0])), dim=1))
        x = self.Up1(x)                             # F_primary [B, main_c, 96, 96, 96]

        # ================= Global Multi-class =================
        logits_global = self.global_head(x)   # [B,3,H,W,D]

        # ================= Auxiliary (stop-grad) =================
        f_aux = self.aux_decoder(x.detach())

        trunk = self.trunk_head(f_aux)
        small = self.small_head(f_aux)
        junc  = self.junc_head(f_aux)

        # ================= Topology-Gated Enhancement =================
        gate = self.gate_act(junc).detach()
        feat_gate = self.trunk_for_gate(f_aux).detach()
        x = x + gate * feat_gate

        # ================= Final Global =================
        logits_global = self.global_head(x)

        return logits_global, trunk, small, junc


def mobileuvit(inch=1, dims=[16, 32, 64, 128], depths=[1, 1, 3, 3, 3], kernels=[3, 3, 7], embed_dim=256):
    return MobileUViT(inch=inch, dims=dims, depths=depths, kernels=kernels, embed_dim=embed_dim)

def mobileuvit_l(inch=1, dims=[32, 64, 128, 256], depths=[1, 1, 3, 3, 4], kernels=[3, 3, 7], embed_dim=512):
    return MobileUViT(inch=inch, dims=dims, depths=depths, kernels=kernels, embed_dim=embed_dim)

# ========================= 推理功能 =========================
# ========================= 基础配置 =========================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth") 
INPUT_IMAGE = os.path.join(os.path.dirname(__file__), "hepaticvessel_437.nii", "hepaticvessel_437.nii")
INPUT_LABEL = os.path.join(os.path.dirname(__file__), "hp437.nii", "hp437.nii")
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "vis_results", current_time)
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