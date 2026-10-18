# -*- coding: utf-8 -*-
"""Loss functions and schedule used to train AICFuse."""
import torch
from torch import nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """1 - SSIM，值越小表示结构越相似"""
    def __init__(self, window_size=11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size

    def forward(self, img1, img2):
        C1 = (0.01 * 1.0) ** 2
        C2 = (0.03 * 1.0) ** 2
        pad = self.window_size // 2

        mu1 = F.avg_pool2d(img1, self.window_size, stride=1, padding=pad)
        mu2 = F.avg_pool2d(img2, self.window_size, stride=1, padding=pad)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu12 = mu1 * mu2

        sig1_sq = F.avg_pool2d(img1 * img1, self.window_size,
                               stride=1, padding=pad) - mu1_sq
        sig2_sq = F.avg_pool2d(img2 * img2, self.window_size,
                               stride=1, padding=pad) - mu2_sq
        sig12 = F.avg_pool2d(img1 * img2, self.window_size,
                             stride=1, padding=pad) - mu12

        ssim_map = ((2 * mu12 + C1) * (2 * sig12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2))
        return 1.0 - ssim_map.mean()


# ============================================================================
# 2. Sobel 梯度损失
# ============================================================================
class SobelGradientLoss(nn.Module):
    """Sobel 梯度 L1 损失：强制重建图像与原图梯度一致"""
    def __init__(self):
        super(SobelGradientLoss, self).__init__()
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32
                               ).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0,  0,  0],
                                [1,  2,  1]], dtype=torch.float32
                               ).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _gradient(self, img):
        gx = F.conv2d(img, self.sobel_x, padding=1)
        gy = F.conv2d(img, self.sobel_y, padding=1)
        return gx, gy

    def forward(self, pred, target):
        gx_p, gy_p = self._gradient(pred)
        gx_t, gy_t = self._gradient(target)
        return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)


# ============================================================================
# 3. ★ 纹理-梯度联合损失 (Texture-Gradient Joint Loss)
# ============================================================================
class TextureGradientLoss(nn.Module):
    """
    纹理-梯度联合损失 = texture_w * SSIM_loss + gradient_w * Gradient_loss

    - SSIM 损失: 保证亮度、对比度、结构相似性
    - 梯度损失: Sobel 算子提取梯度，强制梯度一致，锐化边缘
    """
    def __init__(self, texture_weight=5.0, gradient_weight=10.0):
        super(TextureGradientLoss, self).__init__()
        self.texture_weight = texture_weight
        self.gradient_weight = gradient_weight
        self.ssim_loss = SSIMLoss(window_size=11)
        self.grad_loss = SobelGradientLoss()

    def forward(self, original, reconstructed):
        ssim_l = self.ssim_loss(original, reconstructed)
        grad_l = self.grad_loss(reconstructed, original)
        return self.texture_weight * ssim_l + self.gradient_weight * grad_l


# ============================================================================
# 4. ★ 细节模态对齐损失 (余弦相似度)
# ============================================================================
class DetailAlignmentLoss(nn.Module):
    """
    loss = 1 - cosine_similarity(D_I, D_V)

    从特征方向角度对齐细节特征，范围 [0, 2]。
    """
    def __init__(self):
        super(DetailAlignmentLoss, self).__init__()

    def forward(self, feat_I_D, feat_V_D):
        B = feat_I_D.shape[0]
        f1 = feat_I_D.reshape(B, -1)
        f2 = feat_V_D.reshape(B, -1)
        cos_sim = F.cosine_similarity(f1, f2, dim=1)
        return 1.0 - cos_sim.mean()


# ============================================================================
# 5. ★ 动态损失权重调度器
# ============================================================================
class DynamicLossScheduler:
    """
    训练初期: 重点学习基础对齐和重构 (稳定结构)
    训练后期: 逐渐加强细节对齐 (细化纹理)

    权重策略:
        base_weight    = 1.0                              (始终保持)
        detail_weight  = min(1.0, 0.5 + 0.5 * progress)  (随训练增大)
        recon_weight   = 0.6                              (固定)
        texture_weight = 0.3                              (固定)
        align_weight   = 0.5                              (固定)
    """
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs

    def get_weights(self, current_epoch):
        progress = current_epoch / self.total_epochs  # 0→1
        return {
            'base_weight':    1.0,
            'detail_weight':  min(1.0, 0.5 + 0.5 * progress),
            'recon_weight':   0.6,
            'texture_weight': 0.3,
            'align_weight':   0.5,
        }


# ============================================================================
# 6. 特征归一化
# ============================================================================
def normalize_features(feat):
    """Instance-wise L2 归一化，平衡不同损失的尺度"""
    B = feat.shape[0]
    feat_flat = feat.reshape(B, -1)
    norm = feat_flat.norm(p=2, dim=1, keepdim=True) + 1e-8
    feat_norm = feat_flat / norm
    return feat_norm.reshape_as(feat)

