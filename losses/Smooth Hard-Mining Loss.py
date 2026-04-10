import torch
from torch import nn
import torch.nn.functional as F

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    # 归一化计算，避免除零错误
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-6)
    return x  # 修正：缩进至函数体内

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())  # 等效写法，注释保留
    dist = dist.clamp(min=1e-6).sqrt()  # 数值稳定性处理
    return dist  # 修正：缩进至函数体内

class SHM Loss(nn.Module):
    def __init__(self, margin=0.3, gamma=128, soft_mining=True):
        super(Tripletloss, self).__init__()
        self.margin = margin  # 三元组损失的边界值
        self.gamma = gamma    # 软挖掘的温度系数
        self.soft_mining = soft_mining  # 是否启用软挖掘

    def forward(self, inputs, targets):
        # 对输入特征进行L2归一化
        inputs = F.normalize(inputs, p=2, dim=1)
        # 计算所有样本间的欧式距离
        dist = euclidean_dist(inputs, inputs)
        n = inputs.size(0)
        
        # 构建正负样本掩码：mask_pos为1表示同一类，mask_neg为1表示不同类
        mask_pos = targets.expand(n, n).eq(targets.expand(n, n).t()).float()
        mask_neg = 1 - mask_pos
        mask_pos = mask_pos - torch.eye(n, device=dist.device)  # 排除样本自身

        if self.soft_mining:
            # 软挖掘模式：基于LogSumExp的分布级硬样本挖掘
            dist_ap = dist * mask_pos  # 正样本对距离（非自身）
            dist_an = dist * mask_neg + (1 - mask_neg) * 999.0  # 负样本对距离（填充极大值）
            soft_max_ap = torch.logsumexp(dist_ap * self.gamma, dim=1) / self.gamma
            soft_min_an = -torch.logsumexp(-dist_an * self.gamma, dim=1) / self.gamma
            
            # 计算带margin的三元组损失
            loss = F.relu(soft_max_ap - soft_min_an + self.margin)
        else:
            # 硬挖掘模式：直接取最难正样本和最难负样本
            dist_ap = (dist * mask_pos).max(dim=1)[0]
            dist_an = (dist * mask_neg + (1 - mask_neg) * 999.0).min(dim=1)[0]
            loss = F.relu(dist_ap - dist_an + self.margin)

        return loss.mean()  # 返回批次平均损失