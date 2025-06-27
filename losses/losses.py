"""
损失函数实现
"""
import torch
import torch.nn.functional as F


class DiceLoss(torch.nn.Module):
    """Dice损失函数实现"""
    def __init__(self, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        pred = torch.softmax(pred, dim=1)
        
        intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
        union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
        
        dice_score = (2. * intersection + self.epsilon) / (union + self.epsilon)
        return 1 - dice_score.mean()


class CrossEntropyLoss(torch.nn.Module):
    """交叉熵损失函数实现"""
    def __init__(self, weight=None, ignore_index=255):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        return F.cross_entropy(pred, target, weight=self.weight, ignore_index=self.ignore_index)


class CombinedLoss(torch.nn.Module):
    """组合损失函数（Dice + 交叉熵）"""
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss()
        self.ce_loss = CrossEntropyLoss()

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        return self.alpha * dice + (1 - self.alpha) * ce
