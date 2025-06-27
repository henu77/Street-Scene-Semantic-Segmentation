"""
损失函数模块初始化
"""
try:
    from .losses import DiceLoss, CrossEntropyLoss, CombinedLoss
    __all__ = ['DiceLoss', 'CrossEntropyLoss', 'CombinedLoss']
except ImportError as e:
    print(f"Warning: Could not import loss functions: {e}")
    __all__ = []
