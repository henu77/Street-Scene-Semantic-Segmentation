"""
数据处理模块初始化
"""
try:
    from .dataset import CamVidLoader
    from .augmentation import get_augmentation
    __all__ = ['CamVidLoader', 'get_augmentation']
except ImportError as e:
    print(f"Warning: Could not import data modules: {e}")
    __all__ = []
