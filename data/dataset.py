"""
CamVid数据集加载器
"""
import os
import collections
import torch
import numpy as np
from skimage.transform import resize
from skimage.io import imread
import matplotlib.pyplot as plt
from torch.utils import data
import os.path as pathlib
import torchvision.transforms as T

class_labels_dict = {
    "Sky": 0,
    "Building": 1,
    "Pole": 2,
    "Road": 3,
    "LaneMarking": 4,
    "SideWalk": 5,
    "Pavement": 6,
    "Tree": 7,
    "SignSymbol": 8,
    "Fence": 9,
    "Car_Bus": 10,
    "Pedestrian": 11,
    "Bicyclist": 12,
    "Unlabelled": 13
}

# 类别颜色映射
Sky = [[128, 128, 128]]
Building = [[128, 0, 0]]
Pole = [[192, 192, 128], [0, 0, 64]]
Road = [[128, 64, 128], [128, 128, 192]]
LaneMarking = [[128, 0, 192], [128, 0, 64]]
SideWalk = [[0, 0, 192]]
Pavement = [[60, 40, 222]]
Tree = [[128, 128, 0]]
SignSymbol = [[192, 128, 128]]
Fence = [[64, 64, 128]]
Car_Bus = [[64, 0, 128], [64, 128, 192], [192, 128, 192]]
Pedestrian = [[64, 64, 0]]
Bicyclist = [[0, 128, 192]]
Unlabelled = [[0, 0, 0]]

label_colours = [
    Sky, Building, Pole, Road, LaneMarking, SideWalk, 
    Pavement, Tree, SignSymbol, Fence, Car_Bus, 
    Pedestrian, Bicyclist, Unlabelled,
]


def from_label_to_rgb(label):
    """将标签图像转换为RGB图像"""
    n_classes = len(label_colours)
    r = label.copy()
    g = label.copy()
    b = label.copy()
    for l in range(0, n_classes):
        r[label == l] = label_colours[l][0][0]
        g[label == l] = label_colours[l][0][1]
        b[label == l] = label_colours[l][0][2]

    rgb = np.zeros((label.shape[0], label.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


def from_rgb_to_label(rgb):
    """将RGB图像转换为标签图像"""
    n_classes = len(label_colours)
    label = np.ones((rgb.shape[0], rgb.shape[1])) * (-1)
    r = rgb[:,:,0]
    g = rgb[:,:,1]
    b = rgb[:,:,2]
    for l in range(0, n_classes):
        final_mask = np.zeros_like(label).astype(bool)
        for j in label_colours[l]:
            curr_mask = (r == j[0]) &  (g == j[1]) & (b == j[2])
            final_mask = final_mask | curr_mask
        label[final_mask] = l
    label[label==-1] = 11  # 其他标签都被认为是"unlabelled"
    label = label.astype(np.uint8)
    return label


class CamVidLoader(data.Dataset):
    """CamVid数据集加载器"""
    def __init__(
        self,
        root,
        split="train",
        is_aug=False,
        aug=None,
        is_pytorch_transform=True,
        img_size=None
    ):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.is_aug = is_aug
        self.is_pytorch_transform = is_pytorch_transform
        self.augmentation = aug
        self.mean = np.array([104.00699/255.0, 116.66877/255.0, 122.67892/255.0])
        self.n_classes = 14
        self.files = collections.defaultdict(list)
        
        # 获取文件列表
        for split_name in ["train", "test", "val"]:
            split_path = os.path.join(root, split_name)
            if os.path.exists(split_path):
                file_list = os.listdir(split_path)
                self.files[split_name] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = os.path.join(self.root, self.split, img_name)
        lbl_path = os.path.join(self.root, self.split + "_labels", 
                               pathlib.splitext(img_name)[0] + '_L.png')
        
        # 读取图像和标签
        img = imread(img_path)
        lbl = imread(lbl_path)
        lbl = from_rgb_to_label(lbl)
        
        # 转换为PIL图像
        img = T.ToPILImage()(img)
        lbl = T.ToPILImage()(lbl)
        
        # 数据增强
        if self.is_aug and self.augmentation is not None:
            img, lbl = self.augmentation(img, lbl)
            
        # 变换处理
        img, lbl = self.transform(img, lbl)
        return img, lbl, img_path

    def transform(self, img, lbl):
        """图像变换处理"""
        img, lbl = np.array(img), np.array(lbl)

        # 调整大小
        img = resize(img, (self.img_size[0], self.img_size[1]), order=2)
        lbl = resize(lbl, (self.img_size[0], self.img_size[1]), order=0)
        
        if self.is_pytorch_transform:
            img = img[:, :, ::-1]  # RGB -> BGR
            img = img.astype(np.float64)
            img -= self.mean  # 减去均值
            img = img.transpose(2, 0, 1)  # NHWC -> NCHW
            img = torch.from_numpy(img).float()
            lbl = torch.from_numpy(lbl).long()
        return img, lbl
