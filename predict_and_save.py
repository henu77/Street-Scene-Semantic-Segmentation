import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from models.unet import UNet
from models.improved_unet import UNetImprove
from data.dataset import CamVidLoader
from config import IMG_SIZE


def get_model(model_type, num_classes):
    if model_type == "unet":
        return UNet(3, n_classes=num_classes, width=32, bilinear=True)
    elif model_type == "improved_unet":
        return UNetImprove(3, n_classes=num_classes, width=32, bilinear=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def save_image(np_img, path):
    img = Image.fromarray(np_img.astype(np.uint8))
    img.save(path)


def main():
    parser = argparse.ArgumentParser(description="UNet Prediction Script")
    parser.add_argument(
        "--model_type", type=str, choices=["unet", "improved_unet"], default="improved_unet"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="outputs/part3_small_object_optimization/improved_unet_model/best_model.pth",
    )
    parser.add_argument("--dataset", type=str, default="CamVid/sunny")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_cuda", action="store_true", default=True)
    parser.add_argument(
        "--num_classes", type=int, default=14
    )  # 根据你的数据集类别数调整
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    )

    save_dir = f"visualization/{args.model_type}_predictions/{args.dataset}"

    raw_dir = os.path.join(save_dir, "raw")
    pred_dir = os.path.join(save_dir, "pred")
    gt_dir = os.path.join(save_dir, "gt")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # 加载模型
    model = get_model(args.model_type, args.num_classes)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    # 数据集和DataLoader
    dataset = CamVidLoader(
        root=args.dataset,
        split="test",
        img_size=IMG_SIZE,
        is_aug=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_nums = len(dataset)
    print(f"Total images to predict: {all_nums}")
    with torch.no_grad():
        for idx, (img, gt, img_name) in enumerate(loader):
            img = img.to(device)
            output = model(img)
            pred = torch.argmax(output, dim=1).cpu().numpy()
            img_np = img.cpu().numpy().transpose(0, 2, 3, 1)  # (B, H, W, C), BGR, 已减均值
            gt_np = gt.cpu().numpy()
            for b in range(img_np.shape[0]):
                base_name = os.path.splitext(os.path.basename(img_name[b]))[0]
                # 反归一化原图像（BGR->RGB, 加均值, clip到0-255）
                img_bgr = img_np[b] + np.array([104.00699, 116.66877, 122.67892]) / 255.0
                img_rgb = img_bgr[..., ::-1]  # BGR->RGB
                img_rgb = np.clip(img_rgb * 255, 0, 255).astype(np.uint8)
                save_image(img_rgb, os.path.join(raw_dir, base_name + ".png"))

                # 预测和真值用颜色映射
                from data.dataset import from_label_to_rgb
                pred_rgb = (from_label_to_rgb(pred[b]) * 255).astype(np.uint8)
                gt_rgb = (from_label_to_rgb(gt_np[b]) * 255).astype(np.uint8)
                save_image(pred_rgb, os.path.join(pred_dir, base_name + ".png"))
                save_image(gt_rgb, os.path.join(gt_dir, base_name + ".png"))
            if (idx + 1) % 10 == 0 or idx + 1 == all_nums:
                print(f"Processed {idx + 1}/{all_nums} images.")

if __name__ == "__main__":
    main()
