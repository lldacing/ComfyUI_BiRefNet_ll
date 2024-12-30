import numpy as np
import torch
from PIL import Image


def tensor_to_pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil_to_tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def add_mask_as_alpha(image, mask):
    """
    将 (b, h, w) 形状的 mask 添加为 (b, h, w, 3) 形状的 image 的第 4 个通道（alpha 通道）。
    """
    # 检查输入形状
    assert image.dim() == 4 and image.size(-1) == 3, "image 的形状必须为 (b, h, w, 3)"
    assert mask.dim() == 3, "mask 的形状必须为 (b, h, w)"
    assert image.size(0) == mask.size(0) and image.size(1) == mask.size(1) and image.size(2) == mask.size(2), "image 和 mask 的批次、高、宽维度必须一致"

    # 将 mask 扩展为 (b, h, w, 1)
    mask = mask[..., None]

    # 将 image 和 mask 拼接为 (b, h, w, 4)
    image_with_alpha = torch.cat([image, mask], dim=-1)

    return image_with_alpha


def normalize_mask(mask_tensor):
    max_val = torch.max(mask_tensor)
    min_val = torch.min(mask_tensor)

    if max_val == min_val:
        return mask_tensor

    normalized_mask = (mask_tensor - min_val) / (max_val - min_val)

    return normalized_mask
