import numpy as np
import torch
from PIL import Image


def tensor_to_pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil_to_tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def apply_mask_to_image(image, mask):
    """
    Apply a mask to an image and set non-masked parts to transparent.

    Args:
        image (torch.Tensor): Image tensor of shape (h, w, c) or (1, h, w, c).
        mask (torch.Tensor): Mask tensor of shape (1, 1, h, w) or (h, w).

    Returns:
        torch.Tensor: Masked image tensor of shape (h, w, c+1) with transparency.
    """
    # 判断 image 的形状
    if image.dim() == 3:
        pass
    elif image.dim() == 4:
        image = image.squeeze(0)
    else:
        raise ValueError("Image should be of shape (h, w, c) or (1, h, w, c).")

    h, w, c = image.shape
    # 判断 mask 的形状
    if mask.dim() == 4:
        mask = mask.squeeze(0).squeeze(0)  # 去掉前2个维度 (h,w)
    elif mask.dim() == 3:
        mask = mask.squeeze(0)
    elif mask.dim() == 2:
        pass
    else:
        raise ValueError("Mask should be of shape (1, 1, h, w) or (h, w).")

    assert mask.shape == (h, w), "Mask shape does not match image shape."

    # 将 mask 扩展到与 image 相同的通道数
    image_mask = mask.unsqueeze(-1).expand(h, w, c)

    # 应用遮罩，黑色部分是0，相乘后白色1的部分会被保留，其它部分变为了黑色
    masked_image = image * image_mask

    # 遮罩的黑白当做alpha通道的不透明度，黑色是0表示透明，白色是1表示不透明
    alpha = mask
    # alpha通道拼接到原图像的RGB中
    masked_image_with_alpha = torch.cat((masked_image[:, :, :3], alpha.unsqueeze(2)), dim=2)

    return masked_image_with_alpha.unsqueeze(0)


def normalize_mask(mask_tensor):
    max_val = torch.max(mask_tensor)
    min_val = torch.min(mask_tensor)

    if max_val == min_val:
        return mask_tensor

    normalized_mask = (mask_tensor - min_val) / (max_val - min_val)

    return normalized_mask
