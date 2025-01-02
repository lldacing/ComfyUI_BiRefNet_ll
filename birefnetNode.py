import os
import safetensors.torch
import torch
from torchvision import transforms
from torch.hub import download_url_to_file
import comfy
from comfy import model_management
import folder_paths
from birefnet.models.birefnet import BiRefNet
from birefnet_old.models.birefnet import BiRefNet as OldBiRefNet
from birefnet.utils import check_state_dict
from .util import tensor_to_pil, apply_mask_to_image, normalize_mask, refine_foreground
deviceType = model_management.get_torch_device().type

models_dir_key = "birefnet"

models_path_default = folder_paths.get_folder_paths(models_dir_key)[0]

usage_to_weights_file = {
    'General': 'BiRefNet',
    'General-Lite': 'BiRefNet_T',
    'General-Lite-2K': 'BiRefNet_lite-2K',
    'Portrait': 'BiRefNet-portrait',
    'Matting': 'BiRefNet-matting',
    'DIS': 'BiRefNet-DIS5K',
    'HRSOD': 'BiRefNet-HRSOD',
    'COD': 'BiRefNet-COD',
    'DIS-TR_TEs': 'BiRefNet-DIS5K-TR_TEs'
}

modelNameList = ['General', 'General-Lite', 'General-Lite-2K', 'Portrait', 'Matting', 'DIS', 'HRSOD', 'COD', 'DIS-TR_TEs']


def get_model_path(model_name):
    return os.path.join(models_path_default, f"{model_name}.safetensors")


def download_models(model_root, model_urls):
    if not os.path.exists(model_root):
        os.makedirs(model_root, exist_ok=True)

    for local_file, url in model_urls:
        local_path = os.path.join(model_root, local_file)
        if not os.path.exists(local_path):
            local_path = os.path.abspath(os.path.join(model_root, local_file))
            download_url_to_file(url, dst=local_path)


def download_birefnet_model(model_name):
    """
    Downloading model from huggingface.
    """
    model_root = os.path.join(models_path_default)
    model_urls = (
        (f"{model_name}.safetensors",
         f"https://huggingface.co/ZhengPeng7/{usage_to_weights_file[model_name]}/resolve/main/model.safetensors"),
    )
    download_models(model_root, model_urls)

class ImagePreprocessor():
    def __init__(self, resolution) -> None:
        self.transform_image = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform_image_old = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
        ])

    def proc(self, image) -> torch.Tensor:
        image = self.transform_image(image)
        return image

    def old_proc(self, image) -> torch.Tensor:
        image = self.transform_image_old(image)
        return image

VERSION = ["old", "v1"]
old_models_name = ["BiRefNet-DIS_ep580.pth", "BiRefNet-ep480.pth"]


class AutoDownloadBiRefNetModel:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (modelNameList,),
                "device": (["AUTO", "CPU"],)
            }
        }

    RETURN_TYPES = ("BiRefNetMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "image/BiRefNet"
    DESCRIPTION = "Auto download BiRefNet model from huggingface to models/BiRefNet/{model_name}.safetensors"

    def load_model(self, model_name, device):
        bb_index = 3 if model_name == "General-Lite" or model_name == "General-Lite-2K" else 6
        biRefNet_model = BiRefNet(bb_pretrained=False, bb_index=bb_index)
        model_file_name = f'{model_name}.safetensors'
        model_full_path = folder_paths.get_full_path(models_dir_key, model_file_name)
        if model_full_path is None:
            download_birefnet_model(model_name)
            model_full_path = folder_paths.get_full_path(models_dir_key, model_file_name)
        if device == "AUTO":
            device_type = deviceType
        else:
            device_type = "cpu"
        state_dict = safetensors.torch.load_file(model_full_path, device=device_type)
        biRefNet_model.load_state_dict(state_dict)
        biRefNet_model.to(device_type)
        biRefNet_model.eval()
        return [(biRefNet_model, VERSION[1])]


class LoadRembgByBiRefNetModel:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (folder_paths.get_filename_list(models_dir_key),),
                "device": (["AUTO", "CPU"], )
            },
            "optional": {
                "use_weight": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("BiRefNetMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "rembg/BiRefNet"
    DESCRIPTION = "Load BiRefNet model from folder models/BiRefNet or the path of birefnet configured in the extra YAML file"

    def load_model(self, model, device, use_weight=False):
        if model in old_models_name:
            version = VERSION[0]
            biRefNet_model = OldBiRefNet(bb_pretrained=use_weight)
        else:
            version = VERSION[1]
            bb_index = 3 if model == "General-Lite.safetensors" or model == "General-Lite-2K.safetensors" else 6
            biRefNet_model = BiRefNet(bb_pretrained=use_weight, bb_index=bb_index)

        model_path = folder_paths.get_full_path(models_dir_key, model)
        if device == "AUTO":
            device_type = deviceType
        else:
            device_type = "cpu"
        if model_path.endswith(".safetensors"):
            state_dict = safetensors.torch.load_file(model_path, device=device_type)
        else:
            state_dict = torch.load(model_path, map_location=device_type)
            state_dict = check_state_dict(state_dict)

        biRefNet_model.load_state_dict(state_dict)
        biRefNet_model.to(device_type)
        biRefNet_model.eval()
        return [(biRefNet_model, version)]


class RembgByBiRefNetAdvanced:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BiRefNetMODEL",),
                "images": ("IMAGE",),
                "width": ("INT",
                          {
                              "default": 1024,
                              "min": 0,
                              "max": 16384,
                              "tooltip": "The width of the preprocessed image, does not affect the final output image size"
                          }),
                "height": ("INT",
                           {
                               "default": 1024,
                               "min": 0,
                               "max": 16384,
                               "tooltip": "The height of the preprocessed image, does not affect the final output image size"
                           }),
                "upscale_method": (["bislerp", "nearest-exact", "bilinear", "area", "bicubic"],
                                   {
                                       "default": "bilinear",
                                       "tooltip": "Interpolation method for post-processing mask"
                                   }),
                "blur_size": ("INT", {"default": 91, "min": 1, "max": 255, "step": 2, }),
                "blur_size_two": ("INT", {"default": 7, "min": 1, "max": 255, "step": 2, }),
                "fill_color": ("BOOLEAN", {"default": False}),
                "color": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFF, "step": 1, "display": "color"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "rem_bg"
    CATEGORY = "rembg/BiRefNet"

    def rem_bg(self, model, images, upscale_method='bilinear', width=1024, height=1024, blur_size=91, blur_size_two=7, fill_color=False, color=None):
        model, version = model
        model_device_type = next(model.parameters()).device.type
        _images = []
        _masks = []

        for image in images:
            h, w, c = image.shape
            pil_image = tensor_to_pil(image)

            image_preproc = ImagePreprocessor(resolution=(width, height))
            if VERSION[0] == version:
                im_tensor = image_preproc.old_proc(pil_image).unsqueeze(0)
            else:
                im_tensor = image_preproc.proc(pil_image).unsqueeze(0)

            del image_preproc

            with torch.no_grad():
                mask = model(im_tensor.to(model_device_type))[-1].sigmoid().cpu()

            # 遮罩大小需还原为与原图一致
            mask = comfy.utils.common_upscale(mask, w, h, upscale_method, "disabled")

            # (1, 1, h, w)
            mask = normalize_mask(mask)
            # (c, h, w) => (c, h, w)
            _image_masked = refine_foreground(image.permute(2, 0, 1), mask.squeeze(0), r1=blur_size, r2=blur_size_two).squeeze(0)
            # (c, h, w) => (h, w, c)
            _image_masked = _image_masked.permute(1, 2, 0)
            if fill_color and color is not None:
                r = torch.full([h, w, 1], ((color >> 16) & 0xFF) / 0xFF)
                g = torch.full([h, w, 1], ((color >> 8) & 0xFF) / 0xFF)
                b = torch.full([h, w, 1], (color & 0xFF) / 0xFF)
                # (h, w, 3)
                background_color = torch.cat((r, g, b), dim=-1)
                # (h, w, 1)
                apply_mask = mask.squeeze(0).permute(1, 2, 0).expand_as(_image_masked)
                _image_masked = _image_masked * apply_mask + background_color * (1 - apply_mask)
                # (h, w, 3)=>(1, h, w,3)
                image = _image_masked.unsqueeze(0)
                del background_color, apply_mask
            else:
                # image的非mask对应部分设为透明 => (1, h, w, 4)
                image = apply_mask_to_image(_image_masked.cpu(), mask.cpu())

            _images.append(image)
            _masks.append(mask.squeeze(0))

        out_images = torch.cat(_images, dim=0)
        out_masks = torch.cat(_masks, dim=0)

        return out_images, out_masks


class RembgByBiRefNet(RembgByBiRefNetAdvanced):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BiRefNetMODEL",),
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "rem_bg"
    CATEGORY = "rembg/BiRefNet"

    def rem_bg(self, model, images):
        return super().rem_bg(model, images)


NODE_CLASS_MAPPINGS = {
    "AutoDownloadBiRefNetModel": AutoDownloadBiRefNetModel,
    "LoadRembgByBiRefNetModel": LoadRembgByBiRefNetModel,
    "RembgByBiRefNet": RembgByBiRefNet,
    "RembgByBiRefNetAdvanced": RembgByBiRefNetAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoDownloadBiRefNetModel": "AutoDownloadBiRefNetModel",
    "LoadRembgByBiRefNetModel": "LoadRembgByBiRefNetModel",
    "RembgByBiRefNet": "RembgByBiRefNet",
    "RembgByBiRefNetAdvanced": "RembgByBiRefNetAdvanced",
}
