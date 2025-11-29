"""
Albumentations-based augmentation wrappers and registry.
"""

from typing import Any, Dict, Tuple


import albumentations as A
from PIL import Image


from .albumentations_utils import _apply_albumentations_transform


class AlbumentationsBase:
    def __init__(self, transform, supports_boxes=True, supports_masks=True):
        """
        Wrap a single albumentations transform (or list) to work with DETR-style targets.
        Handles bbox params/mask wiring once, so subclasses only define params.
        """
        if isinstance(transform, (list, tuple)):
            transforms = list(transform)
        else:
            transforms = [transform]
        params = {}
        if supports_boxes:
            params["bbox_params"] = A.BboxParams(
                format="pascal_voc",
                label_fields=["category_ids", "bbox_ids"],
                clip=True,
            )
        self.compose = A.Compose(transforms, **params)
        self.supports_boxes = supports_boxes
        available_keys = getattr(self.compose, "_available_keys", set())
        self.mask_key = (
            "masks" if supports_masks and "masks" in available_keys else None
        )
        self.supports_masks = self.mask_key is not None

    def __call__(
        self, image: Image.Image, target: Dict[str, Any]
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        return _apply_albumentations_transform(
            self.compose,
            image,
            target,
            supports_boxes=self.supports_boxes,
            mask_key=self.mask_key,
        )


class AlbumentationsHorizontalFlip(AlbumentationsBase):
    def __init__(self, p=0.5):
        super().__init__(A.HorizontalFlip(p=p))


class AlbumentationsRotate(AlbumentationsBase):
    def __init__(self, limit=15, p=0.5):
        super().__init__(A.Rotate(limit=limit, p=p))


class AlbumentationsRandomBrightnessContrast(AlbumentationsBase):
    def __init__(self, p=0.2, brightness_limit=0.1, contrast_limit=0.1):
        super().__init__(
            A.RandomBrightnessContrast(
                p=p,
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
            )
        )


class AlbumentationsShiftScaleRotate(AlbumentationsBase):
    def __init__(self, shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5):
        super().__init__(
            A.ShiftScaleRotate(
                shift_limit=shift_limit,
                scale_limit=scale_limit,
                rotate_limit=rotate_limit,
                p=p,
                border_mode=0,
            )
        )


class AlbumentationsGaussNoise(AlbumentationsBase):
    def __init__(self, var_limit=(10.0, 50.0), p=0.3):
        super().__init__(A.GaussNoise(var_limit=var_limit, p=p))


class AlbumentationsColorJitter(AlbumentationsBase):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5):
        super().__init__(
            A.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                p=p,
            )
        )


class AlbumentationsBlur(AlbumentationsBase):
    def __init__(self, blur_limit=7, p=0.3):
        super().__init__(A.Blur(blur_limit=blur_limit, p=p))


class AlbumentationsCoarseDropout(AlbumentationsBase):
    def __init__(self, max_holes=8, max_height=16, max_width=16, p=0.5):
        super().__init__(
            A.CoarseDropout(
                max_holes=max_holes,
                max_height=max_height,
                max_width=max_width,
                p=p,
            )
        )


class AlbumentationsVerticalFlip(AlbumentationsBase):
    def __init__(self, p=0.5):
        super().__init__(A.VerticalFlip(p=p))


class AlbumentationsHueSaturationValue(AlbumentationsBase):
    def __init__(
        self, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
    ):
        super().__init__(
            A.HueSaturationValue(
                hue_shift_limit=hue_shift_limit,
                sat_shift_limit=sat_shift_limit,
                val_shift_limit=val_shift_limit,
                p=p,
            )
        )


class AlbumentationsCLAHE(AlbumentationsBase):
    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8), p=0.5):
        super().__init__(
            A.CLAHE(clip_limit=clip_limit, tile_grid_size=tile_grid_size, p=p)
        )


class AlbumentationsChannelShuffle(AlbumentationsBase):
    def __init__(self, p=0.5):
        super().__init__(A.ChannelShuffle(p=p))


class AlbumentationsRandomCrop(AlbumentationsBase):
    def __init__(self, height=224, width=224, p=0.5):
        super().__init__(A.RandomCrop(height=height, width=width, p=p))


class AlbumentationsAffine(AlbumentationsBase):
    def __init__(
        self,
        scale=(0.9, 1.1),
        translate_percent=(0.1, 0.1),
        rotate=(-15, 15),
        shear=(-10, 10),
        fit_output=False,
        p=0.5,
    ):
        super().__init__(
            A.Affine(
                scale=scale,
                translate_percent=translate_percent,
                rotate=rotate,
                shear=shear,
                p=p,
                fit_output=fit_output,
            )
        )


class AlbumentationsRandomShadow(AlbumentationsBase):
    def __init__(
        self,
        flare_roi=(0, 0.5, 1, 1),
        angle_lower=0.3,
        angle_upper=1.3,
        num_flare_circles_lower=1,
        num_flare_circles_upper=3,
        p=0.5,
    ):
        super().__init__(
            A.RandomSunFlare(
                flare_roi=flare_roi,
                angle_lower=angle_lower,
                angle_upper=angle_upper,
                num_flare_circles_lower=num_flare_circles_lower,
                num_flare_circles_upper=num_flare_circles_upper,
                p=p,
            )
        )


ALBUMENTATIONS_AUGS = {
    "AlbumentationsHorizontalFlip": AlbumentationsHorizontalFlip,
    "AlbumentationsRotate": AlbumentationsRotate,
    "AlbumentationsRandomBrightnessContrast": AlbumentationsRandomBrightnessContrast,
    "AlbumentationsShiftScaleRotate": AlbumentationsShiftScaleRotate,
    "AlbumentationsGaussNoise": AlbumentationsGaussNoise,
    "AlbumentationsColorJitter": AlbumentationsColorJitter,
    "AlbumentationsBlur": AlbumentationsBlur,
    "AlbumentationsCoarseDropout": AlbumentationsCoarseDropout,
    "AlbumentationsVerticalFlip": AlbumentationsVerticalFlip,
    "AlbumentationsHueSaturationValue": AlbumentationsHueSaturationValue,
    "AlbumentationsCLAHE": AlbumentationsCLAHE,
    "AlbumentationsChannelShuffle": AlbumentationsChannelShuffle,
    "AlbumentationsRandomCrop": AlbumentationsRandomCrop,
    "AlbumentationsAffine": AlbumentationsAffine,
    "AlbumentationsRandomShadow": AlbumentationsRandomShadow,
}


def build_albumentations_from_config(config_dict):
    augmentations = []
    for aug_name, params in config_dict.items():
        aug_class = ALBUMENTATIONS_AUGS.get(aug_name)
        if aug_class:
            augmentations.append(aug_class(**params))
        else:
            print(f"Warning: Unknown augmentation {aug_name}")
    return augmentations
