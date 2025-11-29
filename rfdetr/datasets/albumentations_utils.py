"""
Utility helpers for applying albumentations transforms to DETR-style targets.
Centralizes mask/box wiring so individual transforms remain minimal.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import albumentations as A
import numpy as np
import torch
from PIL import Image


def _as_numpy(data: Any) -> Optional[np.ndarray]:
    """Return a detached numpy view for tensors/arrays; passthrough otherwise."""
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _labels_to_list(labels: Any, length: int) -> List[int]:
    """Convert label tensor/iterable to a list with fallback zeros for missing labels."""
    if labels is None:
        return [0] * length
    if isinstance(labels, torch.Tensor):
        return labels.detach().cpu().tolist()
    return list(labels)


def _albumentations_masks_to_tensor(
    mask_list: Iterable[np.ndarray], height: int, width: int
) -> torch.Tensor:
    if not mask_list:
        return torch.zeros((0, height, width), dtype=torch.bool)
    masks_np = np.stack(
        [np.asarray(mask, dtype=np.float32) for mask in mask_list],
        axis=0,
    )
    masks_np = masks_np > 0.5
    return torch.from_numpy(masks_np)


def _resolve_mask_key(compose: A.Compose) -> Optional[str]:
    available_keys = getattr(compose, "_available_keys", set())
    if "masks" in available_keys:
        return "masks"
    if "mask" in available_keys:
        return "mask"
    return None


def _prepare_masks_for_albumentations(masks: Any, mask_key: Optional[str]):
    """Format masks for albumentations compose based on the expected key and rank."""
    if mask_key is None or masks is None:
        return [] if mask_key == "masks" else None

    masks_np = _as_numpy(masks)
    if masks_np.dtype == np.bool_:
        masks_np = masks_np.astype(np.uint8)
    else:
        masks_np = masks_np.astype(np.uint8, copy=False)
    if masks_np.ndim == 2:
        return [masks_np] if mask_key == "masks" else masks_np
    if masks_np.ndim == 3:
        # DETR-style masks have shape [N, H, W]
        if mask_key == "masks":
            return [masks_np[idx] for idx in range(masks_np.shape[0])]
        if mask_key == "mask":
            return np.moveaxis(masks_np, 0, -1)
    raise TypeError(
        f"Unsupported mask tensor with shape {masks_np.shape} for key '{mask_key}'"
    )


def _extract_masks_from_albumentations(
    masks_aug: Any, mask_key: Optional[str]
) -> List[np.ndarray]:
    """Normalize albumentations mask output back to a list of uint8 arrays."""
    if mask_key is None or masks_aug is None:
        return []

    if mask_key == "masks":
        if isinstance(masks_aug, np.ndarray):
            if masks_aug.ndim == 2:
                return [masks_aug.astype(np.uint8, copy=False)]
            if masks_aug.ndim == 3:
                return [
                    masks_aug[idx].astype(np.uint8, copy=False)
                    for idx in range(masks_aug.shape[0])
                ]
        return [np.asarray(mask, dtype=np.uint8) for mask in masks_aug]

    if mask_key == "mask":
        if isinstance(masks_aug, np.ndarray):
            if masks_aug.ndim == 2:
                return [masks_aug.astype(np.uint8, copy=False)]
            if masks_aug.ndim == 3:
                return [
                    masks_aug[..., idx].astype(np.uint8, copy=False)
                    for idx in range(masks_aug.shape[-1])
                ]
        return [np.asarray(mask, dtype=np.uint8) for mask in masks_aug]

    raise TypeError(f"Unsupported augmented mask type for key '{mask_key}'")


def apply_albumentations_transform(
    compose: A.Compose,
    image: Image.Image,
    target: Dict[str, Any],
    supports_boxes: bool = True,
    supports_masks: bool = True,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Apply an albumentations compose to PIL image/target dict, handling boxes/masks."""
    mask_key = _resolve_mask_key(compose) if supports_masks else None
    return _apply_albumentations_transform(
        compose,
        image,
        target,
        supports_boxes=supports_boxes,
        mask_key=mask_key,
    )


def _apply_albumentations_transform(
    compose: A.Compose,
    image: Image.Image,
    target: Dict[str, Any],
    supports_boxes: bool = True,
    mask_key: Optional[str] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Internal albumentations entry point that handles boxes/masks/labels uniformly."""
    image_np = np.array(image)
    data = {"image": image_np}
    target_out = target.copy()

    processors = getattr(compose, "processors", {})
    bbox_processor = processors.get("bboxes") if isinstance(processors, dict) else None
    has_bbox_params = bbox_processor is not None and supports_boxes

    source_masks = target.get("masks")
    if mask_key is None and source_masks is not None:
        raise ValueError(
            "Mask data provided but the albumentations compose does not expose mask support."
        )

    if has_bbox_params:
        if supports_boxes and "boxes" in target and target["boxes"] is not None:
            boxes_np = _as_numpy(target["boxes"]).reshape(-1, 4)
            bboxes = [tuple(map(float, box)) for box in boxes_np.tolist()]
        else:
            boxes_np = np.zeros((0, 4), dtype=np.float32)
            bboxes = []
        bbox_ids = list(range(len(bboxes)))
        labels_list = _labels_to_list(target.get("labels"), len(bboxes))
        data["bboxes"] = bboxes
        data["category_ids"] = labels_list
        data["bbox_ids"] = bbox_ids

    if mask_key is not None:
        mask_payload = _prepare_masks_for_albumentations(source_masks, mask_key)
        data[mask_key] = mask_payload

    augmented = compose(**data)

    aug_image = augmented["image"]
    image_out = Image.fromarray(aug_image)
    target_out["size"] = torch.as_tensor([image_out.height, image_out.width])

    if has_bbox_params:
        bboxes_aug = augmented.get("bboxes", [])
        new_boxes = (
            torch.as_tensor(bboxes_aug, dtype=torch.float32)
            if bboxes_aug
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        target_out["boxes"] = new_boxes

        orig_labels = target.get("labels")
        label_dtype = (
            orig_labels.dtype if isinstance(orig_labels, torch.Tensor) else torch.int64
        )
        target_out["labels"] = torch.as_tensor(
            augmented.get("category_ids", []), dtype=label_dtype
        )

        keep_ids = augmented.get("bbox_ids", list(range(len(new_boxes))))
        keep_tensor = torch.as_tensor(keep_ids, dtype=torch.long)

        if "iscrowd" in target:
            iscrowd = target["iscrowd"]
            if isinstance(iscrowd, torch.Tensor):
                target_out["iscrowd"] = (
                    iscrowd[keep_tensor]
                    if keep_tensor.numel()
                    else iscrowd.new_zeros((0,), dtype=iscrowd.dtype)
                )
            else:
                target_out["iscrowd"] = torch.as_tensor(
                    np.asarray(iscrowd)[keep_ids], dtype=torch.int64
                )

        if "area" in target:
            target_out["area"] = (new_boxes[:, 2] - new_boxes[:, 0]).clamp(min=0) * (
                new_boxes[:, 3] - new_boxes[:, 1]
            ).clamp(min=0)

    if mask_key is not None:
        masks_aug = augmented.get(mask_key, None)
        mask_list = _extract_masks_from_albumentations(masks_aug, mask_key)

        num_instances = (
            target_out["boxes"].shape[0]
            if ("boxes" in target_out and target_out["boxes"] is not None)
            else len(mask_list)
        )

        if has_bbox_params:
            if len(mask_list) == len(target_out.get("labels", [])):
                # Align by bbox ids when present.
                keep_ids = augmented.get("bbox_ids", list(range(len(mask_list))))
                mask_list = [mask_list[idx] for idx in keep_ids]
            elif len(mask_list) != num_instances:
                if len(mask_list) > num_instances:
                    mask_list = mask_list[:num_instances]
                elif len(mask_list) > 0 and num_instances > len(mask_list):
                    pad = np.zeros_like(mask_list[0])
                    mask_list = mask_list + [pad] * (num_instances - len(mask_list))
                else:
                    mask_list = []

        target_out["masks"] = _albumentations_masks_to_tensor(
            mask_list, image_out.height, image_out.width
        )

    return image_out, target_out
