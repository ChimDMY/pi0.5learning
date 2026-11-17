"""Input/output transforms for the single-arm Unitree Z1 dataset."""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image: np.ndarray) -> np.ndarray:
    """Convert a LeRobot image to the HWC uint8 format used by the policy."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        # LeRobot video tensors are commonly returned as float32 in [0, 1].
        image = (255 * image).clip(0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UnitreeZ1Inputs(transforms.DataTransformFn):
    """Map Z1 observations to the common pi0/pi05 model observation format."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # pi05 uses the same image-mask convention as pi0 here.
                "right_wrist_0_rgb": np.False_ if self.model_type != _model.ModelType.PI0_FAST else np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class UnitreeZ1Outputs(transforms.DataTransformFn):
    """Return the six arm targets and one gripper command produced by pi05."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :7], dtype=np.float32)}
