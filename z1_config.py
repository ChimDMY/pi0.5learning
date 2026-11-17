"""Inference-only equivalent of the frozen Z1 simulation training transforms."""
import dataclasses
from pathlib import Path

# Keep native video initialization before JAX, as in the working source environment.
import lerobot.common.datasets.lerobot_dataset  # noqa: F401
from openpi import transforms
from openpi.models.pi0_config import Pi0Config
from openpi.training import config
from unitree_z1_policy import UnitreeZ1Inputs, UnitreeZ1Outputs
from contract import ASSET_ID, PROMPT


@dataclasses.dataclass(frozen=True)
class Z1GraspData(config.DataConfigFactory):
    def create(self, assets_dirs, model_config):
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=transforms.Group(
                inputs=[UnitreeZ1Inputs(model_type=model_config.model_type)],
                outputs=[UnitreeZ1Outputs()],
            ),
            model_transforms=config.ModelTransformFactory(default_prompt=PROMPT)(model_config),
        )


def make_config(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    return config.TrainConfig(
        name="pi05_z1_grasp_deploy",
        model=Pi0Config(
            pi05=True, action_dim=32, action_horizon=10,
            discrete_state_input=True, paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=Z1GraspData(
            repo_id=ASSET_ID,
            assets=config.AssetsConfig(assets_dir=str(checkpoint / "assets"), asset_id=ASSET_ID),
            base_config=config.DataConfig(prompt_from_task=True),
        ),
        batch_size=1, fsdp_devices=1, num_workers=0, wandb_enabled=False,
    )
