"""The frozen grasp policy wire contract. No robot drivers or command execution."""
import numpy as np

PROMPT = "抓起青色方块，抬高至少5厘米并保持2秒。"
ASSET_ID = "z1_sim_20261107"
SCHEMA = "z1_grasp_chunk_v1"


def validate_request(data):
    if data.get("schema") != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA}")
    if data.get("prompt") != PROMPT:
        raise ValueError("Wrong task prompt; this service is only for the frozen grasp task")
    if not isinstance(data.get("request_id"), str) or not data["request_id"]:
        raise ValueError("request_id must be a nonempty string")
    state = np.asarray(data["observation/state"], dtype=np.float32)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError("state must contain 7 finite measured joint angles in radians")
    obs = {"observation/state": state, "prompt": PROMPT}
    for key in ("observation/image", "observation/wrist_image"):
        im = np.asarray(data[key])
        if im.dtype != np.uint8 or im.shape != (240, 320, 3):
            raise ValueError(f"{key}: expected uint8 RGB HWC (240,320,3)")
        obs[key] = im
    seed = data.get("noise_seed")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**63:
        raise ValueError("noise_seed must be an integer in [0, 2**63)")
    noise = np.random.default_rng(int(seed)).standard_normal((10, 32)).astype(np.float32)
    return obs, noise


def validate_actions(actions):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != (10, 7) or not np.isfinite(actions).all():
        raise ValueError("Model returned invalid actions: expected finite (10,7)")
    return actions
