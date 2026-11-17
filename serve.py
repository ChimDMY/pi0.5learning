"""Serve raw Z1 grasp action chunks. This process never connects to the arm."""
import argparse
import datetime
import json
import logging
import os
from pathlib import Path
import time
import uuid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--gpu", required=True, help="Explicit inference host GPU index or UUID")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--log-dir", type=Path, default=Path("runs"))
    args = ap.parse_args()
    # Before importing JAX, Torch or the model.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    logging.basicConfig(level=logging.INFO)
    from check_artifacts import verify, sha256
    identity = verify(args.checkpoint)
    from verify_runtime import verify_runtime
    runtime_identity = verify_runtime()
    from z1_config import make_config
    import importlib.metadata
    import jax
    import numpy as np
    from contract import PROMPT, SCHEMA, validate_request, validate_actions
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("No JAX GPU device; refusing an accidental CPU inference deployment")
    run = args.log_dir / (datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema": SCHEMA, "task": "lift_hold", "prompt": PROMPT,
        "action_shape": [10, 7], "action_units": "absolute joint radians in training convention",
        "joint_order": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"],
        "action_dt_s": 0.04, "suggested_execute_steps": 5, "rtc": False,
        "raw_model_output": True, "hardware_execution_enabled": False,
        **identity,
    }
    policy = create_trained_policy(make_config(args.checkpoint), args.checkpoint.resolve())
    # Warm up once before listening. Synthetic observation is not a task validation.
    warmup = {"schema": SCHEMA, "request_id": "warmup", "prompt": PROMPT, "noise_seed": 812,
              "observation/state": np.zeros(7, dtype=np.float32),
              "observation/image": np.zeros((240, 320, 3), dtype=np.uint8),
              "observation/wrist_image": np.zeros((240, 320, 3), dtype=np.uint8)}
    obs, noise = validate_request(warmup)
    validate_actions(policy.infer(obs, noise=noise)["actions"])
    versions = {name: importlib.metadata.version(name) for name in
                ("jax", "jaxlib", "flax", "orbax-checkpoint", "numpy", "websockets")}
    adapter_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    (run / "loaded_policy.json").write_text(json.dumps({**metadata, "runtime": runtime_identity,
        "package_versions": versions, "adapter_sha256": adapter_hashes,
        "gpu_argument": args.gpu, "jax_devices": [str(d) for d in jax.devices()],
        "warmup_passed": True}, ensure_ascii=False, indent=2), encoding="utf-8")

    class GraspService:
        def infer(self, data):
            obs, noise = validate_request(data)
            start = time.monotonic()
            actions = validate_actions(policy.infer(obs, noise=noise)["actions"])
            elapsed = (time.monotonic() - start) * 1000
            sample_id = uuid.uuid4().hex
            # Full inputs, noise and raw predictions support later deployment diagnosis.
            np.savez_compressed(run / (sample_id + ".npz"),
                overview=obs["observation/image"], wrist=obs["observation/wrist_image"],
                state=obs["observation/state"], noise=noise, actions=actions)
            with (run / "requests.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"request_id": data["request_id"], "sample_id": sample_id,
                    "noise_seed": int(data["noise_seed"]), "unix_time": time.time(),
                    "infer_ms": elapsed}, ensure_ascii=False) + "\n")
            return {"schema": SCHEMA, "request_id": data["request_id"], "actions": actions,
                    "infer_ms": elapsed, "sample_id": sample_id, "action_dt_s": 0.04,
                    "suggested_execute_steps": 5, "raw_model_output": True}

    logging.info("Ready. Raw predictions only; logs: %s", run)
    WebsocketPolicyServer(GraspService(), host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
