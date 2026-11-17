"""One inference request; no SDK import and no robot commands."""
import argparse
import json
from pathlib import Path
import time
import uuid
import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.client import connect
from contract import PROMPT, SCHEMA, validate_actions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--observation", type=Path, help="Optional NPZ with overview, wrist, state")
    ap.add_argument("--seed", type=int, default=812)
    ap.add_argument("--timeout", type=float, default=60, help="Finite seconds per receive")
    args = ap.parse_args()
    if args.observation:
        with np.load(args.observation, allow_pickle=False) as data:
            overview, wrist, state = (data[k].copy() for k in ("overview", "wrist", "state"))
    else:
        overview = np.zeros((240, 320, 3), dtype=np.uint8)
        wrist = overview.copy()
        state = np.zeros(7, dtype=np.float32)
    request_id = uuid.uuid4().hex
    with connect(f"ws://{args.host}:{args.port}", compression=None, max_size=4*1024*1024,
                 open_timeout=min(args.timeout, 10), close_timeout=2) as ws:
        metadata = msgpack_numpy.unpackb(ws.recv(timeout=args.timeout))
        if metadata.get("schema") != SCHEMA or metadata.get("prompt") != PROMPT:
            raise ValueError("Connected to the wrong task/server")
        start = time.monotonic()
        ws.send(msgpack_numpy.Packer().pack({"schema": SCHEMA, "request_id": request_id,
            "noise_seed": args.seed, "prompt": PROMPT, "observation/image": overview,
            "observation/wrist_image": wrist, "observation/state": state}))
        response = ws.recv(timeout=args.timeout)
        if isinstance(response, str):
            raise RuntimeError(response)
        result = msgpack_numpy.unpackb(response)
    if result.get("request_id") != request_id:
        raise ValueError("Response/request mismatch")
    actions = validate_actions(result["actions"])
    print(json.dumps({"passed": True, "action_shape": list(actions.shape),
        "round_trip_ms": (time.monotonic()-start)*1000, "infer_ms": result["infer_ms"],
        "synthetic_input": args.observation is None, "robot_moved": False,
        "model_identity": metadata["model_identity"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
