"""Verify the exact mixed200 parameter files and normalization; standard library only."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(checkpoint):
    checkpoint = Path(checkpoint).expanduser().resolve()
    manifest = json.loads((ROOT / "manifests/grasp_checkpoint.json").read_text(encoding="utf-8"))
    expected = manifest["file_sha256"]
    actual = {
        p.relative_to(checkpoint).as_posix()
        for folder in ("params", "assets")
        for p in (checkpoint / folder).rglob("*") if p.is_file()
    }
    if actual != set(expected):
        raise ValueError(f"Checkpoint file set mismatch; missing={set(expected)-actual}, extra={actual-set(expected)}")
    for rel, digest in expected.items():
        if sha256(checkpoint / rel) != digest:
            raise ValueError(f"SHA256 mismatch: {rel}")
    if (checkpoint / "model.safetensors").exists():
        raise ValueError("Unexpected PyTorch weights would override the intended Orbax checkpoint")
    return {"verified": True, "files": len(expected), "checkpoint": str(checkpoint),
            "model_identity": manifest["model_identity"],
            "manifest_sha256": sha256(ROOT / "manifests/grasp_checkpoint.json")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    print(json.dumps(verify(ap.parse_args().checkpoint), ensure_ascii=False, indent=2))
