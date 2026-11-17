"""Check installed OpenPI sources against the pinned inference source manifest."""
import importlib.util
import json
from pathlib import Path
from check_artifacts import sha256


def verify_runtime():
    spec = importlib.util.find_spec("openpi")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("OpenPI not installed; run scripts/setup.sh first")
    root = Path(next(iter(spec.submodule_search_locations))).resolve().parents[1]
    manifest = json.loads((Path(__file__).parent / "manifests/runtime.json").read_text())
    for rel, expected in manifest["file_sha256"].items():
        p = root / rel
        if not p.is_file() or sha256(p) != expected:
            raise RuntimeError(f"OpenPI source mismatch: {rel}; use the pinned isolated deployment checkout")
    return {"openpi_root": str(root), "openpi_commit": manifest["openpi_commit"],
            "verified_source_files": len(manifest["file_sha256"])}


if __name__ == "__main__":
    print(json.dumps(verify_runtime(), indent=2))
