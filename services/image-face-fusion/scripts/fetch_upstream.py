"""Fetch source only. Does not install upstream or download any model weights."""
import json
import subprocess
import sys
from pathlib import Path

lock = json.loads((Path(__file__).resolve().parents[1] / "upstream.lock.json").read_text())
target = Path(sys.argv[1]).resolve()
subprocess.run(["git", "clone", "--depth", "1", "--branch", lock["version"], lock["repository"], str(target)], check=True)
revision = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
if revision != lock["commit"]:
    raise SystemExit("FaceFusion source revision does not match upstream.lock.json")
print("FaceFusion source verified:", revision)
