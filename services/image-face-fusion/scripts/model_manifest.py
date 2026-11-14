"""Print required local model/hash paths without downloading or initializing models."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.settings import Settings
from src.worker import OfficialBackend

print(json.dumps(OfficialBackend(Settings.from_env(), initialize=False).manifest(), indent=2))
