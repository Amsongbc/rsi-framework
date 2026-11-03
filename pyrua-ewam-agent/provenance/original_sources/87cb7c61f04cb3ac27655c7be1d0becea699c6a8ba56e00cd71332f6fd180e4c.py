#!/usr/bin/env python3
"""Generate host-specific robot paths without modifying the downloaded asset snapshot."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "RoboDojo-resources/Assets"
ROOT = HERE / "runtime_assets"


def main():
    assets = ROOT / "Assets"
    assets.mkdir(parents=True, exist_ok=True)
    for child in SOURCE.iterdir():
        target = assets / child.name
        if child.name == "Robots":
            if not target.exists():
                shutil.copytree(child, target)
        elif not target.exists():
            target.symlink_to(child, target_is_directory=child.is_dir())
    changes = []
    for template in (assets / "Robots").glob("**/*_tmp.yml"):
        source = template.read_text()
        rendered = source.replace("${ASSETS_PATH}", str(ROOT)).replace("$ASSETS_PATH", str(ROOT))
        target = template.with_name(template.name.replace("_tmp.yml", ".yml"))
        target.write_text(rendered)
        changes.append({"template": str(template), "target": str(target),
                        "template_sha256": hashlib.sha256(source.encode()).hexdigest(),
                        "generated_sha256": hashlib.sha256(rendered.encode()).hexdigest()})
    (HERE / "runtime_asset_paths.json").write_text(json.dumps({
        "source_snapshot_unchanged": True, "scope": "Official *_tmp.yml ASSETS_PATH substitution only",
        "generated": changes}, indent=2) + "\n")
    print(f"Generated {len(changes)} host-specific robot configurations")


if __name__ == "__main__":
    main()
