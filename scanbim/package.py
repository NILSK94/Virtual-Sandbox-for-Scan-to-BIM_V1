"""Explicit source allowlist. Local credentials and runtime data never ship."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import zipfile

ROOT_FILES={"README.md","pyproject.toml","requirements.txt","requirements-dev.txt","requirements-lock.txt",
    "Start.ps1","Start.bat","Setup-Experimente.ps1","API-Einstellungen.ps1","API-Einstellungen.bat",
    "settings.example.toml",".gitignore"}
ROOT_DIRS={"scanbim","web","docs","examples","sandbox","tests"}
PRIVATE_DIRS={"data",".venv","__pycache__",".pytest_cache","private","secrets",".git"}
KEY_PATTERN=re.compile(rb"sk-[A-Za-z0-9_-]{20,}")

def permitted(path):
    if any(part in PRIVATE_DIRS for part in path.parts):return False
    if path.name in {"settings.toml", "jev.toml"} or path.name.startswith(".env") or ".local." in path.name:return False
    return (len(path.parts)==1 and path.name in ROOT_FILES) or (len(path.parts)>1 and path.parts[0] in ROOT_DIRS)

def package_project(root,output):
    root=Path(root).resolve();output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    pending=output.with_suffix(output.suffix+".partial")
    count=0
    try:
        with zipfile.ZipFile(pending,"w",zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(root.rglob("*")):
                rel=file.relative_to(root)
                if not file.is_file() or not permitted(rel):continue
                if file.is_symlink():raise ValueError("Verknüpfung nicht im Projektpaket erlaubt: "+str(rel))
                if file.stat().st_size>64*1024*1024:raise ValueError("Große Datendatei bitte separat halten: "+str(rel))
                payload=file.read_bytes()
                if KEY_PATTERN.search(payload):raise ValueError("Möglicher API-Schlüssel; Paket nicht erstellt. Datei: "+str(rel))
                archive.writestr((Path("ScanBIM-Studio")/rel).as_posix(),payload);count+=1
        pending.replace(output)
    except Exception:
        pending.unlink(missing_ok=True);raise
    return {"files":count,"bytes":output.stat().st_size}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Quellpaket ohne lokale Zugangsdaten erstellen")
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    print(package_project(Path(__file__).resolve().parent.parent,args.output))
