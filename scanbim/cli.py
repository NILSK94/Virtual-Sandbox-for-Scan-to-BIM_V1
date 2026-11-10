from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from .cloud import demo_room,write_ply
from .model import SceneStore
from .agents import AgentRunner

def main():
    parser=argparse.ArgumentParser(description="Reproduzierbarer ScanBIM-Referenzdurchlauf")
    parser.add_argument("--output",type=Path,default=Path("data/cli-demo"))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    p,c=demo_room();write_ply(args.output/"reference-room.ply",p,c)
    s=SceneStore(args.output).create(p,c,"Referenzraum",hashlib.sha256(p.tobytes()).hexdigest())
    runner=AgentRunner();runner.start(s)
    runner.threads[s.state["id"]].join(120)
    quality=s.state["quality"]
    report={"agent":s.state["agent"],"elements":len(s.state["elements"]),
            "openings":sum(len(e.get("openings",[])) for e in s.state["elements"]),
            "quality":{k:v for k,v in (quality or {}).items() if k not in ("distances_m","sample_ids")},
            "ifc":str((s.folder/"reconstruction.ifc").resolve())}
    (args.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=True,indent=2))
    if s.state["agent"]["status"]!="completed": raise SystemExit(1)

if __name__=="__main__": main()
