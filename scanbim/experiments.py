"""Bounded, no-network Docker execution. Never executes generated code on host."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid
import numpy as np
from .geometry import region_points

IMAGE="scanbim-experiment:2"

def docker_status():
    if not shutil.which("docker"): return {"available":False,"message":"Docker ist nicht installiert."}
    try:
        info=subprocess.run(["docker","image","inspect",IMAGE],capture_output=True,timeout=6)
        if info.returncode:
            error=info.stderr.decode(errors="replace").lower()
            if "no such image" in error or "no such object" in error:
                return {"available":False,"message":"Experiment-Image fehlt. Setup-Experimente.ps1 ausführen."}
            if "permission denied" in error or "access is denied" in error:
                return {"available":False,"message":"Zugriff auf Docker verweigert. ScanBIM im selben Windows-Konto wie Docker Desktop starten."}
            return {"available":False,"retryable":True,"message":"Docker-Dienst ist nicht erreichbar."}
        details=json.loads(info.stdout)
        return {"available":True,"message":"Isolierter Versuchsraum bereit","image":IMAGE,"image_id":details[0]["Id"]}
    except (OSError,subprocess.TimeoutExpired):
        return {"available":False,"retryable":True,"message":"Docker-Dienst ist nicht erreichbar."}
    except (ValueError, IndexError, KeyError, TypeError):
        return {"available":False,"message":"Docker hat ungültige Image-Informationen geliefert."}

def validate_overlays(overlays):
    if not isinstance(overlays,list) or len(overlays)>30: raise ValueError("Maximal 30 Overlays.")
    cleaned=[]
    for item in overlays:
        if item.get("kind") not in ("points","lines","mesh"): raise ValueError("Overlay: points, lines oder mesh erforderlich.")
        vertices=np.asarray(item.get("vertices",[]),float)
        if vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)>15000 or not np.isfinite(vertices).all():
            raise ValueError("Overlay benötigt höchstens 15.000 endliche 3D-Punkte.")
        obj={"kind":item["kind"],"vertices":vertices.tolist(),"label":str(item.get("label","Versuch"))[:160]}
        if "colors" in item:
            rgb=np.asarray(item["colors"],float)
            if rgb.shape!=vertices.shape or not np.isfinite(rgb).all() or np.any((rgb<0)|(rgb>255)):
                raise ValueError("Overlay-Farben: ein RGB-Tripel (0–255) je Punkt erforderlich.")
            obj["colors"]=rgb.tolist()
        if item["kind"]=="mesh":
            faces=np.asarray(item.get("faces",[]),int)
            if faces.ndim!=2 or faces.shape[1]!=3 or len(faces)>3000 or faces.min()<0 or faces.max()>=len(vertices):
                raise ValueError("Ungültige Overlay-Dreiecke.")
            obj["faces"]=faces.tolist()
        cleaned.append(obj)
    return cleaned

def run_experiment(scene,code,name="Eigenes Verfahren",region=None,timeout_s=20,parameters=None,selection_id=None):
    if not isinstance(code,str) or len(code.encode())>64000: raise ValueError("Programm maximal 64 KB.")
    timeout_s=int(timeout_s)
    if not 1<=timeout_s<=30: raise ValueError("Zeitlimit: 1 bis 30 Sekunden.")
    parameters={} if parameters is None else parameters
    if not isinstance(parameters,dict) or len(json.dumps(parameters,allow_nan=False))>32000:raise ValueError("Werkzeugparameter: JSON-Objekt bis 32 KB.")
    status=docker_status()
    # Desktop may be booting or waking from Resource Saver. Wait before starting
    # the container so its computation timeout remains reserved for the program.
    deadline=time.monotonic()+45
    while status.get("retryable") and time.monotonic()<deadline:
        if scene.stop.wait(2): raise ValueError("Experiment abgebrochen.")
        status=docker_status()
    if not status["available"]: raise ValueError(status["message"]+" Es wird kein Code auf dem Host ausgeführt.")
    eid=uuid.uuid4().hex[:10];folder=scene.folder/f"experiment-{eid}";folder.mkdir()
    if selection_id:
        from .segmentation import sample_selection
        points,colors,source_ids,region_total,selection_sampled=sample_selection(scene,region,200000,selection_id)
    else:
        points,colors,source_ids,region_total=scene.sample(region,200000)
        selection_sampled=len(points)<region_total
    if not len(points): raise ValueError("Versuchsregion enthält keine Punkte.")
    np.savez(folder/"cloud.npz",points=points,colors=colors,source_indices=source_ids)
    data_info={"source_point_count":len(scene.points),"region_point_count":region_total,
        "color_source":(scene.cloud_metadata or {}).get("color_source",scene.state.get("color_source","unknown")),"parameters":parameters,
        "sample_count":len(points),"sampled":selection_sampled,"selection_id":selection_id,"region":region,"unit":"m"}
    (folder/"data-info.json").write_text(json.dumps(data_info),encoding="utf-8")
    # Compatible with the installed worker; source is still only executed inside Docker.
    (folder/"program.py").write_text("parameters = data_info.get('parameters', {})\n"+code,encoding="utf-8")
    container=f"scanbim-{eid}"
    cmd=["docker","run","--rm","--name",container,"--network","none","--read-only",
         "--cap-drop","ALL","--security-opt","no-new-privileges","--user","65534:65534",
         "--cpus","1","--memory","512m","--memory-swap","512m","--pids-limit","64",
         "--log-driver","none","--tmpfs","/tmp:rw,noexec,nosuid,size=32m",
         "--mount",f"type=bind,source={folder.resolve()},target=/input,readonly"]
    if scene.cloud_metadata and not selection_id:
        for filename in ("points.npy","colors.npy"):
            cmd.extend(["--mount",f"type=bind,source={(scene.folder/filename).resolve()},target=/dataset/{filename},readonly"])
    cmd.append(status["image_id"])
    start=time.monotonic();output=bytearray();errors=bytearray();overflow=threading.Event()
    record={"id":eid,"name":str(name)[:120],"code":code,"region":region,"source_hash":scene.state["source_hash"],
            "point_count":len(points),"data_info":data_info,"timeout_s":timeout_s,"image":IMAGE,"image_id":status["image_id"],
            "limits":{"memory_mb":512,"cpus":1,"network":False}}
    proc=None
    def drain(pipe,dest):
        while True:
            chunk=pipe.read(8192)
            if not chunk: break
            if len(dest)+len(chunk)>2_000_000:
                overflow.set();break
            dest.extend(chunk)
    try:
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        readers=[threading.Thread(target=drain,args=(proc.stdout,output),daemon=True),
                 threading.Thread(target=drain,args=(proc.stderr,errors),daemon=True)]
        for thread in readers: thread.start()
        while proc.poll() is None:
            if overflow.is_set(): raise ValueError("Ausgabelimit von 2 MB überschritten.")
            if time.monotonic()-start>timeout_s: raise ValueError("Experiment wegen Zeitlimit beendet.")
            if scene.stop.is_set(): raise ValueError("Experiment abgebrochen.")
            time.sleep(.05)
        for thread in readers: thread.join(timeout=1)
        if overflow.is_set(): raise ValueError("Ausgabelimit von 2 MB überschritten.")
        try: response=json.loads(output.decode())
        except (ValueError,UnicodeDecodeError): raise ValueError("Ungültige Versuchsausgabe: "+errors.decode(errors="replace")[-1000:])
        record.update(response)
        if response.get("ok"):
            result=response["result"]
            overlays=validate_overlays(result.get("overlays",[])) if isinstance(result,dict) else []
            if (isinstance(result,dict) and result.get("vertices") and result.get("faces") and
                    not any(item["kind"]=="mesh" for item in overlays) and len(overlays)<30):
                faces=result["faces"]
                subset=np.linspace(0,len(faces)-1,min(3000,len(faces)),dtype=int)
                try:
                    overlays+=validate_overlays([{"kind":"mesh","label":record["name"],
                        "vertices":result["vertices"],"faces":[faces[i] for i in subset]}])
                except (TypeError,ValueError,IndexError):
                    pass
            for item in overlays: item["experiment_id"]=eid
            scene.state["overlays"]=[o for o in scene.state["overlays"] if o.get("experiment_id")!=eid]+overlays
    except Exception as error:
        record.update(ok=False,error=str(error))
    finally:
        if proc and proc.poll() is None: proc.kill()
        try: subprocess.run(["docker","rm","-f",container],capture_output=True,timeout=5)
        except (OSError,subprocess.TimeoutExpired): pass
    record["duration_s"]=round(time.monotonic()-start,3)
    (folder/"result.json").write_text(json.dumps(record,ensure_ascii=False),encoding="utf-8")
    scene.state["experiments"].append(record)
    return record
