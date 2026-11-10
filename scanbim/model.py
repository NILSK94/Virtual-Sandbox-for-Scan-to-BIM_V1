from __future__ import annotations
import copy
import json
import threading
import time
import uuid
from pathlib import Path
import numpy as np
from .timelapse import record_event

def clean(value):
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    return value

class Scene:
    def __init__(self,folder,points,colors,state=None):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.points,self.colors=points,colors
        self.cloud_metadata=None;self.preview=None
        metadata_path=self.folder/"cloud.json"
        if metadata_path.exists():
            self.cloud_metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
            with np.load(self.folder/"preview.npz",allow_pickle=False) as preview:
                self.preview={k:preview[k] for k in preview.files}
        self.bounds=self.cloud_metadata["bounds"] if self.cloud_metadata else {
            "min":points.min(axis=0).tolist(),"max":points.max(axis=0).tolist()}
        self._sample_cache={}
        self.lock=threading.RLock();self.stop=threading.Event();self.pause=threading.Event()
        self.activity_lock=threading.Lock();self.activity={"seq":0,"status":"idle","tool":None,"message":"Bereit"}
        self.state=state or {"id":self.folder.name,"name":"Neue Szene","source_hash":"","unit":"m",
            "created_at":time.time(),"planes":[],"elements":[],"views":[],"experiments":[],"observations":[],
            "questions":[],"uncertainties":[],"overlays":[],"quality":None,"region":None,"revision":0,
            "agent":{"status":"idle","mode":None,"step":0,"message":"Bereit für die Untersuchung"}}
        for key,value in {"generation":0,"user_guidance":[],"custom_tools":[],"fit_reports":[],"decisions":[],"review":None,"completion":None}.items():
            self.state.setdefault(key,copy.deepcopy(value))
        self.state.setdefault("ifc",{})
        self.state.setdefault("coordination",None)
        self.state.setdefault("model_revision",0)
        self._ifc_preview=None
        self.recording=None
        self._journal=self.folder/"events.jsonl"
        self._events=None if self._journal.is_file() else []
        self.set_activity(status=self.state["agent"]["status"],tool=None,message=self.state["agent"]["message"])

    @property
    def events(self):
        with self.lock:
            if self._events is None:
                events=[]
                with self._journal.open(encoding="utf-8") as journal:
                    for line in journal:
                        try: events.append(json.loads(line))
                        except json.JSONDecodeError: pass
                self._events=events
            return self._events

    @events.setter
    def events(self, events):
        with self.lock:
            self._events=events

    def public(self):
        with self.lock:
            data=copy.deepcopy(self.state)
            data["point_count"]=len(self.points)
            data["bounds"]=self.bounds
            data["storage"]="disk-mapped" if self.cloud_metadata else "memory"
            data["preview_count"]=len(self.preview["points"]) if self.preview else min(len(self.points),200000)
            data["render_origin"]=((np.array(self.bounds["min"])+self.bounds["max"])/2).tolist()
            data["event_count"]=len(self.events)
            data["activity"]=self.activity_snapshot()
            data["color_source"]=(self.cloud_metadata or {}).get("color_source",self.state.get("color_source","unknown"))
            return clean(data)

    def set_activity(self,**values):
        # Separate from the geometry lock: long measurements must remain visible.
        with self.activity_lock:
            self.activity={"seq":self.activity["seq"]+1,"time":time.time(),**values}

    def activity_snapshot(self):
        with self.activity_lock:return copy.deepcopy(self.activity)

    def sample(self,region=None,limit=200000,section=None):
        """Bounded deterministic numeric sample, with original global point IDs.

        Regional/section filters scan indexed blocks, retaining a uniform seeded
        reservoir; no full-source mask or materialized regional cloud is created.
        """
        from .geometry import vector,unit
        limit=int(limit)
        if not 1<=limit<=250000:raise ValueError("Arbeitsstichprobe: 1 bis 250.000 Punkte.")
        low=high=normal=center=None
        if region:
            low,high=vector(region["min"]),vector(region["max"])
            if np.any(low>=high):raise ValueError("Region: min muss kleiner als max sein.")
        if section:
            normal=unit(section["normal"]);center=vector(section["origin"])
            thickness=float(section.get("thickness",.1))
            if thickness<=0:raise ValueError("Schnittdicke muss positiv sein.")
        key=json.dumps([region,limit,section],sort_keys=True)
        if key in self._sample_cache:return self._sample_cache[key]
        if not region and not section:
            if self.preview:
                ids=np.linspace(0,len(self.preview["points"])-1,min(limit,len(self.preview["points"])),dtype=int)
                result=(self.preview["points"][ids],self.preview["colors"][ids],self.preview["source_indices"][ids],len(self.points))
            else:
                ids=np.linspace(0,len(self.points)-1,min(limit,len(self.points)),dtype=np.int64)
                result=(self.points[ids],self.colors[ids],ids,len(self.points))
        else:
            chunks=self.cloud_metadata["chunks"] if self.cloud_metadata else [
                {"start":i,"count":min(131072,len(self.points)-i),"min":self.bounds["min"],"max":self.bounds["max"]}
                for i in range(0,len(self.points),131072)]
            rng=np.random.default_rng(17);reservoir=np.empty(0,dtype=np.int64);keys=np.empty(0);total=0
            for chunk in chunks:
                if region and (np.any(np.array(chunk["max"])<low) or np.any(np.array(chunk["min"])>high)):continue
                if section:
                    midpoint=(np.array(chunk["min"])+chunk["max"])/2
                    radius=np.dot(np.abs(normal),(np.array(chunk["max"])-chunk["min"])/2)
                    if abs((midpoint-center)@normal)>radius+thickness/2:continue
                start=chunk["start"];p=self.points[start:start+chunk["count"]]
                mask=np.ones(len(p),bool)
                if region:mask &= np.all((p>=low)&(p<=high),axis=1)
                if section:mask &= np.abs((p-center)@normal)<=thickness/2
                ids=np.flatnonzero(mask)+start;total+=len(ids)
                reservoir=np.concatenate([reservoir,ids]);keys=np.concatenate([keys,rng.random(len(ids))])
                if len(reservoir)>limit:
                    keep=np.argpartition(keys,limit-1)[:limit];reservoir=reservoir[keep];keys=keys[keep]
            reservoir.sort()
            result=(self.points[reservoir],self.colors[reservoir],reservoir,total)
        if len(self._sample_cache)>=3:self._sample_cache.pop(next(iter(self._sample_cache)))
        self._sample_cache[key]=result
        return result

    def save(self):
        tmp=self.folder/"state.tmp"
        tmp.write_text(json.dumps(clean(self.state),ensure_ascii=False,allow_nan=False),encoding="utf-8")
        tmp.replace(self.folder/"state.json")

    def event(self,tool,args,result,status="ok",duration=0,actor="human"):
        with self.lock:
            self.state["revision"]+=1
            event={"seq":len(self.events)+1,"time":time.time(),"tool":tool,"args":clean(args),
                   "result":clean(result),"status":status,"duration_ms":round(duration*1000),"actor":actor,
                   "snapshot":copy.deepcopy({k:self.state[k] for k in ("elements","region","agent","overlays")})}
            event["snapshot"]["view"]=copy.deepcopy(self.state["views"][-1] if self.state["views"] else None)
            event["snapshot"]["activity"]=self.activity_snapshot()
            event["snapshot"]["ifc"]=copy.deepcopy(self.state["ifc"])
            event["snapshot"]["coordination"]=copy.deepcopy(self.state["coordination"])
            event["snapshot"]["model_revision"]=self.state["model_revision"]
            self.events.append(event)
            with (self.folder/"events.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps(clean(event),ensure_ascii=False,allow_nan=False)+"\n")
            self.save()
            record_event(self,tool)
            return event

    def invalidate_quality(self):
        self.state["quality"]=None;self.state["review"]=None;self.state["completion"]=None
        self.state["coordination"]=None;self._ifc_preview=None
        self.state["model_revision"]+=1

class SceneStore:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.scenes={}
        for path in self.root.glob("*/state.json"):
            try:
                state=json.loads(path.read_text(encoding="utf-8"))
                if (path.parent/"points.npy").exists():
                    points=np.load(path.parent/"points.npy",mmap_mode="r",allow_pickle=False)
                    colors=np.load(path.parent/"colors.npy",mmap_mode="r",allow_pickle=False)
                else:
                    with np.load(path.parent/"source.npz",allow_pickle=False) as p:points,colors=p["points"],p["colors"]
                scene=Scene(path.parent,points,colors,state)
                if scene.state["agent"]["status"] in ("running","paused"):
                    scene.state["agent"].update(status="stopped",message="Server neu gestartet; Zwischenstand wurde wiederhergestellt.")
                self.scenes[state["id"]]=scene
            except (OSError,ValueError,KeyError): continue

    def create(self,points,colors,name,source_hash,unit="m"):
        sid=uuid.uuid4().hex[:12];scene=Scene(self.root/sid,points,colors)
        scene.state.update(name=name,source_hash=source_hash,unit="m",input_unit=unit)
        np.savez_compressed(scene.folder/"source.npz",points=points,colors=colors)
        self.scenes[sid]=scene
        scene.event("import",{"name":name,"input_unit":unit},{"points":len(points),"source_hash":source_hash})
        return scene

    def adopt_cache(self,staging,sid,name,metadata):
        destination=self.root/sid
        if destination.exists():raise ValueError("Import-ID existiert bereits.")
        Path(staging).rename(destination)
        points=np.load(destination/"points.npy",mmap_mode="r",allow_pickle=False)
        colors=np.load(destination/"colors.npy",mmap_mode="r",allow_pickle=False)
        scene=Scene(destination,points,colors)
        scene.state.update(name=name,source_hash=metadata["source_hash"],unit="m",input_unit=metadata["input_unit"],
            source_bytes=metadata["source_bytes"],import_method="streaming-disk-cache")
        scene.event("import",{"name":name,"input_unit":metadata["input_unit"],"method":"streaming"},
            {"points":len(points),"source_hash":metadata["source_hash"],"preview_points":metadata["preview_count"],"cache_bytes":metadata["cache_bytes"]})
        self.scenes[sid]=scene
        return scene

    def get(self,sid):
        if sid not in self.scenes: raise ValueError("Szene nicht gefunden.")
        return self.scenes[sid]
