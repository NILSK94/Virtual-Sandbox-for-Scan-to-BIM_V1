from __future__ import annotations
import copy
from concurrent.futures import ThreadPoolExecutor
import shutil
import threading
import time
import uuid
from pathlib import Path
from .streaming import convert_to_cache,probe,SUPPORTED,MAX_FILE_BYTES

class ImportManager:
    """One conversion at a time; jobs do not block HTTP or existing scenes."""
    def __init__(self,store):
        self.store=store;self.jobs={};self.lock=threading.RLock()
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix="cloud-import")

    def new(self,name,unit,size=0):
        if unit not in ("m","cm","mm"):raise ValueError("Ungültige Einheit.")
        if Path(name).suffix.lower() not in SUPPORTED:raise ValueError("Unterstützt: PLY, PCD und XYZ.")
        if size<0 or size>MAX_FILE_BYTES:raise ValueError("Datei überschreitet das konfigurierte Größenlimit.")
        jid=uuid.uuid4().hex[:12];folder=self.store.root/(".import-"+jid);folder.mkdir()
        job={"id":jid,"name":Path(name).name,"unit":unit,"source_bytes":size,"status":"waiting",
             "progress":0.,"phase":"Bereit","created_at":time.time(),"cancel":threading.Event(),"folder":folder}
        self.jobs[jid]=job;return job

    def public(self,jid):
        if jid not in self.jobs:raise ValueError("Import nicht gefunden.")
        with self.lock:return {k:copy.deepcopy(v) for k,v in self.jobs[jid].items() if k not in ("cancel","folder")}

    def update(self,job,progress,phase,**metadata):
        with self.lock:job.update(progress=round(progress,4),phase=phase,**metadata)

    def local(self,path,unit):
        path=Path(str(path).strip().strip('"')).expanduser()
        spec=probe(path)
        job=self.new(path.name,unit,spec["size"]);job["status"]="queued"
        self.executor.submit(self.run,job,path,False)
        return self.public(job["id"])

    def run(self,job,path,temporary_source):
        job["status"]="running"
        try:
            if job["cancel"].is_set():raise InterruptedError("Import abgebrochen.")
            metadata=convert_to_cache(path,job["folder"],job["unit"],
                progress=lambda p,label,**kw:self.update(job,p,label,**kw),cancel=job["cancel"].is_set)
            if job["cancel"].is_set():raise InterruptedError("Import abgebrochen.")
            if temporary_source:Path(path).unlink()
            scene=self.store.adopt_cache(job["folder"],job["id"],job["name"],metadata)
            self.update(job,1.,"Import abgeschlossen",scene_id=scene.state["id"])
            job["status"]="completed"
        except Exception as error:
            job.update(status="cancelled" if isinstance(error,InterruptedError) else "error",error=str(error),phase=str(error))
            self.cleanup(job)

    def cleanup(self,job):
        folder=job["folder"].resolve();root=self.store.root.resolve()
        if folder.parent==root and folder.name==".import-"+job["id"] and folder.exists():
            shutil.rmtree(folder)

    def cancel(self,jid):
        job=self.jobs.get(jid)
        if job is None:raise ValueError("Import nicht gefunden.")
        job["cancel"].set()
        if job["status"]=="waiting":
            job.update(status="cancelled",phase="Import abgebrochen")
            self.cleanup(job)
        return self.public(jid)
