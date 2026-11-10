from __future__ import annotations
import asyncio
import copy
import hashlib
import io
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI,File,UploadFile,Form,Request,HTTPException
from fastapi.responses import FileResponse,StreamingResponse,JSONResponse,Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from .agents import AgentRunner
from .guidance import add_guidance
from . import timelapse
from .cloud import read_cloud,demo_room,MAX_BYTES
from .experiments import docker_status
from .model import SceneStore
from .tools import ToolRegistry,CATALOG
from .imports import ImportManager
from .streaming import MAX_FILE_BYTES
from .settings import load_credentials,settings_status,save_settings,check_connection,available_models
from .model_tests import function_test
from .reset import preview as reset_preview,reset_scene
from .ifc_export import bim_state, inspect_ifc
from .review import model_signature
from .geometry import triangle_distance

ROOT=Path(__file__).resolve().parent.parent

class Action(BaseModel):
    tool:str
    arguments:dict=Field(default_factory=dict)

class StartAgent(BaseModel):
    mode:str="vlm"
    max_steps:int=Field(default=100,ge=1,le=180)
    bim:bool=True

class GuideAgent(BaseModel):
    text:str=Field(min_length=1,max_length=4000)

class LocalImport(BaseModel):
    path:str
    unit:str="m"

class UploadImport(BaseModel):
    name:str
    size:int=0
    unit:str="m"

class ResetRun(BaseModel):
    confirmed:bool=False
    expected_revision:int

def create_app(data_dir=None):
    store=SceneStore(data_dir or os.getenv("SCANBIM_DATA_DIR",ROOT/"data"))
    tools=ToolRegistry();agents=AgentRunner(tools)
    imports=ImportManager(store)
    def shutdown_imports():
        for job in list(imports.jobs.values()):job["cancel"].set()
        imports.executor.shutdown(wait=False,cancel_futures=True)
    @asynccontextmanager
    async def lifespan(app):
        try:yield
        finally:shutdown_imports()
    app=FastAPI(title="ScanBIM Studio",version="0.3.0",lifespan=lifespan)
    app.state.imports=imports
    app.state.store=store;app.state.agents=agents;app.state.tools=tools

    @app.middleware("http")
    async def local_only(request,call_next):
        host=request.url.hostname
        if host not in ("localhost","127.0.0.1","::1"):
            return JSONResponse({"detail":"Lokaler Dienst: ungültiger Host."},status_code=403)
        if request.method not in ("GET","HEAD","OPTIONS"):
            origin=request.headers.get("origin")
            if (origin and urlsplit(origin).netloc != request.headers.get("host")) or request.headers.get("x-scanbim-client")!="studio":
                return JSONResponse({"detail":"Schreibzugriff nur durch den lokalen Studio-Client."},status_code=403)
        response=await call_next(request)
        response.headers["X-Content-Type-Options"]="nosniff"
        response.headers["Referrer-Policy"]="no-referrer"
        if request.url.path in ("/", "/index.html", "/app.js", "/style.css"):
            response.headers["Cache-Control"]="no-store"
        return response

    def scene(sid):
        try: return store.get(sid)
        except ValueError as e: raise HTTPException(404,str(e))

    @app.exception_handler(ValueError)
    async def value_error(request,exc): return JSONResponse({"detail":str(exc)},status_code=400)

    @app.get("/api/config")
    def config():
        try:credentials=load_credentials();settings_error=None
        except ValueError as error:credentials=None;settings_error=str(error)
        return {"version":"0.3.0","max_file_bytes":MAX_FILE_BYTES,"preview_limit":200000,
                "vlm_configured":bool(credentials and credentials.configured),"settings_error":settings_error,
                "model":credentials.model if credentials else None,"sandbox":docker_status(),"tools":CATALOG}

    @app.get("/api/settings")
    def get_settings(provider:str|None=None):
        return JSONResponse(settings_status(provider),headers={"Cache-Control":"no-store"})

    async def settings_input(request):
        # Parse manually so validation responses never echo a submitted key.
        body=await request.body()
        if len(body)>16384:raise ValueError("Eingabe ist zu groß.")
        try:values=json.loads(body)
        except (ValueError,UnicodeError):raise ValueError("Eingaben konnten nicht gelesen werden.") from None
        if not isinstance(values,dict) or not set(values)<= {"api_key","model","provider","kind"}:
            raise ValueError("Nur API-Schlüssel und Modell sind als Eingaben erlaubt.")
        key,model=values.get("api_key",""),values.get("model","")
        if not isinstance(key,str) or not isinstance(model,str):raise ValueError("API-Schlüssel und Modell müssen Text sein.")
        provider,kind=values.get("provider"),values.get("kind","text")
        if provider is not None and not isinstance(provider,str) or not isinstance(kind,str):
            raise ValueError("Anbieter und Test müssen Text sein.")
        return key,model,provider,kind

    @app.post("/api/settings")
    async def update_settings(request:Request):
        key,model,provider,kind=await settings_input(request)
        result=await asyncio.to_thread(save_settings,key,model,provider)
        return JSONResponse(result,headers={"Cache-Control":"no-store"})

    @app.post("/api/settings/test")
    async def test_settings(request:Request):
        key,model,provider,kind=await settings_input(request)
        result=await asyncio.to_thread(check_connection,key,model,provider)
        return JSONResponse(result,headers={"Cache-Control":"no-store"})

    @app.post("/api/settings/models")
    async def settings_models(request:Request):
        key,model,provider,kind=await settings_input(request)
        result=await asyncio.to_thread(available_models,key,provider)
        return JSONResponse(result,headers={"Cache-Control":"no-store"})

    @app.post("/api/settings/function-test")
    async def test_model_function(request:Request):
        key,model,provider,kind=await settings_input(request)
        result=await asyncio.to_thread(function_test,key,model,provider,kind)
        return JSONResponse(result,headers={"Cache-Control":"no-store"})
    @app.get("/api/scenes")
    def scenes():
        return [{"id":s.state["id"],"name":s.state["name"],"created_at":s.state["created_at"],"point_count":len(s.points)}
                for s in sorted(store.scenes.values(),key=lambda s:s.state["created_at"],reverse=True)]

    @app.post("/api/demo")
    def demo():
        p,c=demo_room();digest=hashlib.sha256(p.tobytes()).hexdigest()
        s=store.create(p,c,"Raum 01 · Referenzscan",digest)
        s.state["observations"].append({"text":"Synthetische Referenzszene: 6 × 4,5 × 3 m, eine Tür- und eine Fensterlücke, 3 mm Rauschen. Diese Sollwerte werden dem Messverfahren nicht übergeben.","actor":"fixture"})
        s.save();return s.public()

    @app.post("/api/upload")
    async def upload(file:UploadFile=File(...),unit:str=Form("m")):
        # Compatibility endpoint. The UI uses local file access or raw streamed
        # upload jobs, avoiding multipart spooling for multi-gigabyte files.
        job=imports.new(file.filename or "scan.xyz",unit)
        path=job["folder"]/("incoming"+Path(job["name"]).suffix)
        try:
            size=0
            with path.open("wb") as target:
                while block:=await file.read(1024*1024):
                    size+=len(block)
                    if size>MAX_FILE_BYTES:raise ValueError("Datei überschreitet das konfigurierte Limit.")
                    await asyncio.to_thread(target.write,block)
            await asyncio.to_thread(imports.run,job,path,True)
            if job["status"]!="completed":raise ValueError(job.get("error","Import fehlgeschlagen."))
            return scene(job["scene_id"]).public()
        except Exception:
            imports.cleanup(job);raise

    @app.post("/api/imports/local")
    def import_local(options:LocalImport):return imports.local(options.path,options.unit)

    @app.post("/api/imports/picker")
    def pick_local_file():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root=tk.Tk();root.withdraw();root.attributes("-topmost",True)
            try:
                path=filedialog.askopenfilename(parent=root,title="Punktwolke auf diesem Rechner öffnen",
                    filetypes=[("Punktwolken","*.ply *.pcd *.xyz *.txt"),("Alle Dateien","*.*")])
            finally:root.destroy()
            return {"path":path or None}
        except Exception as error:
            raise ValueError("Dateiauswahl nicht verfügbar. Dateipfad bitte direkt eintragen.") from error

    @app.post("/api/imports")
    def new_upload(options:UploadImport):
        job=imports.new(options.name,options.unit,options.size)
        return imports.public(job["id"])

    @app.put("/api/imports/{jid}/source")
    async def upload_source(jid,request:Request):
        if jid not in imports.jobs:raise ValueError("Import nicht gefunden.")
        job=imports.jobs[jid]
        if job["status"]!="waiting":raise ValueError("Upload wurde bereits begonnen oder abgebrochen.")
        job["status"]="receiving";path=job["folder"]/("incoming"+Path(job["name"]).suffix)
        size=0
        try:
            with path.open("wb") as target:
                async for block in request.stream():
                    if job["cancel"].is_set():raise InterruptedError("Import abgebrochen.")
                    size+=len(block)
                    if size>MAX_FILE_BYTES:raise ValueError("Datei überschreitet das konfigurierte Limit.")
                    await asyncio.to_thread(target.write,block)
                    imports.update(job,0.,"Datei wird übertragen",bytes_received=size)
            if job["source_bytes"] and size!=job["source_bytes"]:raise ValueError("Datei wurde unvollständig übertragen.")
            job.update(status="queued",phase="Import wartet",source_bytes=size)
            imports.executor.submit(imports.run,job,path,True)
            return imports.public(jid)
        except Exception as error:
            job.update(status="cancelled" if job["cancel"].is_set() else "error",error=str(error),phase=str(error))
            imports.cleanup(job)
            raise ValueError(str(error)) from error

    @app.get("/api/imports/{jid}")
    def import_status(jid):return imports.public(jid)

    @app.post("/api/imports/{jid}/cancel")
    def cancel_import(jid):return imports.cancel(jid)

    @app.get("/api/scenes/{sid}")
    def get_scene(sid): return scene(sid).public()

    @app.get("/api/scenes/{sid}/reset-preview")
    def preview_reset(sid):return reset_preview(scene(sid))

    @app.post("/api/scenes/{sid}/reset")
    def reset(sid,options:ResetRun):return reset_scene(scene(sid),options.expected_revision,options.confirmed)

    @app.get("/api/scenes/{sid}/points")
    def get_points(sid):
        s=scene(sid);p,c,ids,total=s.sample(limit=200000)
        return {"points":np.round(p,6).ravel().tolist(),"colors":c.ravel().tolist(),
                "source_indices":ids.tolist(),"total":len(s.points),"displayed":len(ids)}

    @app.post("/api/scenes/{sid}/action")
    def action(sid,action:Action): return tools.call(scene(sid),action.tool,action.arguments,actor="human")

    @app.post("/api/scenes/{sid}/agent/start")
    def start(sid,options:StartAgent): return agents.start(scene(sid),options.mode,options.max_steps,bim=options.bim)

    @app.post("/api/scenes/{sid}/agent/guide")
    def guide(sid,options:GuideAgent):
        return add_guidance(scene(sid),options.text)

    @app.post("/api/scenes/{sid}/agent/{command}")
    def control(sid,command):
        s=scene(sid)
        if s.state["agent"]["status"] not in ("running","paused"):
            raise ValueError("Kein aktiver Agentenlauf.")
        if command=="pause": s.pause.set();s.state["agent"]["status"]="paused"
        elif command=="resume": s.pause.clear();s.state["agent"]["status"]="running"
        elif command=="stop": s.stop.set();s.pause.clear()
        else: raise ValueError("Unbekannter Steuerbefehl.")
        s.event("agent_control",{"command":command},{"status":s.state["agent"]["status"]},actor="human")
        return {"command":command,"status":s.state["agent"]["status"]}

    @app.get("/api/scenes/{sid}/events")
    def events(sid,after:int=0): return scene(sid).events[max(0,after):]

    @app.get("/api/scenes/{sid}/stream")
    async def stream(sid,request:Request):
        s=scene(sid)
        async def changes():
            revision=-1;activity_seq=-1;generation=-1
            while not await request.is_disconnected():
                activity=s.activity_snapshot()
                if s.state["revision"]!=revision or activity["seq"]!=activity_seq or s.state["generation"]!=generation:
                    revision=s.state["revision"]
                    activity_seq=activity["seq"];generation=s.state["generation"]
                    yield f"data: {json.dumps({'revision':revision,'activity':activity,'generation':generation})}\n\n"
                else: yield ": keepalive\n\n"
                await asyncio.sleep(.8)
        return StreamingResponse(changes(),media_type="text/event-stream",headers={"Cache-Control":"no-cache"})

    @app.get("/api/scenes/{sid}/artifacts/{filename}")
    def artifact(sid,filename,revision:int|None=None):
        s=scene(sid)
        if Path(filename).name!=filename or not (filename.startswith("view-") and filename.endswith(".png") or filename in ("reconstruction.ifc","bim.ifc")):
            raise HTTPException(404)
        path=s.folder/filename
        if filename.endswith(".ifc") and revision is not None:
            with s.lock:
                field="exported_bim_revision" if filename=="bim.ifc" else "exported_ifc_revision"
                if revision!=s.state.get(field) or not path.is_file():
                    raise HTTPException(409,"Ein neuer IFC-Stand wurde erzeugt. Download erneut starten.")
                return Response(path.read_bytes(),media_type="application/x-step",
                                headers={"Content-Disposition":f'attachment; filename="{filename}"'})
        if not path.is_file(): raise HTTPException(404)
        return FileResponse(path,filename=filename if filename.endswith(".ifc") else None)

    @app.get("/api/scenes/{sid}/journal")
    def journal(sid): return FileResponse(scene(sid).folder/"events.jsonl",filename=f"scanbim-{sid}-journal.jsonl")

    @app.get("/api/scenes/{sid}/recordings")
    def list_recordings(sid):
        return timelapse.recordings(scene(sid))

    @app.post("/api/scenes/{sid}/recordings/{run}/gif")
    def create_recording_gif(sid, run):
        timelapse.export_gif(timelapse.recording_path(scene(sid), run))
        return {"url": f"/api/scenes/{sid}/recordings/{run}/gif"}

    @app.get("/api/scenes/{sid}/recordings/{run}/gif")
    def download_recording_gif(sid, run):
        path = timelapse.recording_path(scene(sid), run) / "zeitraffer.gif"
        if path.is_symlink() or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="image/gif", filename=f"scanbim-{run}.gif",
                            headers={"Cache-Control":"no-store"})

    @app.get("/api/scenes/{sid}/model")
    def model(sid):
        s=scene(sid)
        return JSONResponse({"schema":"ScanBIM-parametric-0.1","source_hash":s.state["source_hash"],"elements":s.state["elements"]},
                            headers={"Content-Disposition":'attachment; filename="scanbim-model.json"'})

    @app.get("/api/scenes/{sid}/ifc-model")
    def ifc_model(sid,seq:int|None=None,bim:bool=False):
        s=scene(sid)
        with s.lock:
            if seq is not None and not 1<=seq<=len(s.events):
                raise HTTPException(404,"IFC-Verlaufsstand nicht gefunden.")
            snapshot=s.state if seq is None else s.events[seq-1]["snapshot"]
            state={**{key:s.state[key] for key in ("id","name","source_hash")},
                   "elements":snapshot["elements"],"ifc":snapshot.get("ifc",{})}
            excluded=0
            if bim:
                count=len(state["elements"])
                state=bim_state(state)
                excluded=count-len(state["elements"])
            signature=model_signature([state,bim])
            revision=snapshot.get("model_revision")
            cached=s._ifc_preview
            if cached is not None and cached[0]==signature:
                return {**cached[1],"model_revision":revision,"excluded_surfaces":excluded}
            state=copy.deepcopy(state)
        _,report=inspect_ifc(state,meshes=True)
        with s.lock:
            s._ifc_preview=(signature,report)
        return {**report,"model_revision":revision,"excluded_surfaces":excluded}

    @app.get("/api/scenes/{sid}/ifc-evidence/{element_id}")
    def ifc_evidence(sid,element_id,seq:int|None=None):
        shape=ifc_model(sid,seq)["elements"].get(element_id)
        if shape is None:
            raise HTTPException(404,"IFC-Körper nicht gefunden.")
        s=scene(sid)
        vertices=np.asarray(shape["vertices"],float)
        with s.lock:
            tolerance=(s.state.get("quality") or {}).get("tolerance_m",.03) if seq is None else .03
            bounds={"min":(vertices.min(axis=0)-tolerance).tolist(),
                    "max":(vertices.max(axis=0)+tolerance).tolist()}
            points,_,indices,total=s.sample(bounds,12000)
        if len(points):
            distance=triangle_distance(points,vertices,shape["faces"])
            matches=np.flatnonzero(distance<=tolerance)
            selected=matches[np.linspace(0,len(matches)-1,min(len(matches),1600),dtype=int)]
        else:
            selected=np.array([],dtype=int)
        return {"points":points[selected].tolist(),"source_indices":indices[selected].tolist(),
                "tolerance_m":tolerance,"regional_points":total,"sampled":len(points)<total}

    @app.get("/api/examples/roof")
    def roof_code(): return {"code":(ROOT/"examples"/"roof_experiment.py").read_text(encoding="utf-8")}

    app.mount("/",StaticFiles(directory=ROOT/"web",html=True),name="studio")
    return app

app=create_app()
