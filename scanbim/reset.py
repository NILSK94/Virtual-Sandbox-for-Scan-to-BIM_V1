"""Reset only derived run artifacts in one owned scene; retain source data."""
import copy
import json
import shutil
import uuid

LISTS=("planes","elements","views","experiments","observations","questions","uncertainties","overlays","custom_tools","decisions","fit_reports","user_guidance")

def preview(scene):
    with scene.lock:
        if scene.state["agent"]["status"] in ("running","paused"):
            raise ValueError("Zuerst den Agenten stoppen und auf das Laufende warten.")
        return {"revision":scene.state["revision"],"generation":scene.state["generation"],
                "counts":{k:len(scene.state.get(k,[])) for k in LISTS},"events":len(scene.events),
                "keeps":["Punktwolke und Quellcache","KI-Zugang","andere Szenen"]}

def reset_scene(scene,expected_revision,confirmed=False):
    with scene.lock:
        info=preview(scene)
        if confirmed is not True:raise ValueError("Löschen muss im Dialog bestätigt werden.")
        if expected_revision!=info["revision"]:raise ValueError("Die Szene wurde verändert. Löschdialog bitte erneut öffnen.")
        root=scene.folder.resolve()
        paths=[p for p in root.iterdir() if (p.name.startswith("view-") and p.suffix==".png") or
               p.name.startswith("experiment-") or p.name in ("events.jsonl","reconstruction.ifc","bim.ifc","state.json","agent-results","point-selections")]
        # No recursive operation may follow a link out of this owned directory.
        for p in paths:
            for entry in [p,*(p.rglob("*") if p.is_dir() and not p.is_symlink() else [])]:
                if entry.is_symlink() or not entry.resolve().is_relative_to(root):
                    raise ValueError("Verknüpftes Artefakt kann nicht sicher gelöscht werden.")
        old_state=copy.deepcopy(scene.state);old_events=scene.events[:]
        stage=root/(".reset-"+uuid.uuid4().hex);stage.mkdir();moved=[]
        try:
            for p in paths:p.rename(stage/p.name);moved.append(p.name)
            for key in LISTS:scene.state[key]=[]
            scene.state["working_memory"]=[]
            scene.state["reconstruction_plan"]={"overview":"","items":[]}
            scene.invalidate_quality()
            scene.state["region"]=None
            scene.state["ifc"]={}
            scene.state["generation"]+=1
            scene.state["agent"]={"status":"idle","mode":None,"step":0,"message":"Zurückgesetzt. Bereit für einen neuen Run."}
            scene.events=[];scene.stop.clear();scene.pause.clear();scene._sample_cache.clear()
            scene.set_activity(status="idle",tool=None,message="Bereit für einen neuen Run")
            scene.event("reset",{}, {"generation":scene.state["generation"],"source_preserved":True},actor="human")
        except Exception:
            scene.state=old_state;scene.events=old_events
            for name in moved:
                dest=root/name
                if dest.is_file():dest.unlink()
                (stage/name).rename(dest)
            if not list(stage.iterdir()):stage.rmdir()
            raise
        try:shutil.rmtree(stage)
        except OSError:
            return {"reset":True,"warning":"Lauf zurückgesetzt. Einige alte Artefakte konnten noch nicht entfernt werden.","cleanup_pending":stage.name}
        return {"reset":True,"generation":scene.state["generation"],"source_preserved":True}
