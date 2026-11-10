"""Persistent versions and execution experience for agent-authored tools."""
import copy
import hashlib
import json
import re
import threading
import time
import uuid

from .methods import _score
from .experiments import run_experiment

_lock = threading.RLock()


def _path(scene, memory_id):
    if not isinstance(memory_id, str) or not re.fullmatch(r"[0-9a-f]{12}", memory_id):
        raise ValueError("Ungültige Werkzeug-ID im Gedächtnis.")
    return scene.folder.parent / ".scanbim-tools" / (memory_id + ".json")


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def remember_tool(scene, tool):
    """Save a new revision before exposing it to the scene."""
    with _lock:
        memory_id = tool.get("memory_id") or uuid.uuid4().hex[:12]
        path = _path(scene, memory_id)
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "id": memory_id, "versions": [],
        }
        version_id = hashlib.sha256(json.dumps(
            [tool["code"], tool["description"], tool["parameter_help"]], ensure_ascii=False
        ).encode()).hexdigest()[:16]
        if not any(v["id"] == version_id for v in data["versions"]):
            data["versions"].append({
                "id": version_id, "name": tool["name"], "description": tool["description"],
                "parameter_help": tool["parameter_help"], "code": tool["code"],
                "created_at": time.time(), "trials": [],
            })
        data.update(name=tool["name"], updated_at=time.time())
        _write(path, data)
        return {"memory_id": memory_id, "memory_version": version_id}


def list_tool_memory(scene, query="", limit=12):
    limit = int(limit)
    if not 1 <= limit <= 30 or not isinstance(query, str) or len(query) > 500:
        raise ValueError("Suchtext bis 500 Zeichen und limit zwischen 1 und 30 angeben.")
    terms = query.casefold().split()
    found = []
    with _lock:
        for path in (scene.folder.parent / ".scanbim-tools").glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            for version in data["versions"]:
                text = (version["name"] + " " + version["description"]).casefold()
                relevance = sum(term in text for term in terms)
                if terms and not relevance:
                    continue
                trials = version["trials"]
                found.append({
                    "memory_id": data["id"], "version": version["id"], "name": version["name"],
                    "description": version["description"], "parameter_help": version["parameter_help"],
                    "successful_runs": sum(t["ok"] for t in trials),
                    "failed_runs": sum(not t["ok"] for t in trials),
                    "tested_sources": len({t["source_hash"] for t in trials if t["source_hash"]}),
                    "recent_trials": [{**t, "parameters": _compact_parameters(t["parameters"])}
                                      for t in trials[-3:]], "created_at": version["created_at"],
                    "relevance": relevance,
                })
    found.sort(key=lambda item: (item["relevance"], item["created_at"]), reverse=True)
    return {"tools": found[:limit], "total_matches": len(found),
            "note": "Ausführungserfolg und geometrische Qualität getrennt prüfen. "
                    "Frühere Parameter und räumliche Bereiche gehören zum damaligen Scan."}


def recall_tool(scene, memory_id, version=None):
    with _lock:
        path = _path(scene, memory_id)
        if not path.is_file():
            raise ValueError("Werkzeug nicht im Gedächtnis gefunden.")
        data = json.loads(path.read_text(encoding="utf-8"))
        selected = next((v for v in reversed(data["versions"]) if version is None or v["id"] == version), None)
        if selected is None:
            raise ValueError("Werkzeugversion nicht gefunden.")
        existing = next((t for t in scene.state["custom_tools"] if
                         t.get("memory_id") == memory_id and t.get("memory_version") == selected["id"]), None)
        if existing:
            return existing
        if len(scene.state["custom_tools"]) >= 40:
            raise ValueError("Maximal 40 eigene Werkzeuge je Szene.")
        tool = {key: selected[key] for key in ("name", "description", "code", "parameter_help")}
        tool.update(id=uuid.uuid4().hex[:10], revision=1, memory_id=memory_id, memory_version=selected["id"])
        scene.state["custom_tools"].append(tool)
        return tool


def record_trial(scene, tool, experiment):
    score = None
    result = experiment.get("result")
    if (experiment.get("ok") and isinstance(result, dict) and result.get("vertices")
            and isinstance(result.get("faces"), list) and 0 < len(result["faces"]) <= 5000):
        try:
            score = _score(scene, experiment, .03)
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError):
            pass
    with _lock:
        path = _path(scene, tool["memory_id"])
        data = json.loads(path.read_text(encoding="utf-8"))
        version = next(v for v in data["versions"] if v["id"] == tool["memory_version"])
        trial = {"source_hash": scene.state["source_hash"], "scene_id": scene.state["id"],
                 "experiment_id": experiment["id"], "ok": bool(experiment["ok"]),
                 "error": str(experiment.get("error", ""))[:1000],
                 "parameters": copy.deepcopy(experiment.get("data_info", {}).get("parameters", {})),
                 "region": experiment.get("region"), "duration_s": experiment.get("duration_s"),
                 "score_at_30mm": score, "time": time.time()}
        version["trials"].append(trial)
        _write(path, data)
        return trial


def run_tool(scene, tool, region, timeout_s=20, parameters=None, selection_id=None):
    tool.update(remember_tool(scene, tool))
    try:
        extra={"selection_id":selection_id} if selection_id else {}
        result = run_experiment(scene, tool["code"], tool["name"], region, timeout_s, parameters, **extra)
    except (ValueError, RuntimeError) as error:
        record_trial(scene, tool, {"id": None, "ok": False, "error": str(error), "region": region,
                                  "data_info": {"parameters": parameters or {}}})
        raise
    result["tool_memory_trial"] = record_trial(scene, tool, result)
    result["tool_memory"] = {"memory_id": tool["memory_id"], "version": tool["memory_version"]}
    return result


def _compact_parameters(parameters):
    encoded = json.dumps(parameters, ensure_ascii=False)
    return parameters if len(encoded) <= 1200 else {"preview": encoded[:1200], "truncated": True}
