"""Versioned reconstruction methods, evaluated across independent local scans."""
from __future__ import annotations

import hashlib
import json
import threading
import uuid

from .experiments import run_experiment
from .geometry import validate
from .ifc_tools import parse_mesh

_LOCK = threading.RLock()
_FILENAME = ".scanbim-methods.json"


def _read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _summary(method):
    return {"id": method["id"], "name": method["name"], "scope": method["scope"],
            "versions": [{"id": version["id"], "status": version["status"],
                          "successful_sources": sum(trial["passed"] for trial in version["trials"].values())}
                         for version in method["versions"]]}


def list_methods(scene):
    with _LOCK:
        return {"methods": [_summary(method) for method in _read(scene.folder.parent / _FILENAME).values()]}


def _score(scene, experiment, tolerance):
    result = experiment.get("result")
    if not isinstance(result, dict):
        raise ValueError("Ein bewertbares Verfahren muss vertices und faces liefern.")
    mesh = parse_mesh(result.get("vertices"), result.get("faces"), "auto")
    points, _, _, _ = scene.sample(experiment.get("region"), 12000)
    if not len(points):
        raise ValueError("Die Testregion enthält keine Prüfpunkte.")
    report = validate(points, [{**mesh, "kind": "mesh", "id": "candidate"}], tolerance)
    return {"p95_m": report["p95_m"], "coverage": report["within_tolerance"],
            "surface_support": report["elements"][0]["surface_coverage"]}


def evaluate_method(scene, experiment, baseline, args):
    code = experiment.get("code")
    source = scene.state["source_hash"]
    if not isinstance(code, str) or not code or not source or experiment.get("source_hash") != source:
        raise ValueError("Ein ausgeführtes Verfahren mit eindeutigem Quellscan ist erforderlich.")
    if baseline and (baseline.get("region") != experiment.get("region") or baseline.get("source_hash") != source):
        raise ValueError("Beide Varianten müssen auf demselben Quellscan und derselben Region laufen.")
    tolerance = float(args.get("tolerance", .03))
    candidate_score = _score(scene, experiment, tolerance)
    baseline_score = _score(scene, baseline, tolerance) if baseline else None
    parameters = experiment.get("data_info", {}).get("parameters", {})
    signature = hashlib.sha256(json.dumps([code, parameters], sort_keys=True).encode()).hexdigest()[:16]
    path = scene.folder.parent / _FILENAME
    with _LOCK:
        methods = _read(path)
        method_id = args.get("method_id")
        if method_id and method_id not in methods:
            raise ValueError("Methode nicht in der Bibliothek gefunden.")
        if method_id:
            method = methods[method_id]
        else:
            scope = str(args.get("scope", "")).strip()
            if not scope:
                raise ValueError("Beschreibe den Geltungsbereich des Verfahrens.")
            method_id = uuid.uuid4().hex[:10]
            method = {"id": method_id, "name": str(args.get("name", experiment["name"]))[:120],
                      "scope": scope[:500], "versions": []}
            methods[method_id] = method
        version = next((v for v in method["versions"] if v["id"] == signature), None)
        active = next((v for v in reversed(method["versions"]) if v["status"] == "validated"), None)
        if active and active is not version:
            baseline_used = (baseline or {}).get("library_method", {})
            if baseline_used.get("id") != method_id or baseline_used.get("version") != active["id"]:
                raise ValueError("Eine neue Version muss gegen die aktuelle geprüfte Version antreten.")
        if version is None:
            version = {"id": signature, "status": "candidate", "code": code, "parameters": parameters, "trials": {}}
            method["versions"].append(version)
        passed = candidate_score["coverage"] >= .9 and candidate_score["surface_support"] >= .6
        if baseline_score:
            passed = passed and (
                candidate_score["coverage"] >= baseline_score["coverage"] - .002 and
                candidate_score["surface_support"] >= baseline_score["surface_support"] - .002 and
                candidate_score["p95_m"] <= baseline_score["p95_m"] + .0005 and
                (candidate_score["p95_m"] < baseline_score["p95_m"] - .0005 or
                 candidate_score["coverage"] > baseline_score["coverage"] + .002 or
                 candidate_score["surface_support"] > baseline_score["surface_support"] + .002))
        version["trials"][source] = {"passed": passed, "candidate": candidate_score, "baseline": baseline_score,
                                     "tolerance_m": tolerance}
        trials = version["trials"].values()
        if len(version["trials"]) >= 2 and all(trial["passed"] for trial in trials):
            version["status"] = "validated"
        elif not passed:
            version["status"] = "candidate"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(methods, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
        return {"method": _summary(method), "version": signature, "passed": passed,
                "candidate": candidate_score, "baseline": baseline_score}


def apply_method(scene, args, region):
    with _LOCK:
        method = _read(scene.folder.parent / _FILENAME).get(args["method_id"])
        if method is None:
            raise ValueError("Methode nicht in der Bibliothek gefunden.")
        requested = args.get("version")
        version = next((v for v in reversed(method["versions"]) if
                        (v["id"] == requested if requested else v["status"] == "validated")), None)
        if version is None:
            raise ValueError("Version nicht verfügbar. Kandidaten mit ausdrücklicher Versions-ID testen.")
    record = run_experiment(scene, version["code"], method["name"], region,
                            int(args.get("timeout_s", 20)), version["parameters"])
    record["library_method"] = {"id": method["id"], "version": version["id"], "status": version["status"]}
    return record
