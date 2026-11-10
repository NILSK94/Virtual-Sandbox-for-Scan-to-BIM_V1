"""Building-agnostic acceptance checks, independent of a reconstruction method."""
import hashlib
import json
import numpy as np
from .ifc_export import has_volume, inspect_ifc, is_terrain

def model_signature(elements):
    return hashlib.sha256(json.dumps(elements,sort_keys=True,allow_nan=False).encode()).hexdigest()

def review_model(scene,assessment,unresolved,max_rms_m,min_coverage):
    if not isinstance(unresolved,list) or not all(isinstance(v,str) for v in unresolved):
        raise ValueError("Offene Bereiche als Textliste angeben.")
    if not str(assessment).strip():raise ValueError("Begründung der Modellprüfung fehlt.")
    rms,coverage=float(max_rms_m),float(min_coverage)
    if not np.isfinite(rms) or rms<=0 or not np.isfinite(coverage) or not 0<coverage<=1:
        raise ValueError("Akzeptanzkriterien: RMS > 0 m und Punktabdeckung > 0 bis 1.")
    issues=[];seen=set();elements=scene.state["elements"];quality=scene.state["quality"]
    if not elements:issues.append("Kein Modell vorhanden.")
    for e in elements:
        if scene.state["agent"].get("bim") and not has_volume(e) and not is_terrain(scene.state,e):
            issues.append("Offene Bauteilmessfläche: "+e["name"])
        geometry={k:v for k,v in e.items() if k in ("kind","origin","u","v","normal","width","height","thickness","vertices","faces","cuts","depth_direction")}
        signature=model_signature(geometry)
        if signature in seen:issues.append("Identische doppelte Geometrie: "+e["name"])
        seen.add(signature)
        if not e.get("evidence"):issues.append("Messnachweis fehlt: "+e["name"])
        if e["kind"]=="mesh":
            triangles=np.asarray(e["vertices"])[np.asarray(e["faces"])]
            areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
            if np.any(areas<1e-12):issues.append("Entartete Dreiecke: "+e["name"])
    signature=model_signature(elements)
    if not quality or quality.get("model_signature")!=signature or quality.get("region") is not None:
        issues.append("Aktuelle numerische Prüfung der gesamten Szene fehlt.")
    else:
        if quality["rms_m"]>rms:issues.append("RMS überschreitet das gewählte Akzeptanzkriterium.")
        if quality["within_tolerance"]<coverage:issues.append("Punktabdeckung unterschreitet das gewählte Akzeptanzkriterium.")
        evidence={item["id"]:item for item in quality["elements"]}
        for e in elements:
            support=evidence.get(e["id"],{})
            if e.get("parameter_status",{}).get("geometry")=="assumed":
                anchor=evidence.get(e.get("evidence",{}).get("anchor_id"),{})
                if not anchor.get("supported_points"):
                    issues.append("Annahme ohne gemessenen Anker: "+e["name"])
            elif not support.get("supported_points") and not support.get("surface_coverage"):
                issues.append("Keine Punktstützung für Bauteil: "+e["name"])
            elif (e.get("parameter_status",{}).get("geometry")=="measured" and
                  support.get("assigned_points",0)>=64 and support.get("surface_coverage",0)<.2):
                issues.append("Gemessene Geometrie hat zu wenig Flächenstützung: "+e["name"])
    _,coordination=inspect_ifc(scene.state,check=True)
    scene.state["coordination"]=coordination
    issues.extend(coordination["issues"])
    issues.extend("Schnittvolumen %.6f m³: %s" % (hit["volume_m3"]," / ".join(hit["names"]))
                  for hit in coordination["clashes"])
    report={"accepted":not issues and not unresolved,"assessment":str(assessment)[:4000],
            "unresolved":[v[:2000] for v in unresolved],"issues":issues,"model_signature":signature,
            "criteria":{"max_rms_m":rms,"min_coverage":coverage,"tolerance_m":quality.get("tolerance_m") if quality else None},
            "coordination":coordination,
            "note":"Geometrische Punktstützung und IFC-Koordination mit dokumentierter Unsicherheit."}
    scene.state["review"]=report
    return report
