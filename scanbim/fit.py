"""Local fit diagnostics and comparable before/after measurements."""
import hashlib
import json
import uuid

import numpy as np

from . import geometry
from .ifc_export import inspect_ifc
from .review import model_signature


def inspect_fit(scene, element_id, region=None, tolerance=.03, baseline_id=None, padding=.25, ifc_geometry=False):
    element=next((e for e in scene.state["elements"] if e["id"]==element_id),None)
    if element is None:
        raise ValueError("Bauteil nicht gefunden.")
    tolerance,padding=float(tolerance),float(padding)
    if not np.isfinite(padding) or not .005<=padding<=2:
        raise ValueError("Randabstand zwischen 0,005 und 2 m angeben.")
    baseline=None
    if baseline_id:
        baseline=next((r for r in scene.state["fit_reports"] if r["id"]==baseline_id),None)
        if baseline is None or baseline["element_id"]!=element_id:
            raise ValueError("Passenden früheren Fit-Bericht für dieses Bauteil angeben.")
        if region is not None and region!=baseline["region"]:
            raise ValueError("Für den Vergleich dieselbe Prüfregion verwenden.")
        region=baseline["region"]
        tolerance=baseline["tolerance_m"]
    evaluated=element
    if ifc_geometry or element.get("cuts") or any("vertices" in opening for opening in element.get("openings",[])):
        _,ifc=inspect_ifc(scene.state,meshes=True)
        shape=ifc["elements"].get(element_id)
        if not shape:
            raise ValueError("Ausgewertete IFC-Geometrie fehlt.")
        evaluated={**element,**shape,"kind":"mesh"}
    if evaluated["kind"]=="mesh":
        vertices=np.asarray(evaluated["vertices"])
    else:
        origin,u,v=(np.asarray(evaluated[k]) for k in ("origin","u","v"))
        vertices=np.array([origin+x*u+y*v for x,y in
                           ((0,0),(evaluated["width"],0),(evaluated["width"],evaluated["height"]),(0,evaluated["height"]))])
    if region is None:
        region={"min":(vertices.min(axis=0)-padding).tolist(),"max":(vertices.max(axis=0)+padding).tolist()}
    points,_,ids,total=scene.sample(region,12000)
    if not len(points):
        raise ValueError("Die Prüfregion enthält keine Scanpunkte. Region erweitern.")
    quality=geometry.validate(points,[evaluated],tolerance)
    distances=np.asarray(quality["distances_m"])
    sample_signature=hashlib.sha256(json.dumps(ids.tolist()).encode()).hexdigest()
    stats={"rms_m":quality["rms_m"],"median_m":float(np.median(distances)),
           "p95_m":quality["p95_m"],"within_tolerance":quality["within_tolerance"],
           "surface_support":quality["elements"][0]["surface_coverage"]}
    if not all(np.isfinite(value) for value in stats.values()):
        raise ValueError("Die Geometrie liefert keine endlichen Fit-Prüfwerte.")
    worst=np.argsort(distances)[-12:][::-1]
    report={"id":uuid.uuid4().hex[:12],"element_id":element_id,
            "ifc_geometry":bool(ifc_geometry),
            "element_signature":model_signature(element),"region":region,"tolerance_m":tolerance,
            "sample_signature":sample_signature,"sample_count":len(points),"region_point_count":total,
            "sampled":len(points)<total,"metrics":stats,
            "covers_element_bounds":bool(np.all(np.asarray(region["min"])<=vertices.min(axis=0)+1e-8)
                                        and np.all(np.asarray(region["max"])>=vertices.max(axis=0)-1e-8)),
            "worst_points":[{"source_index":int(ids[i]),"xyz":points[i].tolist(),
                             "distance_m":float(distances[i])} for i in worst],
            "note":"Lokale Prüfung gegen endliche Flächen. Benachbarte Bauteile und Verdeckungen "
                   "können Residuen beeinflussen. Die Region kann unbeachtete Bereiche außerhalb ausschließen."}
    if baseline:
        if baseline["sample_signature"]!=sample_signature:
            raise ValueError("Die Punktstichprobe hat sich geändert. Neue Baseline aufnehmen.")
        previous=baseline["metrics"]
        delta={key:stats[key]-previous[key] for key in stats}
        report["comparison"]={"baseline_id":baseline_id,"delta":delta,
            "improved":stats["rms_m"]<previous["rms_m"]-1e-6
                       and stats["p95_m"]<=previous["p95_m"]+1e-6
                       and stats["within_tolerance"]>=previous["within_tolerance"]-1e-6
                       and stats["surface_support"]>=previous["surface_support"]-1e-6}
    scene.state["fit_reports"].append(report)
    return report
