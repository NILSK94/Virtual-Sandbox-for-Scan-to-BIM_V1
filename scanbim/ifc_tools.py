"""Geometry and metadata contracts for the autonomous IFC authoring tools."""
from __future__ import annotations

import copy
import math
import uuid

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .ifc_export import ASSETS, PART_CLASSES, declaration, inspect_ifc, mesh_topology, model_schema, product_class


def parse_mesh(vertices, faces, role):
    if role not in ("auto", "surface", "solid"):
        raise ValueError("Geometrierolle muss auto, surface oder solid sein.")
    vertices, indices = np.asarray(vertices, dtype=float), np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not 3 <= len(vertices) <= 12000 or not np.isfinite(vertices).all():
        raise ValueError("Ein Netz braucht 3 bis 12.000 endliche 3D-Punkte.")
    if (indices.ndim != 2 or indices.shape[1] != 3 or indices.dtype.kind not in "iu"
            or not 1 <= len(indices) <= 20000 or indices.min() < 0 or indices.max() >= len(vertices)):
        raise ValueError("Ein Netz braucht 1 bis 20.000 gültige Dreiecke mit ganzzahligen Punktindizes.")
    triangles = vertices[indices]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    if np.any(areas < 1e-12):
        raise ValueError("Das Netz enthält entartete Dreiecke.")
    closed, _ = mesh_topology(vertices, indices)
    if role == "solid" and not closed:
        raise ValueError("Ein Volumennetz braucht eine geschlossene und konsistent orientierte Hülle.")
    return {"vertices": vertices.tolist(), "faces": indices.tolist(), "geometry_role": role}


def configure_model(scene, config):
    updated = copy.deepcopy(scene.state.get("ifc", {}))
    if set(config) - {"asset", "schema", "site_name", "asset_name", "asset_type", "levels", "parts", "map"}:
        raise ValueError("Unbekannte IFC-Projekteinstellung.")
    updated.update(config)
    asset = updated.get("asset", "building")
    if asset not in ASSETS:
        raise ValueError("Unbekannter Bauwerkstyp.")
    schema = updated.get("schema", "IFC4")
    if "schema" not in config and asset not in ("building", "site"):
        schema = "IFC4X3_ADD2"
    if schema not in ("IFC4", "IFC4X3_ADD2") or (ASSETS[asset] and not declaration(schema, ASSETS[asset])):
        raise ValueError("Die IFC-Version unterstützt den gewählten Bauwerkstyp nicht.")
    updated["schema"] = schema
    if updated.get("asset_type") and (not ASSETS[asset] or
            updated["asset_type"] not in predefined_values(schema, ASSETS[asset])):
        raise ValueError("Vordefinierter Typ passt nicht zum Bauwerk.")
    parts=updated.get("parts",[])
    if not isinstance(parts,list) or len(parts)>200 or (parts and asset not in PART_CLASSES):
        raise ValueError("Anlagenteile sind für Infrastruktur in IFC 4.3 vorgesehen.")
    part_ids=set()
    for part in parts:
        if (not isinstance(part,dict) or not isinstance(part.get("id"),str) or not part["id"] or
                part["id"] in part_ids or (part.get("parent") and part["parent"] not in part_ids)):
            raise ValueError("Anlagenteile brauchen eindeutige IDs. Übergeordnete Teile zuerst angeben.")
        part["name"]=str(part.get("name") or part["id"])[:120]
        part.setdefault("type","NOTDEFINED")
        part.setdefault("usage","REGION")
        if part["type"] not in predefined_values(schema,PART_CLASSES[asset]) or part["usage"] not in (
                "LATERAL","LONGITUDINAL","REGION","VERTICAL","NOTDEFINED"):
            raise ValueError("Ungültiger Typ oder Nutzungsbereich eines Anlagenteils.")
        part_ids.add(part["id"])
    for key in ("site_name", "asset_name"):
        if key in updated:
            updated[key] = str(updated[key]).strip()[:120]
    levels = updated.get("levels", [])
    if not isinstance(levels, list) or len(levels) > 100:
        raise ValueError("Geschosse als Liste mit bis zu 100 Einträgen angeben.")
    if "levels" in updated and not levels:
        raise ValueError("Mindestens ein Geschoss angeben oder die Standardebene verwenden.")
    seen = set()
    for level in levels:
        if not isinstance(level, dict) or not isinstance(level.get("id"), str) or not level["id"] or level["id"] in seen:
            raise ValueError("Jedes Geschoss braucht eine eindeutige ID.")
        level["name"] = str(level["name"]).strip()[:120]
        level["elevation"] = float(level["elevation"])
        if not math.isfinite(level["elevation"]):
            raise ValueError("Geschosshöhe muss endlich sein.")
        seen.add(level["id"])
    if updated.get("map") is not None:
        georef = updated["map"]
        if not isinstance(georef, dict) or not str(georef.get("crs", "")).strip():
            raise ValueError("Georeferenzierung braucht ein bekanntes Koordinatensystem.")
        updated["map"] = {"crs": str(georef["crs"])[:120]}
        for key, default in (("eastings", None), ("northings", None), ("height", None),
                             ("x_axis_abscissa", 1), ("x_axis_ordinate", 0), ("scale", 1)):
            value = georef.get(key, default)
            if value is None or not math.isfinite(float(value)):
                raise ValueError("Georeferenzierung: fehlender oder ungültiger Wert für " + key)
            updated["map"][key] = float(value)
        if updated["map"]["scale"] <= 0 or not math.hypot(updated["map"]["x_axis_abscissa"], updated["map"]["x_axis_ordinate"]):
            raise ValueError("Georeferenzierung braucht einen positiven Maßstab und eine gültige Achse.")
    for element in scene.state["elements"]:
        meta = element.get("ifc", {})
        if meta.get("class") and not declaration(schema, meta["class"]):
            raise ValueError("IFC-Klasse passt nicht zur neuen Version: " + meta["class"])
        if meta.get("level") and (asset != "building" or meta["level"] not in (seen or {"default"})):
            raise ValueError("Ein Bauteil verweist auf ein Geschoss außerhalb der neuen Struktur.")
        if meta.get("part") and meta["part"] not in part_ids:
            raise ValueError("Ein Bauteil verweist auf einen Anlagenteil außerhalb der neuen Struktur.")
    scene.state["ifc"] = updated
    scene.invalidate_quality()
    return updated


def predefined_values(schema, ifc_class):
    entity=declaration(schema,ifc_class)
    if entity:
        for attribute in entity.all_attributes():
            if attribute.name()=="PredefinedType":
                enum=attribute.type_of_attribute().declared_type().as_enumeration_type()
                return set(enum.enumeration_items())-{"USERDEFINED"}
    return set()


def set_metadata(scene, element, changes):
    if set(changes) - {"class", "predefined_type", "type_name", "material", "layers", "level", "part", "fills", "bounds"}:
        raise ValueError("Unbekannte IFC-Bauteileigenschaft.")
    meta = copy.deepcopy(element.get("ifc", {}))
    meta.update(changes)
    meta = {key: value for key, value in meta.items() if value is not None}
    schema = model_schema(scene.state)
    if "class" in meta and not isinstance(meta["class"],str):
        raise ValueError("IFC-Klasse muss als Text angegeben werden.")
    candidate = {**element, "ifc": meta}
    ifc_class = product_class(candidate, schema)
    entity = declaration(schema, ifc_class)
    if not entity or entity.is_abstract():
        raise ValueError("Unbekannte oder abstrakte IFC-Klasse: " + str(ifc_class))
    parent = entity
    while parent and parent.name() != "IfcElement":
        parent = parent.supertype()
    if not parent and ifc_class != "IfcSpace":
        raise ValueError("Ein Modellbauteil braucht eine IfcElement-Klasse oder IfcSpace.")
    if ifc_class == "IfcSpace" and (element.get("cuts") or element.get("contacts") or element.get("openings") or
            any(element["id"] in e.get("cuts",[]) or element["id"] in e.get("contacts",[]) or
                element["id"] in e.get("ifc",{}).get("bounds",[]) for e in scene.state["elements"])):
        raise ValueError("Vor der Umklassifizierung in einen Raum die Bauteilbeziehungen lösen.")
    for key in ("predefined_type", "type_name", "material", "level", "part", "fills"):
        if key in meta and (not isinstance(meta[key], str) or not 1 <= len(meta[key].strip()) <= 120):
            raise ValueError("Ungültiger Textwert für " + key)
    if "predefined_type" in meta:
        if meta["predefined_type"] not in predefined_values(schema,ifc_class):
            raise ValueError("Vordefinierter Typ passt nicht zur IFC-Klasse.")
    if "type_name" in meta and not declaration(schema, ifc_class + "Type"):
        raise ValueError("Diese IFC-Klasse hat keinen zugehörigen Bauteiltyp.")
    if "layers" in meta:
        layers = meta["layers"]
        if ("material" in meta or ifc_class not in ("IfcWall","IfcSlab") or
                element["kind"] not in ("wall","slab") or not isinstance(layers,list) or not 1<=len(layers)<=30):
            raise ValueError("Schichten brauchen einen planaren Wand- oder Plattenkörper ohne parallele Materialzuordnung.")
        for layer in layers:
            if not isinstance(layer,dict) or not str(layer.get("name","")).strip():
                raise ValueError("Jede Schicht braucht einen Materialnamen.")
            thickness=float(layer["thickness_m"])
            if not math.isfinite(thickness) or thickness<=0:
                raise ValueError("Schichtdicke muss positiv und endlich sein.")
            layer["name"]=str(layer["name"])[:120]
            layer["thickness_m"]=thickness
        if not math.isclose(sum(layer["thickness_m"] for layer in layers),element["thickness"],abs_tol=1e-5):
            raise ValueError("Schichtsumme muss der Bauteildicke entsprechen.")
    elements = {e["id"]: e for e in scene.state["elements"]}
    if "level" in meta:
        config = scene.state.get("ifc", {})
        if config.get("asset", "building") != "building" or meta["level"] not in {
                level["id"] for level in config.get("levels", [{"id": "default"}])}:
            raise ValueError("Geschoss-ID gehört nicht zur Projektstruktur.")
    if "part" in meta and meta["part"] not in {part["id"] for part in scene.state.get("ifc",{}).get("parts",[])}:
        raise ValueError("Anlagenteil-ID gehört nicht zur Projektstruktur.")
    if "fills" in meta:
        host, _, index = meta["fills"].partition(":")
        if (ifc_class not in ("IfcDoor", "IfcWindow") or not index.isdecimal() or not 1 <= int(index) <=
                len(elements.get(host, {}).get("openings", [])) or
                any(e["id"] != element["id"] and e.get("ifc", {}).get("fills") == meta["fills"] for e in elements.values())):
            raise ValueError("Füllung braucht eine freie, vorhandene Öffnung und die Klasse IfcDoor oder IfcWindow.")
    if "bounds" in meta:
        if (ifc_class != "IfcSpace" or not isinstance(meta["bounds"], list) or
                not all(isinstance(k, str) and k in elements and k != element["id"] and
                        product_class(elements[k], schema) != "IfcSpace" for k in meta["bounds"])):
            raise ValueError("Raumgrenzen brauchen vorhandene Bauteile.")
        meta["bounds"] = list(dict.fromkeys(meta["bounds"]))
    element["ifc"] = meta
    scene.invalidate_quality()
    return element


def terrain_mesh(points, cell, max_slope):
    """Sample the lower return band on a grid and triangulate neighbouring samples."""
    if not .02 <= cell <= 100 or not .05 <= max_slope <= 5:
        raise ValueError("Geländeraster: 0,02–100 m und maximale Steigung: 0,05–5 m/m.")
    if len(points) < 3:
        raise ValueError("Die Geländeregion enthält zu wenige Punkte.")
    cells = np.floor((points[:, :2] - points[:, :2].min(axis=0)) / cell).astype(np.int64)
    order = np.lexsort((points[:, 2], cells[:, 1], cells[:, 0]))
    indices = cells[order]
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(indices, axis=0), axis=1)) + 1]
    ends = np.r_[starts[1:], len(points)]
    keys = indices[starts]
    if len(keys) > 12000:
        raise ValueError("Gelände: Region verkleinern oder Raster vergrößern.")
    vertices = points[order[starts + (ends - starts - 1) // 5]]
    lookup = {tuple(key): i for i, key in enumerate(keys)}
    faces = []
    for i, (x, y) in enumerate(keys):
        neighbours = [i, lookup.get((x + 1, y)), lookup.get((x + 1, y + 1)), lookup.get((x, y + 1))]
        present = [j for j in neighbours if j is not None]
        if len(present) == 3:
            candidates = [present]
        elif len(present) == 4:
            candidates = [[present[0], present[1], present[2]], [present[0], present[2], present[3]]]
        else:
            continue
        for candidate in candidates:
            triangle = vertices[candidate]
            edges = triangle - np.roll(triangle, 1, axis=0)
            if np.all(np.abs(edges[:, 2]) <= max_slope * np.linalg.norm(edges[:, :2], axis=1) + .02):
                normal_z = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])[2]
                if abs(normal_z) > 1e-12:
                    faces.append(candidate if normal_z > 0 else candidate[::-1])
    if not faces:
        raise ValueError("Keine zusammenhängende Geländeoberfläche in der gewählten Region.")
    triangles = np.asarray(faces)
    a, b = triangles.ravel(), triangles[:, [1, 2, 0]].ravel()
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(vertices), len(vertices))).tocsr()
    _, labels = connected_components(graph, directed=False)
    component = labels[triangles[:, 0]]
    counts = np.bincount(component, minlength=int(labels.max()) + 1)
    # Of the substantial patches, prefer the one connected to the lowest points.
    substantial = counts >= max(1, .05 * counts.max())
    candidates = np.unique(triangles[substantial[component]])
    seed = candidates[np.argmin(vertices[candidates, 2])]
    faces = triangles[component == labels[seed]]
    used, compact = np.unique(faces, return_inverse=True)
    return vertices[used], compact.reshape(-1, 3)


def add_terrain(scene, args, region):
    cell = float(args.get("cell", .25))
    points, _, ids, total = scene.sample(region)
    vertices, faces = terrain_mesh(points, cell, float(args.get("max_slope", .8)))
    mesh = parse_mesh(vertices, faces, "surface")
    element = {**mesh, "id": uuid.uuid4().hex[:10], "kind": "mesh", "semantic": "terrain",
               "name": str(args.get("name", "Gelände"))[:120], "locked": False,
               "parameter_status": {"geometry": "derived"},
               "uncertainty": "Geländehypothese aus unteren Messpunkten. Raster und Steigung begrenzen die Interpolation.",
               "evidence": {"source_hash": scene.state["source_hash"], "method": "Unteres Punktraster / lokale Dreiecke",
                            "region": region, "cell_m": cell, "support": len(vertices),
                            "sampling": {"analyzed": len(ids), "region_total": total, "sampled": len(ids) < total}}}
    scene.state["elements"].append(element)
    scene.invalidate_quality()
    return element


def change_join(scene, element, target, remove=False, mode="subtract"):
    if mode not in ("subtract","contact"):
        raise ValueError("Anschlussmodus muss subtract oder contact sein.")
    field = "cuts" if mode=="subtract" else "contacts"
    cuts = element.get(field, [])
    key = target["id"]
    if remove:
        element[field] = [other for other in cuts if other != key]
        scene.invalidate_quality()
        return element
    if key == element["id"] or key in element.get("cuts",[]) or key in element.get("contacts",[]):
        raise ValueError("Für den Verschnitt sind zwei verschiedene, noch nicht so verbundene Bauteile erforderlich.")
    if "IfcSpace" in (product_class(element,model_schema(scene.state)),product_class(target,model_schema(scene.state))):
        raise ValueError("Räume werden über Raumgrenzen mit Bauteilen verbunden.")
    elements = {e["id"]: e for e in scene.state["elements"]}
    preview = {**scene.state, "elements": list(elements.values())}
    host = preview["elements"].index(element)
    if mode=="contact":
        preview["elements"][host] = {**element, "contacts": [*cuts, key]}
        _, report=inspect_ifc(preview,check=True)
        if (any(set(hit["elements"])=={key,element["id"]} for hit in report["clashes"]) or
                any(issue=="Anschluss ohne Kontakt: "+element["name"]+" / "+target["name"] for issue in report["issues"]) or
                any(part not in report["elements"] for part in (key,element["id"]))):
            raise ValueError("Ein Kontaktanschluss braucht eine Fuge von höchstens 5 mm ohne Schnittvolumen.")
        element["contacts"] = [*cuts,key]
        scene.invalidate_quality()
        return element
    pending, seen = [key], set()
    while pending:
        current = pending.pop()
        if current == element["id"]:
            raise ValueError("Verschnitte brauchen eine eindeutige Reihenfolge ohne Kreis.")
        if current not in seen:
            seen.add(current)
            pending.extend(elements[current].get("cuts", []))
    for part in (element, target):
        if part["kind"] == "mesh" and (part.get("geometry_role") == "surface" or
                not mesh_topology(np.asarray(part["vertices"]), np.asarray(part["faces"]))[0]):
            raise ValueError("Für einen volumetrischen Verschnitt sind zwei geschlossene Körper erforderlich.")
    _, before = inspect_ifc(preview)
    preview["elements"][host] = {**element, "cuts": [*cuts, key]}
    _, after = inspect_ifc(preview)
    old = before["elements"].get(element["id"], {}).get("net_volume_m3")
    new = after["elements"].get(element["id"], {}).get("net_volume_m3")
    if old is None or new is None or new <= 1e-8 or old - new <= 1e-8:
        raise ValueError("Der Schnitt hat keinen messbaren Überstand entfernt oder ergäbe einen ungültigen Körper.")
    element["cuts"] = [*cuts, key]
    scene.invalidate_quality()
    return element


def add_opening(scene, element, experiment, args):
    if product_class(element,model_schema(scene.state))=="IfcSpace":
        raise ValueError("Ein Ausschnitt braucht ein physisches Bauteil.")
    result = experiment.get("result")
    if not isinstance(result, dict):
        raise ValueError("Der Öffnungsversuch muss vertices und faces liefern.")
    opening = parse_mesh(result.get("vertices"), result.get("faces"), "solid")
    status = args.get("status", "derived")
    if status not in ("measured", "derived", "hypothesis"):
        raise ValueError("Status der Öffnung: measured, derived oder hypothesis.")
    opening.update(experiment_id=experiment["id"], status=status,
                   semantic_hint=str(args.get("semantic_hint", "opening"))[:120],
                   uncertainty=str(args.get("uncertainty", ""))[:2000])
    key = element["id"]
    _, before = inspect_ifc(scene.state)
    candidate = {**element, "openings": [*element.get("openings", []), opening]}
    preview = {**scene.state, "elements": [candidate if e["id"] == key else e for e in scene.state["elements"]]}
    _, after = inspect_ifc(preview)
    old = before["elements"].get(key, {}).get("net_volume_m3")
    new = after["elements"].get(key, {}).get("net_volume_m3")
    if old is None or new is None or new <= 1e-8 or old - new <= 1e-8:
        raise ValueError("Der Ausschnitt muss einen Teil eines geschlossenen Bauteils entfernen.")
    element["openings"] = candidate["openings"]
    scene.invalidate_quality()
    return {"element_id": key, "opening_ref": f"{key}:{len(element['openings'])}",
            "removed_volume_m3": old - new, "status": status}
