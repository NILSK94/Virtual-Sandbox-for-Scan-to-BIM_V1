"""Reconstruct closed, scan-backed bodies from one or two observed surfaces."""
from __future__ import annotations

import numpy as np
import shapely
from scipy import ndimage
from scipy.spatial import cKDTree

from .geometry import unit
from .ifc_export import has_volume, inspect_ifc, is_terrain, mesh_topology
from .ifc_tools import parse_mesh


def projection(surface, direction=None):
    points = np.asarray(surface["vertices"], dtype=float)
    origin = points.mean(axis=0)
    if direction is None:
        _, axes = np.linalg.eigh((points-origin).T @ (points-origin))
        direction = axes[:, 0]
        if direction[np.argmax(np.abs(direction))] < 0:
            direction = -direction
    n = unit(direction)
    tangent=(points-origin)-np.outer((points-origin) @ n,n)
    _,axes=np.linalg.eigh(tangent.T @ tangent)
    u = unit(axes[:,-1]-float(axes[:,-1] @ n)*n)
    if u[np.argmax(np.abs(u))]<0:
        u=-u
    v = np.cross(n, u)
    return origin, np.column_stack((u, v, n))


def closed_caps(upper, lower, faces):
    """Weld cap and side indices, retaining holes as through-voids."""
    upper = np.asarray(upper)
    faces = np.asarray(faces, dtype=int)
    size = len(upper)
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    a, b = edges[counts[inverse] == 1].T
    return np.vstack((upper, lower)), np.vstack((
        faces, faces[:, ::-1] + size,
        np.column_stack((b, a, a + size)), np.column_stack((b, a + size, b + size)),
    ))


def projected_area(points, faces):
    triangles = points[faces, :2]
    delta = triangles[:, 1:] - triangles[:, :1]
    return (delta[:, 0, 0]*delta[:, 1, 1]-delta[:, 0, 1]*delta[:, 1, 0]) / 2


def planar_body(surface, direction, depth, tolerance):
    origin, frame = projection(surface, direction)
    points = (np.asarray(surface["vertices"]) - origin) @ frame
    faces = np.asarray(surface["faces"], dtype=int)
    height = float(np.median(points[:, 2]))
    error = float(np.quantile(np.abs(points[:, 2] - height), .95))
    if error > tolerance:
        raise ValueError(f"Ebene nicht ausreichend gestützt ({error*1000:.0f} mm P95). Gekrümmten Offset oder Gegenfläche nutzen.")
    area = projected_area(points, faces)
    polygons = shapely.polygons(points[faces[np.abs(area)>1e-10], :2])
    footprint = shapely.union_all(polygons, grid_size=1e-7)
    polygons = [part for part in shapely.get_parts(footprint) if part.geom_type == "Polygon" and part.area > 0]
    if not polygons:
        raise ValueError("Die Fläche hat in Materialrichtung kein geschlossenes Profil.")
    outline = max(polygons, key=lambda part: part.area)
    fraction = outline.area / footprint.area
    if fraction < .8:
        raise ValueError("Mehrere getrennte Messflächen. Zuerst bauteilweise aufteilen.")
    # Close only small holes in the measured footprint. Larger observed gaps
    # are carried through the profile and remain voids in the body.
    holes = [ring for ring in outline.interiors if shapely.Polygon(ring).area >= .04]
    filled_holes = len(outline.interiors)-len(holes)
    outline = shapely.orient_polygons(shapely.Polygon(outline.exterior,holes).simplify(
        min(.015,tolerance/3),preserve_topology=True))
    triangles = shapely.get_parts(shapely.constrained_delaunay_triangles(outline))
    xy, indices = np.unique(shapely.get_coordinates(triangles), axis=0, return_inverse=True)
    # Polygon rings repeat their first point.
    caps = indices.reshape(-1, 4)[:, :3]
    points = np.column_stack((xy, np.zeros(len(xy))))
    reverse = projected_area(points, caps)<0
    caps[reverse] = caps[reverse, ::-1]
    origin = origin + frame[:, 2] * height
    low = points @ frame.T + origin
    high = low + frame[:, 2] * depth
    vertices, faces = closed_caps(high, low, caps)
    profile = {"origin": origin.tolist(), "direction": frame[:, 2].tolist(), "x_axis": frame[:, 0].tolist(),
               "outer": [list(p) for p in outline.exterior.coords],
               "holes": [[list(p) for p in ring.coords] for ring in outline.interiors], "depth_m": depth}
    return vertices, faces, {"profile": profile, "footprint_fraction": fraction, "small_gaps_closed":filled_holes,
                             "plane_p95_m": error, "thickness_min_m": depth, "thickness_max_m": depth}


def sample_heights(points, faces, probes, tolerance):
    faces = faces[np.abs(projected_area(points, faces))>1e-10]
    if not len(faces):
        raise ValueError("Messfläche ist in der gewählten Richtung nicht projizierbar.")
    triangles = points[faces]
    tree = shapely.STRtree(shapely.polygons(triangles[:, :, :2]))
    query, triangle_ids = tree.query(shapely.points(probes), predicate="covered_by")
    t = triangles[triangle_ids]
    a, b = t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]
    q = probes[query]-t[:, 0, :2]
    det = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
    z = t[:, 0, 2] + ((q[:, 0]*b[:, 1]-q[:, 1]*b[:, 0])*a[:, 2]
                     +(a[:, 0]*q[:, 1]-a[:, 1]*q[:, 0])*b[:, 2]) / det
    low, high = np.full(len(probes), np.inf), np.full(len(probes), -np.inf)
    np.minimum.at(low, query, z)
    np.maximum.at(high, query, z)
    valid = np.isfinite(low) & (high-low <= tolerance)
    result = np.full(len(probes), np.nan)
    result[valid] = (low[valid]+high[valid])/2
    return result


def unpinch(mask):
    """Exclude corner-only contacts that would create a nonmanifold edge in 3D."""
    mask = mask.copy()
    while True:
        a, b, c, d = mask[:-1, :-1], mask[:-1, 1:], mask[1:, :-1], mask[1:, 1:]
        diagonal = a & d & ~b & ~c
        cross = b & c & ~a & ~d
        if not (diagonal.any() or cross.any()):
            break
        a[diagonal] = False
        b[cross] = False
    labels, count = ndimage.label(mask)
    if not count:
        raise ValueError("Keine zusammenhängende gemeinsame Messfläche mit plausibler Dicke.")
    counts = np.bincount(labels[mask])
    return labels == np.argmax(counts)


def paired_body(first, second, direction, cell, min_depth, max_depth, tolerance, extension=0.):
    origin, frame = projection(first, direction)
    a = (np.asarray(first["vertices"]) - origin) @ frame
    b = (np.asarray(second["vertices"]) - origin) @ frame
    fa, fb = np.asarray(first["faces"], dtype=int), np.asarray(second["faces"], dtype=int)
    lower = np.maximum(a[:, :2].min(0), b[:, :2].min(0))
    upper = np.minimum(a[:, :2].max(0), b[:, :2].max(0))
    if np.any(upper<=lower):
        raise ValueError("Die Flächen überdecken sich in der gewählten Richtung nicht.")
    if extension:
        lower,upper=a[:,:2].min(0),a[:,:2].max(0)
    dimensions = np.ceil((upper-lower)/cell).astype(int)
    if np.prod(dimensions)>4500:
        raise ValueError("Körperraster zu fein. Rasterweite erhöhen oder Bauteil in Abschnitte teilen.")
    x, y = (np.linspace(lo,hi,int(n)+1) for lo,hi,n in zip(lower,upper,dimensions))
    xx, yy = np.meshgrid(x,y)
    vertices_xy = np.column_stack((xx.ravel(),yy.ravel()))
    ids = np.arange(xx.size).reshape(xx.shape)
    quads = np.stack((ids[:-1,:-1],ids[:-1,1:],ids[1:,1:],ids[1:,:-1]),axis=-1)
    center = vertices_xy[quads].mean(axis=-2).reshape(-1,2)
    probes = np.vstack((vertices_xy,center))
    za = sample_heights(a,fa,probes,tolerance)
    zb = sample_heights(b,fb,probes,tolerance)
    separation = za-zb
    finite = np.isfinite(separation)
    if not finite.any():
        raise ValueError("Keine gemeinsamen Stützpunkte in der Projektion.")
    sign = 1 if np.median(separation[finite])>=0 else -1
    separation *= sign
    valid = (separation>=min_depth) & (separation<=max_depth)
    measured=valid.copy()
    reference=za.copy()
    if extension and measured.any():
        known=probes[measured]
        distance, neighbours=cKDTree(known).query(probes,k=min(8,len(known)))
        if distance.ndim==1:
            distance,neighbours=distance[:,None],neighbours[:,None]
        infer=(~valid) & (distance[:,0]<=extension)
        separation[infer]=np.median(separation[measured][neighbours[infer]],axis=1)
        # Bridge only sub-cell holes on the reference surface.
        visible=np.isfinite(reference)
        if visible.any():
            distance, nearest=cKDTree(probes[visible]).query(probes)
            fill=(~visible) & (distance<=min(.24,cell*.55))
            reference[fill]=reference[visible][nearest[fill]]
        valid=(valid | infer) & np.isfinite(reference)
    common = valid[quads].all(axis=-1) & valid[len(vertices_xy):].reshape(quads.shape[:2])
    keep = unpinch(common)
    quads = quads[keep]
    faces = quads[:,[0,1,2,0,2,3]].reshape(-1,3)
    ids, compact = np.unique(faces,return_inverse=True)
    other=reference-separation*sign
    top = np.column_stack((vertices_xy[ids],np.maximum(reference[ids],other[ids]))) @ frame.T + origin
    bottom = np.column_stack((vertices_xy[ids],np.minimum(reference[ids],other[ids]))) @ frame.T + origin
    vertices, faces = closed_caps(top,bottom,compact.reshape(-1,3))
    common_area = float(np.prod((upper-lower)/dimensions)*len(quads))
    return vertices, faces, {"common_area_m2": common_area, "cells": len(quads), "cell_m": cell,
                             "largest_patch_fraction": float(np.count_nonzero(keep)/np.count_nonzero(common)),
                             "inferred_vertex_fraction":float(np.mean(~measured[ids] | ~np.isfinite(za[ids]))),
                             "thickness_median_m":float(np.median(separation[measured])),
                             "thickness_min_m": float(separation[ids].min()),
                             "thickness_max_m": float(separation[ids].max())}


def solidify(scene, element, opposite, args):
    for source in (element,opposite):
        if source is None:
            continue
        if (source["kind"]!="mesh" or is_terrain(scene.state,source) or source.get("geometry_role")=="solid"
                or source.get("cuts") or source.get("openings") or source.get("contacts")):
            raise ValueError("Nur offene, unverschnittene Bauteilmessflächen können überführt werden.")
        if (source.get("geometry_role")!="surface" and
                mesh_topology(np.asarray(source["vertices"]), np.asarray(source["faces"]))[0]):
            raise ValueError("Das Element ist bereits geschlossen.")
        if not source.get("evidence"):
            raise ValueError("Jede Quellfläche braucht einen Messnachweis.")
    mode = args.get("mode","paired" if opposite else "planar")
    direction = args.get("direction")
    tolerance = float(args.get("tolerance",.08))
    if not .001<=tolerance<=.2:
        raise ValueError("Flächentoleranz muss zwischen 1 und 200 mm liegen.")
    uncertainty = str(args.get("reason","")).strip()
    if opposite:
        if element["id"]==opposite["id"] or mode not in ("paired","paired_profile","paired_extend"):
            raise ValueError("Für die Paarbildung zwei verschiedene Flächen auswählen.")
        key = opposite["id"]
        for other in scene.state["elements"]:
            metadata = other.get("ifc",{})
            if (key in other.get("cuts",[]) or key in other.get("contacts",[]) or key in metadata.get("bounds",[])
                    or metadata.get("fills","").startswith(key+":") or other.get("evidence",{}).get("anchor_id")==key):
                raise ValueError("Gegenfläche hat noch Beziehungen. Vor dem Zusammenfassen lösen.")
        if opposite.get("ifc",{}).get("fills"):
            raise ValueError("Eine Füllung kann nicht als Gegenfläche verwendet werden.")
        depth_status = "derived"
    else:
        if mode not in ("planar","offset") or direction is None or not uncertainty:
            raise ValueError("Einseitige Körper brauchen Materialrichtung, Tiefe und eine Begründung.")
        depth = float(args.get("thickness_m",0))
        if not .001<=depth<=30:
            raise ValueError("Materialtiefe muss zwischen 1 mm und 30 m liegen.")
        depth_status = "assumed"
    if mode == "planar":
        vertices, faces, info = planar_body(element,direction,depth,tolerance)
    else:
        other = opposite
        if other is None:
            other = {**element,"vertices":(np.asarray(element["vertices"])+unit(direction)*depth).tolist()}
        cell = float(args.get("cell",.6))
        minimum = float(args.get("min_depth_m",.01 if opposite else depth*.99))
        maximum = float(args.get("max_depth_m",10 if opposite else depth*1.01))
        if not .01<=cell<=20 or not .0005<=minimum<maximum<=50:
            raise ValueError("Ungültiges Raster oder Dickenintervall.")
        extension=float(args.get("extension_m",1.5)) if mode=="paired_extend" else 0.
        if not 0<=extension<=5:
            raise ValueError("Seitliche Ergänzung muss zwischen 0 und 5 m liegen.")
        vertices, faces, info = paired_body(element,other,direction,cell,minimum,maximum,tolerance,extension)
        if extension:
            info["extension_m"]=extension
            if not uncertainty:
                uncertainty=("Gegenfläche aus benachbarten belegten Dicken ergänzt. "
                             "Ergänzte Rückseiten und seitliche Abschlüsse sind abgeleitet.")
        if mode=="paired_profile":
            origin,frame=projection(element,direction)
            n=frame[:,2]
            offset=(np.asarray(other["vertices"])-origin) @ n
            median=np.median(offset)
            p95=float(np.quantile(np.abs(offset-median),.95))
            if p95>tolerance:
                raise ValueError("Gegenfläche ist für ein durchgängiges Profil zu stark gekrümmt.")
            depth=info["thickness_median_m"]
            inward=n if median>0 else -n
            vertices,faces,profile=planar_body(element,inward,depth,tolerance)
            info.update(profile,opposite_plane_p95_m=p95)
            if not uncertainty:
                uncertainty=("Durchgängiges Profil aus gemeinsam gestützter Dicke. "
                             "In Abschnitten ohne Gegenmessung ist die Rückseite abgeleitet.")
    mesh = parse_mesh(vertices,faces,"solid")
    source_records = [{"id":item["id"],"name":item["name"],"evidence":item["evidence"]}
                      for item in (element,opposite) if item is not None]
    measure = {**info,"mode":mode,"source_surfaces":source_records,"thickness_status":depth_status}
    updated = {**element, **mesh, "solidification":measure,
               "name":str(args.get("name",element["name"]))[:120],
               "parameter_status":{**element.get("parameter_status",{}),"geometry":"derived","thickness":depth_status},
               "uncertainty":uncertainty[:2000] or "Volumen aus gemeinsamer Messfläche. Seitlicher Abschluss abgeleitet."}
    preview = {**scene.state, "elements":[updated]}
    # Test the actual native IFC representation before replacing either source.
    _, report = inspect_ifc(preview)
    body = report["elements"].get(element["id"],{})
    if report["issues"] or not body.get("closed") or not body.get("net_volume_m3"):
        raise ValueError("Der resultierende IFC-Körper konnte nicht geschlossen eingelesen werden.")
    element.clear()
    element.update(updated)
    if opposite:
        scene.state["elements"].remove(opposite)
    scene.invalidate_quality()
    summary = {k:v for k,v in measure.items() if k not in ("profile","source_surfaces")}
    return {"id":element["id"],"name":element["name"],"net_volume_m3":body["net_volume_m3"],
            "representation":body.get("representation"),"source_ids":[item["id"] for item in source_records],**summary}


def open_structural_surfaces(state):
    return [part for part in state["elements"] if not has_volume(part) and not is_terrain(state,part)]


def probe_thickness(scene, element, args):
    """Find parallel return bands spread across the footprint of a measured face."""
    if element["kind"]!="mesh" or is_terrain(scene.state,element):
        raise ValueError("Für die Dickenmessung eine Bauteilmessfläche auswählen.")
    maximum=float(args.get("max_depth_m",2.))
    step=float(args.get("bin_m",.04))
    if not .1<=maximum<=10 or not .01<=step<=.2:
        raise ValueError("Messreichweite 0,1 bis 10 m und Raster 10 bis 200 mm.")
    origin,frame=projection(element,args.get("direction"))
    vertices=np.asarray(element["vertices"],float)
    local=(vertices-origin) @ frame
    height=float(np.median(local[:,2]))
    flatness=float(np.quantile(np.abs(local[:,2]-height),.95))
    if flatness>.2:
        raise ValueError("Für stark gekrümmte Flächen lokale Querschnitte oder Paarflächen verwenden.")
    faces=np.asarray(element["faces"],int)
    faces=faces[np.abs(projected_area(local,faces))>1e-10]
    footprint=shapely.union_all(shapely.polygons(local[faces,:2]),grid_size=1e-7)
    if not footprint.area>0:
        raise ValueError("Messfläche hat in dieser Richtung keinen Fußabdruck.")
    radius=np.abs(frame[:,2])*maximum+.03
    points,_,_,total=scene.sample({"min":(vertices.min(0)-radius).tolist(),
                                    "max":(vertices.max(0)+radius).tolist()},120000)
    local_points=(points-origin) @ frame
    distance=local_points[:,2]-height
    mask=(np.abs(distance)<=maximum) & (np.abs(distance)>=max(.05,flatness*1.3+step))
    local_points=local_points[mask]
    inside=shapely.contains_xy(footprint,local_points[:,0],local_points[:,1])
    matches=local_points[inside]
    if not len(matches):
        return {"candidates":[],"analyzed":len(points),"region_total":total,
                "direction":frame[:,2].tolist(),"reference_p95_m":flatness}
    size=max(.12,min(1.,float(np.sqrt(footprint.area)/15)))
    cells=np.floor(matches[:,:2]/size).astype(int)
    bands=np.rint((matches[:,2]-height)/step).astype(int)
    occupied=np.unique(np.column_stack((bands,cells)),axis=0)
    ids, counts=np.unique(occupied[:,0],return_counts=True)
    threshold=max(5,.12*footprint.area/(size*size))
    candidates=[]
    for index in np.argsort(-counts)[:8]:
        offset=float(ids[index]*step)
        if counts[index]<threshold or any(abs(item["signed_depth_m"]-offset)<2*step for item in candidates):
            continue
        candidates.append({"signed_depth_m":offset,"support_cells":int(counts[index]),
                           "material_direction":(frame[:,2]*np.sign(offset)).tolist()})
    return {"candidates":candidates,"analyzed":len(points),"region_total":total,
            "direction":frame[:,2].tolist(),"reference_p95_m":flatness,
            "note":"Parallele Punktbänder sind Kandidaten für Rückseiten. Zugehörigkeit anhand von Ansichten und Querschnitten prüfen."}


def build_available_solids(scene, actor):
    """Use explicit face roles first. Ambiguous cases go to the visual agent."""
    open_parts=open_structural_surfaces(scene.state)
    remaining=[part for part in open_parts if actor=="human" or not part.get("locked")]
    built, skipped = [], []

    def role(part):
        explicit=part.get("parameters",{}).get("surface_role")
        if explicit in ("upper","lower"):
            return explicit
        name=part["name"].lower()
        if "unterseite" in name:
            return "lower"
        if "oberseite" in name:
            return "upper"
        return None

    for part in remaining:
        if part not in scene.state["elements"] or part.get("geometry_role")=="solid":
            continue
        parameters=part.get("parameters",{})
        try:
            if role(part)=="upper":
                options=[other for other in remaining if other in scene.state["elements"] and
                         other.get("geometry_role")!="solid" and role(other)=="lower" and
                         other.get("semantic")==part.get("semantic")]
                if len(options)==1:
                    other=options[0]
                    size=np.ptp(np.asarray(part["vertices"]),axis=0)
                    cell=max(.06,float(np.sort(size)[-2])*.025)
                    built.append(solidify(scene,part,other,{"mode":"paired_extend","direction":[0,0,1],
                        "cell":cell,"extension_m":min(2.,float(np.max(size))*.10),
                        "max_depth_m":min(3.,max(.65,float(np.sort(size)[-2])*.15))}))
                    continue
            if parameters.get("thickness_m") and parameters.get("material_direction"):
                built.append(solidify(scene,part,None,{"mode":"planar","thickness_m":parameters["thickness_m"],
                    "direction":parameters["material_direction"],
                    "reason":"Aus der dokumentierten Dicke und Materialseite der Bauteilhypothese übernommen."}))
        except (ValueError,RuntimeError) as error:
            skipped.append({"id":part["id"],"reason":str(error)})
    return {"built":built,"skipped":skipped,"still_open":[{"id":part["id"],"name":part["name"]}
             for part in open_structural_surfaces(scene.state)]}
