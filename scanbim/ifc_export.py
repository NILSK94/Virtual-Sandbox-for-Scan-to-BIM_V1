"""Author and inspect the same IFC working model used by the studio and export."""
from __future__ import annotations

import json
import uuid

import ifcopenshell
import ifcopenshell.api
import ifcopenshell.geom
import ifcopenshell.util.element
import ifcopenshell.util.shape
import ifcopenshell.validate
import numpy as np
from ifcopenshell.util.shape_builder import ShapeBuilder

ASSETS = {"building": "IfcBuilding", "bridge": "IfcBridge", "road": "IfcRoad",
          "railway": "IfcRailway", "facility": "IfcFacility", "site": None}
PART_CLASSES = {"bridge":"IfcBridgePart","road":"IfcRoadPart","railway":"IfcRailwayPart",
                "facility":"IfcFacilityPartCommon"}
CLASSES = {
    "wall": "IfcWall", "slab": "IfcSlab", "roof": "IfcRoof", "beam": "IfcBeam",
    "column": "IfcColumn", "member": "IfcMember", "plate": "IfcPlate",
    "footing": "IfcFooting", "pile": "IfcPile", "railing": "IfcRailing",
    "stair": "IfcStair", "ramp": "IfcRamp", "covering": "IfcCovering",
    "door": "IfcDoor", "window": "IfcWindow", "space": "IfcSpace",
    "terrain": "IfcGeographicElement", "bearing": "IfcBearing",
    "pavement": "IfcPavement", "kerb": "IfcKerb", "earthworks": "IfcEarthworksFill",
}
QUANTITIES = {
    "IfcWall": "Qto_WallBaseQuantities", "IfcSlab": "Qto_SlabBaseQuantities",
    "IfcBeam": "Qto_BeamBaseQuantities",
    "IfcColumn": "Qto_ColumnBaseQuantities", "IfcFooting": "Qto_FootingBaseQuantities",
}


def model_schema(state):
    return state.get("ifc", {}).get("schema", "IFC4")


def global_id(state, key):
    uid = uuid.uuid5(uuid.NAMESPACE_URL, f"scanbim:{state['id']}:{key}")
    return ifcopenshell.guid.compress(uid.hex)


def declaration(schema, ifc_class):
    try:
        entity = ifcopenshell.ifcopenshell_wrapper.schema_by_name(schema).declaration_by_name(ifc_class).as_entity()
    except RuntimeError:
        entity = None
    return entity


def product_class(element, schema):
    ifc_class = element.get("ifc", {}).get("class")
    if ifc_class:
        return ifc_class
    semantic = str(element.get("semantic", element["kind"])).lower()
    candidate = CLASSES.get(semantic)
    return candidate if candidate and declaration(schema, candidate) else "IfcBuildingElementProxy"


def is_terrain(state, element):
    return (str(element.get("semantic","")).lower()=="terrain" or
            (product_class(element,model_schema(state))=="IfcGeographicElement" and
             element.get("ifc",{}).get("predefined_type")=="TERRAIN"))


def has_volume(element):
    if element["kind"] != "mesh":
        return True
    return element.get("geometry_role")!="surface" and mesh_topology(
        np.asarray(element["vertices"]),np.asarray(element["faces"]))[0]


def bim_state(state):
    selected=[element for element in state["elements"] if has_volume(element) or is_terrain(state,element)]
    keys={element["id"] for element in selected}
    elements=[]
    for element in selected:
        item={**element}
        ifc={**item.get("ifc",{})}
        if ifc.get("fills","").partition(":")[0] not in keys:
            ifc.pop("fills",None)
        item["ifc"]=ifc
        for name in ("contacts","cuts"):
            if name in item:
                item[name]=[key for key in item[name] if key in keys]
        elements.append(item)
    return {**state,"elements":elements}


def mesh_topology(vertices, faces):
    """Check boundary, winding and signed volume before claiming a closed shell."""
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    keys = np.sort(edges, axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    winding = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    triangles = vertices[faces] - vertices[faces[0, 0]]
    volume = float(np.einsum("ij,ij->", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])) / 6)
    return bool(np.all(counts == 2) and np.all(winding == 0) and abs(volume) > 1e-12), volume


def placement(element):
    matrix = np.eye(4)
    normal = np.asarray(element["normal"])
    matrix[:3, 3] = element["origin"]
    if element["kind"] == "slab":
        matrix[:3, :3] = np.column_stack([element["u"], element["v"], normal])
        if element.get("depth_direction") != 1:
            matrix[:3, 3] -= normal * element["thickness"]
    else:
        matrix[:3, :3] = np.column_stack([element["u"], -normal, element["v"]])
        if element.get("depth_direction") == 1:
            matrix[:3, 3] += normal * element["thickness"]
    return matrix


def build_ifc(state):
    schema = model_schema(state)
    model = ifcopenshell.file(schema=schema)

    def api(operation, **kwargs):
        return ifcopenshell.api.run(operation, model, **kwargs)

    def root(ifc_class, key, name, predefined_type=None):
        entity = api("root.create_entity", ifc_class=ifc_class, name=name, predefined_type=predefined_type)
        entity.GlobalId = global_id(state, key)
        return entity

    project = root("IfcProject", "project", state["name"])
    api("unit.assign_unit", units=[api("unit.add_si_unit", unit_type=kind)
                                 for kind in ("LENGTHUNIT", "AREAUNIT", "VOLUMEUNIT")])
    context = api("context.add_context", context_type="Model")
    body = api("context.add_context", context_type="Model", context_identifier="Body",
               target_view="MODEL_VIEW", parent=context)
    config = state.get("ifc", {})
    if config.get("map"):
        georef = config["map"]
        crs = model.create_entity("IfcProjectedCRS", Name=georef["crs"])
        model.create_entity("IfcMapConversion", SourceCRS=context, TargetCRS=crs,
                            Eastings=georef["eastings"], Northings=georef["northings"],
                            OrthogonalHeight=georef["height"], XAxisAbscissa=georef["x_axis_abscissa"],
                            XAxisOrdinate=georef["x_axis_ordinate"], Scale=georef["scale"])
    site = root("IfcSite", "site", config.get("site_name", "Scan-Gelände"))
    api("aggregate.assign_object", products=[site], relating_object=project)
    asset = config.get("asset", "building")
    facility = site
    if ASSETS[asset]:
        facility = root(ASSETS[asset], "facility", config.get("asset_name", state["name"]), config.get("asset_type"))
        api("aggregate.assign_object", products=[facility], relating_object=site)
    containers = {}
    if asset == "building":
        for level in config.get("levels", [{"id": "default", "name": "Rekonstruierte Ebene", "elevation": 0.0}]):
            storey = root("IfcBuildingStorey", "storey:" + level["id"], level["name"])
            storey.Elevation = level["elevation"]
            api("aggregate.assign_object", products=[storey], relating_object=facility)
            matrix = np.eye(4)
            matrix[2, 3] = level["elevation"]
            api("geometry.edit_object_placement", product=storey, matrix=matrix)
            containers[level["id"]] = storey
    parts = {}
    for item in config.get("parts", []):
        part = root(PART_CLASSES[asset],"part:"+item["id"],item["name"],item["type"])
        part.UsageType = item["usage"]
        api("aggregate.assign_object",products=[part],relating_object=parts.get(item.get("parent"),facility))
        parts[item["id"]] = part
    types, materials, openings = {}, {}, {}
    products = {}
    elements = {element["id"]: element for element in state["elements"]}

    def block(width, height, thickness):
        return api("geometry.add_wall_representation", context=body, length=width, height=height, thickness=thickness)

    def slab(width, height, thickness):
        profile = [(0., 0.), (width, 0.), (width, height), (0., height), (0., 0.)]
        return api("geometry.add_slab_representation", context=body, depth=thickness, polyline=profile)

    def planar_body(element, width, height, thickness):
        return (slab if element["kind"] == "slab" else block)(width, height, thickness)

    def rectangular_opening(element, opening):
        hole = planar_body(element, opening["width"], opening["height"], element["thickness"] + .04)
        offset = [opening["x"], opening["y"], -.02] if element["kind"] == "slab" else [opening["x"], -.02, opening["y"]]
        return hole, offset

    def set_pset(product, name, properties):
        if properties:
            pset = api("pset.add_pset", product=product, name=name)
            api("pset.edit_pset", pset=pset, properties=properties)

    def place_solid(item, transform):
        pos = item.Position
        origin = np.array(pos.Location.Coordinates) if pos else np.zeros(3)
        point = tuple(float(v) for v in (transform[:3, :3] @ origin + transform[:3, 3]))
        axis = tuple(float(v) for v in transform[:3, 2])
        reference = tuple(float(v) for v in transform[:3, 0])
        item.Position = model.createIfcAxis2Placement3D(model.createIfcCartesianPoint(point),
                        model.createIfcDirection(axis), model.createIfcDirection(reference))
        if pos:
            ifcopenshell.util.element.remove_deep2(model, pos)

    def mesh_body(element, matrix):
        profile = element.get("solidification",{}).get("profile")
        if profile:
            origin = matrix[:3,:3].T @ (np.array(profile["origin"])-matrix[:3,3])
            normal = matrix[:3,:3].T @ profile["direction"]
            axis = matrix[:3,:3].T @ profile["x_axis"]
            coordinates = lambda values: tuple(float(v) for v in values)
            position = model.createIfcAxis2Placement3D(model.createIfcCartesianPoint(coordinates(origin)),
                         model.createIfcDirection(coordinates(normal)),model.createIfcDirection(coordinates(axis)))

            def ring(points):
                return model.createIfcIndexedPolyCurve(model.createIfcCartesianPointList2D(points))

            outer = ring(profile["outer"])
            holes = [ring(points) for points in profile["holes"]]
            if holes:
                section = model.createIfcArbitraryProfileDefWithVoids("AREA",None,outer,holes)
            else:
                section = model.createIfcArbitraryClosedProfileDef("AREA",None,outer)
            return model.createIfcExtrudedAreaSolid(section,position,model.createIfcDirection((0.,0.,1.)),
                                                     profile["depth_m"])
        vertices = np.asarray(element["vertices"], dtype=float)
        faces = np.asarray(element["faces"], dtype=int)
        closed, signed = mesh_topology(vertices, faces)
        closed = closed and element.get("geometry_role") != "surface"
        if closed and signed < 0:
            faces = faces[:, ::-1]
        local = (vertices - matrix[:3, 3]) @ matrix[:3, :3]
        if closed and element.get("solidification"):
            return ShapeBuilder(model).faceted_brep(local,faces.tolist())
        item = ShapeBuilder(model).polygonal_face_set(local, faces.tolist())
        item.Closed = closed
        return item

    def solid_operand(element, host):
        if element["kind"] == "mesh":
            item = mesh_body(element, host)
        else:
            transform = np.linalg.solve(host, placement(element))
            raw = planar_body(element, element["width"], element["height"], element["thickness"])
            item = raw.Items[0]
            place_solid(item, transform)
            model.remove(raw)
        for opening in element.get("openings", []):
            if "vertices" in opening:
                cutter = mesh_body(opening, host)
            else:
                hole, shift = rectangular_opening(element, opening)
                offset = transform.copy()
                offset[:3, 3] += transform[:3, :3] @ shift
                cutter = hole.Items[0]
                place_solid(cutter, offset)
                model.remove(hole)
            item = model.create_entity("IfcBooleanResult", "DIFFERENCE", item, cutter)
        for key in element.get("cuts", []):
            item = model.create_entity("IfcBooleanResult", "DIFFERENCE", item, solid_operand(elements[key], host))
        return item

    for key, element in elements.items():
        meta = element.get("ifc", {})
        ifc_class = product_class(element, schema)
        predefined = meta.get("predefined_type") or ("TERRAIN" if ifc_class == "IfcGeographicElement"
                                                     and element.get("semantic") == "terrain" else None)
        product = root(ifc_class, "element:" + key, element["name"], predefined)
        products[key] = product
        default_container = next(iter(containers.values()),facility)
        container = site if predefined == "TERRAIN" else parts.get(meta.get("part"),containers.get(meta.get("level"),default_container))
        if product.is_a("IfcSpatialElement"):
            api("aggregate.assign_object", products=[product], relating_object=container)
        else:
            api("spatial.assign_container", products=[product], relating_structure=container)
        matrix = np.eye(4)
        if element["kind"] == "mesh":
            matrix[:3, 3] = element["vertices"][0]
            item = mesh_body(element, matrix)
            kind = {"IfcFacetedBrep":"Brep","IfcExtrudedAreaSolid":"SweptSolid"}.get(item.is_a(),"Tessellation")
            rep = model.create_entity("IfcShapeRepresentation", body, "Body", kind, [item])
        else:
            matrix = placement(element)
            rep = planar_body(element, element["width"], element["height"], element["thickness"])
        for target_id in element.get("cuts", []):
            cutter = solid_operand(elements[target_id], matrix)
            rep.Items = [model.create_entity("IfcBooleanResult", "DIFFERENCE", rep.Items[0], cutter)]
            rep.RepresentationType = "CSG"

        api("geometry.assign_representation", product=product, representation=rep)
        api("geometry.edit_object_placement", product=product, matrix=matrix)
        for index, opening in enumerate(element.get("openings", [])):
            number = index + 1
            opening_key = f"{key}:{number}"
            void = root("IfcOpeningElement", "opening:" + opening_key,
                        f"{element['name']} / Öffnung {number}", "OPENING")
            if "vertices" in opening:
                location = np.eye(4)
                location[:3, 3] = opening["vertices"][0]
                item = mesh_body(opening, location)
                void_rep = model.create_entity("IfcShapeRepresentation", body, "Body", "Tessellation", [item])
            else:
                void_rep, shift = rectangular_opening(element, opening)
                location = matrix.copy()
                location[:3, 3] += matrix[:3, :3] @ shift
            api("geometry.assign_representation", product=void, representation=void_rep)
            api("geometry.edit_object_placement", product=void, matrix=location)
            api("feature.add_feature", feature=void, element=product)
            openings[opening_key] = void
            set_pset(void, "ScanBIM_Evidence", {"Status": opening.get("status", "hypothesis"),
                     "SemanticHint": opening.get("semantic_hint"), "ExperimentId": opening.get("experiment_id"),
                     "Uncertainty": opening.get("uncertainty"), "Resolution_m": opening.get("resolution_m")})

        type_class = product.is_a() + "Type"
        if meta.get("type_name") and declaration(schema, type_class):
            type_key = (type_class, meta["type_name"], predefined)
            if type_key not in types:
                types[type_key] = root(type_class, "type:" + str(type_key), meta["type_name"], predefined)
            api("type.assign_type", related_objects=[product], relating_type=types[type_key],
                should_map_representations=False)
        if meta.get("material"):
            label = meta["material"]
            if label not in materials:
                materials[label] = api("material.add_material", name=label)
            api("material.assign_material", products=[product], material=materials[label])
        if meta.get("layers"):
            layers = []
            for layer in meta["layers"]:
                label = layer["name"]
                if label not in materials:
                    materials[label] = api("material.add_material", name=label)
                layers.append(model.create_entity("IfcMaterialLayer", Material=materials[label],
                                                  LayerThickness=layer["thickness_m"]))
            layer_set = model.create_entity("IfcMaterialLayerSet", MaterialLayers=layers)
            usage = model.create_entity("IfcMaterialLayerSetUsage", ForLayerSet=layer_set,
                        LayerSetDirection="AXIS3" if element["kind"]=="slab" else "AXIS2",
                        DirectionSense="POSITIVE", OffsetFromReferenceLine=0.)
            model.create_entity("IfcRelAssociatesMaterial", GlobalId=global_id(state,"layers:"+key),
                                RelatedObjects=[product], RelatingMaterial=usage)
        set_pset(product, "ScanBIM_Evidence", {
            "ElementId": key, "SourceSHA256": state["source_hash"],
            "SemanticHypothesis": str(element.get("semantic", element["kind"])),
            "Evidence": json.dumps(element.get("evidence", {}), ensure_ascii=False),
            "ParameterStatus": json.dumps(element.get("parameter_status", {}), ensure_ascii=False),
            "Uncertainty": str(element.get("uncertainty", "")),
            "ThicknessAssumed": element.get("parameter_status", {}).get("thickness", "assumed") == "assumed"
            if element["kind"] != "mesh" or element.get("solidification") else False,
        })
        if element.get("solidification"):
            source = element["solidification"]
            set_pset(product,"ScanBIM_Solidification",{
                "Method":source["mode"],"ThicknessStatus":source["thickness_status"],
                "MinimumThickness_m":source["thickness_min_m"],"MaximumThickness_m":source["thickness_max_m"],
                "SourceSurfaces":json.dumps(source["source_surfaces"],ensure_ascii=False),
                "InferredCounterfaceFraction":source.get("inferred_vertex_fraction"),
                "Extension_m":source.get("extension_m"),"PlaneFitP95_m":source.get("plane_p95_m"),
                "SmallFootprintGapsClosed":source.get("small_gaps_closed"),
            })
        if element.get("material"):
            set_pset(product, "ScanBIM_MaterialHypothesis", {
                "Hypothesis": json.dumps(element["material"], ensure_ascii=False), "Status": "Unconfirmed"})

    connected = set()
    for key, element in elements.items():
        for target in [*element.get("cuts", []), *element.get("contacts", [])]:
            pair = tuple(sorted((key, target)))
            if target in products and pair not in connected:
                model.create_entity("IfcRelConnectsElements", GlobalId=global_id(state, "join:" + ":".join(pair)),
                                    RelatingElement=products[target], RelatedElement=products[key])
                connected.add(pair)
        filling = element.get("ifc", {}).get("fills")
        if filling in openings:
            api("feature.add_filling", opening=openings[filling], element=products[key])
        if products[key].is_a("IfcSpace"):
            for boundary in element.get("ifc", {}).get("bounds", []):
                if boundary not in products:
                    continue
                model.create_entity("IfcRelSpaceBoundary", GlobalId=global_id(state, "boundary:" + key + ":" + boundary),
                                    RelatingSpace=products[key], RelatedBuildingElement=products[boundary],
                                    PhysicalOrVirtualBoundary="PHYSICAL", InternalOrExternalBoundary="NOTDEFINED")
    return model, products


def inspect_ifc(state, *, meshes=False, check=False, tolerance=.002):
    model, products = build_ifc(state)
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    issues, data, solids = [], {}, {}
    by_id = {element["id"]: element for element in state["elements"]}
    tree = ifcopenshell.geom.tree() if check else None

    for key, product in products.items():
        source = by_id[key]
        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
            vertices = ifcopenshell.util.shape.get_vertices(shape.geometry)
            faces = ifcopenshell.util.shape.get_faces(shape.geometry)
        except (RuntimeError, ValueError) as error:
            issues.append(f"{product.Name}: IFC-Körper kann nicht gelesen werden ({error}).")
            continue
        if not len(faces):
            issues.append(f"{product.Name}: IFC-Körper ist leer.")
            continue
        closed, volume = mesh_topology(vertices, faces)
        representation_item=product.Representation.Representations[0].Items[0]
        while representation_item.is_a("IfcBooleanResult"):
            representation_item=representation_item.FirstOperand
        expected_solid = source["kind"] != "mesh" or source.get("geometry_role") == "solid" or bool(
            getattr(representation_item,"Closed",False))
        if expected_solid and not closed:
            issues.append(f"{product.Name}: IFC-Körper ist offen oder entartet.")
        data[key] = {"class": product.is_a(), "global_id": product.GlobalId, "closed": closed and expected_solid,
                     "net_volume_m3": abs(volume) if closed and expected_solid else None, "type_name": source.get("ifc", {}).get("type_name"),
                     "material": source.get("ifc", {}).get("material"), "layers": source.get("ifc", {}).get("layers"),
                     "predefined_type": ifcopenshell.util.element.get_predefined_type(product),
                     "representation": product.Representation.Representations[0].RepresentationType}
        if product.is_a("IfcElement"):
            data[key]["connections"] = [*source.get("cuts", []), *source.get("contacts", [])]
        if meshes:
            data[key].update(vertices=vertices.tolist(), faces=faces.tolist())
        if tree is not None:
            tree.add_element(shape)
        if closed and expected_solid and not product.is_a("IfcSpace"):
            solids[key] = (vertices, faces, volume)
        if closed and expected_solid and product.is_a() in QUANTITIES:
            qto = ifcopenshell.api.run("pset.add_qto", model, product=product, name=QUANTITIES[product.is_a()])
            quantities = {"NetVolume": abs(volume)}
            if source["kind"] == "wall" and product.is_a("IfcWall"):
                quantities.update(Length=source["width"], Height=source["height"], Width=source["thickness"])
            if source["kind"] == "slab" and product.is_a("IfcSlab"):
                quantities.update(Length=source["width"], Width=source["height"], Depth=source["thickness"])
            ifcopenshell.api.run("pset.edit_qto", model, qto=qto, properties=quantities)

    clashes = []
    if check:
        logger = ifcopenshell.validate.json_logger()
        ifcopenshell.validate.validate(model, logger)
        issues.extend(f"IFC-Schema: {issue['message']}" for issue in logger.statements)
        if len(solids) >= 2:
            clashes, failures = check_intersections(solids, by_id, tolerance)
            issues.extend(failures)
        for key, element in by_id.items():
            connections = [*element.get("cuts", []), *element.get("contacts", [])]
            anchor = element.get("evidence", {}).get("anchor_id")
            if anchor:
                connections.append(anchor)
            for target in connections:
                if key in data and target in data and not tree.clash_clearance_many(
                        [products[key]], [products[target]], clearance=.03 if target==anchor else .005, check_all=False):
                    issues.append("Anschluss ohne Kontakt: " + element["name"] + " / " + by_id[target]["name"])
    ids = {product.id():key for key,product in products.items()}

    def node(entity):
        child_spatial=[child for relation in getattr(entity,"IsDecomposedBy",()) for child in relation.RelatedObjects]
        child_elements=[child for relation in getattr(entity,"ContainsElements",()) for child in relation.RelatedElements]
        return {"name":entity.Name,"class":entity.is_a(),"type":getattr(entity,"PredefinedType",None),
                "element_id":ids.get(entity.id()),"children":[node(child) for child in child_spatial+child_elements]}

    return model, {"schema": model.schema_identifier, "elements": data, "issues": issues,
                   "tree": node(model.by_type("IfcProject")[0]),
                   "clashes": clashes, "tolerance_m": tolerance}


def check_intersections(solids, by_id, tolerance):
    """Sweep boxes first, then calculate actual intersection even for coplanar faces."""
    bounds = {key: (verts.min(axis=0), verts.max(axis=0)) for key, (verts, _, _) in solids.items()}
    origin = next(iter(bounds.values()))[0]
    axis = np.argmax(np.ptp([low for low, _ in bounds.values()], axis=0))
    ordered = sorted(bounds, key=lambda key: bounds[key][0][axis])
    scratch = ifcopenshell.file(schema="IFC4")
    builder = ShapeBuilder(scratch)
    settings = ifcopenshell.geom.settings()
    operands = {}
    clashes, issues = [], []
    for index, a in enumerate(ordered):
        a_low, a_high = bounds[a]
        for b in ordered[index + 1:]:
            b_low, b_high = bounds[b]
            if b_low[axis] >= a_high[axis] - tolerance:
                break
            size = np.minimum(a_high,b_high) - np.maximum(a_low,b_low)
            if np.any(size <= tolerance):
                continue
            names = [by_id[key]["name"] for key in (a,b)]
            for key in (a,b):
                if key not in operands:
                    vertices, faces, signed = solids[key]
                    operands[key] = builder.faceted_brep(vertices-origin, faces if signed > 0 else faces[:, ::-1])
            result = scratch.create_entity("IfcBooleanResult", "INTERSECTION", operands[a], operands[b])
            try:
                mesh = ifcopenshell.geom.create_shape(settings, result)
                volume = ifcopenshell.util.shape.get_volume(mesh)
            except RuntimeError:
                issues.append("Schnittvolumen konnte nicht geprüft werden: " + " / ".join(names))
                continue
            if volume > 1e-8:
                vertices = ifcopenshell.util.shape.get_vertices(mesh)
                clashes.append({"elements": [a,b], "names": names, "volume_m3": volume,
                                "depth_m": float(np.ptp(vertices,axis=0).min()),
                                "point": (vertices.mean(axis=0)+origin).tolist()})
    return sorted(clashes, key=lambda clash: -clash["volume_m3"]), issues


def export_ifc(scene, path, *, bim_only=False):
    state=bim_state(scene.state) if bim_only else scene.state
    model, report = inspect_ifc(state)
    temporary = path.with_suffix(".ifc.tmp")
    try:
        model.write(str(temporary))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"file": path.name, "elements": len(state["elements"]), "schema": model.schema_identifier,
            "openings": sum(len(e.get("openings", [])) for e in state["elements"]),
            "issues": report["issues"], "note": "IFC-Arbeitsmodell mit Messnachweisen und ausgewiesenen Annahmen."}
