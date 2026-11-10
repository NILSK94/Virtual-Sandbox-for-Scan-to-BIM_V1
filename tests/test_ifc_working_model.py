import ifcopenshell
import numpy as np
import pytest
from fastapi.testclient import TestClient

from scanbim.app import create_app
from scanbim.ifc_export import inspect_ifc
from scanbim.model import SceneStore
from scanbim.tools import ToolRegistry


def scene_with_points(root, source="source", store=None):
    x, y = np.meshgrid(np.linspace(0, 3, 50), np.linspace(0, 2, 40))
    points = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    return (store or SceneStore(root)).create(points, np.zeros_like(points, dtype=np.uint8), "Test", source)


def planar(registry, scene, name, origin, width=2, height=2, thickness=.2):
    return registry.call(scene, "create_element", {"experiment_id": "fixture", "element": {
        "kind": "wall", "name": name, "origin": origin,
        "u": [1, 0, 0], "v": [0, 0, 1], "normal": [0, -1, 0],
        "width": width, "height": height, "thickness": thickness,
    }})


def box(origin, dimensions):
    vertices = np.asarray([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ]) * dimensions + origin
    faces = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
             [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
             [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    return {"vertices": vertices.tolist(), "faces": faces}


def test_intersection_is_removed_from_ifc_body_and_quantities(tmp_path):
    scene = scene_with_points(tmp_path)
    tools = ToolRegistry()
    scene.state["experiments"].append({"id": "fixture", "ok": True})
    host = planar(tools, scene, "A", [0, 0, 0])
    target = planar(tools, scene, "B", [1.5, 0, 0])
    before = tools.call(scene, "check_ifc")
    assert len(before["clashes"]) == 1
    assert before["clashes"][0]["volume_m3"] == pytest.approx(.2)
    tools.call(scene, "join_elements", {"element_id": host["id"], "target_id": target["id"]})
    tools.call(scene, "set_ifc_metadata", {"element_id": host["id"], "changes": {
        "type_name": "Bestandswand", "predefined_type": "STANDARD",
        "layers": [{"name": "Beton", "thickness_m": .15}, {"name": "Putz", "thickness_m": .05}],
    }})
    after = tools.call(scene, "check_ifc")
    assert not after["clashes"] and not after["issues"]
    assert after["elements"][host["id"]]["net_volume_m3"] == pytest.approx(.6)
    assert after["elements"][host["id"]]["global_id"] == before["elements"][host["id"]]["global_id"]
    tools.call(scene, "export_ifc")
    model = ifcopenshell.open(str(scene.folder / "reconstruction.ifc"))
    assert len(model.by_type("IfcRelConnectsElements")) == 1
    assert len(model.by_type("IfcMaterialLayerSetUsage")) == 1
    exported = model.by_guid(after["elements"][host["id"]]["global_id"])
    assert any(q.is_a("IfcQuantityVolume") and q.VolumeValue == pytest.approx(.6)
               for rel in exported.IsDefinedBy if rel.RelatingPropertyDefinition.is_a("IfcElementQuantity")
               for q in rel.RelatingPropertyDefinition.Quantities)


def test_opening_in_free_mesh_and_confirmed_filling(tmp_path):
    scene = scene_with_points(tmp_path)
    tools = ToolRegistry()
    scene.state["experiments"].extend([
        {"id": "wall", "ok": True, "result": box([0, 0, 0], [3, .25, 2.5])},
        {"id": "opening", "ok": True, "result": box([1, -.05, .4], [.8, .35, 1.2])},
        {"id": "window", "ok": True, "result": box([1.05, .08, .45], [.7, .08, 1.1])},
    ])
    wall = tools.call(scene, "create_mesh", {"experiment_id": "wall", "name": "Wand",
                     "semantic": "wall", "geometry_role": "solid"})
    opening = tools.call(scene, "create_opening", {"experiment_id": "opening", "element_id": wall["id"]})
    assert opening["removed_volume_m3"] == pytest.approx(.24)
    window = tools.call(scene, "create_mesh", {"experiment_id": "window", "semantic": "window"})
    tools.call(scene, "set_ifc_metadata", {"element_id": window["id"],
                                          "changes": {"fills": opening["opening_ref"]}})
    model, report = inspect_ifc(scene.state, check=True)
    assert not report["issues"] and not report["clashes"]
    assert len(model.by_type("IfcRelVoidsElement")) == 1
    assert len(model.by_type("IfcRelFillsElement")) == 1


def test_infrastructure_terrain_and_snapshot_api(tmp_path):
    app = create_app(tmp_path)
    scene = scene_with_points(tmp_path, store=app.state.store)
    tools = app.state.tools
    tools.call(scene, "configure_ifc", {"asset": "bridge", "map": {
        "crs": "EPSG:25832", "eastings": 600000, "northings": 5700000, "height": 120,
    }})
    terrain = tools.call(scene, "model_terrain", {"cell": .2})
    sequence = scene.events[-1]["seq"]
    tools.call(scene, "update_element", {"element_id": terrain["id"], "changes": {"name": "Neu"}})
    with TestClient(app, base_url="http://127.0.0.1") as client:
        base = f"/api/scenes/{scene.state['id']}"
        preview = client.get(base + f"/ifc-model?seq={sequence}").json()
        assert preview["schema"] == "IFC4X3_ADD2"
        shape = preview["elements"][terrain["id"]]
        assert shape["class"] == "IfcGeographicElement" and not shape["closed"]
        evidence = client.get(base + f"/ifc-evidence/{terrain['id']}?seq={sequence}").json()
        assert len(evidence["points"]) == len(evidence["source_indices"]) > 0
        export = tools.call(scene, "export_ifc")
        result = client.get(export["url"])
        assert result.status_code == 200
        model = ifcopenshell.file.from_string(result.text)
        assert model.by_guid(shape["global_id"]).Name == "Neu"
        assert len(model.by_type("IfcBridge")) == len(model.by_type("IfcMapConversion")) == 1


def test_method_needs_two_distinct_scan_sources(tmp_path):
    tools = ToolRegistry()
    method_id = None
    for source in ("first", "first", "second"):
        scene = scene_with_points(tmp_path, source=source)
        scene.state["experiments"].append({
            "id": "mesh", "name": "Ebene", "ok": True, "source_hash": source,
            "code": "result = {...}", "region": None, "result": {
                "vertices": [[0, 0, 0], [3, 0, 0], [3, 2, 0], [0, 2, 0]],
                "faces": [[0, 1, 2], [0, 2, 3]],
            }})
        result = tools.call(scene, "evaluate_method", {"experiment_id": "mesh", "method_id": method_id,
                            "scope": "Ebene Flächen", "tolerance": .03})
        method_id = result["method"]["id"]
        assert result["passed"]
        assert result["method"]["versions"][0]["status"] == ("validated" if source == "second" else "candidate")
