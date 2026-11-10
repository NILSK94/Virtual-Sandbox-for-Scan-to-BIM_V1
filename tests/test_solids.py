from __future__ import annotations

import ifcopenshell
import numpy as np
import pytest
from fastapi.testclient import TestClient

from scanbim.agents import AgentRunner
from scanbim.app import create_app
from scanbim.ifc_export import inspect_ifc
from scanbim.model import SceneStore
from scanbim.tools import ToolRegistry


def surface(tools, scene, name, z, semantic="slab", slope=0.):
    vertices = [[0, 0, z], [3, 0, z + 3*slope], [3, 2, z + 3*slope], [0, 2, z]]
    key = "experiment-" + name
    scene.state["experiments"].append({"id": key, "ok": True, "result": {
        "vertices": vertices, "faces": [[0, 1, 2], [0, 2, 3]],
    }})
    return tools.call(scene, "create_mesh", {"experiment_id": key, "name": name,
                      "semantic": semantic, "geometry_role": "surface"})


def create_scene(store):
    x, y = np.meshgrid(np.linspace(0, 3, 41), np.linspace(0, 2, 31))
    points = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    return store.create(points, np.zeros_like(points, dtype=np.uint8), "Volumentest", "scan-source")


def test_single_measured_face_becomes_native_ifc_extrusion(tmp_path):
    scene = create_scene(SceneStore(tmp_path))
    tools = ToolRegistry()
    patch = surface(tools, scene, "Messfläche", 0)
    with pytest.raises(ValueError, match="Begründung"):
        tools.call(scene, "solidify_mesh", {"element_id": patch["id"], "thickness_m": .3})
    assert scene.state["elements"][0]["geometry_role"] == "surface"
    report = tools.call(scene, "solidify_mesh", {"element_id": patch["id"], "direction": [0, 0, -1],
        "thickness_m": .3, "reason": "Einseitige Messung, Dicke als Arbeitsannahme."})
    assert report["representation"] == "SweptSolid"
    assert report["id"] == patch["id"]
    assert report["thickness_status"] == "assumed"
    assert report["net_volume_m3"] == pytest.approx(1.8)
    model, check = inspect_ifc(scene.state, check=True)
    assert len(model.by_type("IfcExtrudedAreaSolid")) == 1
    assert not check["issues"]


def test_pair_uses_variable_measured_depth_and_keeps_one_product(tmp_path):
    scene = create_scene(SceneStore(tmp_path))
    tools = ToolRegistry()
    top = surface(tools, scene, "Oberseite", .6, slope=.1)
    bottom = surface(tools, scene, "Unterseite", 0)
    result = tools.call(scene, "solidify_mesh", {"element_id": top["id"], "opposite_id": bottom["id"],
                        "direction": [0, 0, 1], "cell": .2})
    assert result["representation"] == "Brep"
    assert result["net_volume_m3"] == pytest.approx(4.5)
    assert result["thickness_max_m"] == pytest.approx(.9)
    assert result["thickness_min_m"] == pytest.approx(.6)
    assert result["source_ids"] == [top["id"], bottom["id"]]
    assert [element["id"] for element in scene.state["elements"]] == [top["id"]]
    assert len(scene.events[-2]["snapshot"]["elements"]) == 2


def test_thickness_probe_requires_distributed_point_support(tmp_path):
    tools = ToolRegistry()
    store = SceneStore(tmp_path)
    x,y = np.meshgrid(np.linspace(0,3,34), np.linspace(0,2,29))
    first = np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    second = first + [0,0,.48]
    points = np.vstack((first,second,np.tile([1,1,.8],(500,1))))
    scene = store.create(points,np.zeros_like(points,dtype=np.uint8),"Messung","source")
    patch = surface(tools,scene,"Sichtfläche",0)
    result = tools.call(scene,"probe_thickness",{"element_id":patch["id"],"direction":[0,0,1]})
    assert result["candidates"][0]["signed_depth_m"] == pytest.approx(.48)
    assert all(candidate["signed_depth_m"]!=.8 for candidate in result["candidates"])


def test_bim_view_and_export_share_bridge_hierarchy(tmp_path):
    app = create_app(tmp_path)
    scene = create_scene(app.state.store)
    tools = app.state.tools
    tools.call(scene, "configure_ifc", {"asset": "bridge", "asset_type": "GIRDER",
                    "parts": [{"id": "deck", "name": "Überbau", "type": "DECK"}]})
    top = surface(tools, scene, "Oberseite", .5)
    surface(tools, scene, "Unterseite", 0)
    surface(tools, scene, "Gelände", -.5, "terrain")
    surface(tools, scene, "Offene Sichtfläche", 2, "wall")
    tools.call(scene, "set_ifc_metadata", {"element_id": top["id"], "changes": {"part": "deck"}})
    runner = AgentRunner(tools)
    runner.start(scene, mode="vlm", max_steps=8, bim=True)
    runner.threads[scene.state["id"]].join(25)
    assert scene.state["agent"]["status"] == "partial"
    assert scene.state["completion"]["bim"]["file"] == "bim.ifc"
    with TestClient(app, base_url="http://127.0.0.1") as client:
        base = f"/api/scenes/{scene.state['id']}"
        preview = client.get(base + "/ifc-model?bim=true").json()
        assert preview["excluded_surfaces"] == 1
        assert len(preview["elements"]) == 2
        assert preview["elements"][top["id"]]["closed"]
        download = client.get(scene.state["completion"]["bim"]["url"])
        assert download.status_code == 200
    file = ifcopenshell.file.from_string(download.text)
    part, = file.by_type("IfcBridgePart")
    assert part.PredefinedType == "DECK"
    assert part.ContainsElements[0].RelatedElements[0].GlobalId == preview["elements"][top["id"]]["global_id"]
    assert not file.by_type("IfcWall")
