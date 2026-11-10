import numpy as np
import pytest

from scanbim.model import SceneStore
from scanbim.tools import ToolRegistry, CATALOG


def scene_with_wall(tmp_path, offset):
    y,z=np.meshgrid(np.linspace(0,2,21),np.linspace(0,2,21))
    points=np.c_[np.zeros(y.size),y.ravel(),z.ravel()]
    scene=SceneStore(tmp_path).create(points,np.full(points.shape,200,dtype=np.uint8),"fit","source")
    scene.state["elements"].append({
        "id":"wall","name":"Wand","kind":"wall","origin":[offset,0,0],
        "u":[0,1,0],"v":[0,0,1],"normal":[1,0,0],"width":2.,"height":2.,"thickness":.18,
        "depth_direction":-1,"openings":[],"locked":False,
        "parameter_status":{"geometry":"measured"},"evidence":{"method":"synthetic"},
    })
    scene.state["experiments"].append({"id":"fit-exp","ok":True})
    return scene


def test_fit_comparison_detects_offset_and_uses_identical_points(tmp_path):
    scene=scene_with_wall(tmp_path,.08)
    tools=ToolRegistry()
    baseline=tools.call(scene,"inspect_fit",{"element_id":"wall"})
    assert baseline["metrics"]["rms_m"]==pytest.approx(.08)
    assert baseline["covers_element_bounds"] and baseline["worst_points"]
    tools.call(scene,"update_element",{"element_id":"wall","changes":{"origin":[0,0,0]},
        "evidence":{"experiment_id":"fit-exp"}},actor="agent")
    improved=tools.call(scene,"inspect_fit",{"element_id":"wall","baseline_id":baseline["id"]})
    assert improved["metrics"]["rms_m"]<1e-9
    assert improved["comparison"]["improved"]
    assert improved["sample_signature"]==baseline["sample_signature"]
    with pytest.raises(ValueError,match="dieselbe Prüfregion"):
        tools.call(scene,"inspect_fit",{"element_id":"wall","baseline_id":baseline["id"],
            "region":{"min":[-.1,0,0],"max":[.1,1,1]}})
    reopened=SceneStore(tmp_path).get(scene.state["id"])
    assert len(reopened.state["fit_reports"])==2


def test_vlm_review_does_not_require_optional_component_diagnostics(tmp_path):
    scene=scene_with_wall(tmp_path,0)
    scene.state["agent"]["mode"]="vlm"
    tools=ToolRegistry()
    args={"assessment":"Messwerte geprüft","unresolved":[],"max_rms_m":.01,"min_coverage":.99}
    tools.call(scene,"validate",{"region":None})
    assert tools.call(scene,"review_model",args)["accepted"]
    assert not scene.state["fit_reports"]
    partial=tools.call(scene,"inspect_fit",{"element_id":"wall",
        "region":{"min":[-.1,.2,.2],"max":[.1,1.8,1.8]}})
    assert not partial["covers_element_bounds"]
    assert tools.call(scene,"review_model",args)["accepted"]
    tools.call(scene,"inspect_fit",{"element_id":"wall"})
    assert tools.call(scene,"review_model",args)["accepted"]
    tools.call(scene,"update_element",{"element_id":"wall","changes":{"origin":[.06,0,0]},
        "evidence":{"experiment_id":"fit-exp"}},actor="agent")
    tools.call(scene,"validate",{"region":None})
    report=tools.call(scene,"review_model",args)
    assert not report["accepted"]
    assert not any("lokale Fit" in issue for issue in report["issues"])


def test_agent_catalog_has_no_secondary_model():
    assert "classify_planes" not in CATALOG
    assert "inspect_fit" in CATALOG


def test_orientation_refinement_requires_valid_frame_and_evidence(tmp_path):
    scene=scene_with_wall(tmp_path,0)
    tools=ToolRegistry()
    angle=.03
    change={"u":[-np.sin(angle),np.cos(angle),0],"v":[0,0,1],
            "normal":[np.cos(angle),np.sin(angle),0]}
    with pytest.raises(ValueError,match="Messnachweis"):
        tools.call(scene,"update_element",{"element_id":"wall","changes":change},actor="agent")
    tools.call(scene,"update_element",{"element_id":"wall","changes":change,
        "evidence":{"experiment_id":"fit-exp"}},actor="agent")
    with pytest.raises(ValueError,match="orthonormal"):
        tools.call(scene,"update_element",{"element_id":"wall","changes":{"normal":[0,0,1]},
            "evidence":{"experiment_id":"fit-exp"}},actor="agent")
    assert scene.state["elements"][0]["normal"]==change["normal"]
