import json
import numpy as np
import pytest

from scanbim.model import SceneStore
from scanbim.tools import ToolRegistry
from scanbim import tool_memory


def make_scene(store, source):
    return store.create(np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),
                        np.full((3,3),200,dtype=np.uint8), source, source)


def test_tool_versions_survive_restart_and_transfer_between_scenes(tmp_path):
    store=SceneStore(tmp_path)
    scene=make_scene(store,"source-a")
    tools=ToolRegistry()
    first=tools.call(scene,"define_tool",{
        "name":"Profil","description":"Querschnitte für schlanke Bauteile.",
        "code":"result={'value':1}","parameter_help":"station_m"})
    original_version=first["memory_version"]
    tools.call(scene,"define_tool",{
        "tool_id":first["id"],"name":"Profil","description":"Robuste Querschnitte.",
        "code":"result={'value':2}","parameter_help":"station_m"})
    second=make_scene(SceneStore(tmp_path),"source-b")
    memory=tools.call(second,"list_tool_memory",{"query":"Profil"})
    assert len(memory["tools"])==2
    recalled=tools.call(second,"recall_tool",{"memory_id":first["memory_id"],"version":original_version})
    assert recalled["code"]=="result={'value':1}"
    assert tools.call(second,"recall_tool",{
        "memory_id":first["memory_id"],"version":original_version})["id"]==recalled["id"]
    assert len(second.state["custom_tools"])==1


def test_attempts_and_geometric_fit_are_recorded_separately(tmp_path,monkeypatch):
    scene=make_scene(SceneStore(tmp_path),"source-a")
    tools=ToolRegistry()
    tool=tools.call(scene,"define_tool",{"name":"Test","description":"Testbereich","code":"result={}"})
    result={"id":"experiment-1","ok":True,"source_hash":"source-a","region":None,
            "result":{"vertices":scene.points.tolist(),"faces":[[0,1,2]]},
            "data_info":{"parameters":{"threshold":.01}},"duration_s":.1}
    monkeypatch.setattr(tool_memory,"run_experiment",lambda *args:result)
    tested=tools.call(scene,"call_custom_tool",{"tool_id":tool["id"]})
    assert tested["tool_memory_trial"]["score_at_30mm"]["coverage"]==1
    def unavailable(*args):
        raise ValueError("Docker unavailable")
    monkeypatch.setattr(tool_memory,"run_experiment",unavailable)
    with pytest.raises(ValueError,match="Docker unavailable"):
        tools.call(scene,"call_custom_tool",{"tool_id":tool["id"]})
    memory=tools.call(scene,"list_tool_memory")
    summary=memory["tools"][0]
    assert summary["successful_runs"]==1 and summary["failed_runs"]==1
    assert summary["recent_trials"][-1]["error"]=="Docker unavailable"
    assert summary["recent_trials"][-1]["score_at_30mm"] is None
    assert summary["tested_sources"]==1


def test_free_experiments_are_also_saved(tmp_path,monkeypatch):
    scene=make_scene(SceneStore(tmp_path),"source-a")
    tools=ToolRegistry()
    monkeypatch.setattr(tool_memory,"run_experiment",lambda *args:{
        "id":"free-1","ok":False,"error":"fit failed","region":None,"data_info":{}})
    result=tools.call(scene,"run_experiment",{"name":"First try","code":"result={}"})
    assert result["tool_memory"]["memory_id"]
    assert tools.call(scene,"list_tool_memory")["tools"][0]["failed_runs"]==1
