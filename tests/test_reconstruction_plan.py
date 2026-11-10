import json
import numpy as np
import pytest
from scanbim.model import SceneStore
from scanbim.tools import ToolRegistry
from scanbim.token_context import scene_context


def test_plan_is_generic_persistent_and_incrementally_editable(tmp_path):
    scene=SceneStore(tmp_path).create(np.eye(3),np.zeros((3,3),dtype=np.uint8),'Unknown asset','hash')
    tools=ToolRegistry()
    tools.call(scene,'update_reconstruction_plan',{'overview':'Geometrie erst untersuchen','items':[
        {'id':'curved','label':'Gekruemmte Tragstruktur','status':'measuring','method':'Eigener Profilfit'},
        {'id':'detail','label':'Schlankes Detail','status':'unexamined'}]})
    tools.call(scene,'update_reconstruction_plan',{'items':[{'id':'curved','label':'Gekruemmte Tragstruktur','status':'improving','next_action':'Abweichung lokal untersuchen'}]})
    restored=SceneStore(tmp_path).get(scene.state['id'])
    plan=json.loads(scene_context(restored,30))['reconstruction_plan']
    assert len(plan['items'])==2
    assert plan['items'][0]['method']=='Eigener Profilfit'
    assert plan['items'][0]['status']=='improving'
    assert plan['items'][1]['status']=='unexamined'
    with pytest.raises(ValueError):
        tools.call(restored,'update_reconstruction_plan',{'items':[{'id':'x','label':'X','status':'done','element_ids':['missing']}]})
    assert len(restored.state['reconstruction_plan']['items'])==2
