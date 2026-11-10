import json
import numpy as np
import pytest
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape
import ifcopenshell.validate
from fastapi.testclient import TestClient
from scanbim.cloud import demo_room
from scanbim.model import SceneStore
from scanbim.tools import ToolRegistry
from scanbim.agents import AgentRunner
from scanbim.app import create_app

@pytest.fixture
def scene(tmp_path):
    p,c=demo_room();return SceneStore(tmp_path).create(p,c,"Test-Raum","test-source-sha")

def test_full_pipeline_ifc_geometry_and_persistence(scene):
    runner=AgentRunner();runner.start(scene)
    runner.threads[scene.state['id']].join(30)
    assert scene.state['agent']['status']=='completed',scene.state['agent']
    assert len(scene.state['elements'])==6
    assert sum(len(e.get('openings',[])) for e in scene.state['elements'])==2
    q=scene.state['quality'];assert q['within_tolerance']>.97 and q['rms_m']<.02
    f=ifcopenshell.open(str(scene.folder/'reconstruction.ifc'))
    assert f.schema=='IFC4' and len(f.by_type('IfcWall'))==4
    assert len(f.by_type('IfcOpeningElement'))==2 and len(f.by_type('IfcRelVoidsElement'))==2
    assert len(f.by_type('IfcBuildingStorey'))==1
    assert not f.by_type('IfcDoor') and not f.by_type('IfcWindow')
    settings=ifcopenshell.geom.settings()
    wall=next(w for w in f.by_type('IfcWall') if w.HasOpenings)
    shape=ifcopenshell.geom.create_shape(settings,wall)
    assert len(shape.geometry.verts)>0
    e=next(e for e in scene.state['elements'] if e['name']==wall.Name)
    expected=(e['width']*e['height']-sum(o['width']*o['height'] for o in e['openings']))*e['thickness']
    assert ifcopenshell.util.shape.get_volume(shape.geometry)==pytest.approx(expected,rel=.01)
    logger=ifcopenshell.validate.json_logger();ifcopenshell.validate.validate(f,logger)
    assert not logger.statements,logger.statements
    loaded=SceneStore(scene.folder.parent).get(scene.state['id'])
    assert len(loaded.state['elements'])==6 and len(loaded.events)==len(scene.events)
    assert loaded.events[-1]['snapshot']['elements']==scene.state['elements']
    assert all(e['parameter_status']['thickness']=='assumed' for e in loaded.state['elements'])

def test_manual_lock_and_rejected_edit_is_atomic(scene):
    tools=ToolRegistry();planes=tools.call(scene,'detect_planes')['planes'];e=tools.call(scene,'model_plane',{'plane_id':planes[2]['id']})
    tools.call(scene,'update_element',{'element_id':e['id'],'changes':{'thickness':.22}})
    e=next(item for item in scene.state['elements'] if item['id']==e['id'])
    assert e['locked'] and e['parameter_status']['thickness']=='manual'
    with pytest.raises(ValueError,match='gesperrt'):tools.call(scene,'delete_element',{'element_id':e['id']},actor='agent')
    before=e['width']
    with pytest.raises(ValueError):tools.call(scene,'update_element',{'element_id':e['id'],'changes':{'width':-1}})
    assert e['width']==before and scene.events[-1]['status']=='error'

def test_demo_budget_and_repeat_preserve_model(scene):
    runner=AgentRunner();runner.start(scene,max_steps=2)
    runner.threads[scene.state['id']].join(15)
    assert scene.state['agent']['step']==2 and scene.state['agent']['status']=='stopped'
    runner.start(scene);runner.threads[scene.state['id']].join(20)
    assert len(scene.state['elements'])==6
    ids=[e['id'] for e in scene.state['elements']]
    runner.start(scene);runner.threads[scene.state['id']].join(20)
    assert [e['id'] for e in scene.state['elements']]==ids

def test_experimental_mesh_export(scene):
    tools=ToolRegistry()
    scene.state['experiments'].append({'id':'test-evidence','ok':True,'result':{'measured':True}})
    mesh=tools.call(scene,'create_mesh',{'experiment_id':'test-evidence','name':'Dachhypothese','semantic':'roof',
        'vertices':[[0,0,3],[0,4,3],[2,4,4],[2,0,4]],'faces':[[0,1,2],[0,2,3]],
        'parameters':{'pitch_degrees':26.565},'uncertainty':'Nur beobachtete Fläche; kein Volumen.'})
    assert mesh['evidence']['experiment_id']=='test-evidence'
    tools.call(scene,'export_ifc')
    f=ifcopenshell.open(str(scene.folder/'reconstruction.ifc'))
    roof=f.by_type('IfcRoof')[0]
    shape=ifcopenshell.geom.create_shape(ifcopenshell.geom.settings(),roof)
    assert len(shape.geometry.faces)==6

def test_api_import_actions_and_cross_origin(tmp_path):
    app=create_app(tmp_path);client=TestClient(app,base_url='http://127.0.0.1')
    assert client.post('/api/demo').status_code==403
    headers={'X-ScanBIM-Client':'studio'}
    assert client.post('/api/demo',headers={**headers,'Origin':'https://outside.example'}).status_code==403
    s=client.post('/api/demo',headers=headers).json();sid=s['id']
    result=client.post(f'/api/scenes/{sid}/action',headers=headers,json={'tool':'request_view','arguments':{'section':{'origin':[0,0,1.5],'normal':[0,0,1],'thickness':.15}}})
    assert result.status_code==200
    view=result.json();assert view['section']['thickness']==.15 and view['source_hash']==s['source_hash']
    assert client.get(view['url']).headers['content-type']=='image/png'
    assert client.get(f'/api/scenes/{sid}/artifacts/state.json').status_code==404
    bad=client.post('/api/upload',headers=headers,files={'file':('bad.xyz',b'0 0 0\n1 NaN 0\n0 1 2')},data={'unit':'m'})
    assert bad.status_code==400
    points=client.get(f'/api/scenes/{sid}/points').json()
    assert len(points['source_indices'])*3==len(points['points'])
