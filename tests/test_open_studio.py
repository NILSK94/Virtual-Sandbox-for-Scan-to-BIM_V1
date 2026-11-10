import hashlib
import json
import os
import threading
import time
import numpy as np
import pytest
import ifcopenshell
from fastapi.testclient import TestClient
from scanbim.app import create_app
from scanbim.model import SceneStore
from scanbim.reset import preview,reset_scene,LISTS
from scanbim.tools import ToolRegistry
from scanbim.streaming import convert_to_cache

def make_scene(tmp_path):
    p=np.array([[0.,0,0],[2,0,0],[0,2,0],[.5,.5,0]])
    return SceneStore(tmp_path/'data').create(p,np.full(p.shape,120,np.uint8),'Freie Struktur','sourcehash')

def test_reset_removes_derived_files_and_old_snapshots_but_preserves_sources(tmp_path):
    s=make_scene(tmp_path);t=ToolRegistry()
    original=hashlib.sha256((s.folder/'source.npz').read_bytes()).hexdigest()
    t.call(s,'request_view',{})
    t.call(s,'define_tool',{'name':'Test','description':'Freier Code','code':'result={}'})
    t.call(s,'remember',{'observation':'Alt','uncertainty':'Alte Annahme'})
    (s.folder/'experiment-owned').mkdir();(s.folder/'experiment-owned'/'program.py').write_text('alt')
    (s.folder/'reconstruction.ifc').write_text('alt')
    revision=preview(s)['revision']
    with pytest.raises(ValueError):reset_scene(s,revision,False)
    with pytest.raises(ValueError):reset_scene(s,revision-1,True)
    s.state['agent']['status']='paused'
    with pytest.raises(ValueError):reset_scene(s,revision,True)
    s.state['agent']['status']='idle'
    result=reset_scene(s,revision,True)
    assert result['reset'] and result['source_preserved']
    assert all(s.state[k]==[] for k in LISTS)
    assert s.state['generation']==1 and s.state['quality'] is None
    assert hashlib.sha256((s.folder/'source.npz').read_bytes()).hexdigest()==original
    assert not list(s.folder.glob('view-*.png')) and not list(s.folder.glob('experiment-*'))
    assert not (s.folder/'reconstruction.ifc').exists() and not list(s.folder.glob('.reset-*'))
    assert len(s.events)==1 and s.events[0]['tool']=='reset'
    restored=SceneStore(s.folder.parent).get(s.state['id'])
    assert len(restored.events)==1 and restored.state['generation']==1
    assert np.array_equal(restored.points,s.points)

def test_reset_rolls_back_if_state_commit_fails(tmp_path,monkeypatch):
    s=make_scene(tmp_path);t=ToolRegistry();view=t.call(s,'request_view',{})
    before=(s.folder/'state.json').read_bytes();journal=(s.folder/'events.jsonl').read_bytes()
    def fail():raise OSError('simuliert')
    monkeypatch.setattr(s,'save',fail)
    with pytest.raises(OSError):reset_scene(s,s.state['revision'],True)
    assert (s.folder/'state.json').read_bytes()==before
    assert (s.folder/'events.jsonl').read_bytes()==journal
    assert (s.folder/view['file']).exists() and s.state['generation']==0

def test_reset_endpoint_requires_local_client_and_current_confirmation(tmp_path):
    app=create_app(tmp_path/'scenes');s=app.state.store.create(np.eye(3),np.full((3,3),120,np.uint8),'Test','hash')
    sid=s.state['id'];headers={'X-ScanBIM-Client':'studio'}
    with TestClient(app,base_url='http://127.0.0.1') as c:
        info=c.get(f'/api/scenes/{sid}/reset-preview').json()
        assert c.post(f'/api/scenes/{sid}/reset',json={'expected_revision':info['revision'],'confirmed':True}).status_code==403
        assert c.post(f'/api/scenes/{sid}/reset',headers=headers,json={'expected_revision':info['revision'],'confirmed':False}).status_code==400
        assert c.post(f'/api/scenes/{sid}/reset',headers=headers,json={'expected_revision':info['revision'],'confirmed':True}).json()['reset']

def test_activity_can_be_read_while_long_tool_holds_geometry_lock(tmp_path,monkeypatch):
    s=make_scene(tmp_path);t=ToolRegistry();entered=threading.Event();release=threading.Event()
    def execute(*a):entered.set();release.wait(3);return {'ok':True}
    monkeypatch.setattr(t,'_execute',execute)
    thread=threading.Thread(target=t.call,args=(s,'run_experiment',{}));thread.start()
    try:
        assert entered.wait(2);start=time.monotonic()
        assert s.activity_snapshot()['status']=='working'
        assert time.monotonic()-start<.2
    finally:release.set();thread.join(4)
    assert s.activity_snapshot()['status']=='done'

def test_finish_distinguishes_partial_and_current_accepted_model(tmp_path):
    s=make_scene(tmp_path);t=ToolRegistry()
    with pytest.raises(ValueError):t.call(s,'finish',{'status':'complete','summary':'Fertig'})
    assert t.call(s,'finish',{'status':'partial','summary':'Noch keine Rekonstruktion'})['status']=='partial'
    s.state['experiments'].append({'id':'e','ok':True})
    e=t.call(s,'create_mesh',{'experiment_id':'e','name':'Freie Schale','semantic':'shell',
        'vertices':[[0,0,0],[2,0,0],[0,2,0]],'faces':[[0,1,2]],'uncertainty':'Test'})
    t.call(s,'validate',{'region':None})
    review=t.call(s,'review_model',{'assessment':'Gemeinsame Fläche geprüft','unresolved':[], 'max_rms_m':.01,'min_coverage':.99})
    assert review['accepted']
    t.call(s,'update_element',{'element_id':e['id'],'changes':{'name':'Revidiert'}})
    with pytest.raises(ValueError):t.call(s,'finish',{'status':'complete','summary':'Fertig'})
    t.call(s,'validate',{'region':None})
    t.call(s,'review_model',{'assessment':'Neu geprüft','unresolved':[],'max_rms_m':.01,'min_coverage':.99})
    complete=t.call(s,'finish',{'status':'complete','summary':'Gemessene Schale geprüft'})
    assert complete['ifc']['schema']=='IFC4'
    f=ifcopenshell.open(str(s.folder/'reconstruction.ifc'))
    assert len(f.by_type('IfcBuildingElementProxy'))==1
    assert not f.by_type('IfcWall')
    assert s.events[-1]['snapshot']['activity']['tool']=='finish'

def test_duplicate_geometry_and_unresolved_regions_block_complete(tmp_path):
    s=make_scene(tmp_path);t=ToolRegistry();s.state['experiments'].append({'id':'e','ok':True})
    args={'experiment_id':'e','name':'A','semantic':'vault','vertices':[[0,0,0],[2,0,0],[0,2,0]],'faces':[[0,1,2]]}
    t.call(s,'create_mesh',args);t.call(s,'create_mesh',args);t.call(s,'validate',{'region':None})
    review=t.call(s,'review_model',{'assessment':'Prüfung','unresolved':['Anschluss offen'],'max_rms_m':.01,'min_coverage':.9})
    assert not review['accepted'] and any('doppelte' in v for v in review['issues'])

def test_rgb_source_and_material_hypothesis_provenance(tmp_path):
    source=tmp_path/'scan.xyz';source.write_text('0 0 0 255 0 0\n2 0 0 255 0 0\n0 2 0 255 0 0\n')
    store=SceneStore(tmp_path/'data');meta=convert_to_cache(source,store.root/'.import-rgb')
    s=store.adopt_cache(store.root/'.import-rgb','rgb','RGB',meta);t=ToolRegistry()
    assert t.call(s,'get_points',{})['color_source']=='source_rgb'
    s.state['experiments'].append({'id':'e','ok':True})
    e=t.call(s,'create_mesh',{'experiment_id':'e','name':'Oberfläche','vertices':s.points.tolist(),'faces':[[0,1,2]]})
    material={'element_id':e['id'],'experiment_id':'e','label':'Ziegel?','confidence':.6,'uncertainty':'Nur Farbe beobachtet'}
    assert t.call(s,'annotate_material',material)['status']=='hypothesis'
    t.call(s,'export_ifc',{})
    f=ifcopenshell.open(str(s.folder/'reconstruction.ifc'))
    assert any(p.Name=='ScanBIM_MaterialHypothesis' for p in f.by_type('IfcPropertySet'))
    source.write_text('0 0 0\n2 0 0\n0 2 0\n')
    fallback=convert_to_cache(source,tmp_path/'fallback')
    assert fallback['color_source']=='display_fallback'
    s.cloud_metadata['color_source']='display_fallback'
    with pytest.raises(ValueError,match='Quellfarben'):t.call(s,'annotate_material',material)

@pytest.mark.skipif(os.getenv('SCANBIM_TEST_DOCKER')!='1',reason='Docker integration opt-in')
@pytest.mark.parametrize('semantic',['vault','bridge','industrial_shell'])
def test_custom_algorithm_reused_after_restart_for_different_structures(tmp_path,semantic):
    # Fixed integration fixture: no prescribed planes or room workflow.
    x,y=np.meshgrid(np.linspace(-2,2,7),np.linspace(-1,1,6))
    z={'vault':2+np.sqrt(5-x*x),'bridge':.4*x+.2*y,'industrial_shell':2+.2*x*y}[semantic]
    points=np.c_[x.ravel(),y.ravel(),z.ravel()];store=SceneStore(tmp_path/'data')
    s=store.create(points,np.full(points.shape,140,np.uint8),semantic,'hash');t=ToolRegistry()
    code="from scipy.spatial import Delaunay\ntri=Delaunay(points[:,:2])\nresult={'vertices':points.tolist(),'faces':tri.simplices.tolist(),'parameter':parameters['name'],'overlays':[{'kind':'points','vertices':points.tolist(),'colors':colors.tolist()}]}"
    tool=t.call(s,'define_tool',{'name':'Eigene Flächentriangulation','description':'Für diese Scans XY-Projektion geeignet','code':code})
    s=SceneStore(store.root).get(s.state['id'])
    exp=t.call(s,'call_custom_tool',{'tool_id':tool['id'],'parameters':{'name':semantic}})
    assert exp['ok'],exp
    assert exp['result']['parameter']==semantic and s.state['overlays'][0]['colors']
    result=exp['result'];t.call(s,'create_mesh',{'experiment_id':exp['id'],'name':semantic,'semantic':semantic,'vertices':result['vertices'],'faces':result['faces']})
    decision=t.call(s,'record_decision',{'decision':'Freie Oberfläche verwenden','reason':'Keine Ebenen vorausgesetzt','evidence':[exp['id']],'confidence':.9})
    assert decision['evidence']==[exp['id']] and not s.state['planes']
    t.call(s,'validate',{'region':None})
    assert t.call(s,'review_model',{'assessment':'Alle Quellpunkte numerisch gestützt','unresolved':[],'max_rms_m':.001,'min_coverage':.99})['accepted']
    assert t.call(s,'finish',{'status':'complete','summary':'Fixture geprüft'})['ifc']['schema']=='IFC4'
