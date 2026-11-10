import json
import copy
import numpy as np
import pytest
from scanbim.agents import AgentRunner
from scanbim.cloud import demo_room
from scanbim.model import SceneStore
from scanbim import agents

def test_vlm_receives_requested_image_and_chooses_own_code(tmp_path,monkeypatch,openai_provider):
    p,c=demo_room();s=SceneStore(tmp_path).create(p,c,'Room','hash')
    monkeypatch.setenv('OPENAI_API_KEY','test-key');monkeypatch.setenv('OPENAI_MODEL','test-model')
    real_client=agents.httpx.Client;requests=[]
    choices=[('request_view',{'position':[3,-8,2],'target':[3,2,1.5]}),
             ('run_experiment',{'name':'Own ridge method','code':"result={'custom': 42}"}),
             ('finish',{'summary':'Eigener Versuch ausgewertet; weiterer Messbedarf bleibt offen.'})]
    class Reply:
        status_code=200
        def __init__(self,data):self.data=data
        def json(self):return self.data
    class MockClient:
        def __init__(self,*a,**k):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def post(self,url,headers,json):
            requests.append(copy.deepcopy(json))
            name,args=choices[len(requests)-1]
            return Reply({'output':[{'type':'function_call','name':'act','call_id':str(len(requests)),
                'arguments':__import__('json').dumps({'tool':name,'arguments_json':__import__('json').dumps(args),'rationale':'Neue Messhypothese prüfen.'})}]})
    monkeypatch.setattr(agents.httpx,'Client',MockClient)
    runner=AgentRunner()
    original=runner.registry.call
    def dispatch(scene,name,args,actor):
        if name=='run_experiment': return {'ok':True,'result':{'custom':42}}
        return original(scene,name,args,actor)
    monkeypatch.setattr(runner.registry,'call',dispatch)
    summary=runner._vlm(s,5)
    assert 'Eigener Versuch' in summary
    assert any(isinstance(m.get('content'),list) and any(c.get('type')=='input_image' for c in m['content'])
               for m in requests[0]['input'])
    assert 'tool_memory' in json.loads(requests[0]['input'][0]['content'])
    assert any(isinstance(m.get('content'),list) and any(c.get('type')=='input_image' for c in m['content']) for m in requests[1]['input'])
    names=[json.loads(m['arguments'])['tool'] for m in requests[-1]['input'] if m.get('type')=='function_call']
    assert 'run_experiment' in names and 'detect_planes' not in names
    assert all(r['store'] is False for r in requests)

def test_trusted_shipped_roof_algorithm_tracks_variable_ridge():
    # This is a fixed reviewed example shipped with the app, not generated code.
    from pathlib import Path
    rng=np.random.default_rng(4)
    x,y=np.meshgrid(np.linspace(0,12,150),np.linspace(-2.5,2.5,90))
    z=4+.12*np.sin(x*.4)-np.abs(y)*(.5+.05*np.sin(x*.2))
    points=np.c_[x.ravel(),y.ravel(),z.ravel()]+rng.normal(0,.004,(x.size,3))
    scope={'points':points}
    source=(Path(__file__).resolve().parents[1]/'examples'/'roof_experiment.py').read_text(encoding='utf-8')
    exec(compile(source,'trusted_roof_example','exec'),scope)
    result=scope['result']
    assert result['rms_m']<.015
    assert len(result['sections'])==12 and len(result['faces'])==44
    assert np.ptp([s['ridge_z_m'] for s in result['sections']])>.12
    assert all(20<s['left_pitch_degrees']<33 for s in result['sections'])
    from scanbim.geometry import triangle_distance
    distances=triangle_distance(points,np.array(result['vertices']),np.array(result['faces']))
    assert np.sqrt(np.mean(distances**2))<.02


def test_default_agent_sends_images_and_tool_results_only_to_ollama(tmp_path, monkeypatch):
    import httpx
    from scanbim.ollama import BASE_URL, DEFAULT_MODEL

    monkeypatch.setenv('OPENAI_API_KEY', 'legacy-key')
    monkeypatch.setenv('OPENAI_MODEL', 'legacy-model')
    points, colors = demo_room()
    scene = SceneStore(tmp_path).create(points, colors, 'Room', 'hash')
    requests = []

    def handle(request):
        assert str(request.url).startswith(BASE_URL + '/')
        assert 'authorization' not in request.headers
        payload = json.loads(request.content)
        assert payload['model'] == DEFAULT_MODEL
        if request.url.path == '/api/show':
            return httpx.Response(200, json={'capabilities': ['vision', 'tools']})
        assert request.url.path == '/api/chat'
        requests.append(payload)
        tool = 'request_view' if len(requests) == 1 else 'finish'
        args = {} if tool == 'request_view' else {'summary': 'Lokale Ansicht ausgewertet.'}
        return httpx.Response(200, json={'message': {'tool_calls': [{'function': {
            'name': 'act', 'arguments': {'tool': tool, 'arguments_json': json.dumps(args), 'rationale': 'Lokale Messung'}}}]}})

    real_client = httpx.Client

    def client(**kwargs):
        assert kwargs['trust_env'] is False
        return real_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(agents.httpx, 'Client', client)
    assert 'Lokale Ansicht ausgewertet' in AgentRunner()._vlm(scene, 4)
    assert any(message.get('images') for message in requests[0]['messages'])
    messages = requests[1]['messages']
    assert any(message.get('images') for message in messages)
    assert any(message['role'] == 'tool' and message['tool_name'] == 'act' for message in messages)


def test_agent_recovers_from_missing_tool_call(tmp_path, monkeypatch, openai_provider):
    import httpx
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    monkeypatch.setenv('OPENAI_MODEL', 'test-model')
    points, colors = demo_room()
    scene = SceneStore(tmp_path).create(points, colors, 'Recovery', 'hash')
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(200, json={'output': []})
        return httpx.Response(200, json={'output': [{'type': 'function_call', 'name': 'act', 'call_id': 'recovered',
            'arguments': json.dumps({'tool': 'finish', 'arguments_json': json.dumps({'status': 'partial', 'summary': 'Recovery OK'}), 'rationale': 'Keine weitere Evidenz'})}]})
    real_client = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    assert 'Recovery OK' in AgentRunner()._vlm(scene, 4)
    assert len(requests) == 2


def test_repeated_read_does_not_abort_before_geometry_action(tmp_path, monkeypatch, openai_provider):
    import httpx
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    monkeypatch.setenv('OPENAI_MODEL', 'test-model')
    points, colors = demo_room()
    scene = SceneStore(tmp_path).create(points, colors, 'Read recovery', 'hash')
    scene.state['experiments'].append({'id':'measured', 'name':'Existing evidence', 'ok':True, 'code':'result={}', 'result':{'height':3.0}})
    requests=[]
    def handle(request):
        requests.append(json.loads(request.content))
        tool='read_experiment' if len(requests)<=4 else 'finish'
        args={'experiment_id':'measured'} if tool=='read_experiment' else {'status':'partial','summary':'Reading recovered'}
        return httpx.Response(200,json={'output':[{'type':'function_call','name':'act','call_id':str(len(requests)),
            'arguments':json.dumps({'tool':tool,'arguments_json':json.dumps(args),'rationale':'Read evidence'})}]})
    real=httpx.Client
    monkeypatch.setattr(httpx,'Client',lambda **kw:real(transport=httpx.MockTransport(handle),**kw))
    assert 'Reading recovered' in AgentRunner()._vlm(scene,7)
    assert len(requests)==5
    assert len(scene.state['agent']['result_memory'])==1
