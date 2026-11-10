import json
from pathlib import Path
import zipfile
import pytest
from fastapi.testclient import TestClient
from scanbim.settings import initialize_settings,load_credentials
from scanbim.package import package_project
from scanbim.app import create_app
from scanbim import settings

pytestmark = pytest.mark.usefixtures('openai_provider')

def test_private_file_loaded_dynamically_never_returned_to_browser(tmp_path,monkeypatch):
    private=tmp_path/'outside-project'/'settings.toml'
    monkeypatch.setenv('SCANBIM_SETTINGS_FILE',str(private))
    monkeypatch.delenv('OPENAI_API_KEY',raising=False);monkeypatch.delenv('OPENAI_MODEL',raising=False)
    assert initialize_settings()==private
    assert not load_credentials().configured
    sentinel='unit-test-private-value'
    private.write_text(f'[openai]\napi_key="{sentinel}"\nmodel="test-model"\n',encoding='utf-8')
    assert load_credentials().configured and sentinel not in repr(load_credentials())
    with TestClient(create_app(tmp_path/'data'),base_url='http://127.0.0.1') as client:
        response=client.get('/api/config')
        assert response.json()['vlm_configured'] and sentinel not in response.text
        assert client.get('/settings.toml').status_code==404
    private.write_text('[openai]\napi_key="unterminated-private-value\n',encoding='utf-8')
    with pytest.raises(ValueError) as error:load_credentials()
    assert 'unterminated-private-value' not in str(error.value)

def test_credentials_excluded_from_release_and_accidental_source_key_blocked(tmp_path):
    root=tmp_path/'project';root.mkdir();(root/'scanbim').mkdir()
    (root/'README.md').write_text('Public documentation')
    secret='unit-test-private-sentinel'
    for name in ['.env','.env.local','settings.toml','keys.local.toml']:(root/name).write_text(secret)
    (root/'scanbim'/'settings.toml').write_text(secret)
    archive=tmp_path/'safe.zip';package_project(root,archive)
    with zipfile.ZipFile(archive) as z:
        assert z.namelist()==['ScanBIM-Studio/README.md']
        assert secret.encode() not in b''.join(z.read(n) for n in z.namelist())
    (root/'scanbim'/'mistake.py').write_text('key = "sk-'+('a'*30)+'"')
    with pytest.raises(ValueError,match='Paket nicht erstellt'):package_project(root,tmp_path/'blocked.zip')
    assert not (tmp_path/'blocked.zip').exists() and not (tmp_path/'blocked.zip.partial').exists()

def test_ui_save_recovers_bad_file_and_keeps_key_when_blank(tmp_path):
    private=settings.settings_path();private.parent.mkdir(parents=True)
    private.write_text('[openai]\napi_key = "broken',encoding='utf-8')
    sentinel='unit-test-private-sentinel'
    headers={'X-ScanBIM-Client':'studio'}
    with TestClient(create_app(tmp_path/'data'),base_url='http://127.0.0.1') as client:
        assert client.get('/api/settings').json()['error']
        result=client.post('/api/settings',headers=headers,json={'api_key':sentinel,'model':'test-model'})
        assert result.status_code==200 and result.json()['configured']
        assert sentinel not in result.text and 'no-store' in result.headers['cache-control']
        result=client.post('/api/settings',headers=headers,json={'api_key':'','model':'second-model'})
        assert result.json()['model']=='second-model' and load_credentials().api_key==sentinel
        assert client.get('/api/config').json()['vlm_configured']
        assert sentinel not in client.get('/api/config').text
        assert sentinel not in client.get('/api/settings').text
        assert not list(private.parent.glob('.settings-*'))

@pytest.mark.parametrize('payload',[
    {'api_key':'unit-secret','model':''},
    {'api_key':'unit-secret','model':42},
    {'api_key':'unit-secret','model':'bad model'},
    {'api_key':'unit-secret','model':'test-model','extra':'unit-secret'},
])
def test_ui_validation_never_echoes_key(tmp_path,payload):
    with TestClient(create_app(tmp_path/'data'),base_url='http://127.0.0.1') as client:
        result=client.post('/api/settings',headers={'X-ScanBIM-Client':'studio'},json=payload)
        assert result.status_code==400 and 'unit-secret' not in result.text
        denied=client.post('/api/settings',json={'api_key':'unit-secret','model':'test-model'})
        assert denied.status_code==403 and 'unit-secret' not in denied.text

@pytest.mark.parametrize('encoding',['utf-8-sig','utf-16'])
def test_notepad_encoding_and_file_priority(encoding,monkeypatch):
    private=settings.settings_path();private.parent.mkdir(parents=True)
    private.write_text('[openai]\napi_key="local-test-value"\nmodel="local-model"\n',encoding=encoding)
    monkeypatch.setenv('OPENAI_API_KEY','environment-test-value')
    assert load_credentials().api_key=='local-test-value'
    assert load_credentials().model=='local-model'

def test_model_lookup_and_connection_check_do_not_echo_provider_errors(tmp_path,monkeypatch):
    import httpx
    original_response=httpx.Response
    sentinel='unit-test-private-sentinel'
    requests=[];status=200
    class FakeClient:
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def get(self,url,headers):
            requests.append((url,headers))
            if status!=200:return original_response(status,json={'error':{'message':sentinel}})
            return original_response(200,json={'data':[{'id':'test-model'},{'id':'other-model'},{'id':'sk-'+('a'*30)}]})
        def post(self,url,headers,json,**kwargs):
            requests.append((url,headers))
            assert json["input"] == "Hallo" and json["store"] is False
            if status!=200:return original_response(status,json={'error':{'message':sentinel}})
            return original_response(200,json={'status':'completed'})
    monkeypatch.setattr(settings.httpx,'Client',FakeClient)
    headers={'X-ScanBIM-Client':'studio'}
    with TestClient(create_app(tmp_path/'data'),base_url='http://127.0.0.1') as client:
        client.post('/api/settings',headers=headers,json={'api_key':sentinel,'model':'test-model'})
        before=settings.settings_path().read_bytes()
        result=client.post('/api/settings/models',headers=headers,json={'api_key':''})
        assert result.json()['models']==['other-model','test-model'] and sentinel not in result.text
        result=client.post('/api/settings/test',headers=headers,json={'api_key':'','model':'test-model'})
        assert result.json()['ok'] and sentinel not in result.text
        assert settings.settings_path().read_bytes()==before
        for code in (401,403,404,429,500):
            status=code
            result=client.post('/api/settings/test',headers=headers,json={'api_key':'','model':'test-model'})
            assert not result.json()['ok'] and sentinel not in result.text
        assert {url for url,_ in requests} == {'https://api.openai.com/v1/models',
            'https://api.openai.com/v1/responses'}
        assert all(header['Authorization']=='Bearer '+sentinel for _,header in requests)
