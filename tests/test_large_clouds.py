import hashlib,json,struct,time
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from scanbim.streaming import convert_to_cache,probe
from scanbim.model import SceneStore
from scanbim.app import create_app

def ply(path,points,binary=False,endian='<'):
    fmt=('binary_little_endian' if endian=='<' else 'binary_big_endian') if binary else 'ascii'
    head=f'ply\nformat {fmt} 1.0\nelement vertex {len(points)}\nproperty double x\nproperty double y\nproperty double z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'.encode()
    with path.open('wb') as f:
        f.write(head)
        for point in points:
            if binary:f.write(struct.pack(endian+'dddBBB',*point,100,170,200))
            else:f.write((' '.join(map(str,point))+' 100 170 200\n').encode())

@pytest.mark.parametrize('mode',['ascii','little','big'])
def test_chunked_ply_exact_coordinates_units_and_restart(tmp_path,mode):
    source=tmp_path/'Punktwolke Ä.ply';expected=np.arange(3*257,dtype=float).reshape(-1,3)
    ply(source,expected,mode!='ascii','>' if mode=='big' else '<')
    stages=[];staging=tmp_path/'data'/'.import-test'
    meta=convert_to_cache(source,staging,'mm',batch_size=31,progress=lambda p,phase,**k:stages.append(p))
    scene=SceneStore(tmp_path/'data').adopt_cache(staging,'test','Test',meta)
    assert isinstance(scene.points,np.memmap) and scene.points.mode=='r'
    assert np.allclose(scene.points,expected*.001)
    assert np.array_equal(scene.colors,np.tile([100,170,200],(len(expected),1)))
    assert meta['source_hash']==hashlib.sha256(source.read_bytes()).hexdigest()
    assert stages==sorted(stages) and len(meta['chunks'])==9
    restored=SceneStore(tmp_path/'data').get('test')
    assert isinstance(restored.points,np.memmap) and restored.public()['unit']=='m'
    assert restored.public()['input_unit']=='mm'
    p,c,ids,total=restored.sample({'min':[.1,.1,.1],'max':[.5,.5,.5]},limit=13)
    all_ids=np.flatnonzero(np.all((expected*.001>=.1)&(expected*.001<=.5),axis=1))
    assert total==len(all_ids) and len(p)==13
    assert np.array_equal(p,restored.points[ids])

def test_no_million_point_limit_and_bounded_preview(tmp_path):
    n=1_100_007;source=tmp_path/'million.ply'
    header=f'ply\nformat binary_little_endian 1.0\nelement vertex {n}\nproperty float x\nproperty float y\nproperty float z\nend_header\n'.encode()
    with source.open('wb') as f:f.write(header);f.truncate(len(header)+n*12)
    staging=tmp_path/'data'/'.import-million';meta=convert_to_cache(source,staging)
    scene=SceneStore(tmp_path/'data').adopt_cache(staging,'million','Million',meta)
    p,c,ids,total=scene.sample()
    assert total==n and len(p)==200000 and ids[-1]==n-1
    assert scene.public()['point_count']==n and scene.public()['storage']=='disk-mapped'
    assert np.array_equal(scene.points[ids],p)

def test_xyz_comments_count_and_source_stays_unchanged(tmp_path):
    source=tmp_path/'source.xyz';raw=b'# scan\n0 1 2\n\n3 4 5\n6 7 8\n';source.write_bytes(raw)
    meta=convert_to_cache(source,tmp_path/'cache',batch_size=2)
    assert meta['point_count']==3 and source.read_bytes()==raw
    assert np.load(tmp_path/'cache'/'points.npy').tolist()==[[0,1,2],[3,4,5],[6,7,8]]

def test_pcd_over_one_million_points(tmp_path):
    n=1100001;source=tmp_path/'large.pcd'
    header=f'VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {n}\nHEIGHT 1\nPOINTS {n}\nDATA binary\n'.encode()
    with source.open('wb') as f:f.write(header);f.truncate(len(header)+n*12)
    meta=convert_to_cache(source,tmp_path/'cache')
    assert meta['point_count']==n and np.load(tmp_path/'cache'/'points.npy',mmap_mode='r').shape==(n,3)

def test_cancel_before_allocation_and_free_disk_guard(tmp_path,monkeypatch):
    source=tmp_path/'source.xyz';source.write_text('0 0 0\n1 0 0\n0 1 0\n')
    with pytest.raises(InterruptedError):convert_to_cache(source,tmp_path/'cancel',cancel=lambda:True)
    assert not (tmp_path/'cancel'/'points.npy').exists()
    class Disk:free=1
    monkeypatch.setattr('scanbim.streaming.shutil.disk_usage',lambda _:Disk())
    with pytest.raises(ValueError,match='Cache-Speicher'):convert_to_cache(source,tmp_path/'no-space')
    assert source.exists()

def test_import_job_api_local_and_raw_upload(tmp_path):
    source=tmp_path/'source.ply';ply(source,np.arange(300,dtype=float).reshape(-1,3),True)
    app=create_app(tmp_path/'data');headers={'X-ScanBIM-Client':'studio'}
    def wait(client,jid):
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            job=client.get('/api/imports/'+jid).json()
            if job['status'] in ('completed','error','cancelled'):return job
            time.sleep(.01)
        pytest.fail('Import did not finish')
    with TestClient(app,base_url='http://127.0.0.1') as client:
        r=client.post('/api/imports/local',headers=headers,json={'path':str(source),'unit':'m'})
        assert r.status_code==200
        job=wait(client,r.json()['id']);assert job['status']=='completed',job
        state=client.get('/api/scenes/'+job['scene_id']).json()
        assert state['storage']=='disk-mapped' and state['point_count']==100
        job=client.post('/api/imports',headers=headers,json={'name':'uploaded.ply','size':source.stat().st_size}).json()
        r=client.put('/api/imports/'+job['id']+'/source',headers=headers,content=source.read_bytes())
        assert r.status_code==200
        assert wait(client,job['id'])['status']=='completed'
        waiting=client.post('/api/imports',headers=headers,json={'name':'cancel.ply','size':100}).json()
        assert client.post('/api/imports/'+waiting['id']+'/cancel',headers=headers).json()['status']=='cancelled'
        assert source.is_file() and not list((tmp_path/'data').glob('.import-*'))
