import os
import pytest
from scanbim import experiments
from scanbim.cloud import demo_room
from scanbim.model import SceneStore

def test_no_docker_never_executes_generated_code_on_host(tmp_path,monkeypatch):
    p,c=demo_room();s=SceneStore(tmp_path).create(p,c,'Room','hash')
    marker=tmp_path/'must-not-exist'
    monkeypatch.setattr(experiments,'docker_status',lambda:{'available':False,'message':'Docker unavailable'})
    with pytest.raises(ValueError,match='kein Code auf dem Host'):
        experiments.run_experiment(s,f"open({str(marker)!r}, 'w').write('bad')\nresult={{}}")
    assert not marker.exists()

@pytest.mark.skipif(os.getenv('SCANBIM_TEST_DOCKER')!='1',reason='Docker integration opt-in')
def test_real_container_execution_isolation_and_timeout(tmp_path):
    p,c=demo_room();s=SceneStore(tmp_path).create(p,c,'Room','hash')
    result=experiments.run_experiment(s,"import os,socket\ntry:\n    socket.create_connection(('1.1.1.1',80),timeout=.3)\n    blocked=False\nexcept OSError:\n    blocked=True\nresult={'count':len(points), 'key_present':'OPENAI_API_KEY' in os.environ, 'network_blocked':blocked, 'uid':os.getuid(), 'memory_max':int(open('/sys/fs/cgroup/memory.max').read()), 'overlays':[{'kind':'points','vertices':[points.mean(axis=0)]}]}")
    assert result['ok'] and result['result']['count']==len(p)
    assert not result['result']['key_present']
    assert result['result']['network_blocked'] and result['result']['uid']==65534
    assert result['result']['memory_max']==512*1024*1024
    failure=experiments.run_experiment(s,"open('/input/host-write', 'w').write('x')\nresult={}")
    assert not failure['ok'] and 'Read-only' in failure['error']
    timeout=experiments.run_experiment(s,"while True: pass",timeout_s=2)
    assert not timeout['ok'] and 'Zeitlimit' in timeout['error']

@pytest.mark.skipif(os.getenv('SCANBIM_TEST_DOCKER')!='1',reason='Docker integration opt-in')
def test_experiment_streams_all_selected_points_beyond_preview(tmp_path):
    import numpy as np
    from scanbim.streaming import convert_to_cache
    n=250111;source=tmp_path/'source.ply'
    header=f'ply\nformat binary_little_endian 1.0\nelement vertex {n}\nproperty float x\nproperty float y\nproperty float z\nend_header\n'.encode()
    points=np.zeros((n,3),dtype='<f4');points[:,0]=np.arange(n)
    with source.open('wb') as f:f.write(header);f.write(points.tobytes())
    store=SceneStore(tmp_path/'data');staging=tmp_path/'data'/'.import-test'
    meta=convert_to_cache(source,staging);scene=store.adopt_cache(staging,'test','Streaming',meta)
    code="count=0\nlast=-1\nfor block, colors, ids in iter_point_blocks(4096):\n    count+=len(block)\n    last=int(ids[-1])\nresult={'count':count,'sample':len(points),'last_id':last,'source_total':data_info['source_point_count']}"
    result=experiments.run_experiment(scene,code,region={'min':[1000,-1,-1],'max':[n-1000,1,1]})
    assert result['ok'],result
    assert result['result']['count']==n-1999
    assert result['result']['sample']==200000 and result['result']['last_id']==n-1000
    assert result['result']['source_total']==n
