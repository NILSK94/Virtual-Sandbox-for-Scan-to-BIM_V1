import numpy as np
from PIL import Image
from scanbim.app import StartAgent
from scanbim.model import SceneStore
from scanbim.tools import ToolRegistry
from scanbim.timelapse import recordings, export_gif, frame_paths


def test_default_request_targets_bim():
    assert StartAgent().bim is True


def test_analysis_steps_survive_recording_and_gif(tmp_path):
    s=SceneStore(tmp_path).create(np.array([[0.,0,0],[2.,0,0],[0,2.,0],[0,0,2.]]),np.full((4,3),150,dtype=np.uint8),'GIF probe','hash')
    s.event('agent_start',{}, {},actor='system')
    t=ToolRegistry()
    t.call(s,'request_view',{})
    t.call(s,'remember',{'observation':'Analyse ohne Geometrieaenderung'})
    s.event('agent_end',{}, {},actor='system')
    runs=recordings(s)
    folder=s.folder/runs[0]['id']
    assert len(frame_paths(folder)) == 4
    with Image.open(export_gif(folder)) as image:
        assert image.n_frames >= 3
        assert image.width == 960
