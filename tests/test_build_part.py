import numpy as np
import ifcopenshell
from scanbim.model import SceneStore
from scanbim.tools import ToolRegistry


def test_build_part_exports_semantic_ifc_and_returns_actual_geometry_fit(tmp_path):
    vertices=[[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]
    faces=[[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
    points=np.array(vertices,dtype=float)
    scene=SceneStore(tmp_path).create(points,np.full(points.shape,120,dtype=np.uint8),'Part test','hash')
    scene.state['experiments'].append({'id':'measured','ok':True,'result':{'vertices':vertices,'faces':faces}})
    registry=ToolRegistry()
    result=registry.call(scene,'build_and_check_part',{'name':'Gemessener Traeger','semantic':'beam','experiment_id':'measured'})
    assert result['ok'],result
    assert result['fit']['ifc_geometry'] is True
    assert result['fit']['metrics']['rms_m']<1e-8
    assert result['quality_accepted'] is False
    model=ifcopenshell.open(str(scene.folder/'reconstruction.ifc'))
    assert len(model.by_type('IfcBeam'))==1
    assert len(scene.state['elements'])==1
    again=registry.call(scene,'build_and_check_part',{'name':'Gemessener Traeger','semantic':'beam','experiment_id':'measured',
                         'element_id':result['element_id'],'baseline_id':result['fit']['id']})
    assert again['ok'],again
    assert len(scene.state['elements'])==1
