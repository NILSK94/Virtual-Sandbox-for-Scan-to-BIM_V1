import io
import struct
import numpy as np
import pytest
from scanbim.cloud import read_cloud,demo_room
from scanbim.geometry import detect_planes,find_openings,plane_distance,validate

def test_xyz_units_and_nonfinite():
    p,c,_=read_cloud(b"0 0 0\n1000 0 0\n0 1000 2000\n","x.xyz","mm")
    assert np.allclose(p[2],[0,1,2]) and c.shape==(3,3)
    with pytest.raises(ValueError,match="NaN/Inf"): read_cloud(b"0 0 0\n1 0 0\n0 nan 2","x.xyz")

@pytest.mark.parametrize("binary",[False,True])
def test_ply(binary):
    header=f"ply\nformat {'binary_little_endian' if binary else 'ascii'} 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode()
    values=[(0.,0.,0.,255,0,0),(1.,0.,0.,0,255,0),(0.,1.,2.,0,0,255)]
    body=b"".join(struct.pack('<fffBBB',*v) for v in values) if binary else b"0 0 0 255 0 0\n1 0 0 0 255 0\n0 1 2 0 0 255\n"
    p,c,_=read_cloud(header+body,"x.ply")
    assert np.allclose(p[2],[0,1,2]) and c[2,2]==255

@pytest.mark.parametrize("binary",[False,True])
def test_pcd(binary):
    header=f"VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 3\nHEIGHT 1\nPOINTS 3\nDATA {'binary' if binary else 'ascii'}\n".encode()
    body=struct.pack('<fffffffff',0,0,0,1,0,0,0,1,2) if binary else b"0 0 0\n1 0 0\n0 1 2\n"
    p,_,_=read_cloud(header+body,"x.pcd");assert p[2,2]==2

@pytest.fixture(scope="module")
def room():
    p,c=demo_room();return p,detect_planes(p)

def test_seeded_room_reconstruction_and_holes(room):
    p,planes=room
    assert len(planes)==6 and sum(x['kind']=='wall' for x in planes)==4
    assert detect_planes(p)==planes
    holes=[o for plane in planes for o in find_openings(p,plane)]
    assert len(holes)==2
    door=next(o for o in holes if o['semantic_hint']=='door')
    window=next(o for o in holes if o['semantic_hint']=='window')
    assert door['width']==pytest.approx(1.,abs=.13)
    assert door['height']==pytest.approx(2.15,abs=.13)
    assert window['width']==pytest.approx(1.6,abs=.15)
    assert all(o['status']=='hypothesis' for o in holes)

def test_validation_respects_finite_bounds_and_opening():
    e={'origin':[0,0,0],'u':[1,0,0],'v':[0,0,1],'normal':[0,-1,0],'width':4,'height':3,
       'openings':[{'x':1,'y':0,'width':1,'height':2}]}
    p=np.array([[.5,.01,1],[1.5,0,1],[5,0,1]])
    assert np.allclose(plane_distance(p,e),[.01,.5,1])
