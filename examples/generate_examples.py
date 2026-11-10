"""Reproduce the two synthetic point-cloud fixtures without external assets."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scanbim.cloud import demo_room,write_ply

folder=Path(__file__).resolve().parent
p,c=demo_room();write_ply(folder/'reference-room.ply',p,c)
rng=np.random.default_rng(4)
x,y=np.meshgrid(np.linspace(0,12,150),np.linspace(-2.5,2.5,90))
z=4+.12*np.sin(x*.4)-np.abs(y)*(.5+.05*np.sin(x*.2))
p=np.c_[x.ravel(),y.ravel(),z.ravel()]+rng.normal(0,.004,(x.size,3))
c=np.tile([142,171,183],(len(p),1));write_ply(folder/'variable-roof.ply',p,c)
print('reference-room.ply and variable-roof.ply generated')
