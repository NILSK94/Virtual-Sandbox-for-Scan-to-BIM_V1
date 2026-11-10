"""Runs ONLY inside the restricted experiment container. No host fallback."""
import contextlib
import json
import sys
import traceback
from pathlib import Path
import numpy as np
import scipy

class LimitedLog:
    def __init__(self): self.text=""
    def write(self,text):
        self.text=(self.text+str(text))[-16000:]
        return len(text)
    def flush(self): pass

def convert(x):
    if isinstance(x,np.ndarray): return convert(x.tolist())
    if isinstance(x,np.generic): return convert(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    if isinstance(x,dict): return {str(k):convert(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [convert(v) for v in x]
    return x

log=LimitedLog()
try:
    data=np.load("/input/cloud.npz",allow_pickle=False)
    info_path=Path("/input/data-info.json")
    info=json.loads(info_path.read_text()) if info_path.exists() else {}
    full_points=np.load("/dataset/points.npy",mmap_mode="r",allow_pickle=False) if Path("/dataset/points.npy").exists() else None
    full_colors=np.load("/dataset/colors.npy",mmap_mode="r",allow_pickle=False) if full_points is not None else None
    def iter_point_blocks(block_size=131072):
        block_size=int(block_size)
        if not 1024<=block_size<=262144:raise ValueError("Blockgröße: 1024 bis 262144 Punkte.")
        p=full_points if full_points is not None else data["points"]
        c=full_colors if full_points is not None else data["colors"]
        region=info.get("region")
        for start in range(0,len(p),block_size):
            block=p[start:start+block_size];colors=c[start:start+block_size]
            ids=np.arange(start,start+len(block),dtype=np.int64) if full_points is not None else data["source_indices"][start:start+len(block)]
            if region:
                mask=np.all((block>=region["min"])&(block<=region["max"]),axis=1)
                block,colors,ids=block[mask],colors[mask],ids[mask]
            if len(block):yield block,colors,ids
    scope={"points":data["points"],"colors":data["colors"],"np":np,"scipy":scipy,
           "source_indices":data["source_indices"],"data_info":info,"iter_point_blocks":iter_point_blocks,
           "__name__":"__experiment__"}
    source=open("/input/program.py",encoding="utf-8").read()
    with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        exec(compile(source,"experiment.py","exec"),scope)
    if "result" not in scope: raise ValueError("Setze result auf ein JSON-kompatibles Objekt.")
    output={"ok":True,"result":convert(scope["result"]),"log":log.text}
    encoded=json.dumps(output,allow_nan=False)
    if len(encoded.encode())>1_800_000: raise ValueError("Ergebnis größer als 1,8 MB. Bitte reduzieren.")
    print(encoded)
except BaseException as error:
    print(json.dumps({"ok":False,"error":str(error),"traceback":traceback.format_exc()[-12000:],"log":log.text}))
    sys.exit(1)
