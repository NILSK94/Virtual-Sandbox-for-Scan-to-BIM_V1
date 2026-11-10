"""Bounded-memory import. Original files are read-only; normalized points stay on disk."""
from __future__ import annotations
import hashlib
import itertools
import json
import os
import shutil
from pathlib import Path
import numpy as np
from .cloud import PLY_TYPES,_columns

BATCH=131072
PREVIEW_POINTS=200000
MAX_FILE_BYTES=int(float(os.getenv("SCANBIM_MAX_FILE_GB","64"))*1024**3)
SUPPORTED={".ply",".pcd",".xyz",".txt"}

def probe(path):
    path=Path(path)
    if not path.is_file(): raise ValueError("Punktdatei wurde nicht gefunden.")
    if path.suffix.lower() not in SUPPORTED: raise ValueError("Hier werden PLY, PCD und XYZ unterstützt.")
    size=path.stat().st_size
    if size>MAX_FILE_BYTES: raise ValueError("Datei überschreitet SCANBIM_MAX_FILE_GB (Standard: 64 GiB).")
    spec={"path":path,"size":size,"offset":0,"count":None,"format":"ascii","names":None,"dtype":None}
    if path.suffix.lower() in (".xyz",".txt"): return spec
    with path.open("rb") as f:
        if path.suffix.lower()==".ply":
            if f.readline().strip()!=b"ply": raise ValueError("Ungültiger PLY-Kopf.")
            props=[];vertex=False
            for _ in range(4096):
                raw=f.readline(65537)
                if len(raw)>65536 or not raw: raise ValueError("Unvollständiger PLY-Kopf.")
                line=raw.decode("ascii").split()
                if not line: continue
                if line[0]=="format": spec["format"]=line[1]
                elif line[0]=="element":
                    if spec["count"] is None and line[1]!="vertex": raise ValueError("PLY: vertex muss das erste Element sein.")
                    vertex=line[1]=="vertex"
                    if vertex: spec["count"]=int(line[2])
                elif line[0]=="property" and vertex:
                    if line[1]=="list": raise ValueError("PLY-Vertex-Listen werden nicht unterstützt.")
                    props.append((line[2],PLY_TYPES[line[1]]))
                elif line[0]=="end_header": break
            else: raise ValueError("PLY-Kopf zu lang.")
            spec["names"]=[n for n,_ in props]
            if spec["format"] in ("binary_little_endian","binary_big_endian"):
                endian="<" if spec["format"]=="binary_little_endian" else ">"
                spec["dtype"]=np.dtype([(n,endian+t) for n,t in props])
            elif spec["format"]!="ascii": raise ValueError("Unbekanntes PLY-Format.")
        else:
            header={}
            for _ in range(128):
                raw=f.readline(65537)
                if len(raw)>65536 or not raw: raise ValueError("Unvollständiger PCD-Kopf.")
                parts=raw.decode("ascii").split()
                if not parts or parts[0].startswith("#"): continue
                header[parts[0].upper()]=parts[1:]
                if parts[0].upper()=="DATA": break
            fields=header.get("FIELDS",header.get("FIELD",[]))
            counts=list(map(int,header.get("COUNT",["1"]*len(fields))))
            spec["count"]=int(header.get("POINTS",[str(int(header.get("WIDTH",[0])[0])*int(header.get("HEIGHT",[1])[0]))])[0])
            spec["format"]=header["DATA"][0]
            spec["names"]=[k if j==0 else f"{k}_{j}" for k,n in zip(fields,counts) for j in range(n)]
            if spec["format"]=="binary":
                types={"F":"f","I":"i","U":"u"}
                spec["dtype"]=np.dtype([(k,"<"+types[t]+s,(n,)) if n>1 else (k,"<"+types[t]+s)
                    for k,t,s,n in zip(fields,header["TYPE"],header["SIZE"],counts)])
                spec["names"]=fields
            elif spec["format"]!="ascii": raise ValueError("PCD binary_compressed bitte als binary oder ASCII speichern.")
        spec["offset"]=f.tell()
    if spec["count"] is None or not 3<=spec["count"]<=2**40: raise ValueError("Ungültige deklarierte Punktanzahl.")
    if not all(n in spec["names"] for n in ("x","y","z")): raise ValueError("x, y und z fehlen im Dateikopf.")
    if spec["dtype"] is not None and spec["offset"]+spec["count"]*spec["dtype"].itemsize>size:
        raise ValueError("Binärdatei ist kürzer als die deklarierte Punktanzahl.")
    return spec

def _hash_source(spec,progress,cancel):
    digest=hashlib.sha256();consumed=0;count=0;pending=b""
    count_rows=spec["count"] is None
    with spec["path"].open("rb") as f:
        while block:=f.read(4*1024*1024):
            if cancel(): raise InterruptedError("Import abgebrochen.")
            digest.update(block);consumed+=len(block)
            if count_rows:
                rows=(pending+block).split(b"\n");pending=rows.pop()
                if len(pending)>65536: raise ValueError("XYZ-Zeile ist ungewöhnlich lang.")
                count+=sum(bool(row.strip()) and not row.lstrip().startswith(b"#") for row in rows)
            progress(.15*consumed/max(spec["size"],1),"Quelle prüfen" if not count_rows else "XYZ-Punkte zählen",bytes_read=consumed)
    if count_rows:
        count+=int(bool(pending.strip()) and not pending.lstrip().startswith(b"#"))
        spec["count"]=count
    if spec["count"]<3: raise ValueError("Mindestens drei Punkte erforderlich.")
    return digest.hexdigest()

def _blocks(spec,batch_size):
    with spec["path"].open("rb") as f:
        f.seek(spec["offset"])
        if spec["dtype"] is not None:
            # fromfile reads only one bounded block, never the whole source.
            remaining=spec["count"]
            while remaining:
                n=min(batch_size,remaining);a=np.fromfile(f,dtype=spec["dtype"],count=n)
                if len(a)!=n: raise ValueError("Binärdaten wurden während des Imports verkürzt.")
                yield _columns(a,spec["names"]);remaining-=n
        else:
            rows=[];consumed=0
            for line in f:
                if len(line)>65536: raise ValueError("Datenzeile zu lang.")
                if not line.strip() or line.lstrip().startswith(b"#"): continue
                rows.append(line);consumed+=1
                if len(rows)==batch_size or consumed==spec["count"]:
                    a=np.loadtxt(rows,ndmin=2);rows=[]
                    if spec["names"]: yield _columns(a,spec["names"])
                    else:
                        if a.shape[1]<3: raise ValueError("XYZ benötigt mindestens drei Spalten.")
                        yield a[:,:3],a[:,3:6] if a.shape[1]>=6 else None
                if consumed==spec["count"]: break
            if rows:
                a=np.loadtxt(rows,ndmin=2)
                yield _columns(a,spec["names"]) if spec["names"] else (a[:,:3],a[:,3:6] if a.shape[1]>=6 else None)
            if consumed!=spec["count"]: raise ValueError("Punktanzahl stimmt nicht mit der Datei überein.")

def convert_to_cache(path,folder,unit="m",progress=None,cancel=None,batch_size=BATCH):
    progress=progress or (lambda *args,**kwargs:None);cancel=cancel or (lambda:False)
    if unit not in ("m","cm","mm"): raise ValueError("Einheit muss m, cm oder mm sein.")
    spec=probe(path);before=spec["path"].stat();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    digest=_hash_source(spec,progress,cancel);count=spec["count"]
    needed=count*27+64*1024*1024
    if shutil.disk_usage(folder).free<needed:
        raise ValueError(f"Zu wenig freier Cache-Speicher. Benötigt werden ungefähr {needed/1024**3:.1f} GiB zusätzlich zur Quelldatei.")
    progress(.15,"Festplattencache anlegen",point_count=count,cache_bytes=count*27,source_bytes=spec["size"])
    positions=colors=None
    try:
        # Sequential file writes keep even mapped dirty pages out of the
        # importer working set. Only completed arrays are opened with mmap.
        positions=(folder/"points.npy").open("wb");colors=(folder/"colors.npy").open("wb")
        np.lib.format.write_array_header_2_0(positions,{"descr":"<f8","fortran_order":False,"shape":(count,3)})
        np.lib.format.write_array_header_2_0(colors,{"descr":"|u1","fortran_order":False,"shape":(count,3)})
        preview_ids=np.linspace(0,count-1,min(count,PREVIEW_POINTS),dtype=np.int64)
        preview_p=np.empty((len(preview_ids),3),float);preview_c=np.empty((len(preview_ids),3),np.uint8)
        chunks=[];offset=0;lo=np.full(3,np.inf);hi=-lo;scale={"m":1.,"cm":.01,"mm":.001}[unit];has_rgb=True
        for p,c in _blocks(spec,batch_size):
            if cancel(): raise InterruptedError("Import abgebrochen.")
            p=np.asarray(p,dtype=np.float64)*scale;n=len(p)
            if not np.isfinite(p).all(): raise ValueError("NaN/Inf in Punktkoordinaten; Import abgelehnt.")
            if np.max(np.abs(p))>1e9: raise ValueError("Koordinaten außerhalb des unterstützten Bereichs.")
            if offset+n>count: raise ValueError("Datei enthält mehr Punkte als deklariert.")
            if c is None:
                has_rgb=False;c=np.tile(np.array([101,164,181],np.uint8),(n,1))
            else:
                c=np.asarray(c,float)
                if c.size and np.max(c)<=1:c=c*255
                c=np.nan_to_num(np.clip(c,0,255)).astype(np.uint8)
            positions.write(np.ascontiguousarray(p,dtype="<f8").tobytes())
            colors.write(np.ascontiguousarray(c,dtype=np.uint8).tobytes())
            low,high=p.min(axis=0),p.max(axis=0);lo=np.minimum(lo,low);hi=np.maximum(hi,high)
            chunks.append({"start":offset,"count":n,"min":low.tolist(),"max":high.tolist()})
            a,b=np.searchsorted(preview_ids,[offset,offset+n]);local=preview_ids[a:b]-offset
            preview_p[a:b]=p[local];preview_c[a:b]=c[local]
            offset+=n
            # The normalized file is never accumulated as Python lists/arrays.
            progress(.15+.80*offset/count,"Punkte blockweise importieren",points_read=offset,point_count=count)
        if offset!=count:raise ValueError("Punktanzahl stimmt nicht.")
        positions.flush();colors.flush()
    finally:
        for handle in (positions,colors):
            if handle is not None:handle.close()
    after=spec["path"].stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError("Die Quelldatei wurde während des Imports verändert. Bitte erneut importieren.")
    np.savez(folder/"preview.npz",points=preview_p,colors=preview_c,source_indices=preview_ids)
    metadata={"schema":"disk-cloud-1","point_count":count,"source_bytes":spec["size"],"source_hash":digest,
        "bounds":{"min":lo.tolist(),"max":hi.tolist()},"input_unit":unit,"unit":"m",
        "format":spec["format"],"preview_count":len(preview_ids),"batch_points":batch_size,
        "cache_bytes":count*27,"chunks":chunks,"color_source":"source_rgb" if has_rgb else "display_fallback"}
    (folder/"cloud.json").write_text(json.dumps(metadata),encoding="utf-8")
    progress(.98,"Vorschau fertigstellen")
    return metadata
